"""CPU regressions; synthetic accuracy is never an experiment result."""
import argparse
import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path
import torch
from torch import nn
import dino_experiment as d

class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.front = nn.Parameter(torch.tensor([1., 2.]))
        self.backbone = nn.Module()
        self.backbone.norm = nn.Linear(2, 1, bias=False)
        self.head = nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            for p in self.parameters():
                p.fill_(1.)

class ExperimentTests(unittest.TestCase):
    def test_deterministic_resize_matches_bicubic_value_and_gradient(self):
        d.seed_all(17)
        model = d.Backbone(48)
        x = torch.randn(1, 144, 48, requires_grad=True)
        reference = d.F.interpolate(x.transpose(1, 2).reshape(1, 48, 12, 12),
                                    size=(6, 6), mode="bicubic", align_corners=False)
        actual = (model.local_pos_projection @ x).transpose(1, 2).reshape(1, 48, 6, 6)
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
        weights = torch.randn_like(reference)
        ref_grad, = torch.autograd.grad((reference*weights).sum(), x)
        act_grad, = torch.autograd.grad((actual*weights).sum(), x)
        torch.testing.assert_close(act_grad, ref_grad, rtol=1e-5, atol=1e-6)

    def test_reference_self_test(self):
        d.self_test()

    def test_clipping_timing_and_adaptive_resume(self):
        student = Toy()
        teacher = copy.deepcopy(student)
        with torch.no_grad():
            for name, p in teacher.named_parameters():
                p.mul_({"front": .999, "back": .99, "head": .01}[d.group_name(name)])
        for method in "BCG":
            with self.subTest(method=method):
                t = copy.deepcopy(teacher)
                updater = d.TeacherUpdater(student, t, method)
                for epoch in range(5):
                    stats = updater.step(.004, epoch)
                    self.assertTrue(all(stats["ratio_"+g] == 1 for g in d.GROUPS))
                self.assertEqual(updater.n, 0)
                stats = updater.step(.004, 5)
                timing = updater.last_timing
                self.assertTrue(all(v >= 0 for v in timing.values()))
                self.assertAlmostEqual(timing["algorithm_s"],
                    timing["ema_s"]+timing["coefficient_s"]+
                    (timing["gap_s"] if method != "B" else 0.))
                if method != "B":
                    self.assertEqual(stats["clip_low_front"].item(), 1)
                    self.assertEqual(stats["clip_high_head"].item(), 1)
                    self.assertEqual(updater.n, 1)
                if method == "C":
                    torch.testing.assert_close(stats["ratio_front"], stats["ratio_head"])
                resumed_t = copy.deepcopy(t)
                resumed = d.TeacherUpdater(student, resumed_t, method)
                resumed.load_state_dict(copy.deepcopy(updater.state_dict()))
                updater.step(.003, 6)
                resumed.step(.003, 6)
                for p, q in zip(t.parameters(), resumed_t.parameters()):
                    torch.testing.assert_close(p, q, rtol=0, atol=0)
                torch.testing.assert_close(updater.dbar, resumed.dbar, rtol=0, atol=0)

    def test_group_partition(self):
        model = d.DINO(smoke=True)
        for name, _ in model.named_parameters():
            if "blocks." in name:
                expected = "front" if int(name.split(".")[2]) < 6 else "back"
                self.assertEqual(d.group_name(name), expected)
        for name in ("backbone.cls_token", "backbone.pos_embed", "backbone.patch_embed.weight"):
            self.assertEqual(d.group_name(name), "front")
        self.assertEqual(d.group_name("backbone.norm.weight"), "back")
        self.assertEqual(d.group_name("head.last_weight"), "head")

    def test_full_training_resume_at_adaptive_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            def args(out, until, resume=False):
                # Isolate exact checkpoint replay from CPU parallel reduction rounding.
                # Production thread counts remain unchanged in server.py.
                return argparse.Namespace(device="cpu", threads=1, workers=0,
                    batch=10, eval_batch=10, gpu_memory_gib=10.5, until=until,
                    momentum=.996, clip_low=.5, clip_high=2., data=str(Path(folder)/"data"),
                    smoke=True, seed=0, method="G", out=str(Path(folder)/out), resume=resume)
            with contextlib.redirect_stdout(io.StringIO()):
                a = args("continuous", 6)
                d.configure_runtime(a)
                d.train(a)
                d.train(args("resumed", 5))
                d.train(args("resumed", 6, True))
            full = d.load_checkpoint(Path(folder)/"continuous/SMOKE_G_seed0/last.pt")
            resumed = d.load_checkpoint(Path(folder)/"resumed/SMOKE_G_seed0/last.pt")
            for model in ("student", "teacher"):
                for name, tensor in full[model].items():
                    torch.testing.assert_close(tensor, resumed[model][name], rtol=0, atol=0)
            torch.testing.assert_close(full["center"], resumed["center"], rtol=0, atol=0)
            torch.testing.assert_close(full["updater"]["dbar"], resumed["updater"]["dbar"], rtol=0, atol=0)
            self.assertEqual(full["step"], resumed["step"])
            self.assertEqual(full["updater"]["n"], 2)
            for r, q in zip(full["history"], resumed["history"]):
                for key in ("loss", "gap_front", "gap_back", "gap_head", "ratio_front", "ratio_head"):
                    self.assertEqual(r[key], q[key])
            self.assertEqual(full["history"][-1]["teacher_knn"],
                             resumed["history"][-1]["teacher_knn"])
            changed = args("resumed", 7, True)
            changed.gpu_memory_gib = 9.
            with self.assertRaises(ValueError):
                d.train(changed)

if __name__ == "__main__":
    unittest.main(verbosity=2)
