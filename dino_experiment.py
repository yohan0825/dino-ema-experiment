"""DINO 학생탐구: B/C/G/R, single-GPU FP32, Python >=3.10.
Run: python dino_experiment.py --help
The smoke test uses synthetic data: its accuracy is NOT an experiment result.
"""
import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

VERSION = "1.1"
GROUPS = ("front", "back", "head")
EPS = 1e-8


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"].cpu())
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in s["cuda"]])


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def atomic_checkpoint(path, obj):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def load_checkpoint(path):
    # Only load checkpoints made by this script and trusted by you.
    return torch.load(path, map_location="cpu", weights_only=False)


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, n, 3, 3, d // 3).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        x = x + self.proj(y.transpose(1, 2).reshape(b, n, d))
        return x + self.mlp(self.norm2(x))


class Backbone(nn.Module):
    def __init__(self, dim=192):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, 8, 8)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 145, dim))
        self.blocks = nn.ModuleList([Block(dim) for _ in range(12)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        # Bicubic resize is linear. Precompute its fixed 12x12 -> 6x6
        # operator on CPU; matmul avoids nondeterministic CUDA resize backward.
        basis = torch.eye(144).reshape(144, 1, 12, 12)
        resize = F.interpolate(basis, size=(6, 6), mode="bicubic", align_corners=False)
        self.register_buffer("local_pos_projection",
                             resize.reshape(144, 36).T.contiguous(), persistent=False)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, image):
        x = self.patch_embed(image)
        b, d, h, w = x.shape
        pos = self.pos_embed[:, 1:].reshape(1, 12, 12, d).permute(0, 3, 1, 2)
        if (h, w) == (6, 6):
            pos = (self.local_pos_projection @ self.pos_embed[:, 1:]).transpose(1, 2).reshape(1, d, 6, 6)
        elif (h, w) != (12, 12):
            raise ValueError("This experiment supports only 96px global / 48px local crops.")
        pos = torch.cat([self.pos_embed[:, :1], pos.flatten(2).transpose(1, 2)], 1)
        x = torch.cat([self.cls_token.expand(b, -1, -1), x.flatten(2).transpose(1, 2)], 1) + pos
        for block in self.blocks:
            x = block(x)
        return self.norm(x)[:, 0]


class Head(nn.Module):
    def __init__(self, dim=192, out_dim=4096, hidden=2048, bottleneck=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, bottleneck))
        # Unit row norm is weight normalization with fixed scale g=1.
        self.last_weight = nn.Parameter(torch.empty(out_dim, bottleneck))
        nn.init.trunc_normal_(self.last_weight, std=0.02)

    def forward(self, x):
        x = F.normalize(self.mlp(x), dim=-1, eps=1e-6)
        return F.linear(x, F.normalize(self.last_weight, dim=1, eps=1e-6))


class DINO(nn.Module):
    def __init__(self, smoke=False):
        super().__init__()
        dim = 48 if smoke else 192
        self.backbone = Backbone(dim)
        self.head = Head(dim, 64 if smoke else 4096, 96 if smoke else 2048,
                         32 if smoke else 256)
        self.apply(self.initialize)

    @staticmethod
    def initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, crops):
        # Concatenate same-sized views, preserving global0/global1/local0/local1 order.
        globals_ = self.backbone(torch.cat(crops[:2]))
        if len(crops) == 2:
            return self.head(globals_).chunk(2)
        locals_ = self.backbone(torch.cat(crops[2:]))
        return self.head(torch.cat([globals_, locals_])).chunk(4)


def dino_loss(student, teacher, center):
    targets = [F.softmax((x.detach().float() - center) / 0.04, dim=-1) for x in teacher]
    predictions = [F.log_softmax(x.float() / 0.1, dim=-1) for x in student]
    terms = [-(p * predictions[j]).sum(-1).mean()
             for i, p in enumerate(targets) for j in range(4) if i != j]
    return torch.stack(terms).mean()


def group_name(name):
    if name.startswith("head."):
        return "head"
    if name.startswith("backbone.blocks."):
        return "front" if int(name.split(".")[2]) < 6 else "back"
    if name.startswith("backbone.norm."):
        return "back"
    return "front"


class TeacherUpdater:
    """Document equations: statistics begin at epoch 6, after optimizer.step."""
    def __init__(self, student, teacher, method, low=0.5, high=2.0):
        self.method, self.low, self.high = method, low, high
        tp = dict(teacher.named_parameters())
        self.pairs = {g: [] for g in GROUPS}
        for name, p in student.named_parameters():
            self.pairs[group_name(name)].append((name, p, tp[name]))
        self.count = {g: sum(p.numel() for _, p, _ in ps) for g, ps in self.pairs.items()}
        device = next(student.parameters()).device
        self.dbar = torch.zeros(3, device=device)
        self.v = {}
        self.n = 0

    def state_dict(self):
        return dict(dbar=self.dbar, v=self.v, n=self.n)

    def load_state_dict(self, state):
        device = self.dbar.device
        self.dbar = state["dbar"].to(device)
        self.v = {k: v.to(device, dtype=torch.float32) for k, v in state["v"].items()}
        self.n = state["n"]

    @torch.no_grad()
    def step(self, a, epoch):
        if self.method in ("B", "C", "G"):
            return self.step_bcg(a, epoch)
        active = epoch >= 5 and self.method != "B"  # epoch is zero-based.
        gaps = []
        for g, ps in self.pairs.items():
            delta2 = sum((s.float() - t.float()).square().sum() for _, s, t in ps)
            student2 = sum(s.float().square().sum() for _, s, _ in ps)
            gaps.append((delta2 / self.count[g]).sqrt() /
                        ((student2 / self.count[g]).sqrt() + EPS))
        gaps = torch.stack(gaps)
        ratios = torch.ones_like(gaps)
        if active and self.method in ("C", "G"):
            self.dbar.mul_(0.99).add_(gaps, alpha=0.01)
            raw = self.dbar / (self.dbar.mean() + EPS)
            ratios = torch.where(self.dbar.sum() == 0, torch.ones_like(raw),
                                 raw.clamp(self.low, self.high))
            if self.method == "C":
                sizes = ratios.new_tensor([self.count[g] for g in GROUPS])
                ratios = ((ratios * sizes).sum() / sizes.sum()).expand(3)
        if active and self.method == "R":
            self.n += 1
        logs = {}
        for j, (g, ps) in enumerate(self.pairs.items()):
            low_count = gaps.new_zeros(())
            high_count = gaps.new_zeros(())
            ratio_sum = gaps.new_zeros(())
            v_mean = gaps.new_zeros(())
            if active and self.method == "R":
                correction = 1 - 0.99 ** self.n
                for name, s, t in ps:
                    if name not in self.v:
                        self.v[name] = torch.zeros_like(s, dtype=torch.float32)
                    delta = s.float() - t.float()
                    self.v[name].mul_(0.99).addcmul_(delta, delta, value=0.01)
                # Element-weighted mean within group, NOT mean of tensor means.
                v_mean = sum(v.sum() for name, _, _ in ps
                             for v in [self.v[name]]) / (correction * self.count[g])
                q = v_mean.sqrt()
                for name, s, t in ps:
                    root = (self.v[name] / correction).sqrt()
                    raw = (q + EPS) / (root + EPS)
                    ratio = raw.clamp(self.low, self.high)
                    low_count += (raw < self.low).sum()
                    high_count += (raw > self.high).sum()
                    ratio_sum += ratio.sum()
                    t.add_((s - t) * ratio * a)
                mean_ratio = ratio_sum / self.count[g]
            else:
                mean_ratio = ratios[j]
                for _, s, t in ps:
                    t.add_((s - t) * (a * mean_ratio))
            logs.update({f"gap_{g}": gaps[j], f"a_{g}": mean_ratio * a,
                         f"ratio_{g}": mean_ratio, f"vhat_mean_{g}": v_mean,
                         f"clip_low_{g}": low_count / self.count[g],
                         f"clip_high_{g}": high_count / self.count[g]})
        return logs


    @torch.no_grad()
    def step_bcg(self, a, epoch):
        """C/G gap cost is algorithmic after warmup; B gap cost is diagnostic."""
        device = self.dbar.device
        active = epoch >= 5 and self.method in ("C", "G")
        sync(device)
        started = time.perf_counter()
        gaps = []
        for g, ps in self.pairs.items():
            delta2 = sum((s-t).square().sum() for _, s, t in ps)
            student2 = sum(s.square().sum() for _, s, _ in ps)
            gaps.append((delta2/self.count[g]).sqrt() /
                        ((student2/self.count[g]).sqrt()+EPS))
        gaps = torch.stack(gaps)
        sync(device)
        gap_s = time.perf_counter()-started
        started = time.perf_counter()
        ratios = torch.ones_like(gaps)
        raw = ratios
        if active:
            self.n += 1
            self.dbar.mul_(0.99).add_(gaps, alpha=0.01)
            raw = self.dbar/(self.dbar.mean()+EPS)
            raw = torch.where(self.dbar.sum() == 0, torch.ones_like(raw), raw)
            ratios = raw.clamp(self.low, self.high)
            if self.method == "C":
                sizes = ratios.new_tensor([self.count[g] for g in GROUPS])
                ratios = ((ratios*sizes).sum()/sizes.sum()).expand(3)
        sync(device)
        coefficient_s = time.perf_counter()-started
        started = time.perf_counter()
        for j, ps in enumerate(self.pairs.values()):
            coefficient = a*ratios[j]
            for _, s, t in ps:
                t.add_((s-t)*coefficient)
        sync(device)
        ema_s = time.perf_counter()-started
        started = time.perf_counter()
        logs = {}
        for j, g in enumerate(GROUPS):
            logs.update({f"gap_{g}": gaps[j], f"ratio_{g}": ratios[j],
                         f"raw_ratio_{g}": raw[j], f"a_{g}": a*ratios[j],
                         f"clip_low_{g}": (raw[j] <= self.low).float(),
                         f"clip_high_{g}": (raw[j] >= self.high).float()})
        sync(device)
        logging_s = time.perf_counter()-started
        self.last_timing = dict(
            gap_s=gap_s, coefficient_s=coefficient_s, ema_s=ema_s,
            algorithm_s=ema_s+coefficient_s+(gap_s if active else 0.),
            diagnostic_s=logging_s+(0. if active else gap_s))
        return logs


class CropTransform:
    def __init__(self):
        from torchvision import transforms as T
        interpolation = T.InterpolationMode.BICUBIC
        normalize = [T.ToTensor(), T.Normalize((0.485, 0.456, 0.406),
                                              (0.229, 0.224, 0.225))]
        color = [T.RandomHorizontalFlip(),
                 T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                 T.RandomGrayscale(p=0.2)]
        def transform(size, scale, blur, solarize):
            return T.Compose([T.RandomResizedCrop(size, scale=scale, interpolation=interpolation)]
                             + color + [T.RandomApply([T.GaussianBlur(9, (0.1, 2.0))], p=blur),
                                        T.RandomSolarize(128, p=solarize)] + normalize)
        self.transforms = [transform(96, (0.4, 1.0), 1.0, 0.0),
                           transform(96, (0.4, 1.0), 0.1, 0.2),
                           transform(48, (0.05, 0.4), 0.5, 0.0),
                           transform(48, (0.05, 0.4), 0.5, 0.0)]

    def __call__(self, image):
        return [t(image) for t in self.transforms]


class ViewDataset(Dataset):
    def __init__(self, base, indices, train=False, smoke=False):
        self.base, self.indices, self.train, self.smoke = base, indices, train, smoke
        if not smoke:
            from torchvision import transforms as T
            self.transform = CropTransform() if train else T.Compose([
                T.Resize((96, 96), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        idx = self.indices[index]
        if self.smoke:
            gen = torch.Generator().manual_seed(90000 + idx)
            image = torch.rand(3, 96, 96, generator=gen)
            if self.train:
                return [image, image.flip(-1), image[:, :48, :48], image[:, -48:, -48:]], idx % 10
            return image, idx % 10
        image, label = self.base[idx]
        return self.transform(image), label


def make_data(root, smoke=False, test=False):
    if smoke:
        return None, dict(train=list(range(20)), val=list(range(20, 30))), None
    from torchvision.datasets import CIFAR10
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    base = CIFAR10(str(root), train=True, download=True)
    split_path = root / "split_seed2026_10000_5000.json"
    labels = np.asarray(base.targets)
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
    else:
        gen = np.random.default_rng(2026)
        split = dict(train=[], val=[])
        for cls in range(10):
            indices = gen.permutation(np.flatnonzero(labels == cls))
            split["train"].extend(indices[:1000].tolist())
            split["val"].extend(indices[1000:1500].tolist())
        atomic_json(split_path, split)
    tr, va = split["train"], split["val"]
    assert len(tr) == len(set(tr)) == 10000 and len(va) == len(set(va)) == 5000
    assert not set(tr).intersection(va)
    assert min(tr + va) >= 0 and max(tr + va) < len(base)
    assert np.all(np.bincount(labels[tr], minlength=10) == 1000)
    assert np.all(np.bincount(labels[va], minlength=10) == 500)
    # CIFAR archive includes test files; never instantiate/read the test set until requested.
    testbase = CIFAR10(str(root), train=False, download=False) if test else None
    return base, split, testbase


def worker_seed(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def loader(ds, batch, shuffle, workers, seed):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      drop_last=shuffle, pin_memory=torch.cuda.is_available(),
                      generator=torch.Generator().manual_seed(seed),
                      worker_init_fn=worker_seed, persistent_workers=False)


@torch.no_grad()
def features(model, dl, device, center):
    model.eval()
    xs, ys = [], []
    ent_sum = 0.0
    prob_sum = None
    for image, label in dl:
        f = model.backbone(image.to(device, non_blocking=True))
        prob = ((model.head(f).float() - center) / 0.04).softmax(-1)
        ent_sum += (-(prob * prob.clamp_min(1e-12).log()).sum(-1)).sum().item()
        prob_sum = prob.sum(0) if prob_sum is None else prob_sum + prob.sum(0)
        xs.append(F.normalize(f.float(), dim=-1).cpu())
        ys.append(label)
    x, y = torch.cat(xs), torch.cat(ys)
    avg = prob_sum / len(y)
    return x, y, dict(feature_variance=x.var(0, unbiased=False).mean().item(),
                      entropy_individual=ent_sum / len(y),
                      entropy_batch=(-(avg * avg.clamp_min(1e-12).log()).sum()).item())


@torch.no_grad()
def knn(train_x, train_y, query_x, query_y, device):
    # Cosine k=20, unweighted majority vote, ties -> smallest class index.
    assert len(train_x) >= 20
    ref = train_x.to(device)
    labels = train_y.to(device)
    correct = 0
    for start in range(0, len(query_x), 256):
        q = query_x[start:start + 256].to(device)
        ids = (q @ ref.T).topk(20, dim=1).indices
        votes = F.one_hot(labels[ids], num_classes=10).sum(1)
        pred = votes.argmax(1).cpu()
        correct += (pred == query_y[start:start + len(q)]).sum().item()
    return 100 * correct / len(query_y)


def evaluate(student, teacher, train_ds, val_ds, device, center, workers, batch=64):
    tr = loader(train_ds, batch, False, workers, 100)
    va = loader(val_ds, batch, False, workers, 101)
    output, val_features = {}, []
    for name, model in (("teacher", teacher), ("student", student)):
        train_x, train_y, _ = features(model, tr, device, center)
        val_x, val_y, diagnostics = features(model, va, device, center)
        output[name + "_knn"] = knn(train_x, train_y, val_x, val_y, device)
        output.update({name + "_" + key: value for key, value in diagnostics.items()})
        val_features.append(val_x[:256])
    output["cls_cosine_gap"] = (1 - (val_features[0] * val_features[1]).sum(1)).mean().item()
    return output


def cosine(start, end, step, length):
    return end + (start - end) * (1 + math.cos(math.pi * step / max(length - 1, 1))) / 2


def schedules(step, steps_per_epoch, batch, momentum):
    total, warmup = 100 * steps_per_epoch, 10 * steps_per_epoch
    peak = 0.0005 * batch / 256
    lr = peak * step / max(warmup - 1, 1) if step < warmup else cosine(
        peak, 1e-6, step - warmup, total - warmup)
    return lr, cosine(0.04, 0.4, step, total), cosine(momentum, 1.0, step, total)


def environment():
    packages = {}
    for pkg in ("torch", "torchvision", "numpy", "Pillow", "matplotlib"):
        try:
            packages[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            packages[pkg] = "not installed"
    return dict(python=sys.version, platform=platform.platform(), packages=packages,
                cuda_runtime=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                implementation_version=VERSION,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                dino_reference="https://github.com/facebookresearch/dino/blob/main/main_dino.py",
                note="Independent implementation; no downloaded DINO source or pretrained weights.")


def choose_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다. 학교 GPU 서버의 Python을 선택하세요. "
                           "CPU 점검은 --device cpu 또는 smoke 명령으로 가능합니다.")
    return torch.device(name)


def configure_runtime(args):
    if args.threads < 1 or args.workers < 0 or args.eval_batch < 1:
        raise ValueError("threads/eval-batch must be positive; workers nonnegative")
    if not (0 < args.gpu_memory_gib <= 12):
        raise ValueError("GPU allocator budget must be positive and at most 12 GiB")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == "cuda":
        device = choose_device(args.device)
        # The memory-fraction API rejects torch.device('cuda') without an index.
        # Resolve the current logical device, respecting CUDA_VISIBLE_DEVICES.
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device_index).total_memory
        budget = args.gpu_memory_gib*2**30
        if budget > total:
            raise ValueError("GPU allocator budget exceeds device capacity")
        torch.cuda.set_per_process_memory_fraction(budget/total, device_index)


def train(args):
    device = choose_device(args.device)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not (1 <= args.until <= 100 and args.batch >= 1):
        raise ValueError("--until은 1~100, --batch는 양수여야 합니다.")
    if not (0 < args.momentum < 1 and 0 < args.clip_low <= 1 <= args.clip_high
            and (1 - args.momentum) * args.clip_high < 1):
        raise ValueError("momentum/clip 범위가 잘못되었습니다.")
    base, split, _ = make_data(args.data, args.smoke)
    datasets_ = [ViewDataset(base, split["train"], True, args.smoke),
                 ViewDataset(base, split["train"], False, args.smoke),
                 ViewDataset(base, split["val"], False, args.smoke)]
    nsteps = len(datasets_[0]) // args.batch
    if not nsteps:
        raise ValueError("batch size가 학습 이미지 수보다 큽니다.")
    seed_all(args.seed)
    run_dir = Path(args.out) / (("SMOKE_" if args.smoke else "") + args.method + "_seed" + str(args.seed))
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "last.pt"
    if checkpoint_path.exists() and not args.resume:
        raise FileExistsError(f"{checkpoint_path}가 있습니다. --resume을 추가하거나 --out을 바꾸세요.")
    if args.resume and not checkpoint_path.exists():
        raise FileNotFoundError(f"재개할 checkpoint가 없습니다: {checkpoint_path}")
    config = dict(version=VERSION, method=args.method, seed=args.seed, batch=args.batch,
                  threads=args.threads, eval_batch=args.eval_batch,
                  gpu_memory_gib=args.gpu_memory_gib,
                  momentum=args.momentum, clip_low=args.clip_low, clip_high=args.clip_high,
                  workers=args.workers, smoke=args.smoke, schedule_epochs=100,
                  split_sha256=hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest(),
                  precision="fp32", eval_every=2 if args.smoke else 5,
                  model="tiny-smoke" if args.smoke else "vit-tiny-p8-d192-depth12-head4096",
                  drop_path=0.0, data=str(Path(args.data).resolve()))
    student = DINO(args.smoke).to(device)
    teacher = copy.deepcopy(student).to(device).eval()
    teacher.requires_grad_(False)
    decayed = [p for name, p in student.named_parameters() if p.ndim > 1 and not name.endswith(".bias")]
    other = [p for name, p in student.named_parameters() if p.ndim <= 1 or name.endswith(".bias")]
    opt = torch.optim.AdamW([dict(params=decayed), dict(params=other, weight_decay=0)], lr=0)
    updater = TeacherUpdater(student, teacher, args.method, args.clip_low, args.clip_high)
    center = torch.zeros(1, 64 if args.smoke else 4096, device=device)
    history, first_epoch, step, train_seconds, eval_seconds, previous_wall = [], 0, 0, 0., 0., 0.
    if args.resume:
        ck = load_checkpoint(checkpoint_path)
        if ck["config"] != config:
            raise ValueError("저장 설정과 현재 설정이 다릅니다. batch/workers/경로/계수 등을 유지하세요.")
        current_hash = environment()["script_sha256"]
        if ck["environment"]["script_sha256"] != current_hash:
            raise ValueError("학습 코드가 변경되었습니다. 기존 실험 재개 대신 별도 --out으로 실행하세요.")
        student.load_state_dict(ck["student"])
        teacher.load_state_dict(ck["teacher"])
        opt.load_state_dict(ck["optimizer"])
        updater.load_state_dict(ck["updater"])
        center.copy_(ck["center"])
        first_epoch, step = ck["epoch"], ck["step"]
        history = ck["history"]
        train_seconds, eval_seconds, previous_wall = ck["train_seconds"], ck["eval_seconds"], ck["wall_seconds"]
        restore_rng(ck["rng"])
    else:
        atomic_json(run_dir / "config.json", config)
        atomic_json(run_dir / "environment.json", environment())
    if first_epoch >= args.until:
        print("요청한 epoch까지 이미 완료했습니다.", flush=True)
        return
    session_start = time.perf_counter()
    print(f"{run_dir} | {device} | {nsteps} steps/epoch | FP32", flush=True)
    for epoch in range(first_epoch, args.until):
        # Explicit epoch seed + separate loader generator match augmentations/order on resume.
        seed_all(args.seed * 100003 + epoch)
        dl = loader(datasets_[0], args.batch, True, args.workers, args.seed * 100003 + epoch)
        student.train()
        teacher.eval()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sync(device)
        start = time.perf_counter()
        update_seconds, step_seconds = 0., 0.
        timing_sums = {}
        sums = {}
        loss_sum = 0.
        for views, _ in dl:  # Labels are NEVER used by self-supervised training.
            sync(device)
            step_start = time.perf_counter()
            views = [v.to(device, non_blocking=True) for v in views]
            lr, wd, momentum = schedules(step, nsteps, args.batch, args.momentum)
            for pg in opt.param_groups:
                pg["lr"] = lr
            opt.param_groups[0]["weight_decay"] = wd
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_out = teacher(views[:2])
            student_out = student(views)
            loss = dino_loss(student_out, teacher_out, center)
            if not torch.isfinite(loss):
                raise RuntimeError("loss가 NaN/Inf입니다. 기준선 설정을 점검하세요.")
            loss.backward()
            if epoch < 1:
                student.head.last_weight.grad = None
            nn.utils.clip_grad_norm_(student.parameters(), 3.0, error_if_nonfinite=True)
            opt.step()
            # FP32 only: no GradScaler, no skipped optimizer steps.
            sync(device)
            update_start = time.perf_counter()
            stats = updater.step(1 - momentum, epoch)
            sync(device)
            update_seconds += time.perf_counter() - update_start
            for key, value in getattr(updater, "last_timing", {}).items():
                timing_sums[key] = timing_sums.get(key, 0.) + value
            with torch.no_grad():
                center.mul_(0.9).add_(torch.cat(teacher_out).mean(0, keepdim=True), alpha=0.1)
            for key, value in stats.items():
                sums[key] = sums.get(key, torch.zeros_like(value)) + value
            loss_sum += loss.item()
            step += 1
            sync(device)
            step_seconds += time.perf_counter() - step_start
        sync(device)
        elapsed = time.perf_counter() - start
        train_seconds += elapsed
        row = dict(epoch=epoch + 1, loss=loss_sum / nsteps, lr=lr, weight_decay=wd,
                   momentum=momentum, train_epoch_s=elapsed, train_s=train_seconds,
                   step_ms=1000 * step_seconds / nsteps,
                   teacher_update_ms=1000 * update_seconds / nsteps,
                   train_peak_vram_mb=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.,
                   adaptive_updates=updater.n)
        row.update({key: value.item() / nsteps for key, value in sums.items()})
        row.update({"teacher_" + key[:-2] + "_ms": 1000 * value / nsteps
                    for key, value in timing_sums.items()})
        if (epoch + 1) % config["eval_every"] == 0:
            sync(device)
            eval_start = time.perf_counter()
            row.update(evaluate(student, teacher, datasets_[1], datasets_[2], device, center, args.workers, args.eval_batch))
            sync(device)
            row["eval_epoch_s"] = time.perf_counter() - eval_start
            eval_seconds += row["eval_epoch_s"]
        row["eval_s"] = eval_seconds
        row["wall_s"] = previous_wall + time.perf_counter() - session_start
        history.append(row)
        atomic_checkpoint(checkpoint_path, dict(
            config=config, environment=environment(), student=student.state_dict(),
            teacher=teacher.state_dict(), optimizer=opt.state_dict(), scaler=None,
            updater=updater.state_dict(), center=center, epoch=epoch + 1, step=step,
            rng=rng_state(), history=history, train_seconds=train_seconds,
            eval_seconds=eval_seconds, wall_seconds=row["wall_s"]))
        write_csv(run_dir / "metrics.csv", history)
        accuracy = f" teacher kNN={row['teacher_knn']:.2f}%" if "teacher_knn" in row else ""
        print(f"epoch {epoch + 1:3d}/100 loss={row['loss']:.4f}{accuracy} "
              f"train={elapsed:.1f}s update={row['teacher_update_ms']:.2f}ms", flush=True)
    atomic_json(run_dir / "session_timing.json",
                dict(active_wall_seconds=previous_wall + time.perf_counter() - session_start,
                     train_seconds=train_seconds, eval_seconds=eval_seconds,
                     note="wall includes evaluation/log/checkpoint I/O; excludes paused time and data setup."))
    print(f"저장 완료: {checkpoint_path}", flush=True)


def freeze_target(args):
    root = Path(args.out)
    ck = load_checkpoint(root / "B_seed0" / "last.pt")
    if ck["epoch"] != 20:
        raise ValueError("기준선 seed 0을 정확히 20 epochs까지 실행한 직후 목표를 고정하세요.")
    row = next(r for r in ck["history"] if r["epoch"] == 20)
    target = dict(teacher_knn=row["teacher_knn"], source="B_seed0", epoch=20,
                  config=ck["config"])
    path = root / "target.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != target:
            raise ValueError("이미 다른 목표값이 고정되어 있습니다. 별도 --out을 사용하세요.")
    else:
        atomic_json(path, target)
    print(f"목표 검증 kNN 정확도 고정: {target['teacher_knn']:.2f}%")


def linear_probe(train_x, train_y, val_x, val_y, device, seed):
    seed_all(seed)
    head = nn.Linear(train_x.shape[1], 10).to(device)
    opt = torch.optim.SGD(head.parameters(), lr=0.1, momentum=0.9, weight_decay=0)
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(100):
        opt.param_groups[0]["lr"] = cosine(0.1, 0., epoch, 100)
        head.train()
        for ids in torch.randperm(len(train_y), generator=generator).split(256):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(train_x[ids].to(device)), train_y[ids].to(device))
            loss.backward()
            opt.step()
    return head, classifier_accuracy(head, val_x, val_y, device)


@torch.no_grad()
def classifier_accuracy(head, x, y, device):
    head.eval()
    correct = 0
    for start in range(0, len(y), 256):
        pred = head(x[start:start + 256].to(device)).argmax(-1).cpu()
        correct += (pred == y[start:start + 256]).sum().item()
    return 100 * correct / len(y)


def final_evaluation(args):
    device = choose_device(args.device)
    torch.set_num_threads(args.threads)
    ck_path = Path(args.out) / (args.method + "_seed" + str(args.seed)) / "last.pt"
    ck = load_checkpoint(ck_path)
    if ck["epoch"] != 100 or ck["config"]["smoke"]:
        raise ValueError("최종 평가는 본 실험 100 epoch checkpoint만 사용합니다.")
    base, split, testbase = make_data(ck["config"]["data"], test=args.test)
    fingerprint = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
    if fingerprint != ck["config"]["split_sha256"]:
        raise ValueError("학습 때의 데이터 분할과 다릅니다.")
    train_ds, val_ds = [ViewDataset(base, split[k]) for k in ("train", "val")]
    out = dict(epoch=100, test_evaluated=args.test, method=args.method, seed=args.seed)
    for name in ("teacher", "student"):
        model = DINO().to(device)
        model.load_state_dict(ck[name])
        model.eval().requires_grad_(False)
        center = ck["center"].to(device)
        tx, ty, _ = features(model, loader(train_ds, args.eval_batch, False, args.workers, 100), device, center)
        vx, vy, _ = features(model, loader(val_ds, args.eval_batch, False, args.workers, 101), device, center)
        head, val_accuracy = linear_probe(tx, ty, vx, vy, device, args.seed)
        out[name + "_linear_val"] = val_accuracy
        out[name + "_knn_val"] = knn(tx, ty, vx, vy, device)
        torch.save(head.cpu().state_dict(), ck_path.parent / (name + "_linear.pt"))
        head.to(device)
        if args.test:
            ds = ViewDataset(testbase, list(range(len(testbase))))
            x, y, _ = features(model, loader(ds, args.eval_batch, False, args.workers, 102), device, center)
            out[name + "_linear_test"] = classifier_accuracy(head, x, y, device)
            out[name + "_knn_test"] = knn(tx, ty, x, y, device)
    atomic_json(ck_path.parent / ("final_test.json" if args.test else "linear_val.json"), out)
    print(json.dumps(out, indent=2))


def comparable(config):
    return {k: v for k, v in config.items() if k not in ("method", "seed")}


def summarize(args):
    root = Path(args.out)
    target_path = root / "target.json"
    target = json.loads(target_path.read_text(encoding="utf-8")) if target_path.exists() else None
    runs, rows, common = [], [], None
    for path in sorted(root.glob("[BCGR]_seed*/metrics.csv")):
        cfg = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
        if common is None:
            common = comparable(cfg)
        if comparable(cfg) != common or (target and comparable(target["config"]) != common):
            raise ValueError("비교할 실험들의 공통 설정이 다릅니다. 별도 폴더에서 비교하세요.")
        with path.open(encoding="utf-8-sig", newline="") as f:
            history = [{k: float(v) for k, v in row.items() if v != ""} for row in csv.DictReader(f)]
        evals = [r for r in history if "teacher_knn" in r]
        if not evals:
            continue
        last = evals[-1]
        hit = next((r for r in evals if r["teacher_knn"] > target["teacher_knn"]), None) if target else None
        row = dict(method=cfg["method"], seed=cfg["seed"], epoch=int(last["epoch"]),
                   teacher_knn=last["teacher_knn"], student_knn=last["student_knn"],
                   train_s=last["train_s"], eval_s=last["eval_s"],
                   target_status=("reached" if hit else "not_reached") if target else "not_fixed",
                   target_epoch=int(hit["epoch"]) if hit else "",
                   target_train_s=hit["train_s"] if hit else "",
                   peak_vram_mb=max(r["train_peak_vram_mb"] for r in history))
        for name in ("linear_val.json", "final_test.json"):
            file = path.parent / name
            if file.exists():
                final = json.loads(file.read_text(encoding="utf-8"))
                row.update({k: v for k, v in final.items() if "_linear_" in k or "_knn_" in k})
        rows.append(row)
        runs.append((cfg, history, evals))
    if not rows:
        raise ValueError("비교할 본 실험 metrics.csv가 없습니다.")
    write_csv(root / "comparison_per_seed.csv", rows)
    aggregated = []
    for method in sorted({r["method"] for r in rows}):
        for epoch in sorted({r["epoch"] for r in rows if r["method"] == method}):
            subset = [r for r in rows if r["method"] == method and r["epoch"] == epoch]
            ag = dict(method=method, epoch=epoch, n=len(subset),
                      seeds=" ".join(str(r["seed"]) for r in subset),
                      target_reached_n=sum(r["target_status"] == "reached" for r in subset))
            for key in ("teacher_knn", "student_knn", "train_s", "peak_vram_mb",
                        "teacher_linear_val", "student_linear_val",
                        "teacher_linear_test", "student_linear_test"):
                vals = [r[key] for r in subset if key in r]
                if vals:
                    ag[key + "_mean"] = statistics.mean(vals)
                    ag[key + "_sd"] = statistics.stdev(vals) if len(vals) > 1 else ""
            aggregated.append(ag)
    write_csv(root / "comparison_mean_sd.csv", aggregated)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("CSV 생성 완료. 그래프는 matplotlib 설치 후 summary를 다시 실행하세요.")
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for cfg, hist, ev in runs:
        label = cfg["method"] + "/seed" + str(cfg["seed"])
        axes[0, 0].plot([r["train_s"]/3600 for r in ev], [r["teacher_knn"] for r in ev], marker=".", label=label)
        axes[0, 1].plot([r["epoch"] for r in ev], [r["teacher_feature_variance"] for r in ev], label=label)
        axes[1, 0].plot([r["epoch"] for r in ev], [r["cls_cosine_gap"] for r in ev], label=label)
        axes[1, 1].plot([r["epoch"] for r in hist], [r["teacher_update_ms"] for r in hist], label=label)
    if target:
        axes[0, 0].axhline(target["teacher_knn"], ls="--", color="black", label="fixed target")
    for ax, title, xlabel in zip(axes.flat,
                                ("Teacher kNN (%)", "Normalized feature variance",
                                 "Teacher/student CLS cosine gap", "Teacher update (ms/step)"),
                                ("Training hours", "Epoch", "Epoch", "Epoch")):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root / "comparison.png", dpi=160)
    plt.close(fig)
    for cfg, hist, _ in runs:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for g in GROUPS:
            for ax, key in zip(axes, ("ratio_", "gap_", "clip_high_")):
                ax.plot([r["epoch"] for r in hist], [r[key+g] for r in hist], label=g)
        for ax, title in zip(axes, ("Update multiplier", "Relative parameter gap", "Upper clip fraction")):
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.legend()
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(root / (cfg["method"] + "_seed" + str(cfg["seed"]) + "_statistics.png"), dpi=160)
        plt.close(fig)
    print("CSV와 그래프 저장:", root.resolve())


def self_test():
    torch.set_num_threads(2)
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.front = nn.Parameter(torch.tensor([1., 2.]))
            self.backbone = nn.Module()
            self.backbone.norm = nn.Linear(2, 1, bias=False)
            self.head = nn.Linear(2, 2, bias=False)
    seed_all(7)
    student = Toy()
    with torch.no_grad():
        for p in student.parameters():
            p.copy_(torch.arange(1, p.numel()+1).reshape(p.shape).float())
    teacher = copy.deepcopy(student)
    with torch.no_grad():
        for p in teacher.parameters():
            p.mul_(0.8)
    a = 0.004
    for method in "BCGR":
        t = copy.deepcopy(teacher)
        u = TeacherUpdater(student, t, method)
        before = {k: p.clone() for k, p in t.named_parameters()}
        u.step(a, 4)
        for name, p in t.named_parameters():
            torch.testing.assert_close(p, before[name] + a * (dict(student.named_parameters())[name] - before[name]))
        assert u.n == 0 and len(u.v) == 0 and u.dbar.sum() == 0
    # R first update: vhat=delta^2, including element-weighted group q and clipping.
    t = copy.deepcopy(teacher)
    u = TeacherUpdater(student, t, "R")
    expected = {}
    for g, ps in u.pairs.items():
        q = (sum((s-tp).square().sum() for _, s, tp in ps) / u.count[g]).sqrt()
        for name, s, tp in ps:
            delta = s-tp
            expected[name] = tp + a*((q+EPS)/(delta.abs()+EPS)).clamp(0.5, 2)*delta
    stats = u.step(a, 5)
    for name, p in t.named_parameters():
        torch.testing.assert_close(p, expected[name])
    assert u.n == 1 and all(v.dtype == torch.float32 for v in u.v.values())
    # G/C: analytic formula with different group gaps and unequal group sizes.
    varied_teacher = copy.deepcopy(student)
    with torch.no_grad():
        for name, p in varied_teacher.named_parameters():
            p.mul_({"front": 0.99, "back": 0.8, "head": 0.2}[group_name(name)])
    for method in ("C", "G"):
        t2 = copy.deepcopy(varied_teacher)
        up = TeacherUpdater(student, t2, method)
        ds = []
        for g, ps in up.pairs.items():
            ds.append((sum((s-p).square().sum() for _, s, p in ps)/up.count[g]).sqrt()
                      / ((sum(s.square().sum() for _, s, _ in ps)/up.count[g]).sqrt()+EPS))
        means = torch.stack(ds)*0.01
        ratios = (means/(means.mean()+EPS)).clamp(0.5, 2)
        if method == "C":
            sizes = torch.tensor([up.count[g] for g in GROUPS])
            ratios = ((ratios*sizes).sum()/sizes.sum()).expand(3)
        before = {name: p.clone() for name, p in t2.named_parameters()}
        up.step(a, 5)
        for name, p in t2.named_parameters():
            j = GROUPS.index(group_name(name))
            torch.testing.assert_close(p, before[name]+a*ratios[j]*(dict(student.named_parameters())[name]-before[name]))
    t2 = copy.deepcopy(t)
    u2 = TeacherUpdater(student, t2, "R")
    u2.load_state_dict(copy.deepcopy(u.state_dict()))
    u.step(a, 6)
    u2.step(a, 6)
    for p, q in zip(t.parameters(), t2.parameters()):
        torch.testing.assert_close(p, q)
    # Six cross-view terms, detached teacher targets.
    s = [torch.randn(2, 8, requires_grad=True) for _ in range(4)]
    tlogits = [torch.randn(2, 8, requires_grad=True) for _ in range(2)]
    center = torch.zeros(1, 8)
    loss = dino_loss(s, tlogits, center)
    expected_loss = sum(-(F.softmax(tlogits[i].detach()/0.04, -1) *
                          F.log_softmax(s[j]/0.1, -1)).sum(-1).mean()
                        for i in range(2) for j in range(4) if i != j)/6
    torch.testing.assert_close(loss, expected_loss)
    loss.backward()
    assert all(x.grad is None for x in tlogits) and all(x.grad is not None for x in s)
    net = DINO(smoke=True)
    logits = net([torch.randn(2, 3, size, size) for size in (96, 96, 48, 48)])
    assert len(logits) == 4 and all(x.shape == (2, 64) for x in logits)
    assert schedules(0, 156, 64, 0.996)[0] == 0
    assert math.isclose(schedules(15600-1, 156, 64, 0.996)[2], 1.)
    print("PASS: B warmup, C/G/R formulas, R resume, six-view loss, teacher stop-grad, "
          "multi-crop ViT, 100-epoch schedules.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "self-test", "smoke", "train", "freeze-target", "final", "summary"])
    parser.add_argument("--method", choices=list("BCGR"), default="B")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--until", type=int, default=20, help="Stop at this epoch; schedule ALWAYS spans 100 epochs.")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0, help="Windows-safe default. Keep identical for all methods.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--eval-batch", type=int, default=64)
    parser.add_argument("--gpu-memory-gib", type=float, default=10.5,
                        help="Allocator cap, leaving 1.5 GiB of a 12 GiB budget for CUDA overhead.")
    parser.add_argument("--momentum", type=float, default=0.996)
    parser.add_argument("--clip-low", type=float, default=0.5)
    parser.add_argument("--clip-high", type=float, default=2.0)
    parser.add_argument("--data", default="./data")
    parser.add_argument("--out", default="./runs")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--test", action="store_true", help="Explicitly enable final test-set evaluation.")
    args = parser.parse_args()
    args.smoke = args.command == "smoke"
    if args.smoke:
        args.batch, args.until, args.device, args.workers = 2, 6, "cpu", 0
    if args.command in ("train", "smoke", "final"):
        configure_runtime(args)
    if args.command == "check":
        print(json.dumps(environment(), ensure_ascii=False, indent=2))
    elif args.command == "self-test":
        self_test()
    elif args.command in ("train", "smoke"):
        if args.smoke:
            args.batch, args.until, args.device, args.workers = 2, 6, "cpu", 0
        train(args)
    elif args.command == "freeze-target":
        freeze_target(args)
    elif args.command == "final":
        final_evaluation(args)
    elif args.command == "summary":
        summarize(args)


if __name__ == "__main__":
    main()
