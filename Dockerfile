FROM python:3.11.13-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLCONFIGDIR=/tmp/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 \
      --index-url https://download.pytorch.org/whl/cu126 \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt
ENTRYPOINT ["python", "server.py"]
CMD ["check"]
