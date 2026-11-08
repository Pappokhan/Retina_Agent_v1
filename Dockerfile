# CPU image. For a GPU, build FROM a CUDA-enabled PyTorch base image instead
# (for example pytorch/pytorch:<version>-cuda<ver>-cudnn<ver>-runtime), drop torch/torchvision from the
# pip install step, and run with:  docker run --gpus all ...
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .
RUN useradd --create-home app && mkdir -p uploads && chown -R app /app
USER app

# weights are mounted, not baked in:  -v "$PWD/weights:/app/weights"
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status==200 else 1)"

# One worker: the model cache and the GPU are per process. Threads handle concurrent requests.
CMD ["gunicorn", "-w", "1", "--threads", "4", "--timeout", "120", "-b", "0.0.0.0:8000", "app:create_app()"]
