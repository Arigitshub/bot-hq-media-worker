# Exact official base used for the first live A40 image/video/voice proof.
FROM runpod/pytorch:2.2.0-rc.229-cu1290-torch291-ubuntu2404@sha256:7b0a0df6e9a34a163eeaefdfaf986331ed5cb2bd865ec3df9bb8a92273682c50
USER root
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/cache/huggingface HF_HUB_CACHE=/cache/huggingface/hub \
    HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg espeak-ng libsndfile1 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/worker
COPY requirements.txt ./
# Reuse the base's CUDA PyTorch; do not download another multi-GB torch wheel.
RUN python3 -m pip install --no-cache-dir -r requirements.txt \
    && python3 -m pip check \
    && python3 -c "import sys,torch; assert sys.version_info[:2] == (3,12); assert torch.__version__.split('+')[0] == '2.9.1'; assert torch.version.cuda in ('12.8','12.9')" \
    && python3 -m pip freeze > build-manifest.txt
RUN useradd --create-home --uid 10001 hqworker \
    && mkdir -p /cache/huggingface/hub \
    && chown -R hqworker:hqworker /cache /opt/worker
COPY --chown=hqworker:hqworker models.json policy.py runtime.py runner.py handler.py http_server.py serverless_entry.py ./
COPY --chown=hqworker:hqworker LICENSE THIRD_PARTY_NOTICES.md ./
USER hqworker
# Disable the Pod image's Jupyter/SSH entrypoint on queue workers.
ENTRYPOINT []
CMD ["python3", "serverless_entry.py"]
