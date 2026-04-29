# Dockerfile for STAT-agent demo on Hugging Face Spaces.
# HF Spaces runs containers as user `user` (uid 1000) and expects port 7860.

FROM python:3.11-slim

# Minimal system deps for scientific Python wheels (most have prebuilt wheels).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces convention: non-root user with uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /home/user/app

# Install Python deps first (layer cache friendly).
COPY --chown=user:user pyproject.toml README.md ./
COPY --chown=user:user stat_agent ./stat_agent
RUN pip install --user --upgrade pip && \
    pip install --user . huggingface_hub

# CPU-only skill dependencies. We intentionally drop torch-based skills
# (scvi-tools, tangram-sc, cell2location, SpaGCN, GraphST) because the
# free HF Spaces tier has no GPU and these run impractically slow on CPU.
RUN pip install --user \
    'squidpy>=1.2' \
    'esda>=2.4.0' \
    'gseapy>=1.0' \
    'liana' \
    'cellphonedb>=5.0,<6.0' \
    'plotnine>=0.12' \
    'harmonypy>=0.0.9' \
    'bbknn>=1.5.0' \
    'scanorama>=1.7.0' \
    'qpsolvers>=4.0' \
    'ray>=2.0' \
    'flashdeconv>=0.1.0' \
    'palantir>=1.0.0' \
    'infercnvpy>=0.4.0' \
    'NaiveDE>=0.1' \
    'SpatialDE>=1.1' \
    'scikit-misc>=0.5' \
    'igraph>=0.9,<1.0' \
    'paste-bio>=1.0.0' \
    'POT>=0.9.0,<0.9.6' \
    'pynrrd>=1.0'

# Demo configuration — secrets (POE_API_KEY) come from HF Space settings at runtime.
# POE provider is auto-detected from the `poe/` model prefix.
ENV STAT_DEMO_MODE=1 \
    STAT_DEMO_DATA_DIR=/home/user/app/data \
    STAT_DEMO_HF_DATASET=CyhVVVV/stat-agent-demo-data \
    SPATIAL_AGENT_MODEL=poe/Claude-Sonnet-4.6 \
    HF_HOME=/tmp/huggingface \
    PORT=7860

EXPOSE 7860

CMD ["python", "-m", "stat_agent.web.app", "--host", "0.0.0.0", "--port", "7860"]
