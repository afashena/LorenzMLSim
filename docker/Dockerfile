FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install uv
RUN pip install --upgrade pip && pip install uv

# Copy only dependency files first (better caching)
COPY ../pyproject.toml /workspace/pyproject.toml
COPY ../uv.lock /workspace/uv.lock

# Install deps into the system python inside container
RUN uv sync --frozen

# Copy the rest of the repo
COPY . /workspace

# Don't auto-run training
CMD ["bash"]
