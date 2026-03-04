FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Update & install deps
RUN apt-get update && apt-get --no-install-recommends install -y \
    curl \
    python3 \
    python3-pip \
    && apt-get clean \
    && curl --proto "=https" --tlsv1.2 -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"

# Workdir
WORKDIR /app

# Install dependencies
COPY ./engine/.python-version ./
COPY ./engine/pyproject.toml ./
COPY ./engine/uv.lock ./

RUN uv sync

# Ports
EXPOSE 8000

# Entry
CMD ["/bin/bash"]
