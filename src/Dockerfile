FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app/src

# System dependencies for scientific Python stack and plotting backends
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        graphviz \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        curl \
        git && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies from the project requirements
COPY src/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Default to running Python in unbuffered mode; script provided at runtime
ENTRYPOINT ["python", "-u"]
CMD ["main.py"]
