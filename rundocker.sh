#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=${IMAGE_NAME:-missing-scripts}
SHM_SIZE=${SHM_SIZE:-1g}
WORKDIR=${WORKDIR:-/app/src}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "--build" ]]; then
  shift
  docker build -f "$PROJECT_ROOT/src/Dockerfile" -t "$IMAGE_NAME" "$PROJECT_ROOT"
fi

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 [--build] <script> [args...]" >&2
  exit 1
fi

docker run --rm \
  -v "$PROJECT_ROOT":/app \
  -w "$WORKDIR" \
  --shm-size="$SHM_SIZE" \
  "$IMAGE_NAME" "$@"
