#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=${IMAGE_NAME:-missing-scripts}
SHM_SIZE=${SHM_SIZE:-1g}
WORKDIR=${WORKDIR:-/app/src}
DETACH_ARGS=""

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)
      docker build -f "$PROJECT_ROOT/src/Dockerfile" -t "$IMAGE_NAME" "$PROJECT_ROOT"
      shift
      ;;
    -d)
      DETACH_ARGS="-d"
      shift
      ;;
    --)
      shift
      break
    ;;
    *)
      break
      ;;
  esac
done

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 [--build] [-d] <script> [args...]" >&2
  exit 1
fi

if [[ -n "$DETACH_ARGS" ]]; then
  container_id=$(docker run --rm \
    $DETACH_ARGS \
    -v "$PROJECT_ROOT":/app \
    -w "$WORKDIR" \
    --shm-size="$SHM_SIZE" \
    "$IMAGE_NAME" "$@")
  echo "Container started in detached mode: $container_id"
else
  docker run --rm \
    -v "$PROJECT_ROOT":/app \
    -w "$WORKDIR" \
    --shm-size="$SHM_SIZE" \
    "$IMAGE_NAME" "$@"
fi
