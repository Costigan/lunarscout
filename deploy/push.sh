#!/usr/bin/env bash
# Push the lunarscout horizon worker image to the NRP GitLab registry.
#
# Push the current git short commit tag:
#   deploy/push.sh
#
# Push an explicit tag, and also push :latest:
#   deploy/push.sh --tag-latest --tag v0.1.0rc5
#
# Build the image first with deploy/build.sh.
#
# You must log in to the registry once before the first push:
#   docker login gitlab-registry.nrp-nautilus.io
#
# Registry, project, and image name can be overridden via environment variables:
#   REGISTRY (default: gitlab-registry.nrp-nautilus.io)
#   USERNAME (default: costigan)
#   IMAGE    (default: lunarscout-worker)

set -euo pipefail

REGISTRY="${REGISTRY:-gitlab-registry.nrp-nautilus.io}"
USERNAME="${USERNAME:-costigan}"
IMAGE="${IMAGE:-lunarscout-worker}"

TAG_LATEST=0
TAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag-latest) TAG_LATEST=1 ;;
        --tag) TAG="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [--tag-latest] [--tag TAG]" >&2
            exit 0
            ;;
        *) echo "error: unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "$TAG" ]]; then
    TAG="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo dev)"
fi

FULL_REF="${REGISTRY}/${USERNAME}/${IMAGE}:${TAG}"

echo "Image : ${FULL_REF}"

echo "Pushing ${FULL_REF} ..."
docker push "${FULL_REF}"

if [[ "$TAG_LATEST" == "1" ]]; then
    docker push "${REGISTRY}/${USERNAME}/${IMAGE}:latest"
fi
