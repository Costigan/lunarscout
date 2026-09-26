#!/usr/bin/env bash
# Build the lunarscout horizon worker image for the NRP GitLab registry.
#
# Build (tagged with the current git short commit by default):
#   deploy/build.sh
#
# Build with an explicit tag, and also tag :latest:
#   deploy/build.sh --tag-latest --tag v0.1.0rc5
#
# Push separately with deploy/push.sh.
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

echo "Repository root : ${REPO_ROOT}"
echo "Image           : ${FULL_REF}"

echo "Building ${FULL_REF} ..."
# --provenance=false keeps the image a plain single manifest. BuildKit adds a
# provenance attestation manifest by default, which GitLab's registry rejects
# with "blob unknown to registry" on push.
docker build \
    --provenance=false \
    -f "${SCRIPT_DIR}/Dockerfile" \
    -t "${FULL_REF}" \
    "${REPO_ROOT}"

if [[ "$TAG_LATEST" == "1" ]]; then
    docker tag "${FULL_REF}" "${REGISTRY}/${USERNAME}/${IMAGE}:latest"
    echo "Tagged ${REGISTRY}/${USERNAME}/${IMAGE}:latest"
fi
