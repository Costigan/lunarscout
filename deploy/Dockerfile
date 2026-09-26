# Lunarscout horizon worker image for the scheduler-free batch Job.
#
# Installs lunarscout (with the CUDA extra). The CUDA runtime libraries come
# from the ``cuda-toolkit`` package that ``numba-cuda[cu12]`` pulls in; the GPU
# itself is requested at the Kubernetes pod level (``nvidia.com/gpu: 1``), not
# baked into this image.
#
# Build from the repository root so ``src/`` and ``pyproject.toml`` are in the
# build context. Installing the package also installs the
# ``lunarscout-horizon-worker`` console script (the Job's entry point).

FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    NUMBA_CACHE_DIR=/tmp/numba-cache

WORKDIR /opt/lunarscout

# Copy only what ``pip install .`` needs so unrelated changes do not
# invalidate this layer more than necessary.
COPY pyproject.toml README.md LICENSE MANIFEST.in ./
COPY src ./src

RUN pip install ".[cuda]"

# System runtime libraries that the prebuilt wheels link against but do not
# bundle. ``python:3.12-slim`` omits them:
#   libexpat1  - rasterio/GDAL link it; wheels intentionally rely on the OS
#                copy so expat security patches apply without a rebuild.
#   libgomp1   - numba's OpenMP-parallel CPU loops.
# This layer is kept after the large pip layer so adding a system library does
# not force a re-download of the Python packages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

CMD ["lunarscout-horizon-worker"]
