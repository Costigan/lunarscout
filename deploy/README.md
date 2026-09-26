# Horizon Worker Image and Job

This directory builds the container image used by the scheduler-free horizon
batch on NRP, and the Kubernetes Job that runs it. The image contains Lunarscout
(with its CUDA extra).

Files:

- `Dockerfile` — the worker image definition (installs `lunarscout[cuda]` and
  the `lunarscout-horizon-worker` console script).
- `build.sh` — builds the image (and optionally tags `:latest`).
- `push.sh` — pushes the image to the registry.
- `horizon-job.yaml` — the Indexed Job plus its per-scenario ConfigMap in one
  file (one `apply`, one `delete`).

## Prerequisites

- Docker installed and the daemon running.
- One-time registry login (only needed before the first push):

  ```bash
  docker login gitlab-registry.nrp-nautilus.io
  ```

## Build and push

From the repository root:

```bash
deploy/build.sh
deploy/push.sh
```

This builds and pushes `gitlab-registry.nrp-nautilus.io/costigan/lunarscout-worker:<git-short-sha>`.
The registry, username, image name, and tag can be overridden via `REGISTRY`,
`USERNAME`, `IMAGE`, and `--tag`. See the header of `build.sh` / `push.sh`.

Code changes require a rebuild and re-push to reach the pods; the Job itself
does not change when the code changes.

## Run a batch

Edit the scenario values at the top of `deploy/horizon-job.yaml` (DEM paths,
output/log directories, pod count), then:

```bash
kubectl apply  -f deploy/horizon-job.yaml -n <namespace>
kubectl delete -f deploy/horizon-job.yaml -n <namespace>
```

One file, one `apply`, one `delete` (it holds both the ConfigMap and the Job).
See `docs/strided-generation-nrp.md` for the full procedure.

## Image notes

- The image installs `lunarscout[cuda]` from the local `src/` tree, so it
  always reflects the working copy being built.
- The CUDA runtime libraries come from the `cuda-toolkit` package pulled in by
  `numba-cuda[cu12]`. The GPU is requested per pod at runtime via
  `nvidia.com/gpu: 1`, not baked into the image.
- `NUMBA_CACHE_DIR` defaults to `/tmp/numba-cache` (per-pod, always writable).
  Do not point it at a shared directory: concurrent pods racing on CUDA kernel
  compilation can load a corrupt cache entry.
- The first build downloads several gigabytes (numba, cuda-toolkit, rasterio,
  scipy); later builds reuse Docker's layer cache.
