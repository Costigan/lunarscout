# Strided Horizon Generation on NRP

Scheduler-free batch horizon generation on NRP. Instead of a Dask cluster, a
plain Kubernetes `Job` runs one pod per partition, and each pod generates a
strided share of the 128 by 128 patches with the existing sequential generator.
The skip-if-exists logic coordinates resume across pods through the shared
output directory.

The pieces are split by what changes and how often:

- **Code (image):** the worker is the installed console script
  `lunarscout-horizon-worker` (`lunarscout._numba_horizon.worker`). It processes
  a strided share of the patch list (`--pod-nth` of `--pod-count`) and tees its
  verbose progress (with `s/patch` and `ETA`) to a durable per-pod log. Because
  it is code, it is baked into the worker image — rebuild and re-push after
  changing it.
- **Scenario (data):** the ConfigMap at the top of `deploy/horizon-job.yaml`
  holds the DEM paths, output/log directories, and pod count. Edit those for a
  new scenario.
- **Executor (shape):** the Job at the bottom of the same file is the stable
  Indexed Job (one pod per partition, one GPU each).

## Procedure

1. Build and push the worker image (see `deploy/README.md`):

   ```bash
   deploy/build.sh
   deploy/push.sh
   ```

2. Edit `deploy/horizon-job.yaml`: the DEM paths, output/log directories, and
   `POD_COUNT` (keep it in sync with the Job's `completions`/`parallelism`),
   plus the one-time image tag and PVC `claimName`.

3. Run it (one file, one `apply`):

   ```bash
   kubectl apply -f deploy/horizon-job.yaml -n <namespace>
   ```

4. Stop it (one `delete` removes the Job and the ConfigMap):

   ```bash
   kubectl delete -f deploy/horizon-job.yaml -n <namespace>
   ```

5. Watch progress:

   ```bash
   kubectl get pods -n <namespace>
   kubectl logs -f <pod> -n <namespace>
   ```

   Each pod tees stdout/stderr to `$LOG_DIR/pod-<nth>-<timestamp>.log` on the
   shared filesystem (a unique name per attempt, so a retried pod never
   clobbers an earlier attempt), because Kubernetes pod logs disappear when the
   pod is deleted. Each log ends with an explicit `SUCCESS: ...` or
   `FAILED: ...` line.

## Worker configuration

The worker reads its configuration from environment variables that the Job
supplies from the scenario ConfigMap via `envFrom`:

| Variable           | Purpose                                  |
| ------------------ | ---------------------------------------- |
| `PRIMARY_DEM`      | primary DEM (defines the output grid)    |
| `SURROUNDING_DEMS` | comma-separated surrounding DEMs         |
| `OUTPUT_DIR`       | shared horizon output directory          |
| `LOG_DIR`          | directory for `pod-<nth>-<timestamp>.log` |
| `POD_COUNT`        | number of partitions (matches the Job)   |
| `OBSERVER_HEIGHT_M`| observer height in metres                |
| `COMPRESS`         | `1` for `.cbin`, unset for `.bin`        |

The same `lunarscout-horizon-worker` command accepts those as `--` flags, so it
runs directly for local testing:

```bash
lunarscout-horizon-worker \
    --primary-dem /shared/dem.tif --surrounding-dems /shared/regional.tif \
    --output /shared/horizons --pod-count 4 --pod-nth 0
```

`generate_horizons` exposes the same partitioning directly as
`patch_offset`/`patch_stride`:

```python
# Pod k of N processes patches k, k+N, k+2N, ... so N pods cover every patch once.
horizons = ls.generate_horizons(
    "/shared/horizons",
    ["/shared/dem.tif"],
    patch_offset=k,
    patch_stride=N,
)
```

## Notes

- `NUMBA_CACHE_DIR` must stay per-pod (`/tmp/numba-cache`). A shared cache
  directory makes concurrent pods race on CUDA kernel compilation and can load
  a corrupt entry (observed as `CUDA_ERROR_ILLEGAL_ADDRESS` /
  `CUDA_ERROR_LAUNCH_FAILED`).
- `PRIMARY_DEM` and `SURROUNDING_DEMS` must resolve to the same paths on every
  pod (shared filesystem) and on the driver.
- The worker is resumable: re-running the Job skips already-completed tiles.
