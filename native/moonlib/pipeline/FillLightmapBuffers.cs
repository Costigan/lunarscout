using moonlib.horizon;
using moonlib.math;
using moonlib.spice;
using Serilog;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading.Channels;

namespace moonlib.pipeline
{
    public enum FillLightmapRunState
    {
        NotStarted,
        Running,
        Cancelling,
        Cancelled,
        Completed,
        Failed
    }

    public enum FillLightmapBufferState
    {
        Filled,
        Error
    }

    public sealed record FillLightmapBuffersRequest(
        string DemPath,
        string HorizonDir,
        DateTime[] TimestampsUtc,
        int PatchWidth = FillLightmapBuffers.Width,
        int PatchHeight = FillLightmapBuffers.Height,
        int MaxReadParallelism = 12,
        int MaxComputeParallelism = 20,
        int QueueCapacity = 40,
        bool UseSpiceSunVectors = true
    );

    public sealed record FillLightmapAvailableBuffer(
        int BufferId,
        long Pointer,
        int ByteLength
    );

    public sealed record FillLightmapFilledBuffer(
        int BufferId,
        long TileId,
        int PatchRow,
        int PatchCol,
        int Width,
        int Height,
        int TimeCount,
        FillLightmapBufferState State,
        string? Message = null
    );

    public sealed record FillLightmapPollResult(
        FillLightmapFilledBuffer[] FilledBuffers,
        FillLightmapRunState State,
        double Progress01,
        long TilesProduced,
        long TilesReturned,
        int OwnedBufferCount,
        string? Message = null
    );

    /// <summary>
    /// Fills Python-owned lightmap patch buffers with C# worker threads.
    ///
    /// Python owns all buffer memory. Each poll hands C# currently reusable buffers as
    /// (buffer id, pointer, byte length) records. C# may keep those records only while
    /// queued, being filled, or waiting to be returned. Once a filled/error envelope is
    /// returned by Poll, C# forgets that buffer and Python may reuse or retire it.
    /// </summary>
    public sealed class FillLightmapBuffers : IDisposable
    {
        public const int Width = 128;
        public const int Height = 128;
        public const int HorizonSamples = 1440;

        private readonly object _stateGate = new();
        private readonly ConcurrentDictionary<int, byte> _ownedBufferIds = new();
        private readonly Channel<BufferLease> _availableBuffers;
        private readonly Channel<FillLightmapFilledBuffer> _filledBuffers;
        private CancellationTokenSource? _cancellation;
        private Task? _producerTask;
        private FillLightmapBuffersRequest? _request;
        private int _timeCount;
        private int _expectedByteLength;
        private FillLightmapRunState _state = FillLightmapRunState.NotStarted;
        private string? _message;
        private double _progress01;
        private long _tilesProduced;
        private long _tilesReturned;

        public FillLightmapBuffers()
        {
            _availableBuffers = Channel.CreateUnbounded<BufferLease>(new UnboundedChannelOptions
            {
                SingleReader = false,
                SingleWriter = false
            });
            _filledBuffers = Channel.CreateUnbounded<FillLightmapFilledBuffer>(new UnboundedChannelOptions
            {
                SingleReader = false,
                SingleWriter = false
            });
        }

        public void Start(FillLightmapBuffersRequest request)
        {
            ArgumentNullException.ThrowIfNull(request);
            ValidateRequest(request);

            lock (_stateGate)
            {
                if (_state != FillLightmapRunState.NotStarted)
                    throw new InvalidOperationException("FillLightmapBuffers can only be started once.");

                _request = request;
                _timeCount = request.TimestampsUtc.Length;
                _expectedByteLength = checked(request.PatchWidth * request.PatchHeight * _timeCount);
                _cancellation = new CancellationTokenSource();
                _state = FillLightmapRunState.Running;
                _message = "Lightmap buffer filling started.";
                _producerTask = Task.Run(ProducerLoop);
            }
        }

        public FillLightmapPollResult Poll(FillLightmapAvailableBuffer[]? availableBuffers, int timeoutMs)
        {
            AcceptAvailableBuffers(availableBuffers ?? Array.Empty<FillLightmapAvailableBuffer>());

            if (timeoutMs < 0)
                timeoutMs = 0;

            if (!_filledBuffers.Reader.TryPeek(out _) && timeoutMs > 0 && IsRunningLike())
            {
                using var timeout = new CancellationTokenSource(timeoutMs);
                try
                {
                    _filledBuffers.Reader.WaitToReadAsync(timeout.Token).AsTask().GetAwaiter().GetResult();
                }
                catch (OperationCanceledException)
                {
                    // Poll timeout; return status without filled buffers.
                }
            }

            var filled = new List<FillLightmapFilledBuffer>();
            while (_filledBuffers.Reader.TryRead(out var item))
            {
                _ownedBufferIds.TryRemove(item.BufferId, out _);
                Interlocked.Increment(ref _tilesReturned);
                filled.Add(item);
            }

            var (state, progress, produced, returned, owned, message) = SnapshotStatus();
            return new FillLightmapPollResult(
                FilledBuffers: filled.ToArray(),
                State: state,
                Progress01: progress,
                TilesProduced: produced,
                TilesReturned: returned,
                OwnedBufferCount: owned,
                Message: message
            );
        }

        public FillLightmapPollResult GetStatus()
        {
            var (state, progress, produced, returned, owned, message) = SnapshotStatus();
            return new FillLightmapPollResult(
                FilledBuffers: Array.Empty<FillLightmapFilledBuffer>(),
                State: state,
                Progress01: progress,
                TilesProduced: produced,
                TilesReturned: returned,
                OwnedBufferCount: owned,
                Message: message
            );
        }

        public void Cancel()
        {
            lock (_stateGate)
            {
                if (_state == FillLightmapRunState.Running)
                {
                    _state = FillLightmapRunState.Cancelling;
                    _message = "Cancellation requested.";
                }
            }
            _cancellation?.Cancel();
        }

        private void AcceptAvailableBuffers(IReadOnlyList<FillLightmapAvailableBuffer> buffers)
        {
            if (buffers.Count == 0)
                return;
            if (!IsRunningLike())
                throw new InvalidOperationException("Buffers can only be offered while the run is active.");

            foreach (var buffer in buffers)
            {
                if (buffer.BufferId < 0)
                    throw new ArgumentOutOfRangeException(nameof(buffers), "BufferId must be >= 0.");
                if (buffer.Pointer <= 0)
                    throw new ArgumentOutOfRangeException(nameof(buffers), "Pointer must be non-zero.");
                if (buffer.ByteLength != _expectedByteLength)
                    throw new ArgumentException(
                        $"Buffer {buffer.BufferId} byte length {buffer.ByteLength} does not match expected length {_expectedByteLength}.",
                        nameof(buffers));
                if (!_ownedBufferIds.TryAdd(buffer.BufferId, 0))
                    throw new InvalidOperationException($"Buffer {buffer.BufferId} is already owned by FillLightmapBuffers.");

                if (!_availableBuffers.Writer.TryWrite(new BufferLease(buffer.BufferId, new IntPtr(buffer.Pointer), buffer.ByteLength)))
                {
                    _ownedBufferIds.TryRemove(buffer.BufferId, out _);
                    throw new InvalidOperationException("Failed to queue available buffer.");
                }
            }
        }

        private void ProducerLoop()
        {
            Debug.Assert(_request != null && _cancellation != null);
            var request = _request;
            var cancellationToken = _cancellation.Token;

            try
            {
                var horizonFiles = new HorizonTileStore(request.HorizonDir)
                    .EnumerateFiles(observerElevationMeters: 0f)
                    .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
                    .ToList();

                if (horizonFiles.Count == 0)
                {
                    SetProgress(1.0);
                    SetTerminal(FillLightmapRunState.Completed, "No horizon tiles found.");
                    return;
                }

                var dem = new ElevationMap(request.DemPath);
                var sunVectors = request.UseSpiceSunVectors
                    ? request.TimestampsUtc.Select(t => SpiceManager.SunPosition(EnsureUtc(t)) * 1000.0).ToList()
                    : request.TimestampsUtc.Select(BuildSyntheticSunVector).ToList();

                var pipeline = new Pipeline<FillLightmapWorkItem>();
                pipeline.AddStep(
                    ReadHorizons,
                    maxDegreeOfParallelism: request.MaxReadParallelism,
                    boundedCapacity: request.QueueCapacity,
                    ensureOrdered: false);
                pipeline.AddStep(
                    item => FillBuffer(item, dem, sunVectors, horizonFiles.Count, cancellationToken),
                    maxDegreeOfParallelism: request.MaxComputeParallelism,
                    boundedCapacity: request.QueueCapacity,
                    ensureOrdered: false);
                pipeline.AddTerminalStep(
                    _ => Task.CompletedTask,
                    maxDegreeOfParallelism: 1,
                    boundedCapacity: request.QueueCapacity,
                    ensureOrdered: false);

                pipeline.ProcessAsync(BuildWorkItems(horizonFiles, cancellationToken)).GetAwaiter().GetResult();
                SetProgress(1.0);
                SetTerminal(FillLightmapRunState.Completed, "Lightmap buffer filling completed.");
            }
            catch (OperationCanceledException)
            {
                SetTerminal(FillLightmapRunState.Cancelled, "Lightmap buffer filling cancelled.");
            }
            catch (Exception ex)
            {
                Log.Error(ex, "FillLightmapBuffers failed.");
                SetTerminal(FillLightmapRunState.Failed, ex.Message);
            }
        }

        private IEnumerable<FillLightmapWorkItem> BuildWorkItems(IReadOnlyList<string> horizonFiles, CancellationToken cancellationToken)
        {
            long tileId = 0;
            foreach (var horizonPath in horizonFiles)
            {
                cancellationToken.ThrowIfCancellationRequested();
                yield return new FillLightmapWorkItem(Interlocked.Increment(ref tileId), horizonPath);
            }
        }

        private static Task<FillLightmapWorkItem> ReadHorizons(FillLightmapWorkItem item)
        {
            try
            {
                (item.PatchCol, item.PatchRow, _) = QuadTreeHorizonGenerator.ParseHorizonFilename(item.HorizonPath);
                item.Horizons = HorizonFile.ReadHorizonFile(item.HorizonPath);
                item.ErrorMessage = null;
            }
            catch (Exception ex)
            {
                item.ErrorMessage = ex.Message;
            }

            return Task.FromResult(item);
        }

        private Task<FillLightmapWorkItem> FillBuffer(
            FillLightmapWorkItem item,
            ElevationMap dem,
            List<Vector3d> sunVectors,
            int totalTileCount,
            CancellationToken cancellationToken)
        {
            BufferLease? lease = null;
            try
            {
                cancellationToken.ThrowIfCancellationRequested();

                lease = AcquireBuffer(cancellationToken);

                if (!string.IsNullOrWhiteSpace(item.ErrorMessage))
                    throw new InvalidDataException(item.ErrorMessage);
                if (item.Horizons is null)
                    throw new InvalidDataException($"Missing horizon data for {item.HorizonPath}.");

                WritePatch(
                    lease.Pointer,
                    lease.ByteLength,
                    item.Horizons,
                    dem,
                    item.PatchRow,
                    item.PatchCol,
                    _request!.PatchWidth,
                    _request.PatchHeight,
                    sunVectors);

                PublishFilled(new FillLightmapFilledBuffer(
                    BufferId: lease.BufferId,
                    TileId: item.TileId,
                    PatchRow: item.PatchRow,
                    PatchCol: item.PatchCol,
                    Width: _request.PatchWidth,
                    Height: _request.PatchHeight,
                    TimeCount: sunVectors.Count,
                    State: FillLightmapBufferState.Filled
                ));
                lease = null;
            }
            catch (OperationCanceledException)
            {
                if (lease is not null)
                    ReturnForgottenLease(lease);
                throw;
            }
            catch (Exception ex)
            {
                if (lease is not null)
                {
                    PublishFilled(new FillLightmapFilledBuffer(
                        BufferId: lease.BufferId,
                        TileId: item.TileId,
                        PatchRow: item.PatchRow,
                        PatchCol: item.PatchCol,
                        Width: _request!.PatchWidth,
                        Height: _request.PatchHeight,
                        TimeCount: sunVectors.Count,
                        State: FillLightmapBufferState.Error,
                        Message: ex.Message
                    ));
                    lease = null;
                }
                else
                {
                    Log.Warning(ex, "Failed before acquiring a Python buffer for horizon tile {HorizonPath}.", item.HorizonPath);
                }
            }
            finally
            {
                Interlocked.Increment(ref _tilesProduced);
                var produced = Interlocked.Read(ref _tilesProduced);
                SetProgress(Math.Clamp((double)produced / Math.Max(1, totalTileCount), 0.0, 1.0));
            }

            return Task.FromResult(item);
        }

        private BufferLease AcquireBuffer(CancellationToken cancellationToken)
        {
            while (true)
            {
                cancellationToken.ThrowIfCancellationRequested();
                return _availableBuffers.Reader.ReadAsync(cancellationToken).AsTask().GetAwaiter().GetResult();
            }
        }

        private void PublishFilled(FillLightmapFilledBuffer filled)
        {
            _filledBuffers.Writer.WriteAsync(filled).AsTask().GetAwaiter().GetResult();
        }

        private void ReturnForgottenLease(BufferLease lease)
        {
            _ownedBufferIds.TryRemove(lease.BufferId, out _);
        }

        private unsafe static void WritePatch(
            IntPtr targetPtr,
            int targetByteLength,
            float[] horizons,
            ElevationMap dem,
            int patchRow,
            int patchCol,
            int patchWidth,
            int patchHeight,
            List<Vector3d> sunVectors)
        {
            if (patchRow < 0 || patchCol < 0)
                throw new ArgumentOutOfRangeException("Patch row/col must be >= 0.");
            if (patchRow + patchHeight > dem.Height || patchCol + patchWidth > dem.Width)
                throw new ArgumentOutOfRangeException("Patch row/col extends beyond DEM bounds.");

            int expectedHorizonSamples = checked(patchWidth * patchHeight * HorizonSamples);
            if (horizons.Length < expectedHorizonSamples)
                throw new InvalidDataException(
                    $"Unexpected horizon sample count. Expected at least {expectedHorizonSamples}, got {horizons.Length}.");

            int expectedBytes = checked(patchWidth * patchHeight * sunVectors.Count);
            if (targetByteLength != expectedBytes)
                throw new InvalidDataException($"Unexpected output byte length. Expected {expectedBytes}, got {targetByteLength}.");

            var destination = new Span<byte>(targetPtr.ToPointer(), targetByteLength);
            var matrices = new Matrix4d[patchHeight, patchWidth];
            for (int y = 0; y < patchHeight; y++)
            {
                int line = patchRow + y;
                for (int x = 0; x < patchWidth; x++)
                {
                    int sample = patchCol + x;
                    matrices[y, x] = dem.GetMoonMEToENU(line, sample);
                }
            }

            for (int y = 0; y < patchHeight; y++)
            {
                for (int x = 0; x < patchWidth; x++)
                {
                    int pixelIdx = y * patchWidth + x;
                    int horizonBase = pixelIdx * HorizonSamples;
                    Matrix4d matrix = matrices[y, x];
                    int outputBase = pixelIdx * sunVectors.Count;

                    for (int t = 0; t < sunVectors.Count; t++)
                    {
                        var (azRad, elRad) = dem.GetAzEl(sunVectors[t], matrix);
                        float azDeg = azRad * 57.2957795f;
                        float elDeg = elRad * 57.2957795f;
                        float fraction = LightmapGenerator.BuilderSunFraction(horizons, horizonBase, azDeg, elDeg);
                        destination[outputBase + t] = (byte)(255f * fraction);
                    }
                }
            }
        }

        private static void ValidateRequest(FillLightmapBuffersRequest request)
        {
            if (string.IsNullOrWhiteSpace(request.DemPath))
                throw new ArgumentException("DEM path is required.", nameof(request));
            if (string.IsNullOrWhiteSpace(request.HorizonDir))
                throw new ArgumentException("Horizon directory is required.", nameof(request));
            if (!File.Exists(request.DemPath))
                throw new FileNotFoundException("DEM path does not exist.", request.DemPath);
            if (!Directory.Exists(request.HorizonDir))
                throw new DirectoryNotFoundException($"Horizon directory does not exist: {request.HorizonDir}");
            if (request.TimestampsUtc is null || request.TimestampsUtc.Length == 0)
                throw new ArgumentException("At least one timestamp is required.", nameof(request));
            if (request.PatchWidth != Width || request.PatchHeight != Height)
                throw new NotSupportedException("Only 128x128 horizon tile geometry is currently supported.");
            if (request.MaxReadParallelism < 1)
                throw new ArgumentOutOfRangeException(nameof(request), "MaxReadParallelism must be >= 1.");
            if (request.MaxComputeParallelism < 1)
                throw new ArgumentOutOfRangeException(nameof(request), "MaxComputeParallelism must be >= 1.");
            if (request.QueueCapacity < 1)
                throw new ArgumentOutOfRangeException(nameof(request), "QueueCapacity must be >= 1.");
        }

        private bool IsRunningLike()
        {
            lock (_stateGate)
            {
                return _state is FillLightmapRunState.Running or FillLightmapRunState.Cancelling;
            }
        }

        private void SetProgress(double progress)
        {
            lock (_stateGate)
            {
                _progress01 = Math.Clamp(progress, 0.0, 1.0);
            }
        }

        private void SetTerminal(FillLightmapRunState state, string message)
        {
            lock (_stateGate)
            {
                _state = state;
                _message = message;
            }
        }

        private (FillLightmapRunState State, double Progress, long Produced, long Returned, int Owned, string? Message) SnapshotStatus()
        {
            lock (_stateGate)
            {
                return (
                    _state,
                    _progress01,
                    Interlocked.Read(ref _tilesProduced),
                    Interlocked.Read(ref _tilesReturned),
                    _ownedBufferIds.Count,
                    _message
                );
            }
        }

        private static DateTime EnsureUtc(DateTime value)
        {
            if (value.Kind == DateTimeKind.Utc)
                return value;
            if (value.Kind == DateTimeKind.Unspecified)
                return DateTime.SpecifyKind(value, DateTimeKind.Utc);
            return value.ToUniversalTime();
        }

        private static Vector3d BuildSyntheticSunVector(DateTime timestampUtc)
        {
            var utc = EnsureUtc(timestampUtc);
            double seconds = (utc - DateTime.UnixEpoch).TotalSeconds;
            double angle = seconds / 86400.0 * 2.0 * Math.PI;
            var vector = new Vector3d(Math.Cos(angle), Math.Sin(angle), 0.2);
            vector.Normalize();
            return vector * 1737400.0;
        }

        public void Dispose()
        {
            Cancel();
            try
            {
                _producerTask?.Wait(TimeSpan.FromSeconds(5));
            }
            catch
            {
                // Best-effort shutdown for Python callers.
            }
            _cancellation?.Dispose();
        }

        private sealed record BufferLease(int BufferId, IntPtr Pointer, int ByteLength);

        private sealed class FillLightmapWorkItem
        {
            public long TileId { get; }
            public string HorizonPath { get; }
            public int PatchRow { get; set; } = -1;
            public int PatchCol { get; set; } = -1;
            public float[]? Horizons { get; set; }
            public string? ErrorMessage { get; set; }

            public FillLightmapWorkItem(long tileId, string horizonPath)
            {
                TileId = tileId;
                HorizonPath = horizonPath;
            }
        }
    }
}
