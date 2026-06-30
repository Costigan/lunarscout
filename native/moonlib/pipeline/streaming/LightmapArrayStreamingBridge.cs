using moonlib.horizon;
using moonlib.math;
using moonlib.spice;
using System.Collections.Concurrent;
using System.Threading.Channels;

namespace moonlib.pipeline.streaming
{
    public sealed record LightmapArrayStreamRequest(
        string ScenarioRootDir,
        string DemPath,
        IReadOnlyList<string>? SurroundingDemPaths,
        string HorizonDir,
        DateTime StartUtc,
        DateTime StopUtc,
        double TimeStepHours,
        float ObserverElevationMeters,
        int PatchWidth = 128,
        int PatchHeight = 128,
        int MaxReadParallelism = 4,
        int MaxComputeParallelism = 24,
        int ReadyQueueCapacity = 64,
        bool UseSpiceSunVectors = true
    );

    public enum StreamTileState
    {
        Ready,
        Error,
        Terminal
    }

    public sealed record TileEnvelope(
        string JobId,
        long TileId,
        int BufferId,
        int PatchRow,
        int PatchCol,
        int TimeCount,
        int Width,
        int Height,
        StreamTileState State,
        string? Message = null
    );

    public enum StreamJobState
    {
        Queued,
        Running,
        Cancelling,
        Cancelled,
        Completed,
        Failed
    }

    public sealed record LightmapArrayStreamStatus(
        string JobId,
        StreamJobState State,
        double Progress01,
        long TilesProduced,
        long TilesConsumed,
        int ReadyQueueDepth,
        int FreeBufferCount,
        string? Message = null
    );

    /// <summary>
    /// Streams computed lightmap tile arrays from C# directly into Python-owned NumPy buffers.
    ///
    /// Usage contract:
    /// 1. Call <see cref="StartLightmapArrayStreaming"/> with a <see cref="LightmapArrayStreamRequest"/>.
    /// 2. Register one or more output buffers with <see cref="RegisterOutputBuffer"/>.
    ///    Each buffer must be a fixed-size contiguous uint8 tile payload with shape (time, 128, 128).
    /// 3. Poll <see cref="TryGetNextTile"/> for <see cref="StreamTileState.Ready"/> envelopes.
    ///    The envelope identifies which registered buffer now holds a finished tile.
    /// 4. Read/consume that buffer on the Python side, then return it with <see cref="ReleaseBuffer"/>.
    /// 5. Observe status via <see cref="GetJobStatus"/>, and finish with <see cref="DisposeJob"/>.
    ///
    /// Backpressure behavior:
    /// - Ready tiles are published through a bounded ready queue (ReadyQueueCapacity).
    /// - Producers can only write into caller-registered buffers from the free-buffer pool.
    /// - If the Python consumer stops polling or stops releasing buffers, producers block rather than
    ///   growing memory unbounded.
    ///
    /// Output contract (current implementation):
    /// - Tile geometry: 128 x 128 pixels.
    /// - Payload type: uint8 lightmap fractions scaled to [0,255].
    /// - Tensor layout in each registered buffer: (time_count, height, width).
    /// </summary>
    public sealed partial class LightmapArrayStreamingBridge
    {
        private readonly ConcurrentDictionary<string, StreamingJob> _jobs = new();
        private readonly ConcurrentDictionary<string, StreamingJobV2> _jobsV2 = new();

        public string StartLightmapArrayStreaming(LightmapArrayStreamRequest request)
        {
            ArgumentNullException.ThrowIfNull(request);

            if (request.UseSpiceSunVectors)
            {
                var spice = SpiceManager.Singleton;
                _ = spice;
            }

            Console.WriteLine("C#: StartLightmapArrayStreaming");

            ValidateRequest(request);
            var timeCount = ComputeTimeCount(request.StartUtc, request.StopUtc, request.TimeStepHours);
            var expectedBytesLong = (long)timeCount * request.PatchWidth * request.PatchHeight;
            if (expectedBytesLong <= 0 || expectedBytesLong > int.MaxValue)
                throw new ArgumentOutOfRangeException(nameof(request), "Expected tile byte length must fit within Int32.");

            string jobId = Guid.NewGuid().ToString("N");
            var job = new StreamingJob(
                jobId: jobId,
                request: request,
                timeCount: timeCount,
                expectedTileByteLength: (int)expectedBytesLong
            );

            if (!_jobs.TryAdd(jobId, job))
            {
                job.Dispose();
                throw new InvalidOperationException("Failed to create a unique streaming job identifier.");
            }

            return jobId;
        }

        public bool RegisterOutputBuffer(string jobId, int bufferId, long ptr, int byteLength)
        {
            if (!_jobs.TryGetValue(jobId, out var job))
                return false;
            return job.RegisterBuffer(bufferId, ptr, byteLength);
        }

        public TileEnvelope? TryGetNextTile(string jobId, int timeoutMs)
        {
            //Console.WriteLine("C#: TryGetNextTile");
            if (!_jobs.TryGetValue(jobId, out var job))
                return null;
            return job.TryGetNextTile(timeoutMs);
        }

        public bool ReleaseBuffer(string jobId, int bufferId)
        {
            Console.WriteLine("C#: ReleaseBuffer");
            if (_jobs.TryGetValue(jobId, out var job))
                return job.ReleaseBuffer(bufferId);
            if (_jobsV2.TryGetValue(jobId, out var jobV2))
                return jobV2.ReleaseBuffer(bufferId);
            return false;
        }

        public LightmapArrayStreamStatus GetJobStatus(string jobId)
        {
            Console.WriteLine("C#: GetJobStatus");
            if (_jobs.TryGetValue(jobId, out var job))
                return job.GetStatus();
            if (_jobsV2.TryGetValue(jobId, out var jobV2))
                return jobV2.GetStatus();
            throw new KeyNotFoundException($"Streaming job not found: {jobId}");
        }

        public bool CancelJob(string jobId)
        {
            Console.WriteLine("C#: CancelJob");
            if (_jobs.TryGetValue(jobId, out var job))
            {
                job.Cancel();
                return true;
            }
            if (_jobsV2.TryGetValue(jobId, out var jobV2))
            {
                jobV2.Cancel();
                return true;
            }
            return false;
        }

        public bool DisposeJob(string jobId)
        {
            Console.WriteLine("C#: DisposeJob");
            if (_jobs.TryRemove(jobId, out var job))
            {
                job.Dispose();
                return true;
            }
            if (_jobsV2.TryRemove(jobId, out var jobV2))
            {
                jobV2.Dispose();
                return true;
            }
            return false;
        }

        private static void ValidateRequest(LightmapArrayStreamRequest request)
        {
            if (string.IsNullOrWhiteSpace(request.ScenarioRootDir))
                throw new ArgumentException("ScenarioRootDir is required.", nameof(request));
            if (string.IsNullOrWhiteSpace(request.DemPath))
                throw new ArgumentException("DemPath is required.", nameof(request));
            if (string.IsNullOrWhiteSpace(request.HorizonDir))
                throw new ArgumentException("HorizonDir is required.", nameof(request));
            if (!File.Exists(request.DemPath))
                throw new FileNotFoundException("DEM path does not exist.", request.DemPath);
            if (!Directory.Exists(request.HorizonDir))
                throw new DirectoryNotFoundException($"Horizon directory does not exist: {request.HorizonDir}");
            if (request.TimeStepHours <= 0.0)
                throw new ArgumentOutOfRangeException(nameof(request), "TimeStepHours must be > 0.");
            if (request.StopUtc < request.StartUtc)
                throw new ArgumentOutOfRangeException(nameof(request), "StopUtc must be >= StartUtc.");
            if (request.PatchWidth != 128 || request.PatchHeight != 128)
                throw new NotSupportedException("Only 128x128 horizon tile geometry is currently supported.");
            if (request.ReadyQueueCapacity < 1)
                throw new ArgumentOutOfRangeException(nameof(request), "ReadyQueueCapacity must be >= 1.");
            if (request.MaxReadParallelism < 1)
                throw new ArgumentOutOfRangeException(nameof(request), "MaxReadParallelism must be >= 1.");
            if (request.MaxComputeParallelism < 1)
                throw new ArgumentOutOfRangeException(nameof(request), "MaxComputeParallelism must be >= 1.");
        }

        private static int ComputeTimeCount(DateTime startUtc, DateTime stopUtc, double timeStepHours)
        {
            var start = EnsureUtc(startUtc);
            var stop = EnsureUtc(stopUtc);
            if (stop < start)
                throw new ArgumentOutOfRangeException(nameof(stopUtc), "StopUtc must be >= StartUtc.");

            var step = TimeSpan.FromHours(timeStepHours);
            if (step <= TimeSpan.Zero)
                throw new ArgumentOutOfRangeException(nameof(timeStepHours), "TimeStepHours must be > 0.");

            int count = 0;
            var current = start;
            while (current <= stop)
            {
                checked { count += 1; }
                current = current.Add(step);
            }

            if (count < 1)
                count = 1;
            return count;
        }

        private static DateTime EnsureUtc(DateTime value)
        {
            if (value.Kind == DateTimeKind.Utc)
                return value;
            if (value.Kind == DateTimeKind.Unspecified)
                return DateTime.SpecifyKind(value, DateTimeKind.Utc);
            return value.ToUniversalTime();
        }

        private sealed class StreamingJob : IDisposable
        {
            private readonly ConcurrentDictionary<int, RegisteredBuffer> _buffers = new();
            private readonly Channel<int> _freeBufferIds;
            private readonly Channel<TileEnvelope> _readyTiles;
            private readonly CancellationTokenSource _cancellation = new();
            private readonly object _statusGate = new();
            private readonly Task _producerTask;

            private StreamJobState _state = StreamJobState.Queued;
            private string? _message;
            private double _progress01;
            private long _tilesProduced;
            private long _tilesConsumed;
            private long _nextTileId;
            private int _readyDepth;
            private int _freeBufferCount;

            public string JobId { get; }
            public LightmapArrayStreamRequest Request { get; }
            public int TimeCount { get; }
            public int ExpectedTileByteLength { get; }

            public StreamingJob(
                string jobId,
                LightmapArrayStreamRequest request,
                int timeCount,
                int expectedTileByteLength)
            {
                JobId = jobId;
                Request = request;
                TimeCount = timeCount;
                ExpectedTileByteLength = expectedTileByteLength;

                _freeBufferIds = Channel.CreateUnbounded<int>(new UnboundedChannelOptions
                {
                    SingleReader = false,
                    SingleWriter = false
                });
                _readyTiles = Channel.CreateBounded<TileEnvelope>(new BoundedChannelOptions(Math.Max(1, request.ReadyQueueCapacity))
                {
                    FullMode = BoundedChannelFullMode.Wait,
                    SingleReader = false,
                    SingleWriter = false
                });
                _producerTask = Task.Run(ProducerLoop);
            }

            public bool RegisterBuffer(int bufferId, long ptr, int byteLength)
            {
                if (bufferId < 0 || ptr <= 0 || byteLength <= 0)
                    return false;
                if (byteLength != ExpectedTileByteLength)
                    return false;
                if (IsTerminalState())
                    return false;

                var buffer = new RegisteredBuffer(bufferId, new IntPtr(ptr), byteLength);
                if (!_buffers.TryAdd(bufferId, buffer))
                    return false;

                if (!_freeBufferIds.Writer.TryWrite(bufferId))
                {
                    _buffers.TryRemove(bufferId, out _);
                    return false;
                }

                Interlocked.Increment(ref _freeBufferCount);
                return true;
            }

            public TileEnvelope? TryGetNextTile(int timeoutMs)
            {
                if (timeoutMs < 0)
                    timeoutMs = 0;

                if (_readyTiles.Reader.TryRead(out var envelope))
                {
                    Interlocked.Decrement(ref _readyDepth);
                    return envelope;
                }

                if (timeoutMs == 0)
                    return null;

                using var timeout = new CancellationTokenSource(timeoutMs);
                try
                {
                    var available = _readyTiles.Reader.WaitToReadAsync(timeout.Token).AsTask().GetAwaiter().GetResult();
                    if (!available)
                        return null;
                }
                catch (OperationCanceledException)
                {
                    return null;
                }

                if (_readyTiles.Reader.TryRead(out envelope))
                {
                    Interlocked.Decrement(ref _readyDepth);
                    return envelope;
                }
                return null;
            }

            public bool ReleaseBuffer(int bufferId)
            {
                if (!_buffers.TryGetValue(bufferId, out var buffer))
                    return false;
                if (!buffer.TryMarkFree())
                    return false;
                bool wrote = _freeBufferIds.Writer.TryWrite(bufferId);
                if (wrote)
                    Interlocked.Increment(ref _freeBufferCount);
                else if (!IsTerminalState())
                    return false;
                Interlocked.Increment(ref _tilesConsumed);
                return true;
            }

            public LightmapArrayStreamStatus GetStatus()
            {
                StreamJobState state;
                string? message;
                double progress;
                lock (_statusGate)
                {
                    state = _state;
                    message = _message;
                    progress = _progress01;
                }
                return new LightmapArrayStreamStatus(
                    JobId: JobId,
                    State: state,
                    Progress01: progress,
                    TilesProduced: Interlocked.Read(ref _tilesProduced),
                    TilesConsumed: Interlocked.Read(ref _tilesConsumed),
                    ReadyQueueDepth: Volatile.Read(ref _readyDepth),
                    FreeBufferCount: Volatile.Read(ref _freeBufferCount),
                    Message: message
                );
            }

            public void Cancel()
            {
                lock (_statusGate)
                {
                    if (_state == StreamJobState.Queued || _state == StreamJobState.Running)
                    {
                        _state = StreamJobState.Cancelling;
                        _message = "Cancellation requested.";
                    }
                }
                _cancellation.Cancel();
            }

            private void ProducerLoop()
            {
                try
                {
                    SetState(StreamJobState.Running, "Streaming started.");

                    var horizonFiles = EnumerateHorizonFiles(Request.HorizonDir).ToList();

                    if (horizonFiles.Count == 0)
                    {
                        SetProgress(1.0);
                        EnqueueTerminal("No horizon tiles found.");
                        SetState(StreamJobState.Completed, "Streaming completed with no tiles.");
                        return;
                    }

                    var times = BuildTimes(Request.StartUtc, Request.StopUtc, Request.TimeStepHours);
                    var sunVectors = BuildSunVectors(times, Request.UseSpiceSunVectors);
                    var dem = new ElevationMap(Request.DemPath);
                    var boundedCapacity = Math.Max(1, Request.ReadyQueueCapacity);

                    var readStep = new PipelineStep<StreamingTileWorkItem, StreamingTileWorkItem>(ReadHorizonsStepAsync);
                    var computeStep = new PipelineStep<StreamingTileWorkItem, StreamingTileWorkItem>(
                        item => ComputeTileStepAsync(item, dem, sunVectors, horizonFiles.Count));

                    var pipeline = new Pipeline<StreamingTileWorkItem>();
                    pipeline.AddStep(
                        readStep.Func,
                        maxDegreeOfParallelism: Request.MaxReadParallelism,
                        boundedCapacity: boundedCapacity,
                        ensureOrdered: false);
                    pipeline.AddStep(
                        computeStep.Func,
                        maxDegreeOfParallelism: Request.MaxComputeParallelism,
                        boundedCapacity: boundedCapacity,
                        ensureOrdered: false);
                    pipeline.AddTerminalStep(
                        _ => Task.CompletedTask,
                        maxDegreeOfParallelism: 1,
                        boundedCapacity: boundedCapacity,
                        ensureOrdered: false);

                    pipeline
                        .ProcessAsync(BuildWorkItems(horizonFiles, _cancellation.Token))
                        .GetAwaiter()
                        .GetResult();

                    SetProgress(1.0);
                    EnqueueTerminal("Streaming completed.");
                    SetState(StreamJobState.Completed, "Streaming completed.");
                }
                catch (OperationCanceledException)
                {
                    EnqueueTerminal("Streaming cancelled.");
                    SetState(StreamJobState.Cancelled, "Streaming cancelled.");
                }
                catch (Exception ex)
                {
                    EnqueueTerminal($"Streaming failed: {ex.Message}");
                    SetState(StreamJobState.Failed, ex.Message);
                }
                finally
                {
                    if (GetState() is StreamJobState.Cancelled or StreamJobState.Failed)
                        ReclaimInUseBuffers();
                    _readyTiles.Writer.TryComplete();
                }
            }

            private IEnumerable<StreamingTileWorkItem> BuildWorkItems(
                IReadOnlyList<string> horizonFiles,
                CancellationToken cancellationToken)
            {
                foreach (var horizonPath in horizonFiles)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    yield return new StreamingTileWorkItem(
                        tileId: Interlocked.Increment(ref _nextTileId),
                        horizonPath: horizonPath
                    );
                }
            }

            private Task<StreamingTileWorkItem> ReadHorizonsStepAsync(StreamingTileWorkItem item)
            {
                _cancellation.Token.ThrowIfCancellationRequested();

                try
                {
                    (item.PatchCol, item.PatchRow, _) = QuadTreeHorizonGenerator.ParseHorizonFilename(item.HorizonPath);
                    if (item.PatchRow < 0 || item.PatchCol < 0)
                        throw new InvalidDataException($"Invalid horizon tile filename: {item.HorizonPath}");

                    item.Horizons = HorizonFile.ReadHorizonFile(item.HorizonPath);
                    item.ErrorMessage = null;
                }
                catch (OperationCanceledException)
                {
                    throw;
                }
                catch (Exception ex)
                {
                    item.ErrorMessage = ex.Message;
                }

                return Task.FromResult(item);
            }

            private Task<StreamingTileWorkItem> ComputeTileStepAsync(
                StreamingTileWorkItem item,
                ElevationMap dem,
                List<Vector3d> sunVectors,
                int totalTileCount)
            {
                int bufferId = -1;

                try
                {
                    _cancellation.Token.ThrowIfCancellationRequested();

                    if (!string.IsNullOrWhiteSpace(item.ErrorMessage))
                    {
                        EnqueueError(item, bufferId, item.ErrorMessage!);
                        return Task.FromResult(item);
                    }

                    if (item.Horizons is null)
                        throw new InvalidDataException($"Missing horizons for tile: {item.HorizonPath}");

                    if (item.PatchRow < 0 || item.PatchCol < 0)
                        throw new InvalidDataException($"Missing patch row/col for tile: {item.HorizonPath}");

                    bufferId = AcquireNextFreeBuffer(_cancellation.Token);
                    if (!_buffers.TryGetValue(bufferId, out var buffer))
                        throw new InvalidOperationException($"Unknown buffer id returned by free pool: {bufferId}");

                    WriteTileToRegisteredBuffer(
                        targetPtr: buffer.Pointer,
                        targetByteLength: buffer.ByteLength,
                        horizons: item.Horizons,
                        dem: dem,
                        patchRow: item.PatchRow,
                        patchCol: item.PatchCol,
                        patchWidth: Request.PatchWidth,
                        patchHeight: Request.PatchHeight,
                        sunVectors: sunVectors
                    );

                    EnqueueReady(new TileEnvelope(
                        JobId: JobId,
                        TileId: item.TileId,
                        BufferId: bufferId,
                        PatchRow: item.PatchRow,
                        PatchCol: item.PatchCol,
                        TimeCount: TimeCount,
                        Width: Request.PatchWidth,
                        Height: Request.PatchHeight,
                        State: StreamTileState.Ready
                    ));
                }
                catch (OperationCanceledException)
                {
                    if (bufferId >= 0)
                        ReturnBufferToFreePool(bufferId, countConsumed: false);
                    throw;
                }
                catch (Exception ex)
                {
                    if (bufferId >= 0)
                        ReturnBufferToFreePool(bufferId, countConsumed: false);

                    EnqueueError(item, bufferId, ex.Message);
                }
                finally
                {
                    Interlocked.Increment(ref _tilesProduced);
                    var produced = Interlocked.Read(ref _tilesProduced);
                    SetProgress(Math.Clamp((double)produced / Math.Max(1, totalTileCount), 0.0, 1.0));
                }

                return Task.FromResult(item);
            }

            private void EnqueueError(StreamingTileWorkItem item, int bufferId, string message)
            {
                EnqueueReady(new TileEnvelope(
                    JobId: JobId,
                    TileId: item.TileId,
                    BufferId: bufferId,
                    PatchRow: item.PatchRow,
                    PatchCol: item.PatchCol,
                    TimeCount: TimeCount,
                    Width: Request.PatchWidth,
                    Height: Request.PatchHeight,
                    State: StreamTileState.Error,
                    Message: message
                ));
            }

            private static List<DateTime> BuildTimes(DateTime startUtc, DateTime stopUtc, double timeStepHours)
            {
                var start = LightmapArrayStreamingBridge.EnsureUtc(startUtc);
                var stop = LightmapArrayStreamingBridge.EnsureUtc(stopUtc);
                if (stop < start)
                    throw new ArgumentOutOfRangeException(nameof(stopUtc), "StopUtc must be >= StartUtc.");

                var step = TimeSpan.FromHours(timeStepHours);
                if (step <= TimeSpan.Zero)
                    throw new ArgumentOutOfRangeException(nameof(timeStepHours), "TimeStepHours must be > 0.");

                var output = new List<DateTime>();
                var current = start;
                while (current <= stop)
                {
                    output.Add(current);
                    current = current.Add(step);
                }
                if (output.Count == 0)
                    output.Add(start);
                return output;
            }

            private static List<Vector3d> BuildSunVectors(IReadOnlyList<DateTime> times, bool useSpice)
            {
                if (useSpice)
                    return times.Select(t => SpiceManager.SunPosition(t) * 1000.0).ToList();

                return times.Select(BuildSyntheticSunVector).ToList();
            }

            private static Vector3d BuildSyntheticSunVector(DateTime timestampUtc)
            {
                var utc = LightmapArrayStreamingBridge.EnsureUtc(timestampUtc);
                double seconds = (utc - DateTime.UnixEpoch).TotalSeconds;
                double angle = seconds / 86400.0 * 2.0 * Math.PI;
                var vector = new Vector3d(Math.Cos(angle), Math.Sin(angle), 0.2);
                vector.Normalize();
                return vector * 1737400.0;
            }

            private static IEnumerable<string> EnumerateHorizonFiles(string horizonDir)
            {
                return new HorizonTileStore(horizonDir)
                    .EnumerateFiles()
                    .OrderBy(path => path, StringComparer.OrdinalIgnoreCase);
            }

            private int AcquireNextFreeBuffer(CancellationToken cancellationToken)
            {
                while (true)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var bufferId = _freeBufferIds.Reader.ReadAsync(cancellationToken).AsTask().GetAwaiter().GetResult();
                    Interlocked.Decrement(ref _freeBufferCount);
                    if (!_buffers.TryGetValue(bufferId, out var buffer))
                        continue;
                    if (buffer.TryMarkInUse())
                        return bufferId;
                }
            }

            private void ReturnBufferToFreePool(int bufferId, bool countConsumed)
            {
                if (!_buffers.TryGetValue(bufferId, out var buffer))
                    return;
                if (!buffer.TryMarkFree())
                    return;
                if (!_freeBufferIds.Writer.TryWrite(bufferId))
                    return;
                if (countConsumed)
                    Interlocked.Increment(ref _tilesConsumed);
                Interlocked.Increment(ref _freeBufferCount);
            }

            private void ReclaimInUseBuffers()
            {
                foreach (var pair in _buffers)
                {
                    var buffer = pair.Value;
                    if (!buffer.TryMarkFree())
                        continue;
                    if (_freeBufferIds.Writer.TryWrite(buffer.BufferId))
                        Interlocked.Increment(ref _freeBufferCount);
                }
            }

            private void EnqueueReady(TileEnvelope envelope)
            {
                _readyTiles.Writer.WriteAsync(envelope, _cancellation.Token).AsTask().GetAwaiter().GetResult();
                Interlocked.Increment(ref _readyDepth);
            }

            private void EnqueueTerminal(string message)
            {
                var terminal = new TileEnvelope(
                    JobId: JobId,
                    TileId: Interlocked.Increment(ref _nextTileId),
                    BufferId: -1,
                    PatchRow: -1,
                    PatchCol: -1,
                    TimeCount: TimeCount,
                    Width: Request.PatchWidth,
                    Height: Request.PatchHeight,
                    State: StreamTileState.Terminal,
                    Message: message
                );
                if (_readyTiles.Writer.TryWrite(terminal))
                    Interlocked.Increment(ref _readyDepth);
            }

            private static unsafe void WriteTileToRegisteredBuffer(
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

                var horizonSampleCount = patchWidth * patchHeight * LightmapGenerator.HorizonSamples;
                if (horizons.Length < horizonSampleCount)
                    throw new InvalidDataException(
                        $"Unexpected horizon sample count. Expected at least {horizonSampleCount}, got {horizons.Length}.");

                var expectedBytes = sunVectors.Count * patchWidth * patchHeight;
                if (targetByteLength != expectedBytes)
                    throw new InvalidDataException(
                        $"Unexpected output byte length. Expected {expectedBytes}, got {targetByteLength}.");

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

                var litCount = 0;

                int tilePixelCount = patchWidth * patchHeight;
                for (int timeIndex = 0; timeIndex < sunVectors.Count; timeIndex++)
                {
                    var sunVec = sunVectors[timeIndex];
                    int outputBase = timeIndex * tilePixelCount;
                    for (int y = 0; y < patchHeight; y++)
                    {
                        for (int x = 0; x < patchWidth; x++)
                        {
                            int pixelIndex = y * patchWidth + x;
                            int horizonBase = pixelIndex * LightmapGenerator.HorizonSamples;
                            var (azimuthRad, elevationRad) = dem.GetAzEl(sunVec, matrices[y, x]);
                            float azimuthDeg = azimuthRad * 57.2957795f;
                            float elevationDeg = elevationRad * 57.2957795f;

                            float sunFraction = LightmapGenerator.BuilderSunFraction(
                                horizons,
                                horizonBase,
                                azimuthDeg,
                                elevationDeg
                            );
                            destination[outputBase + pixelIndex] = (byte)(255f * sunFraction);

                            if (sunFraction > 0f) litCount++;
                        }
                    }
                }

                Console.WriteLine($"C#: litCount={litCount}");
            }

            private void SetState(StreamJobState state, string? message)
            {
                lock (_statusGate)
                {
                    _state = state;
                    _message = message;
                }
            }

            private void SetProgress(double progress)
            {
                lock (_statusGate)
                {
                    _progress01 = Math.Clamp(progress, 0.0, 1.0);
                }
            }

            private bool IsTerminalState()
            {
                var state = GetState();
                return state is StreamJobState.Cancelled or StreamJobState.Completed or StreamJobState.Failed;
            }

            private StreamJobState GetState()
            {
                lock (_statusGate)
                {
                    return _state;
                }
            }

            public void Dispose()
            {
                _cancellation.Cancel();
                try
                {
                    _producerTask.Wait(TimeSpan.FromSeconds(5));
                }
                catch
                {
                    // Best-effort shutdown; callers can dispose repeatedly.
                }
                _cancellation.Dispose();
            }
        }

        private sealed class StreamingTileWorkItem
        {
            public long TileId { get; }
            public string HorizonPath { get; }
            public int PatchRow { get; set; } = -1;
            public int PatchCol { get; set; } = -1;
            public float[]? Horizons { get; set; }
            public string? ErrorMessage { get; set; }

            public StreamingTileWorkItem(long tileId, string horizonPath)
            {
                TileId = tileId;
                HorizonPath = horizonPath;
            }
        }

        private sealed class RegisteredBuffer
        {
            private int _inUse;

            public int BufferId { get; }
            public IntPtr Pointer { get; }
            public int ByteLength { get; }

            public RegisteredBuffer(int bufferId, IntPtr pointer, int byteLength)
            {
                BufferId = bufferId;
                Pointer = pointer;
                ByteLength = byteLength;
                _inUse = 0;
            }

            public bool TryMarkInUse()
            {
                return Interlocked.CompareExchange(ref _inUse, 1, 0) == 0;
            }

            public bool TryMarkFree()
            {
                return Interlocked.CompareExchange(ref _inUse, 0, 1) == 1;
            }
        }
    }
}
