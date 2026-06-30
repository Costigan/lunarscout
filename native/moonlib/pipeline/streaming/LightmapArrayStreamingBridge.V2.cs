using moonlib.horizon;
using moonlib.math;
using moonlib.spice;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using System.Threading.Channels;

namespace moonlib.pipeline.streaming
{
    public enum LightmapStreamMode
    {
        SignalStream = 1,
        NativeReduce = 2
    }

    public enum TemporalSignalKind
    {
        SunFractionU8 = 1,
        SunCenterMarginDegF32 = 2,
        EarthCenterMarginDegF32 = 3
    }

    public enum StreamScalarType
    {
        UInt8 = 1,
        Float32 = 2
    }

    public enum TemporalThresholdReference
    {
        CenterMargin = 1,
        LowerLimbMargin = 2,
        UpperLimbMargin = 3
    }

    public enum NativeReducerKind
    {
        AverageSunFraction = 1,
        CumulativeDurationWhere = 2,
        MaxContiguousDurationWhere = 3,
        CombinedSunEarthContiguousDuration = 4
    }

    public enum DurationOutputUnit
    {
        Samples = 1,
        Hours = 2
    }

    public enum ReducedTileOutputType
    {
        UInt8 = 1,
        UInt16 = 2,
        Float32 = 3
    }

    public sealed record TemporalSignalSpec(
        TemporalSignalKind Signal,
        int ChannelIndex,
        bool Enabled = true
    );

    public sealed record TemporalSignalStreamLayout(
        int ChunkTimeCount = 256,
        bool InterleaveChannels = false
    );

    public sealed record ThresholdPredicateSpec(
        TemporalSignalKind Signal,
        TemporalThresholdReference Reference = TemporalThresholdReference.CenterMargin,
        float ThresholdValue = 0f,
        bool GreaterThanOrEqual = true,
        float BodyRadiusDegOverride = float.NaN
    );

    public sealed record SunFractionPredicateSpec(
        byte MinSunFractionU8 = 1,
        bool GreaterThanOrEqual = true
    );

    public abstract record NativeReducerSpec(
        NativeReducerKind Kind,
        ReducedTileOutputType OutputType = ReducedTileOutputType.Float32
    );

    public sealed record AverageSunFractionReducerSpec(
        bool OutputNormalized01 = true,
        ReducedTileOutputType OutputType = ReducedTileOutputType.Float32
    ) : NativeReducerSpec(NativeReducerKind.AverageSunFraction, OutputType);

    public sealed record CumulativeDurationWhereReducerSpec(
        ThresholdPredicateSpec? MarginPredicate = null,
        SunFractionPredicateSpec? SunPredicate = null,
        DurationOutputUnit Unit = DurationOutputUnit.Hours,
        ReducedTileOutputType OutputType = ReducedTileOutputType.Float32
    ) : NativeReducerSpec(NativeReducerKind.CumulativeDurationWhere, OutputType);

    public sealed record MaxContiguousDurationWhereReducerSpec(
        ThresholdPredicateSpec? MarginPredicate = null,
        SunFractionPredicateSpec? SunPredicate = null,
        DurationOutputUnit Unit = DurationOutputUnit.Hours,
        ReducedTileOutputType OutputType = ReducedTileOutputType.Float32
    ) : NativeReducerSpec(NativeReducerKind.MaxContiguousDurationWhere, OutputType);

    public sealed record CombinedSunEarthContiguousDurationReducerSpec(
        SunFractionPredicateSpec SunPredicate,
        ThresholdPredicateSpec EarthMarginPredicate,
        DurationOutputUnit Unit = DurationOutputUnit.Hours,
        ReducedTileOutputType OutputType = ReducedTileOutputType.Float32
    ) : NativeReducerSpec(NativeReducerKind.CombinedSunEarthContiguousDuration, OutputType);

    public sealed record NativeReduceRasterOutputSpec(
        string OutputPath,
        double NoDataValue = -9999.0,
        int SelectedBandIndex = 1,
        string Compression = "DEFLATE"
    );

    public sealed record NativeReduceRasterResult(
        string OutputPath,
        int TilesWritten,
        double? ValueMin,
        double? ValueMax,
        long SizeBytes
    );

    public sealed record LightmapArrayStreamRequestV2(
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

        LightmapStreamMode Mode = LightmapStreamMode.SignalStream,
        IReadOnlyList<TemporalSignalSpec>? Signals = null,
        TemporalSignalStreamLayout? SignalLayout = null,
        IReadOnlyList<NativeReducerSpec>? Reducers = null,
        
        // These are used in tests. In production, SPICE vectors are always used.
        bool UseSpiceSunVectors = true,
        bool UseSpiceEarthVectors = true,
        NativeReduceRasterOutputSpec? RasterOutput = null
    );

    public sealed record TileEnvelopeV2(
        string JobId,
        long TileId,
        int BufferId,
        int PatchRow,
        int PatchCol,
        int Width,
        int Height,
        StreamTileState State,
        StreamScalarType ScalarType,
        int Rank,
        int Dim0,
        int Dim1,
        int Dim2,
        int Dim3,
        int TimeOffset,
        int TimeCount,
        int ChannelCount,
        string? Message = null
    );

    public sealed partial class LightmapArrayStreamingBridge
    {
        public string StartLightmapArrayStreamingV2(LightmapArrayStreamRequestV2 request)
        {
            ArgumentNullException.ThrowIfNull(request);

            Console.WriteLine("C#: StartLightmapArrayStreamingV2");
            var jobConfig = ValidateAndBuildV2Config(request);
            if (request.UseSpiceSunVectors || (jobConfig.HasEarthSignals && request.UseSpiceEarthVectors))
            {
                var spice = SpiceManager.Singleton;
                _ = spice;
            }
            var timeCount = ComputeTimeCount(request.StartUtc, request.StopUtc, request.TimeStepHours);

            var bytesPerScalar = jobConfig.ScalarType switch
            {
                StreamScalarType.UInt8 => 1,
                StreamScalarType.Float32 => 4,
                _ => throw new NotSupportedException($"Unsupported scalar type: {jobConfig.ScalarType}")
            };

            var expectedBytesLong = request.Mode switch
            {
                LightmapStreamMode.SignalStream =>
                    (long)jobConfig.MaxChunkTimeCount * jobConfig.ChannelCount * request.PatchWidth * request.PatchHeight * bytesPerScalar,
                LightmapStreamMode.NativeReduce =>
                    (long)jobConfig.ChannelCount * request.PatchWidth * request.PatchHeight * bytesPerScalar,
                _ => throw new NotSupportedException($"Unsupported V2 mode: {request.Mode}")
            };
            if (expectedBytesLong <= 0 || expectedBytesLong > int.MaxValue)
                throw new ArgumentOutOfRangeException(nameof(request), "Expected V2 tile byte length must fit within Int32.");

            string jobId = Guid.NewGuid().ToString("N");
            var job = new StreamingJobV2(
                jobId: jobId,
                request: request,
                timeCount: timeCount,
                expectedTileByteLength: (int)expectedBytesLong,
                config: jobConfig
            );

            if (!_jobsV2.TryAdd(jobId, job))
            {
                job.Dispose();
                throw new InvalidOperationException("Failed to create a unique streaming V2 job identifier.");
            }

            return jobId;
        }

        public bool RegisterOutputBufferV2(string jobId, int bufferId, long ptr, int byteLength)
        {
            if (!_jobsV2.TryGetValue(jobId, out var job))
                return false;
            return job.RegisterBuffer(bufferId, ptr, byteLength);
        }

        public TileEnvelopeV2? TryGetNextTileV2(string jobId, int timeoutMs)
        {
            if (!_jobsV2.TryGetValue(jobId, out var job))
                return null;
            return job.TryGetNextTile(timeoutMs);
        }

        public NativeReduceRasterResult? GetNativeReduceRasterResult(string jobId)
        {
            if (!_jobsV2.TryGetValue(jobId, out var job))
                return null;
            return job.GetRasterResult();
        }

        private static V2JobConfig ValidateAndBuildV2Config(LightmapArrayStreamRequestV2 request)
        {
            ValidateRequest(new LightmapArrayStreamRequest(
                request.ScenarioRootDir,
                request.DemPath,
                request.SurroundingDemPaths,
                request.HorizonDir,
                request.StartUtc,
                request.StopUtc,
                request.TimeStepHours,
                request.ObserverElevationMeters,
                request.PatchWidth,
                request.PatchHeight,
                request.MaxReadParallelism,
                request.MaxComputeParallelism,
                request.ReadyQueueCapacity,
                request.UseSpiceSunVectors
            ));

            var signalLayout = request.SignalLayout ?? new TemporalSignalStreamLayout();
            if (signalLayout.ChunkTimeCount < 1)
                throw new ArgumentOutOfRangeException(nameof(request), "SignalLayout.ChunkTimeCount must be >= 1.");
            if (signalLayout.InterleaveChannels)
                throw new NotSupportedException("V2 supports only [time, channel, height, width] layout.");

            if (request.Mode == LightmapStreamMode.SignalStream)
            {
                if (request.Reducers is { Count: > 0 })
                    throw new NotSupportedException("SignalStream mode does not accept reducers.");

                var enabledSignals = (request.Signals is null || request.Signals.Count == 0)
                    ? new List<TemporalSignalSpec> { new(TemporalSignalKind.SunFractionU8, 0, true) }
                    : request.Signals.Where(s => s.Enabled).ToList();

                if (enabledSignals.Count == 0)
                    throw new ArgumentException("At least one enabled V2 signal is required.", nameof(request));

                var orderedSignals = enabledSignals.OrderBy(s => s.ChannelIndex).ToList();
                for (int i = 0; i < orderedSignals.Count; i++)
                    ValidateSignalSpec(orderedSignals[i], i);

                bool hasSunFraction = orderedSignals.Any(s => s.Signal == TemporalSignalKind.SunFractionU8);
                bool hasSunCenterMargin = orderedSignals.Any(s => s.Signal == TemporalSignalKind.SunCenterMarginDegF32);
                bool hasEarthCenterMargin = orderedSignals.Any(s => s.Signal == TemporalSignalKind.EarthCenterMarginDegF32);
                bool hasAngleSignals = hasSunCenterMargin || hasEarthCenterMargin;
                bool hasEarthSignals = hasEarthCenterMargin;

                if (hasEarthSignals && !request.UseSpiceEarthVectors)
                    throw new NotSupportedException("Earth margin signals currently require UseSpiceEarthVectors=true.");

                var scalarType = hasAngleSignals ? StreamScalarType.Float32 : StreamScalarType.UInt8;
                if (scalarType == StreamScalarType.UInt8 &&
                    (orderedSignals.Count != 1 || orderedSignals[0].Signal != TemporalSignalKind.SunFractionU8))
                {
                    throw new NotSupportedException("UInt8 SignalStream payloads support exactly one signal: SunFractionU8.");
                }

                return new V2JobConfig(
                    Mode: request.Mode,
                    ScalarType: scalarType,
                    MaxChunkTimeCount: signalLayout.ChunkTimeCount,
                    ChannelCount: orderedSignals.Count,
                    Signals: orderedSignals,
                    Reducers: Array.Empty<NativeReducerSpec>(),
                    HasSunFraction: hasSunFraction,
                    HasSunCenterMargin: hasSunCenterMargin,
                    HasEarthCenterMargin: hasEarthCenterMargin,
                    HasEarthSignals: hasEarthSignals
                );
            }

            if (request.Mode != LightmapStreamMode.NativeReduce)
                throw new NotSupportedException($"Unsupported V2 mode: {request.Mode}");

            if (request.RasterOutput is not null)
            {
                if (string.IsNullOrWhiteSpace(request.RasterOutput.OutputPath))
                    throw new ArgumentException("NativeReduce raster output path must be non-empty.", nameof(request));
                if (request.RasterOutput.SelectedBandIndex < 1)
                    throw new ArgumentOutOfRangeException(nameof(request), "Selected raster band index must be >= 1.");
                string compression = request.RasterOutput.Compression.Trim().ToUpperInvariant();
                if (compression is not ("DEFLATE" or "LZW" or "ZSTD"))
                    throw new NotSupportedException($"Unsupported NativeReduce raster compression: {compression}");
            }

            if (request.Signals is { Count: > 0 })
                throw new NotSupportedException("NativeReduce mode does not accept explicit Signals.");

            var reducers = request.Reducers?.ToList() ?? new List<NativeReducerSpec>();
            if (reducers.Count == 0)
                throw new ArgumentException("NativeReduce mode requires at least one reducer.", nameof(request));

            bool needSunFraction = false;
            bool needSunCenterMargin2 = false;
            bool needEarthCenterMargin2 = false;

            for (int i = 0; i < reducers.Count; i++)
            {
                var reducer = reducers[i] ?? throw new ArgumentException($"Null reducer at index {i}.", nameof(request));
                if (reducer.OutputType != ReducedTileOutputType.Float32)
                    throw new NotSupportedException("Phase 3 V2 NativeReduce supports only Float32 output.");

                switch (reducer)
                {
                    case AverageSunFractionReducerSpec:
                        needSunFraction = true;
                        break;
                    case CumulativeDurationWhereReducerSpec spec:
                        ValidatePredicateCombination(spec.MarginPredicate, spec.SunPredicate);
                        InferReducerSignals(spec.MarginPredicate, spec.SunPredicate, ref needSunFraction, ref needSunCenterMargin2, ref needEarthCenterMargin2);
                        break;
                    case MaxContiguousDurationWhereReducerSpec spec:
                        ValidatePredicateCombination(spec.MarginPredicate, spec.SunPredicate);
                        InferReducerSignals(spec.MarginPredicate, spec.SunPredicate, ref needSunFraction, ref needSunCenterMargin2, ref needEarthCenterMargin2);
                        break;
                    case CombinedSunEarthContiguousDurationReducerSpec spec:
                        if (spec.SunPredicate is null)
                            throw new ArgumentException("CombinedSunEarthContiguousDuration requires SunPredicate.", nameof(request));
                        if (spec.EarthMarginPredicate is null)
                            throw new ArgumentException("CombinedSunEarthContiguousDuration requires EarthMarginPredicate.", nameof(request));
                        if (spec.EarthMarginPredicate.Signal != TemporalSignalKind.EarthCenterMarginDegF32)
                            throw new NotSupportedException("CombinedSunEarthContiguousDuration Earth predicate must target EarthCenterMarginDegF32.");
                        needSunFraction = true;
                        needEarthCenterMargin2 = true;
                        break;
                    default:
                        throw new NotSupportedException($"Unsupported NativeReduce reducer spec type: {reducer.GetType().Name}");
                }
            }

            if (needEarthCenterMargin2 && !request.UseSpiceEarthVectors)
                throw new NotSupportedException("Earth-dependent reducers currently require UseSpiceEarthVectors=true.");

            return new V2JobConfig(
                Mode: request.Mode,
                ScalarType: StreamScalarType.Float32,
                MaxChunkTimeCount: signalLayout.ChunkTimeCount,
                ChannelCount: reducers.Count,
                Signals: Array.Empty<TemporalSignalSpec>(),
                Reducers: reducers,
                HasSunFraction: needSunFraction,
                HasSunCenterMargin: needSunCenterMargin2,
                HasEarthCenterMargin: needEarthCenterMargin2,
                HasEarthSignals: needEarthCenterMargin2
            );
        }

        private static void ValidateSignalSpec(TemporalSignalSpec spec, int expectedChannelIndex)
        {
            if (spec.ChannelIndex != expectedChannelIndex)
                throw new NotSupportedException("V2 signals must use contiguous ChannelIndex values starting at 0.");
            if (spec.Signal is not (TemporalSignalKind.SunFractionU8 or TemporalSignalKind.SunCenterMarginDegF32 or TemporalSignalKind.EarthCenterMarginDegF32))
                throw new NotSupportedException($"Unsupported V2 signal kind: {spec.Signal}");
        }

        private static void ValidatePredicateCombination(ThresholdPredicateSpec? marginPredicate, SunFractionPredicateSpec? sunPredicate)
        {
            if (marginPredicate is null && sunPredicate is null)
                throw new ArgumentException("Reducer predicate requires at least one of MarginPredicate or SunPredicate.");
            if (marginPredicate is not null && marginPredicate.Signal == TemporalSignalKind.SunFractionU8)
                throw new NotSupportedException("ThresholdPredicateSpec must target a center-margin signal, not SunFractionU8.");
        }

        private static void InferReducerSignals(
            ThresholdPredicateSpec? marginPredicate,
            SunFractionPredicateSpec? sunPredicate,
            ref bool needSunFraction,
            ref bool needSunCenterMargin,
            ref bool needEarthCenterMargin)
        {
            if (sunPredicate is not null)
                needSunFraction = true;
            if (marginPredicate is null)
                return;

            switch (marginPredicate.Signal)
            {
                case TemporalSignalKind.SunCenterMarginDegF32:
                    needSunCenterMargin = true;
                    break;
                case TemporalSignalKind.EarthCenterMarginDegF32:
                    needEarthCenterMargin = true;
                    break;
                default:
                    throw new NotSupportedException($"Unsupported threshold predicate signal: {marginPredicate.Signal}");
            }
        }

        private sealed record V2JobConfig(
            LightmapStreamMode Mode,
            StreamScalarType ScalarType,
            int MaxChunkTimeCount,
            int ChannelCount,
            IReadOnlyList<TemporalSignalSpec> Signals,
            IReadOnlyList<NativeReducerSpec> Reducers,
            bool HasSunFraction,
            bool HasSunCenterMargin,
            bool HasEarthCenterMargin,
            bool HasEarthSignals
        );

        private sealed class StreamingJobV2 : IDisposable
        {
            private readonly ConcurrentDictionary<int, RegisteredBuffer> _buffers = new();
            private readonly Channel<int> _freeBufferIds;
            private readonly Channel<TileEnvelopeV2> _readyTiles;
            private readonly CancellationTokenSource _cancellation = new();
            private readonly object _statusGate = new();
            private readonly Task _producerTask;
            private readonly V2JobConfig _config;

            private StreamJobState _state = StreamJobState.Queued;
            private string? _message;
            private double _progress01;
            private long _tilesProduced;
            private long _tilesConsumed;
            private long _tilesCompleted;
            private long _nextEnvelopeId;
            private int _readyDepth;
            private int _freeBufferCount;
            private NativeReduceRasterResult? _rasterResult;

            public string JobId { get; }
            public LightmapArrayStreamRequestV2 Request { get; }
            public int TimeCount { get; }
            public int ExpectedTileByteLength { get; }

            public StreamingJobV2(
                string jobId,
                LightmapArrayStreamRequestV2 request,
                int timeCount,
                int expectedTileByteLength,
                V2JobConfig config)
            {
                JobId = jobId;
                Request = request;
                TimeCount = timeCount;
                ExpectedTileByteLength = expectedTileByteLength;
                _config = config;

                _freeBufferIds = Channel.CreateUnbounded<int>(new UnboundedChannelOptions
                {
                    SingleReader = false,
                    SingleWriter = false
                });
                _readyTiles = Channel.CreateBounded<TileEnvelopeV2>(new BoundedChannelOptions(Math.Max(1, request.ReadyQueueCapacity))
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

            public TileEnvelopeV2? TryGetNextTile(int timeoutMs)
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

            public NativeReduceRasterResult? GetRasterResult()
            {
                lock (_statusGate)
                    return _rasterResult;
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
                NativeReduceGeoTiffWriter? rasterWriter = null;
                try
                {
                    SetState(StreamJobState.Running, "V2 streaming started.");

                    if (Request.RasterOutput is not null)
                    {
                        rasterWriter = new NativeReduceGeoTiffWriter(
                            Request.ScenarioRootDir,
                            Request.DemPath,
                            Request.RasterOutput,
                            _config.ChannelCount);
                    }

                    var horizonFiles = EnumerateHorizonFiles(Request.HorizonDir).ToList();
                    if (horizonFiles.Count == 0)
                    {
                        if (rasterWriter is not null)
                        {
                            var emptyResult = rasterWriter.Commit();
                            lock (_statusGate)
                                _rasterResult = emptyResult;
                        }
                        SetProgress(1.0);
                        EnqueueTerminal("No horizon tiles found.");
                        SetState(StreamJobState.Completed, "V2 streaming completed with no tiles.");
                        return;
                    }

                    var times = BuildTimes(Request.StartUtc, Request.StopUtc, Request.TimeStepHours);
                    var sunVectors = BuildSunVectors(times, Request.UseSpiceSunVectors);
                    var earthVectors = _config.HasEarthSignals
                        ? BuildEarthVectors(times, Request.UseSpiceEarthVectors)
                        : null;
                    var dem = new ElevationMap(Request.DemPath);
                    var boundedCapacity = Math.Max(1, Request.ReadyQueueCapacity);

                    var readStep = new PipelineStep<StreamingTileWorkItem, StreamingTileWorkItem>(ReadHorizonsStepAsync);
                    var computeStep = new PipelineStep<StreamingTileWorkItem, StreamingTileWorkItem>(
                        item => ComputeTileStepAsync(item, dem, sunVectors, earthVectors, horizonFiles.Count, rasterWriter));

                    var pipeline = new Pipeline<StreamingTileWorkItem>();
                    pipeline.AddStep(readStep.Func, Request.MaxReadParallelism, boundedCapacity, false);
                    pipeline.AddStep(computeStep.Func, Request.MaxComputeParallelism, boundedCapacity, false);
                    pipeline.AddTerminalStep(_ => Task.CompletedTask, 1, boundedCapacity, false);

                    pipeline
                        .ProcessAsync(BuildWorkItems(horizonFiles, _cancellation.Token))
                        .GetAwaiter()
                        .GetResult();

                    if (rasterWriter is not null)
                    {
                        var result = rasterWriter.Commit();
                        lock (_statusGate)
                            _rasterResult = result;
                    }

                    SetProgress(1.0);
                    EnqueueTerminal("V2 streaming completed.");
                    SetState(StreamJobState.Completed, "V2 streaming completed.");
                }
                catch (OperationCanceledException)
                {
                    rasterWriter?.Dispose();
                    rasterWriter = null;
                    EnqueueTerminal("V2 streaming cancelled.");
                    SetState(StreamJobState.Cancelled, "V2 streaming cancelled.");
                }
                catch (Exception ex)
                {
                    rasterWriter?.Dispose();
                    rasterWriter = null;
                    EnqueueTerminal($"V2 streaming failed: {ex.Message}");
                    SetState(StreamJobState.Failed, ex.Message);
                }
                finally
                {
                    rasterWriter?.Dispose();
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
                        tileId: Interlocked.Increment(ref _nextEnvelopeId),
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
                List<Vector3d>? earthVectors,
                int totalTileCount,
                NativeReduceGeoTiffWriter? rasterWriter)
            {
                try
                {
                    _cancellation.Token.ThrowIfCancellationRequested();

                    if (!string.IsNullOrWhiteSpace(item.ErrorMessage))
                    {
                        if (rasterWriter is not null)
                            throw new InvalidDataException(item.ErrorMessage);
                        EnqueueError(item, -1, item.ErrorMessage!);
                        return Task.FromResult(item);
                    }

                    if (item.Horizons is null)
                        throw new InvalidDataException($"Missing horizons for tile: {item.HorizonPath}");
                    if (item.PatchRow < 0 || item.PatchCol < 0)
                        throw new InvalidDataException($"Missing patch row/col for tile: {item.HorizonPath}");

                    WriteTileSignalChunksToReadyBuffers(
                        horizons: item.Horizons,
                        dem: dem,
                        patchRow: item.PatchRow,
                        patchCol: item.PatchCol,
                        patchWidth: Request.PatchWidth,
                        patchHeight: Request.PatchHeight,
                        sunVectors: sunVectors,
                        earthVectors: earthVectors,
                        maxChunkTimeCount: _config.MaxChunkTimeCount,
                        rasterWriter: rasterWriter);
                }
                catch (OperationCanceledException)
                {
                    throw;
                }
                catch (Exception ex)
                {
                    if (rasterWriter is not null)
                        throw;
                    EnqueueError(item, -1, ex.Message);
                }
                finally
                {
                    Interlocked.Increment(ref _tilesCompleted);
                    var completed = Interlocked.Read(ref _tilesCompleted);
                    SetProgress(Math.Clamp((double)completed / Math.Max(1, totalTileCount), 0.0, 1.0));
                }

                return Task.FromResult(item);
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

            private static List<Vector3d> BuildEarthVectors(IReadOnlyList<DateTime> times, bool useSpice)
            {
                if (!useSpice)
                    throw new NotSupportedException("Earth signal streaming currently requires SPICE Earth vectors.");
                return times.Select(t => SpiceManager.EarthPosition(t) * 1000.0).ToList();
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

            private void WriteTileSignalChunksToReadyBuffers(
                float[] horizons,
                ElevationMap dem,
                int patchRow,
                int patchCol,
                int patchWidth,
                int patchHeight,
                List<Vector3d> sunVectors,
                List<Vector3d>? earthVectors,
                int maxChunkTimeCount,
                NativeReduceGeoTiffWriter? rasterWriter)
            {
                if (_config.Mode == LightmapStreamMode.NativeReduce)
                {
                    WriteNativeReducedTile(
                        horizons: horizons,
                        dem: dem,
                        patchRow: patchRow,
                        patchCol: patchCol,
                        patchWidth: patchWidth,
                        patchHeight: patchHeight,
                        sunVectors: sunVectors,
                        earthVectors: earthVectors,
                        rasterWriter: rasterWriter);
                    return;
                }
                if (_config.Mode != LightmapStreamMode.SignalStream)
                    throw new NotSupportedException("V2 job config mode mismatch.");

                for (int timeOffset = 0; timeOffset < sunVectors.Count; timeOffset += maxChunkTimeCount)
                {
                    _cancellation.Token.ThrowIfCancellationRequested();
                    int chunkTimeCount = Math.Min(maxChunkTimeCount, sunVectors.Count - timeOffset);
                    int bufferId = AcquireNextFreeBuffer(_cancellation.Token);

                    try
                    {
                        if (!_buffers.TryGetValue(bufferId, out var buffer))
                            throw new InvalidOperationException($"Unknown buffer id returned by free pool: {bufferId}");

                        if (_config.ScalarType == StreamScalarType.UInt8)
                        {
                            WriteSunFractionSignalChunkToRegisteredBuffer(
                                targetPtr: buffer.Pointer,
                                targetByteLength: buffer.ByteLength,
                                horizons: horizons,
                                dem: dem,
                                patchRow: patchRow,
                                patchCol: patchCol,
                                patchWidth: patchWidth,
                                patchHeight: patchHeight,
                                sunVectors: sunVectors,
                                timeOffset: timeOffset,
                                chunkTimeCount: chunkTimeCount,
                                maxChunkTimeCount: maxChunkTimeCount);
                        }
                        else
                        {
                            WriteFloat32SignalChunkToRegisteredBuffer(
                                targetPtr: buffer.Pointer,
                                targetByteLength: buffer.ByteLength,
                                horizons: horizons,
                                dem: dem,
                                patchRow: patchRow,
                                patchCol: patchCol,
                                patchWidth: patchWidth,
                                patchHeight: patchHeight,
                                sunVectors: sunVectors,
                                earthVectors: earthVectors,
                                timeOffset: timeOffset,
                                chunkTimeCount: chunkTimeCount,
                                maxChunkTimeCount: maxChunkTimeCount,
                                signalsByChannel: _config.Signals,
                                channelCount: _config.ChannelCount,
                                hasSunFraction: _config.HasSunFraction,
                                hasSunCenterMargin: _config.HasSunCenterMargin,
                                hasEarthCenterMargin: _config.HasEarthCenterMargin);
                        }

                        EnqueueReady(new TileEnvelopeV2(
                            JobId: JobId,
                            TileId: Interlocked.Increment(ref _nextEnvelopeId),
                            BufferId: bufferId,
                            PatchRow: patchRow,
                            PatchCol: patchCol,
                            Width: patchWidth,
                            Height: patchHeight,
                            State: StreamTileState.Ready,
                            ScalarType: _config.ScalarType,
                            Rank: 4,
                            Dim0: chunkTimeCount,
                            Dim1: _config.ChannelCount,
                            Dim2: patchHeight,
                            Dim3: patchWidth,
                            TimeOffset: timeOffset,
                            TimeCount: chunkTimeCount,
                            ChannelCount: _config.ChannelCount,
                            Message: null
                        ));
                    }
                    catch
                    {
                        ReturnBufferToFreePool(bufferId, countConsumed: false);
                        throw;
                    }
                }
            }

            private void WriteNativeReducedTile(
                float[] horizons,
                ElevationMap dem,
                int patchRow,
                int patchCol,
                int patchWidth,
                int patchHeight,
                List<Vector3d> sunVectors,
                List<Vector3d>? earthVectors,
                NativeReduceGeoTiffWriter? rasterWriter)
            {
                if (_config.Mode != LightmapStreamMode.NativeReduce)
                    throw new NotSupportedException("NativeReduce writer invoked for non-NativeReduce job.");
                if (_config.ScalarType != StreamScalarType.Float32)
                    throw new NotSupportedException("NativeReduce currently supports only float32 payloads.");

                if (rasterWriter is not null)
                {
                    var reduced = new float[_config.ChannelCount * patchWidth * patchHeight];
                    var reducedHandle = GCHandle.Alloc(reduced, GCHandleType.Pinned);
                    try
                    {
                        WriteNativeReduceFloat32TileToRegisteredBuffer(
                            targetPtr: reducedHandle.AddrOfPinnedObject(),
                            targetByteLength: reduced.Length * sizeof(float),
                            horizons: horizons,
                            dem: dem,
                            patchRow: patchRow,
                            patchCol: patchCol,
                            patchWidth: patchWidth,
                            patchHeight: patchHeight,
                            sunVectors: sunVectors,
                            earthVectors: earthVectors,
                            reducers: _config.Reducers,
                            hasSunFraction: _config.HasSunFraction,
                            hasSunCenterMargin: _config.HasSunCenterMargin,
                            hasEarthCenterMargin: _config.HasEarthCenterMargin,
                            timeStepHours: (float)Request.TimeStepHours);
                    }
                    finally
                    {
                        reducedHandle.Free();
                    }

                    rasterWriter.WriteTile(patchCol, patchRow, patchWidth, patchHeight, reduced);
                    Interlocked.Increment(ref _tilesProduced);
                    return;
                }

                int bufferId = AcquireNextFreeBuffer(_cancellation.Token);
                try
                {
                    if (!_buffers.TryGetValue(bufferId, out var buffer))
                        throw new InvalidOperationException($"Unknown buffer id returned by free pool: {bufferId}");

                    WriteNativeReduceFloat32TileToRegisteredBuffer(
                        targetPtr: buffer.Pointer,
                        targetByteLength: buffer.ByteLength,
                        horizons: horizons,
                        dem: dem,
                        patchRow: patchRow,
                        patchCol: patchCol,
                        patchWidth: patchWidth,
                        patchHeight: patchHeight,
                        sunVectors: sunVectors,
                        earthVectors: earthVectors,
                        reducers: _config.Reducers,
                        hasSunFraction: _config.HasSunFraction,
                        hasSunCenterMargin: _config.HasSunCenterMargin,
                        hasEarthCenterMargin: _config.HasEarthCenterMargin,
                        timeStepHours: (float)Request.TimeStepHours);

                    EnqueueReady(new TileEnvelopeV2(
                        JobId: JobId,
                        TileId: Interlocked.Increment(ref _nextEnvelopeId),
                        BufferId: bufferId,
                        PatchRow: patchRow,
                        PatchCol: patchCol,
                        Width: patchWidth,
                        Height: patchHeight,
                        State: StreamTileState.Ready,
                        ScalarType: StreamScalarType.Float32,
                        Rank: 3,
                        Dim0: _config.ChannelCount,
                        Dim1: patchHeight,
                        Dim2: patchWidth,
                        Dim3: 1,
                        TimeOffset: 0,
                        TimeCount: 0,
                        ChannelCount: _config.ChannelCount,
                        Message: null
                    ));
                }
                catch
                {
                    ReturnBufferToFreePool(bufferId, countConsumed: false);
                    throw;
                }
            }

            private void EnqueueError(StreamingTileWorkItem item, int bufferId, string message)
            {
                int rank = _config.Mode == LightmapStreamMode.NativeReduce ? 3 : 4;
                EnqueueReady(new TileEnvelopeV2(
                    JobId: JobId,
                    TileId: Interlocked.Increment(ref _nextEnvelopeId),
                    BufferId: bufferId,
                    PatchRow: item.PatchRow,
                    PatchCol: item.PatchCol,
                    Width: Request.PatchWidth,
                    Height: Request.PatchHeight,
                    State: StreamTileState.Error,
                    ScalarType: _config.ScalarType,
                    Rank: rank,
                    Dim0: 0,
                    Dim1: _config.ChannelCount,
                    Dim2: Request.PatchHeight,
                    Dim3: Request.PatchWidth,
                    TimeOffset: 0,
                    TimeCount: 0,
                    ChannelCount: _config.ChannelCount,
                    Message: message
                ));
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

            private void EnqueueReady(TileEnvelopeV2 envelope)
            {
                _readyTiles.Writer.WriteAsync(envelope, _cancellation.Token).AsTask().GetAwaiter().GetResult();
                Interlocked.Increment(ref _readyDepth);
                if (envelope.State == StreamTileState.Ready)
                    Interlocked.Increment(ref _tilesProduced);
            }

            private void EnqueueTerminal(string message)
            {
                int rank = _config.Mode == LightmapStreamMode.NativeReduce ? 3 : 4;
                var terminal = new TileEnvelopeV2(
                    JobId: JobId,
                    TileId: Interlocked.Increment(ref _nextEnvelopeId),
                    BufferId: -1,
                    PatchRow: -1,
                    PatchCol: -1,
                    Width: Request.PatchWidth,
                    Height: Request.PatchHeight,
                    State: StreamTileState.Terminal,
                    ScalarType: _config.ScalarType,
                    Rank: rank,
                    Dim0: 0,
                    Dim1: _config.ChannelCount,
                    Dim2: Request.PatchHeight,
                    Dim3: Request.PatchWidth,
                    TimeOffset: 0,
                    TimeCount: 0,
                    ChannelCount: _config.ChannelCount,
                    Message: message
                );
                if (_readyTiles.Writer.TryWrite(terminal))
                    Interlocked.Increment(ref _readyDepth);
            }

            private static unsafe void WriteSunFractionSignalChunkToRegisteredBuffer(
                IntPtr targetPtr,
                int targetByteLength,
                float[] horizons,
                ElevationMap dem,
                int patchRow,
                int patchCol,
                int patchWidth,
                int patchHeight,
                List<Vector3d> sunVectors,
                int timeOffset,
                int chunkTimeCount,
                int maxChunkTimeCount)
            {
                if (patchRow < 0 || patchCol < 0)
                    throw new ArgumentOutOfRangeException("Patch row/col must be >= 0.");
                if (patchRow + patchHeight > dem.Height || patchCol + patchWidth > dem.Width)
                    throw new ArgumentOutOfRangeException("Patch row/col extends beyond DEM bounds.");
                if (timeOffset < 0 || chunkTimeCount < 0 || maxChunkTimeCount < 1 || timeOffset + chunkTimeCount > sunVectors.Count)
                    throw new ArgumentOutOfRangeException("Invalid time chunk range.");

                var horizonSampleCount = patchWidth * patchHeight * LightmapGenerator.HorizonSamples;
                if (horizons.Length < horizonSampleCount)
                    throw new InvalidDataException(
                        $"Unexpected horizon sample count. Expected at least {horizonSampleCount}, got {horizons.Length}.");

                var expectedBytes = maxChunkTimeCount * patchWidth * patchHeight;
                if (targetByteLength != expectedBytes)
                    throw new InvalidDataException(
                        $"Unexpected V2 output byte length. Expected {expectedBytes}, got {targetByteLength}.");

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

                int tilePixelCount = patchWidth * patchHeight;
                for (int localTimeIndex = 0; localTimeIndex < chunkTimeCount; localTimeIndex++)
                {
                    int globalTimeIndex = timeOffset + localTimeIndex;
                    var sunVec = sunVectors[globalTimeIndex];
                    int outputBase = localTimeIndex * tilePixelCount;
                    for (int y = 0; y < patchHeight; y++)
                    {
                        for (int x = 0; x < patchWidth; x++)
                        {
                            int pixelIndex = y * patchWidth + x;
                            int horizonBase = pixelIndex * LightmapGenerator.HorizonSamples;
                            var (azimuthRad, elevationRad) = dem.GetAzEl(sunVec, matrices[y, x]);
                            float azimuthDeg = azimuthRad * 57.2957795f;
                            float elevationDeg = elevationRad * 57.2957795f;
                            float sunFraction = LightmapGenerator.BuilderSunFraction(horizons, horizonBase, azimuthDeg, elevationDeg);
                            destination[outputBase + pixelIndex] = (byte)(255f * sunFraction);
                        }
                    }
                }
            }

            private static unsafe void WriteFloat32SignalChunkToRegisteredBuffer(
                IntPtr targetPtr,
                int targetByteLength,
                float[] horizons,
                ElevationMap dem,
                int patchRow,
                int patchCol,
                int patchWidth,
                int patchHeight,
                List<Vector3d> sunVectors,
                List<Vector3d>? earthVectors,
                int timeOffset,
                int chunkTimeCount,
                int maxChunkTimeCount,
                IReadOnlyList<TemporalSignalSpec> signalsByChannel,
                int channelCount,
                bool hasSunFraction,
                bool hasSunCenterMargin,
                bool hasEarthCenterMargin)
            {
                if (patchRow < 0 || patchCol < 0)
                    throw new ArgumentOutOfRangeException("Patch row/col must be >= 0.");
                if (patchRow + patchHeight > dem.Height || patchCol + patchWidth > dem.Width)
                    throw new ArgumentOutOfRangeException("Patch row/col extends beyond DEM bounds.");
                if (timeOffset < 0 || chunkTimeCount < 0 || maxChunkTimeCount < 1 || timeOffset + chunkTimeCount > sunVectors.Count)
                    throw new ArgumentOutOfRangeException("Invalid time chunk range.");
                if (channelCount < 1 || signalsByChannel.Count != channelCount)
                    throw new ArgumentOutOfRangeException(nameof(channelCount), "Invalid V2 channel count.");
                if (hasEarthCenterMargin && (earthVectors is null || earthVectors.Count != sunVectors.Count))
                    throw new InvalidOperationException("Earth vectors are required for Earth center margin signals.");

                var horizonSampleCount = patchWidth * patchHeight * LightmapGenerator.HorizonSamples;
                if (horizons.Length < horizonSampleCount)
                    throw new InvalidDataException(
                        $"Unexpected horizon sample count. Expected at least {horizonSampleCount}, got {horizons.Length}.");

                var expectedBytes = maxChunkTimeCount * channelCount * patchWidth * patchHeight * sizeof(float);
                if (targetByteLength != expectedBytes)
                    throw new InvalidDataException(
                        $"Unexpected V2 float32 output byte length. Expected {expectedBytes}, got {targetByteLength}.");

                var destination = new Span<float>(targetPtr.ToPointer(), targetByteLength / sizeof(float));
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

                int tilePixelCount = patchWidth * patchHeight;
                for (int localTimeIndex = 0; localTimeIndex < chunkTimeCount; localTimeIndex++)
                {
                    int globalTimeIndex = timeOffset + localTimeIndex;
                    var sunVec = sunVectors[globalTimeIndex];
                    var earthVec = hasEarthCenterMargin ? earthVectors![globalTimeIndex] : default;

                    for (int y = 0; y < patchHeight; y++)
                    {
                        for (int x = 0; x < patchWidth; x++)
                        {
                            int pixelIndex = y * patchWidth + x;
                            int horizonBase = pixelIndex * LightmapGenerator.HorizonSamples;

                            byte sunFractionU8 = 0;
                            float sunCenterMarginDeg = 0f;
                            float earthCenterMarginDeg = 0f;

                            if (hasSunFraction || hasSunCenterMargin)
                            {
                                var (sunAzimuthRad, sunElevationRad) = dem.GetAzEl(sunVec, matrices[y, x]);
                                float sunAzimuthDeg = sunAzimuthRad * 57.2957795f;
                                float sunElevationDeg = sunElevationRad * 57.2957795f;
                                if (hasSunFraction)
                                {
                                    float sunFraction = LightmapGenerator.BuilderSunFraction(horizons, horizonBase, sunAzimuthDeg, sunElevationDeg);
                                    sunFractionU8 = (byte)(255f * sunFraction);
                                }
                                if (hasSunCenterMargin)
                                {
                                    float horizonDeg = SampleHorizonElevationDeg(horizons, horizonBase, sunAzimuthDeg);
                                    sunCenterMarginDeg = sunElevationDeg - horizonDeg;
                                }
                            }

                            if (hasEarthCenterMargin)
                            {
                                var (earthAzimuthRad, earthElevationRad) = dem.GetAzEl(earthVec, matrices[y, x]);
                                float earthAzimuthDeg = earthAzimuthRad * 57.2957795f;
                                float earthElevationDeg = earthElevationRad * 57.2957795f;
                                float horizonDeg = SampleHorizonElevationDeg(horizons, horizonBase, earthAzimuthDeg);
                                earthCenterMarginDeg = earthElevationDeg - horizonDeg;
                            }

                            for (int channelIndex = 0; channelIndex < channelCount; channelIndex++)
                            {
                                float value = signalsByChannel[channelIndex].Signal switch
                                {
                                    TemporalSignalKind.SunFractionU8 => sunFractionU8,
                                    TemporalSignalKind.SunCenterMarginDegF32 => sunCenterMarginDeg,
                                    TemporalSignalKind.EarthCenterMarginDegF32 => earthCenterMarginDeg,
                                    _ => throw new NotSupportedException($"Unsupported signal in float32 V2 writer: {signalsByChannel[channelIndex].Signal}")
                                };

                                int dstIndex = ((localTimeIndex * channelCount) + channelIndex) * tilePixelCount + pixelIndex;
                                destination[dstIndex] = value;
                            }
                        }
                    }
                }
            }

            private static unsafe void WriteNativeReduceFloat32TileToRegisteredBuffer(
                IntPtr targetPtr,
                int targetByteLength,
                float[] horizons,
                ElevationMap dem,
                int patchRow,
                int patchCol,
                int patchWidth,
                int patchHeight,
                List<Vector3d> sunVectors,
                List<Vector3d>? earthVectors,
                IReadOnlyList<NativeReducerSpec> reducers,
                bool hasSunFraction,
                bool hasSunCenterMargin,
                bool hasEarthCenterMargin,
                float timeStepHours)
            {
                if (patchRow < 0 || patchCol < 0)
                    throw new ArgumentOutOfRangeException("Patch row/col must be >= 0.");
                if (patchRow + patchHeight > dem.Height || patchCol + patchWidth > dem.Width)
                    throw new ArgumentOutOfRangeException("Patch row/col extends beyond DEM bounds.");
                if (reducers.Count < 1)
                    throw new ArgumentOutOfRangeException(nameof(reducers), "At least one reducer is required.");
                if (hasEarthCenterMargin && (earthVectors is null || earthVectors.Count != sunVectors.Count))
                    throw new InvalidOperationException("Earth vectors are required for Earth reducers.");

                var horizonSampleCount = patchWidth * patchHeight * LightmapGenerator.HorizonSamples;
                if (horizons.Length < horizonSampleCount)
                    throw new InvalidDataException(
                        $"Unexpected horizon sample count. Expected at least {horizonSampleCount}, got {horizons.Length}.");

                var expectedBytes = reducers.Count * patchWidth * patchHeight * sizeof(float);
                if (targetByteLength != expectedBytes)
                    throw new InvalidDataException(
                        $"Unexpected NativeReduce float32 output byte length. Expected {expectedBytes}, got {targetByteLength}.");

                var destination = new Span<float>(targetPtr.ToPointer(), targetByteLength / sizeof(float));
                destination.Clear();

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

                int tilePixelCount = patchWidth * patchHeight;
                int timeCount = sunVectors.Count;
                for (int y = 0; y < patchHeight; y++)
                {
                    for (int x = 0; x < patchWidth; x++)
                    {
                        int pixelIndex = y * patchWidth + x;
                        int horizonBase = pixelIndex * LightmapGenerator.HorizonSamples;

                        var accum = new float[reducers.Count];
                        var currentRun = new float[reducers.Count];
                        var maxRun = new float[reducers.Count];

                        for (int timeIndex = 0; timeIndex < timeCount; timeIndex++)
                        {
                            byte sunFractionU8 = 0;
                            float sunCenterMarginDeg = 0f;
                            float earthCenterMarginDeg = 0f;

                            if (hasSunFraction || hasSunCenterMargin)
                            {
                                var (sunAzimuthRad, sunElevationRad) = dem.GetAzEl(sunVectors[timeIndex], matrices[y, x]);
                                float sunAzimuthDeg = sunAzimuthRad * 57.2957795f;
                                float sunElevationDeg = sunElevationRad * 57.2957795f;
                                if (hasSunFraction)
                                {
                                    float sunFraction = LightmapGenerator.BuilderSunFraction(horizons, horizonBase, sunAzimuthDeg, sunElevationDeg);
                                    sunFractionU8 = (byte)(255f * sunFraction);
                                }
                                if (hasSunCenterMargin)
                                {
                                    sunCenterMarginDeg = sunElevationDeg - SampleHorizonElevationDeg(horizons, horizonBase, sunAzimuthDeg);
                                }
                            }

                            if (hasEarthCenterMargin)
                            {
                                var (earthAzimuthRad, earthElevationRad) = dem.GetAzEl(earthVectors![timeIndex], matrices[y, x]);
                                float earthAzimuthDeg = earthAzimuthRad * 57.2957795f;
                                float earthElevationDeg = earthElevationRad * 57.2957795f;
                                earthCenterMarginDeg = earthElevationDeg - SampleHorizonElevationDeg(horizons, horizonBase, earthAzimuthDeg);
                            }

                            for (int reducerIndex = 0; reducerIndex < reducers.Count; reducerIndex++)
                            {
                                ApplyReducerStep(
                                    reducers[reducerIndex],
                                    sunFractionU8,
                                    sunCenterMarginDeg,
                                    earthCenterMarginDeg,
                                    timeStepHours,
                                    ref accum[reducerIndex],
                                    ref currentRun[reducerIndex],
                                    ref maxRun[reducerIndex]);
                            }
                        }

                        for (int reducerIndex = 0; reducerIndex < reducers.Count; reducerIndex++)
                        {
                            float finalValue = FinalizeReducerValue(
                                reducers[reducerIndex],
                                accum[reducerIndex],
                                currentRun[reducerIndex],
                                maxRun[reducerIndex],
                                timeCount);

                            int dstIndex = reducerIndex * tilePixelCount + pixelIndex;
                            destination[dstIndex] = finalValue;
                        }
                    }
                }
            }

            private static void ApplyReducerStep(
                NativeReducerSpec reducer,
                byte sunFractionU8,
                float sunCenterMarginDeg,
                float earthCenterMarginDeg,
                float timeStepHours,
                ref float accum,
                ref float currentRun,
                ref float maxRun)
            {
                switch (reducer)
                {
                    case AverageSunFractionReducerSpec:
                        accum += sunFractionU8;
                        return;
                    case CumulativeDurationWhereReducerSpec spec:
                    {
                        if (EvaluatePredicate(spec.MarginPredicate, spec.SunPredicate, sunFractionU8, sunCenterMarginDeg, earthCenterMarginDeg))
                            accum += spec.Unit == DurationOutputUnit.Hours ? timeStepHours : 1f;
                        return;
                    }
                    case MaxContiguousDurationWhereReducerSpec spec:
                    {
                        if (EvaluatePredicate(spec.MarginPredicate, spec.SunPredicate, sunFractionU8, sunCenterMarginDeg, earthCenterMarginDeg))
                        {
                            currentRun += spec.Unit == DurationOutputUnit.Hours ? timeStepHours : 1f;
                            if (currentRun > maxRun) maxRun = currentRun;
                        }
                        else
                        {
                            currentRun = 0f;
                        }
                        return;
                    }
                    case CombinedSunEarthContiguousDurationReducerSpec spec:
                    {
                        bool sunOk = EvaluateSunFractionPredicate(spec.SunPredicate, sunFractionU8);
                        bool earthOk = EvaluateThresholdPredicate(spec.EarthMarginPredicate, sunCenterMarginDeg, earthCenterMarginDeg);
                        if (sunOk && earthOk)
                        {
                            currentRun += spec.Unit == DurationOutputUnit.Hours ? timeStepHours : 1f;
                            if (currentRun > maxRun) maxRun = currentRun;
                        }
                        else
                        {
                            currentRun = 0f;
                        }
                        return;
                    }
                    default:
                        throw new NotSupportedException($"Unsupported reducer type: {reducer.GetType().Name}");
                }
            }

            private static float FinalizeReducerValue(
                NativeReducerSpec reducer,
                float accum,
                float currentRun,
                float maxRun,
                int timeCount)
            {
                return reducer switch
                {
                    AverageSunFractionReducerSpec avg => avg.OutputNormalized01
                        ? (timeCount > 0 ? (accum / Math.Max(1, timeCount)) / 255f : 0f)
                        : (timeCount > 0 ? accum / Math.Max(1, timeCount) : 0f),
                    CumulativeDurationWhereReducerSpec => accum,
                    MaxContiguousDurationWhereReducerSpec => Math.Max(maxRun, currentRun),
                    CombinedSunEarthContiguousDurationReducerSpec => Math.Max(maxRun, currentRun),
                    _ => throw new NotSupportedException($"Unsupported reducer type: {reducer.GetType().Name}")
                };
            }

            private static bool EvaluatePredicate(
                ThresholdPredicateSpec? marginPredicate,
                SunFractionPredicateSpec? sunPredicate,
                byte sunFractionU8,
                float sunCenterMarginDeg,
                float earthCenterMarginDeg)
            {
                bool marginOk = marginPredicate is null || EvaluateThresholdPredicate(marginPredicate, sunCenterMarginDeg, earthCenterMarginDeg);
                bool sunOk = sunPredicate is null || EvaluateSunFractionPredicate(sunPredicate, sunFractionU8);
                return marginOk && sunOk;
            }

            private static bool EvaluateSunFractionPredicate(SunFractionPredicateSpec spec, byte sunFractionU8)
            {
                return spec.GreaterThanOrEqual
                    ? sunFractionU8 >= spec.MinSunFractionU8
                    : sunFractionU8 > spec.MinSunFractionU8;
            }

            private static bool EvaluateThresholdPredicate(
                ThresholdPredicateSpec spec,
                float sunCenterMarginDeg,
                float earthCenterMarginDeg)
            {
                float centerMargin = spec.Signal switch
                {
                    TemporalSignalKind.SunCenterMarginDegF32 => sunCenterMarginDeg,
                    TemporalSignalKind.EarthCenterMarginDegF32 => earthCenterMarginDeg,
                    _ => throw new NotSupportedException($"Unsupported threshold predicate signal: {spec.Signal}")
                };

                float bodyRadius = float.IsNaN(spec.BodyRadiusDegOverride)
                    ? GetDefaultBodyRadiusDeg(spec.Signal)
                    : spec.BodyRadiusDegOverride;

                float margin = spec.Reference switch
                {
                    TemporalThresholdReference.CenterMargin => centerMargin,
                    TemporalThresholdReference.LowerLimbMargin => centerMargin - bodyRadius,
                    TemporalThresholdReference.UpperLimbMargin => centerMargin + bodyRadius,
                    _ => throw new NotSupportedException($"Unsupported threshold reference: {spec.Reference}")
                };

                return spec.GreaterThanOrEqual
                    ? margin >= spec.ThresholdValue
                    : margin > spec.ThresholdValue;
            }

            private static float GetDefaultBodyRadiusDeg(TemporalSignalKind signal)
            {
                return signal switch
                {
                    TemporalSignalKind.SunCenterMarginDegF32 => 0.266f,
                    TemporalSignalKind.EarthCenterMarginDegF32 => 0.95f,
                    _ => throw new NotSupportedException($"No body radius for signal: {signal}")
                };
            }

            private static float SampleHorizonElevationDeg(float[] horizons, int horizonBase, float azimuthDeg)
            {
                const float BucketWidthDeg = 360f / LightmapGenerator.HorizonSamplesF;
                const float BucketHalfWidthDeg = BucketWidthDeg / 2f;

                float azWrapped = azimuthDeg % 360f;
                if (azWrapped < 0f)
                    azWrapped += 360f;

                float leftBucketFloat = (azWrapped - BucketHalfWidthDeg) * (LightmapGenerator.HorizonSamplesF / 360f);
                int leftBucket = (int)MathF.Floor(leftBucketFloat);
                float frac = leftBucketFloat - leftBucket;

                while (leftBucket < 0)
                    leftBucket += LightmapGenerator.HorizonSamples;
                while (leftBucket >= LightmapGenerator.HorizonSamples)
                    leftBucket -= LightmapGenerator.HorizonSamples;

                int rightBucket = leftBucket + 1;
                if (rightBucket >= LightmapGenerator.HorizonSamples)
                    rightBucket = 0;

                float left = horizons[horizonBase + leftBucket];
                float right = horizons[horizonBase + rightBucket];
                return left + frac * (right - left);
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
    }
}
