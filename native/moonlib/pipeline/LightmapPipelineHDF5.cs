using HDF.PInvoke;
using moonlib.horizon;
using moonlib.math;
using moonlib.spice;
using Serilog;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace moonlib.pipeline
{
    public sealed class LightmapPipelineHDF5
    {
        public const int Width = 128;
        public const int Height = 128;
        public const int HorizonSamples = 1440;

        private List<DateTime>? _times;
        private List<Vector3d>? _sunvecsMe;
        private ElevationMap? _dem;
        private Hdf5LightCurveWriter? _writer;
        private long _readHorizonsTicks;
        private long _readHorizonsMaxTicks;
        private int _readHorizonsCount;
        private long _generateMatricesTicks;
        private long _generateMatricesMaxTicks;
        private int _generateMatricesCount;
        private long _generateShadowsTicks;
        private long _generateShadowsMaxTicks;
        private int _generateShadowsCount;
        private long _writeDataTicks;
        private long _writeDataMaxTicks;
        private int _writeDataCount;

        public async Task ExecuteAsync(
            List<DateTime> timestamps,
            string outputPath,
            ElevationMap elevationMap,
            string horizonDir,
            IProgress<float>? progress = null,
            Func<DateTime, Vector3d>? sunVectorProvider = null)
        {
            if (timestamps.Count == 0)
                throw new ArgumentException("At least one timestamp is required.", nameof(timestamps));
            if (string.IsNullOrWhiteSpace(outputPath))
                throw new ArgumentException("Output HDF5 path must be non-empty.", nameof(outputPath));

            _dem = elevationMap;
            _times = timestamps;
            _sunvecsMe = sunVectorProvider is null
                ? _times.Select(t => SpiceManager.SunPosition(t) * 1000.0).ToList()
                : _times.Select(sunVectorProvider).ToList();
            ResetMetrics();

            progress ??= new Progress<float>(percent => Console.WriteLine($"Progress: {100 * percent}%"));

            using var writer = new Hdf5LightCurveWriter(
                outputPath,
                elevationMap.Width,
                elevationMap.Height,
                timestamps,
                elevationMap.Projection,
                elevationMap.GeoTransform);
            _writer = writer;

            var pipeline = new Pipeline<Hdf5HorizonProcessingToken>();
            pipeline.AddStep(ReadHorizons, maxDegreeOfParallelism: 12, boundedCapacity: 24, ensureOrdered: false);
            pipeline.AddStep(GenerateMatrices, maxDegreeOfParallelism: 20, boundedCapacity: 40, ensureOrdered: false);
            pipeline.AddStep(GenerateShadows, maxDegreeOfParallelism: 20, boundedCapacity: 40, ensureOrdered: false);

            int processedCount = 0;
            int totalCount = 0;
            pipeline.AddTerminalStep(token =>
            {
                WriteData(token);
                int current = Interlocked.Increment(ref processedCount);
                progress.Report((float)current / totalCount);
                return Task.CompletedTask;
            }, maxDegreeOfParallelism: 1, boundedCapacity: 40, ensureOrdered: false);

            var horizonFilenames = new HorizonTileStore(horizonDir)
                .EnumerateFiles(observerElevationMeters: 0f)
                .ToList();
            totalCount = horizonFilenames.Count;
            Log.Information("Found {Count} horizon files for HDF5 lightmap generation.", horizonFilenames.Count);

            var pipelineStopwatch = Stopwatch.StartNew();
            try
            {
                await pipeline.ProcessAsync(horizonFilenames.Select(f => new Hdf5HorizonProcessingToken
                {
                    Filename = f,
                    Dem = _dem,
                    SunVectorsMe = _sunvecsMe,
                }));
                writer.Commit();
            }
            finally
            {
                pipelineStopwatch.Stop();
                LogMetrics(pipelineStopwatch.Elapsed, totalCount);
                _writer = null;
            }
        }

        private Task<Hdf5HorizonProcessingToken> ReadHorizons(Hdf5HorizonProcessingToken token)
        {
            long start = Stopwatch.GetTimestamp();
            try
            {
                (token.Col, token.Row, token.ObserverElevation) = QuadTreeHorizonGenerator.ParseHorizonFilename(token.Filename);
                try
                {
                    token.Horizons = HorizonFile.ReadHorizonFile(token.Filename);
                }
                catch (Exception ex)
                {
                    Log.Error(ex, "Failed to read horizon file: {Filename}", token.Filename);
                    token.Horizons = null;
                }
                return Task.FromResult(token);
            }
            finally
            {
                RecordStage(
                    ref _readHorizonsTicks,
                    ref _readHorizonsMaxTicks,
                    ref _readHorizonsCount,
                    start);
            }
        }

        private Task<Hdf5HorizonProcessingToken> GenerateMatrices(Hdf5HorizonProcessingToken token)
        {
            long start = Stopwatch.GetTimestamp();
            try
            {
                Debug.Assert(token.Dem != null);
                token.Matrices = new Matrix4d[Height, Width];

                for (int y = 0; y < Height; y++)
                {
                    int line = token.Row + y;
                    for (int x = 0; x < Width; x++)
                    {
                        int sample = token.Col + x;
                        token.Matrices[y, x] = token.Dem.GetMoonMEToENU(line, sample);
                    }
                }

                return Task.FromResult(token);
            }
            finally
            {
                RecordStage(
                    ref _generateMatricesTicks,
                    ref _generateMatricesMaxTicks,
                    ref _generateMatricesCount,
                    start);
            }
        }

        private unsafe Task<Hdf5HorizonProcessingToken> GenerateShadows(Hdf5HorizonProcessingToken token)
        {
            long start = Stopwatch.GetTimestamp();
            try
            {
                if (token.Horizons == null)
                {
                    Log.Warning("Skipping shadow generation for {Filename} due to missing horizon data.", token.Filename);
                    return Task.FromResult(token);
                }

                Debug.Assert(_times != null && token.SunVectorsMe != null && token.Matrices != null && token.Dem != null);

                int numTimes = _times.Count;
                byte[] patchBuffer = new byte[checked(Width * Height * numTimes)];
                float[] horizons = token.Horizons;
                Matrix4d[,] matrices = token.Matrices;
                ElevationMap dem = token.Dem;
                List<Vector3d> sunvecsMe = token.SunVectorsMe;

                fixed (float* horizonsPtr = horizons)
                {
                    for (int y = 0; y < Height; y++)
                    {
                        for (int x = 0; x < Width; x++)
                        {
                            int pixelIdx = y * Width + x;
                            float* pixelHorizons = horizonsPtr + pixelIdx * HorizonSamples;
                            Matrix4d mat = matrices[y, x];
                            int outputBase = pixelIdx * numTimes;

                            for (int t = 0; t < numTimes; t++)
                            {
                                var (azRad, elRad) = dem.GetAzEl(sunvecsMe[t], mat);
                                float azDeg = azRad * 57.2957795f;
                                float elDeg = elRad * 57.2957795f;
                                float frac = LightmapGenerator.BuilderSunFraction(horizons, pixelIdx * HorizonSamples, azDeg, elDeg);
                                patchBuffer[outputBase + t] = (byte)(255f * frac);
                            }
                        }
                    }
                }

                token.ShadowPatchBuffer = patchBuffer;
                return Task.FromResult(token);
            }
            finally
            {
                RecordStage(
                    ref _generateShadowsTicks,
                    ref _generateShadowsMaxTicks,
                    ref _generateShadowsCount,
                    start);
            }
        }

        private void WriteData(Hdf5HorizonProcessingToken token)
        {
            long start = Stopwatch.GetTimestamp();
            try
            {
                if (token.ShadowPatchBuffer == null)
                    return;
                var writer = _writer ?? throw new InvalidOperationException("HDF5 writer is not initialized.");
                writer.WritePatch(token.Col, token.Row, Width, Height, token.ShadowPatchBuffer);
            }
            finally
            {
                RecordStage(
                    ref _writeDataTicks,
                    ref _writeDataMaxTicks,
                    ref _writeDataCount,
                    start);
            }
        }

        private void ResetMetrics()
        {
            _readHorizonsTicks = 0;
            _readHorizonsMaxTicks = 0;
            _readHorizonsCount = 0;
            _generateMatricesTicks = 0;
            _generateMatricesMaxTicks = 0;
            _generateMatricesCount = 0;
            _generateShadowsTicks = 0;
            _generateShadowsMaxTicks = 0;
            _generateShadowsCount = 0;
            _writeDataTicks = 0;
            _writeDataMaxTicks = 0;
            _writeDataCount = 0;
        }

        private static void RecordStage(ref long totalTicks, ref long maxTicks, ref int count, long startTimestamp)
        {
            long elapsed = Stopwatch.GetTimestamp() - startTimestamp;
            Interlocked.Add(ref totalTicks, elapsed);
            Interlocked.Increment(ref count);

            long currentMax;
            do
            {
                currentMax = Volatile.Read(ref maxTicks);
                if (elapsed <= currentMax)
                    return;
            }
            while (Interlocked.CompareExchange(ref maxTicks, elapsed, currentMax) != currentMax);
        }

        private void LogMetrics(TimeSpan elapsed, int expectedCount)
        {
            Log.Information(
                "HDF5 lightmap pipeline elapsed={ElapsedSeconds:F3}s expected_patches={ExpectedCount}",
                elapsed.TotalSeconds,
                expectedCount);
            LogStageMetrics("read_horizons", _readHorizonsCount, _readHorizonsTicks, _readHorizonsMaxTicks, elapsed);
            LogStageMetrics("generate_matrices", _generateMatricesCount, _generateMatricesTicks, _generateMatricesMaxTicks, elapsed);
            LogStageMetrics("generate_shadows", _generateShadowsCount, _generateShadowsTicks, _generateShadowsMaxTicks, elapsed);
            LogStageMetrics("write_data", _writeDataCount, _writeDataTicks, _writeDataMaxTicks, elapsed);
        }

        private static void LogStageMetrics(string name, int count, long totalTicks, long maxTicks, TimeSpan wallTime)
        {
            double totalSeconds = TicksToSeconds(totalTicks);
            double wallSeconds = Math.Max(wallTime.TotalSeconds, 1e-9);
            double avgMilliseconds = count == 0 ? 0.0 : totalSeconds * 1000.0 / count;
            Log.Information(
                "HDF5 stage {Stage}: count={Count} total={TotalSeconds:F3}s avg={AverageMilliseconds:F3}ms max={MaxMilliseconds:F3}ms cpu_equiv_parallelism={Parallelism:F2}",
                name,
                count,
                totalSeconds,
                avgMilliseconds,
                TicksToSeconds(maxTicks) * 1000.0,
                totalSeconds / wallSeconds);
        }

        private static double TicksToSeconds(long ticks)
        {
            return (double)ticks / Stopwatch.Frequency;
        }

        private sealed class Hdf5HorizonProcessingToken
        {
            public ElevationMap? Dem { get; init; }
            public required string Filename { get; init; }
            public int Row { get; set; }
            public int Col { get; set; }
            public float ObserverElevation { get; set; }
            public float[]? Horizons { get; set; }
            public Matrix4d[,]? Matrices { get; set; }
            public List<Vector3d>? SunVectorsMe { get; init; }
            public byte[]? ShadowPatchBuffer { get; set; }
        }

        private sealed class Hdf5LightCurveWriter : IDisposable
        {
            private const string DatasetName = "light_curves";
            private readonly object _writeGate = new();
            private readonly string _outputPath;
            private readonly string _temporaryPath;
            private readonly int _width;
            private readonly int _height;
            private readonly int _timeCount;
            private long _file = -1;
            private long _dataset = -1;
            private bool _committed;

            public Hdf5LightCurveWriter(
                string outputPath,
                int width,
                int height,
                IReadOnlyList<DateTime> timestamps,
                string projection,
                double[] geoTransform)
            {
                _outputPath = Path.GetFullPath(outputPath);
                _width = width;
                _height = height;
                _timeCount = timestamps.Count;

                string? outputDirectory = Path.GetDirectoryName(_outputPath);
                if (string.IsNullOrWhiteSpace(outputDirectory))
                    throw new InvalidOperationException("HDF5 output path has no parent directory.");
                Directory.CreateDirectory(outputDirectory);
                _temporaryPath = Path.Combine(
                    outputDirectory,
                    $".{Path.GetFileName(_outputPath)}.{Guid.NewGuid():N}.tmp");

                CreateFile(timestamps, projection, geoTransform);
            }

            public void WritePatch(int xOffset, int yOffset, int patchWidth, int patchHeight, byte[] values)
            {
                if (values.Length != checked(patchWidth * patchHeight * _timeCount))
                    throw new ArgumentException("Patch buffer length does not match patch dimensions.", nameof(values));

                int writeWidth = Math.Min(patchWidth, _width - xOffset);
                int writeHeight = Math.Min(patchHeight, _height - yOffset);
                if (xOffset < 0 || yOffset < 0 || writeWidth <= 0 || writeHeight <= 0)
                    throw new ArgumentOutOfRangeException(nameof(xOffset), "Patch lies outside the HDF5 output extent.");

                lock (_writeGate)
                {
                    long fileSpace = -1;
                    long memSpace = -1;
                    GCHandle handle = default;
                    try
                    {
                        fileSpace = H5D.get_space(_dataset);
                        CheckId(fileSpace, "H5D.get_space");
                        ulong[] fileStart = { (ulong)yOffset, (ulong)xOffset, 0 };
                        ulong[] fileCount = { (ulong)writeHeight, (ulong)writeWidth, (ulong)_timeCount };
                        CheckNonNegative(
                            H5S.select_hyperslab(fileSpace, H5S.seloper_t.SET, fileStart, null, fileCount, null),
                            "H5S.select_hyperslab(file)");

                        ulong[] memDims = { (ulong)patchHeight, (ulong)patchWidth, (ulong)_timeCount };
                        memSpace = H5S.create_simple(3, memDims, null);
                        CheckId(memSpace, "H5S.create_simple(memory)");
                        ulong[] memStart = { 0, 0, 0 };
                        ulong[] memCount = { (ulong)writeHeight, (ulong)writeWidth, (ulong)_timeCount };
                        CheckNonNegative(
                            H5S.select_hyperslab(memSpace, H5S.seloper_t.SET, memStart, null, memCount, null),
                            "H5S.select_hyperslab(memory)");

                        handle = GCHandle.Alloc(values, GCHandleType.Pinned);
                        CheckNonNegative(
                            H5D.write(
                                _dataset,
                                H5T.NATIVE_UINT8,
                                memSpace,
                                fileSpace,
                                H5P.DEFAULT,
                                handle.AddrOfPinnedObject()),
                            "H5D.write");
                    }
                    finally
                    {
                        if (handle.IsAllocated)
                            handle.Free();
                        if (memSpace >= 0)
                            H5S.close(memSpace);
                        if (fileSpace >= 0)
                            H5S.close(fileSpace);
                    }
                }
            }

            public void Commit()
            {
                lock (_writeGate)
                {
                    if (_committed)
                        throw new InvalidOperationException("HDF5 light-curve writer has already been committed.");
                    CloseHandles();
                    File.Move(_temporaryPath, _outputPath, overwrite: true);
                    _committed = true;
                }
            }

            public void Dispose()
            {
                lock (_writeGate)
                {
                    CloseHandles();
                    if (!_committed && File.Exists(_temporaryPath))
                    {
                        try { File.Delete(_temporaryPath); }
                        catch { /* best-effort scratch cleanup */ }
                    }
                }
            }

            private void CreateFile(IReadOnlyList<DateTime> timestamps, string projection, double[] geoTransform)
            {
                _file = H5F.create(_temporaryPath, H5F.ACC_TRUNC);
                CheckId(_file, "H5F.create");

                long fileSpace = -1;
                long plist = -1;
                try
                {
                    ulong[] dims = { (ulong)_height, (ulong)_width, (ulong)_timeCount };
                    fileSpace = H5S.create_simple(3, dims, null);
                    CheckId(fileSpace, "H5S.create_simple(file)");

                    plist = H5P.create(H5P.DATASET_CREATE);
                    CheckId(plist, "H5P.create(DATASET_CREATE)");
                    ulong[] chunks = { 64, 64, 64 };
                    CheckNonNegative(H5P.set_chunk(plist, 3, chunks), "H5P.set_chunk");
                    CheckNonNegative(H5P.set_deflate(plist, 1), "H5P.set_deflate");

                    _dataset = H5D.create(
                        _file,
                        DatasetName,
                        H5T.STD_U8LE,
                        fileSpace,
                        H5P.DEFAULT,
                        plist,
                        H5P.DEFAULT);
                    CheckId(_dataset, "H5D.create");

                    WriteStringAttribute(_dataset, "axis_order", "y,x,time");
                    WriteStringAttribute(_dataset, "signal_name", "sun_fraction");
                    WriteStringAttribute(_dataset, "units", "uint8_scaled_0_255");
                    WriteStringAttribute(_dataset, "projection_wkt", projection ?? string.Empty);
                    WriteDoubleArrayAttribute(_dataset, "geo_transform", geoTransform);
                    WriteStringArrayAttribute(_dataset, "timestamps_utc", timestamps.Select(FormatUtc).ToArray());
                }
                finally
                {
                    if (plist >= 0)
                        H5P.close(plist);
                    if (fileSpace >= 0)
                        H5S.close(fileSpace);
                }
            }

            private void CloseHandles()
            {
                if (_dataset >= 0)
                {
                    H5D.close(_dataset);
                    _dataset = -1;
                }
                if (_file >= 0)
                {
                    H5F.flush(_file, H5F.scope_t.GLOBAL);
                    H5F.close(_file);
                    _file = -1;
                }
            }

            private static string FormatUtc(DateTime timestamp)
            {
                DateTime utc = timestamp.Kind == DateTimeKind.Utc
                    ? timestamp
                    : timestamp.ToUniversalTime();
                return utc.ToString("yyyy-MM-ddTHH:mm:ss.fffffffZ");
            }

            private static void WriteStringAttribute(long target, string name, string value)
            {
                byte[] bytes = Encoding.UTF8.GetBytes(value);
                long type = H5T.copy(H5T.C_S1);
                CheckId(type, "H5T.copy(C_S1)");
                long space = -1;
                long attr = -1;
                GCHandle handle = default;
                try
                {
                    CheckNonNegative(H5T.set_size(type, new IntPtr(bytes.Length)), "H5T.set_size");
                    CheckNonNegative(H5T.set_cset(type, H5T.cset_t.UTF8), "H5T.set_cset");
                    space = H5S.create(H5S.class_t.SCALAR);
                    CheckId(space, "H5S.create(attribute)");
                    attr = H5A.create(target, name, type, space);
                    CheckId(attr, "H5A.create");
                    handle = GCHandle.Alloc(bytes, GCHandleType.Pinned);
                    CheckNonNegative(H5A.write(attr, type, handle.AddrOfPinnedObject()), "H5A.write");
                }
                finally
                {
                    if (handle.IsAllocated)
                        handle.Free();
                    if (attr >= 0)
                        H5A.close(attr);
                    if (space >= 0)
                        H5S.close(space);
                    if (type >= 0)
                        H5T.close(type);
                }
            }

            private static void WriteDoubleArrayAttribute(long target, string name, IReadOnlyList<double> values)
            {
                long space = -1;
                long attr = -1;
                GCHandle handle = default;
                try
                {
                    ulong[] dims = { (ulong)values.Count };
                    space = H5S.create_simple(1, dims, null);
                    CheckId(space, "H5S.create_simple(double attribute)");
                    attr = H5A.create(target, name, H5T.IEEE_F64LE, space);
                    CheckId(attr, "H5A.create(double attribute)");
                    double[] copy = values.ToArray();
                    handle = GCHandle.Alloc(copy, GCHandleType.Pinned);
                    CheckNonNegative(H5A.write(attr, H5T.NATIVE_DOUBLE, handle.AddrOfPinnedObject()), "H5A.write(double attribute)");
                }
                finally
                {
                    if (handle.IsAllocated)
                        handle.Free();
                    if (attr >= 0)
                        H5A.close(attr);
                    if (space >= 0)
                        H5S.close(space);
                }
            }

            private static void WriteStringArrayAttribute(long target, string name, IReadOnlyList<string> values)
            {
                int width = Math.Max(1, values.Max(value => Encoding.UTF8.GetByteCount(value)));
                byte[] bytes = new byte[checked(values.Count * width)];
                for (int index = 0; index < values.Count; index++)
                {
                    byte[] encoded = Encoding.UTF8.GetBytes(values[index]);
                    Array.Copy(encoded, 0, bytes, index * width, encoded.Length);
                }

                long type = H5T.copy(H5T.C_S1);
                CheckId(type, "H5T.copy(C_S1 array)");
                long space = -1;
                long attr = -1;
                GCHandle handle = default;
                try
                {
                    CheckNonNegative(H5T.set_size(type, new IntPtr(width)), "H5T.set_size(array)");
                    CheckNonNegative(H5T.set_cset(type, H5T.cset_t.UTF8), "H5T.set_cset(array)");
                    ulong[] dims = { (ulong)values.Count };
                    space = H5S.create_simple(1, dims, null);
                    CheckId(space, "H5S.create_simple(string array attribute)");
                    attr = H5A.create(target, name, type, space);
                    CheckId(attr, "H5A.create(string array attribute)");
                    handle = GCHandle.Alloc(bytes, GCHandleType.Pinned);
                    CheckNonNegative(H5A.write(attr, type, handle.AddrOfPinnedObject()), "H5A.write(string array attribute)");
                }
                finally
                {
                    if (handle.IsAllocated)
                        handle.Free();
                    if (attr >= 0)
                        H5A.close(attr);
                    if (space >= 0)
                        H5S.close(space);
                    if (type >= 0)
                        H5T.close(type);
                }
            }

            private static void CheckId(long id, string operation)
            {
                if (id < 0)
                    throw new InvalidOperationException($"{operation} failed with id={id}.");
            }

            private static void CheckNonNegative(int status, string operation)
            {
                if (status < 0)
                    throw new InvalidOperationException($"{operation} failed with status={status}.");
            }
        }
    }
}
