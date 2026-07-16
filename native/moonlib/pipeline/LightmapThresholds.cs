using HDF.PInvoke;
using moonlib.horizon;
using moonlib.math;
using moonlib.spice;
using OSGeo.GDAL;
using Serilog;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;

namespace moonlib.pipeline
{
    /// <summary>
    /// Pipeline-based HDF5 generation for per-time Sun/Earth horizon-margin
    /// threshold bitsets. Timestamps are assumed to be evenly spaced; this is
    /// recorded as start/stop/step metadata rather than validated exhaustively.
    /// </summary>
    public sealed class LightmapThresholds
    {
        public const int Width = 128;
        public const int Height = 128;

        /// <summary>
        /// Load a DEM from <paramref name="demPath"/> and generate a threshold-bit HDF5 cube.
        /// </summary>
        public Task WriteLightmapThresholdFile(
            List<DateTime> timestamps,
            string outputPath,
            string demPath,
            string horizonDir,
            float[] sunThresholdsDeg,
            float[] earthThresholdsDeg,
            IProgress<float>? progress = null,
            Func<DateTime, Vector3d>? sunVectorProvider = null,
            Func<DateTime, Vector3d>? earthVectorProvider = null)
        {
            if (string.IsNullOrWhiteSpace(demPath))
                throw new ArgumentException("DEM path must be non-empty.", nameof(demPath));
            var elevationMap = new ElevationMap(demPath);
            return WriteLightmapThresholdFile(
                timestamps,
                outputPath,
                elevationMap,
                horizonDir,
                sunThresholdsDeg,
                earthThresholdsDeg,
                progress,
                sunVectorProvider,
                earthVectorProvider);
        }

        /// <summary>
        /// Generate a threshold-bit HDF5 cube from an already loaded DEM.
        /// The timestamp list is assumed to be evenly spaced.
        /// </summary>
        public async Task WriteLightmapThresholdFile(
            List<DateTime> timestamps,
            string outputPath,
            ElevationMap elevationMap,
            string horizonDir,
            float[] sunThresholdsDeg,
            float[] earthThresholdsDeg,
            IProgress<float>? progress = null,
            Func<DateTime, Vector3d>? sunVectorProvider = null,
            Func<DateTime, Vector3d>? earthVectorProvider = null)
        {
            ValidateInputs(
                timestamps,
                outputPath,
                elevationMap,
                horizonDir,
                sunThresholdsDeg,
                earthThresholdsDeg);

            List<Vector3d> sunVectorsMe = BuildSunVectors(timestamps, sunVectorProvider);
            List<Vector3d> earthVectorsMe = BuildEarthVectors(timestamps, earthVectorProvider);

            using var writer = new Hdf5ThresholdBitWriter(
                outputPath,
                elevationMap.Width,
                elevationMap.Height,
                timestamps,
                elevationMap.Projection,
                elevationMap.GeoTransform,
                sunThresholdsDeg,
                earthThresholdsDeg);

            using var lightmaps = new Lightmaps();
            var queue = lightmaps.StreamThresholdPatches(
                elevationMap,
                horizonDir,
                sunVectorsMe,
                earthVectorsMe,
                sunThresholdsDeg,
                earthThresholdsDeg,
                progress);

            await Task.Run(() =>
            {
                foreach (PatchThresholdResult patch in queue.GetConsumingEnumerable())
                    writer.WritePatch(patch.PatchCol, patch.PatchRow, Width, Height, patch.Data);
            });

            if (lightmaps.BackgroundTaskError is not null)
                throw new InvalidOperationException(
                    "Threshold-bit generation failed.",
                    lightmaps.BackgroundTaskError);

            writer.Commit();
        }

        /// <summary>
        /// Load a DEM from <paramref name="demPath"/> and write per-time threshold bit counts to CSV.
        /// </summary>
        public Task CountThresholdedPixels(
            List<DateTime> timestamps,
            string outputCsvPath,
            string demPath,
            string horizonDir,
            float[] sunThresholdsDeg,
            float[] earthThresholdsDeg,
            IProgress<float>? progress = null,
            Func<DateTime, Vector3d>? sunVectorProvider = null,
            Func<DateTime, Vector3d>? earthVectorProvider = null)
        {
            if (string.IsNullOrWhiteSpace(demPath))
                throw new ArgumentException("DEM path must be non-empty.", nameof(demPath));
            var elevationMap = new ElevationMap(demPath);
            return CountThresholdedPixels(
                timestamps,
                outputCsvPath,
                elevationMap,
                horizonDir,
                sunThresholdsDeg,
                earthThresholdsDeg,
                progress,
                sunVectorProvider,
                earthVectorProvider);
        }

        /// <summary>
        /// Write one CSV row per timestamp with counts of set bits 0-7 across all generated patches.
        /// The timestamp list is assumed to be evenly spaced.
        /// </summary>
        public async Task CountThresholdedPixels(
            List<DateTime> timestamps,
            string outputCsvPath,
            ElevationMap elevationMap,
            string horizonDir,
            float[] sunThresholdsDeg,
            float[] earthThresholdsDeg,
            IProgress<float>? progress = null,
            Func<DateTime, Vector3d>? sunVectorProvider = null,
            Func<DateTime, Vector3d>? earthVectorProvider = null)
        {
            ValidateInputs(
                timestamps,
                outputCsvPath,
                elevationMap,
                horizonDir,
                sunThresholdsDeg,
                earthThresholdsDeg);

            List<Vector3d> sunVectorsMe = BuildSunVectors(timestamps, sunVectorProvider);
            List<Vector3d> earthVectorsMe = BuildEarthVectors(timestamps, earthVectorProvider);
            long[,] counts = new long[timestamps.Count, 8];

            using var lightmaps = new Lightmaps();
            var queue = lightmaps.StreamThresholdPatches(
                elevationMap,
                horizonDir,
                sunVectorsMe,
                earthVectorsMe,
                sunThresholdsDeg,
                earthThresholdsDeg,
                progress);

            await Task.Run(() =>
            {
                foreach (PatchThresholdResult patch in queue.GetConsumingEnumerable())
                    CountPatchBits(patch, timestamps.Count, counts);
            });

            if (lightmaps.BackgroundTaskError is not null)
                throw new InvalidOperationException(
                    "Threshold-bit counting failed.",
                    lightmaps.BackgroundTaskError);

            WriteCountsCsv(outputCsvPath, timestamps, counts);
        }

        /// <summary>
        /// Write one GDT_UInt16 geotiff per month within the timestamps list's range giving the duration in hours of the longest mission window starting that month.
	///   DataType.GDT_UInt16;
        /// </summary>
        public Task WriteLandedMissionDurations(
            List<DateTime> timestamps,
            string outputFilePattern,
            string demPath,
            string horizonDir,
            float[] sunThresholdsDeg,
            float[] earthThresholdsDeg,
            IProgress<float>? progress = null,
            Func<DateTime, Vector3d>? sunVectorProvider = null,
            Func<DateTime, Vector3d>? earthVectorProvider = null)
        {
            if (string.IsNullOrWhiteSpace(demPath))
                throw new ArgumentException("DEM path must be non-empty.", nameof(demPath));
            var elevationMap = new ElevationMap(demPath);
            return WriteLandedMissionDurations(
                timestamps,
                outputFilePattern,
                elevationMap,
                horizonDir,
                sunThresholdsDeg,
                earthThresholdsDeg,
                progress,
                sunVectorProvider,
                earthVectorProvider);
        }

        /// <summary>
        /// Write one CSV row per timestamp with counts of set bits 0-7 across all generated patches.
        /// The timestamp list is assumed to be evenly spaced.
        /// </summary>
        public async Task WriteLandedMissionDurations(
            List<DateTime> timestamps,
            string outputFilePattern,
            ElevationMap elevationMap,
            string horizonDir,
            float[] sunThresholdsDeg,
            float[] earthThresholdsDeg,
            IProgress<float>? progress = null,
            Func<DateTime, Vector3d>? sunVectorProvider = null,
            Func<DateTime, Vector3d>? earthVectorProvider = null)
        {
            ValidateInputs(
                timestamps,
                outputFile,
                elevationMap,
                horizonDir,
                sunThresholdsDeg,
                earthThresholdsDeg);

            var driver = Gdal.GetDriverByName("GTiff")
                         ?? throw new InvalidOperationException("GDAL GTiff driver is unavailable.");

            List<Vector3d> sunVectorsMe = BuildSunVectors(timestamps, sunVectorProvider);
            List<Vector3d> earthVectorsMe = BuildEarthVectors(timestamps, earthVectorProvider);
            long[,] counts = new long[timestamps.Count, 8];

	    var (months, month_indices) = GetMonthStartsAndIndices(timestamps);

	    string[] _creationOptions = { "TILED=YES", "BLOCKXSIZE=128", "BLOCKYSIZE=128", "COMPRESS=LZW", "BIGTIFF=YES", "SPARSE_OK=TRUE" };

            System.IO.Directory.CreateDirectory(System.IO.Path.GetDirectoryName(outputFile)!);
	    
            var dataset = driver.Create(outputFile, elevationMap.Width, elevationMap.Height, months.Count, DataType.GDT_Int16, _creationOptions);
            if (dataset is null)
                throw new InvalidOperationException($"Failed to create dataset '{outputPath}'.");

	    for (int b = 1; b <= months.Count; b++)
                dataset.GetRasterBand(b).SetNoDataValue(0);

            ds.SetProjection(elevationMap.Projection);
            ds.SetGeoTransform(elevationMap.GeoTransform);

	    try
	    {
		using var lightmaps = new Lightmaps();
		var queue = lightmaps.StreamThresholdPatches(
		    elevationMap,
		    horizonDir,
		    sunVectorsMe,
		    earthVectorsMe,
		    sunThresholdsDeg,
		    earthThresholdsDeg,
		    progress);

		await Task.Run(() =>
		{
		    Parallel.ForEach(
			queue.GetConsumingEnumerable(),
			new ParallelOptions { MaxDegreeOfParallelism = 8 },
			patchResult => {
			    FindLongestMissionDurations(patchResult, month_indices, dataset);
			    });
		});

		if (lightmaps.BackgroundTaskError is not null)
		    throw new InvalidOperationException(
			"Threshold-bit counting failed.",
			lightmaps.BackgroundTaskError);
	    }
	    finally
	    {
		dataset?.Dispose();
	    }
        }

	private static void FindLongestMissionDurations(PatchThresholdResult result, List<int> month_indices, Dataset[] datasets)
	{
	    // I'm not sure whether the bit in this mask is in the right position.
	    byte mask = 1;
	    var (col, row) = (result.PatchCol, result.PatchRow);
	    
	    
	}

        /// <summary>
        /// Write a 128×128 patch of flat pixel data into band 1 of <paramref name="dataset"/>
        /// at the given upper‑left column / row.
        /// </summary>
        public static void WritePatch<T>(this Dataset dataset, int col, int row, T[] data) where T : unmanaged
        {
            var band = dataset.GetRasterBand(1);
            var handle = GCHandle.Alloc(data, GCHandleType.Pinned);
            try
            {
                band.WriteRaster(col, row, PatchSize, PatchSize, handle.AddrOfPinnedObject(),
                    PatchSize, PatchSize, GetGdalType<T>(), 0, 0);
            }
            finally
            {
                handle.Free();
            }
        }

        private static DataType GetGdalType<T>() where T : unmanaged =>
            typeof(T) switch
            {
                Type t when t == typeof(byte)   => DataType.GDT_Byte,
                Type t when t == typeof(float)  => DataType.GDT_Float32,
                Type t when t == typeof(double) => DataType.GDT_Float64,
                Type t when t == typeof(int)    => DataType.GDT_Int32,
                Type t when t == typeof(short)  => DataType.GDT_Int16,
                Type t when t == typeof(uint)   => DataType.GDT_UInt32,
                Type t when t == typeof(ushort) => DataType.GDT_UInt16,
                _ => throw new ArgumentException($"Unsupported pixel type '{typeof(T).Name}'.")
            };

        private static List<Vector3d> BuildSunVectors(
            IReadOnlyList<DateTime> timestamps,
            Func<DateTime, Vector3d>? sunVectorProvider)
        {
            return sunVectorProvider is null
                ? timestamps.Select(t => SpiceManager.SunPosition_meters(t)).ToList()
                : timestamps.Select(sunVectorProvider).ToList();
        }

        private static List<Vector3d> BuildEarthVectors(
            IReadOnlyList<DateTime> timestamps,
            Func<DateTime, Vector3d>? earthVectorProvider)
        {
            return earthVectorProvider is null
                ? timestamps.Select(t => SpiceManager.EarthPosition_meters(t)).ToList()
                : timestamps.Select(earthVectorProvider).ToList();
        }

        internal static void CountPatchBits(PatchThresholdResult patch, int timeCount, long[,] counts)
        {
            int expectedLength = checked(Width * Height * timeCount);
            if (patch.Data.Length != expectedLength)
                throw new ArgumentException("Patch buffer length does not match threshold output dimensions.", nameof(patch));

            for (int index = 0; index < patch.Data.Length; index++)
            {
                byte value = patch.Data[index];
                int timeIndex = index % timeCount;
                for (int bit = 0; bit < 8; bit++)
                {
                    if ((value & (1 << bit)) != 0)
                        counts[timeIndex, bit]++;
                }
            }
        }

        private static void WriteCountsCsv(
            string outputCsvPath,
            IReadOnlyList<DateTime> timestamps,
            long[,] counts)
        {
            string fullPath = Path.GetFullPath(outputCsvPath);
            string? outputDirectory = Path.GetDirectoryName(fullPath);
            if (!string.IsNullOrWhiteSpace(outputDirectory))
                Directory.CreateDirectory(outputDirectory);

            using var writer = new StreamWriter(fullPath, append: false, new UTF8Encoding(false));
            writer.WriteLine(
                "timestamp_utc,sun_threshold_0_count,sun_threshold_1_count,sun_threshold_2_count,sun_threshold_3_count,earth_threshold_0_count,earth_threshold_1_count,earth_threshold_2_count,earth_threshold_3_count");

            for (int timeIndex = 0; timeIndex < timestamps.Count; timeIndex++)
            {
                writer.Write(FormatUtc(timestamps[timeIndex]));
                for (int bit = 0; bit < 8; bit++)
                {
                    writer.Write(',');
                    writer.Write(counts[timeIndex, bit].ToString(CultureInfo.InvariantCulture));
                }
                writer.WriteLine();
            }
        }

        private static string FormatUtc(DateTime timestamp)
        {
            DateTime utc = timestamp.Kind == DateTimeKind.Utc
                ? timestamp
                : timestamp.ToUniversalTime();
            return utc.ToString("yyyy-MM-ddTHH:mm:ss.fffffffZ", CultureInfo.InvariantCulture);
        }

        private static void ValidateInputs(
            IReadOnlyList<DateTime> timestamps,
            string outputPath,
            ElevationMap elevationMap,
            string horizonDir,
            IReadOnlyList<float> sunThresholdsDeg,
            IReadOnlyList<float> earthThresholdsDeg)
        {
            if (timestamps.Count == 0)
                throw new ArgumentException("At least one timestamp is required.", nameof(timestamps));
            if (string.IsNullOrWhiteSpace(outputPath))
                throw new ArgumentException("Output HDF5 path must be non-empty.", nameof(outputPath));
            if (elevationMap == null)
                throw new ArgumentNullException(nameof(elevationMap));
            if (string.IsNullOrWhiteSpace(horizonDir))
                throw new ArgumentException("Horizon directory path must be non-empty.", nameof(horizonDir));
            if (sunThresholdsDeg.Count != 4)
                throw new ArgumentException("Exactly four Sun thresholds are required.", nameof(sunThresholdsDeg));
            if (earthThresholdsDeg.Count != 4)
                throw new ArgumentException("Exactly four Earth thresholds are required.", nameof(earthThresholdsDeg));
            if (sunThresholdsDeg.Any(value => !float.IsFinite(value)))
                throw new ArgumentException("Sun thresholds must be finite degree values.", nameof(sunThresholdsDeg));
            if (earthThresholdsDeg.Any(value => !float.IsFinite(value)))
                throw new ArgumentException("Earth thresholds must be finite degree values.", nameof(earthThresholdsDeg));
        }

        private sealed class Hdf5ThresholdBitWriter : IDisposable
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

            public Hdf5ThresholdBitWriter(
                string outputPath,
                int width,
                int height,
                IReadOnlyList<DateTime> timestamps,
                string projection,
                double[] geoTransform,
                IReadOnlyList<float> sunThresholdsDeg,
                IReadOnlyList<float> earthThresholdsDeg)
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

                CreateFile(
                    timestamps,
                    projection,
                    geoTransform,
                    sunThresholdsDeg,
                    earthThresholdsDeg);
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
                        throw new InvalidOperationException("HDF5 threshold-bit writer has already been committed.");
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

            private void CreateFile(
                IReadOnlyList<DateTime> timestamps,
                string projection,
                double[] geoTransform,
                IReadOnlyList<float> sunThresholdsDeg,
                IReadOnlyList<float> earthThresholdsDeg)
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
                    WriteStringAttribute(_dataset, "signal_name", "sun_earth_threshold_bits");
                    WriteStringAttribute(_dataset, "units", "bitset");
                    WriteStringAttribute(_dataset, "bit_layout", "sun:0-3,earth:4-7");
                    WriteStringAttribute(_dataset, "threshold_comparison", "margin_deg >= threshold_deg");
                    WriteStringAttribute(_dataset, "time_range_assumption", "timestamps_utc are evenly spaced");
                    WriteStringAttribute(_dataset, "projection_wkt", projection ?? string.Empty);
                    WriteDoubleArrayAttribute(_dataset, "geo_transform", geoTransform);
                    WriteFloatArrayAttribute(_dataset, "sun_thresholds_deg", sunThresholdsDeg);
                    WriteFloatArrayAttribute(_dataset, "earth_thresholds_deg", earthThresholdsDeg);
                    WriteStringArrayAttribute(_dataset, "timestamps_utc", timestamps.Select(FormatUtc).ToArray());
                    WriteStringAttribute(_dataset, "start_time_utc", FormatUtc(timestamps[0]));
                    WriteStringAttribute(_dataset, "stop_time_utc", FormatUtc(timestamps[^1]));
                    WriteDoubleAttribute(_dataset, "time_step_seconds", TimeStepSeconds(timestamps));
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

            private static double TimeStepSeconds(IReadOnlyList<DateTime> timestamps)
            {
                if (timestamps.Count < 2)
                    return 0.0;
                DateTime startUtc = timestamps[0].Kind == DateTimeKind.Utc
                    ? timestamps[0]
                    : timestamps[0].ToUniversalTime();
                DateTime secondUtc = timestamps[1].Kind == DateTimeKind.Utc
                    ? timestamps[1]
                    : timestamps[1].ToUniversalTime();
                return (secondUtc - startUtc).TotalSeconds;
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

            private static void WriteDoubleAttribute(long target, string name, double value)
            {
                long space = -1;
                long attr = -1;
                GCHandle handle = default;
                try
                {
                    space = H5S.create(H5S.class_t.SCALAR);
                    CheckId(space, "H5S.create(double attribute)");
                    attr = H5A.create(target, name, H5T.IEEE_F64LE, space);
                    CheckId(attr, "H5A.create(double attribute)");
                    double[] copy = { value };
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

            private static void WriteFloatArrayAttribute(long target, string name, IReadOnlyList<float> values)
            {
                long space = -1;
                long attr = -1;
                GCHandle handle = default;
                try
                {
                    ulong[] dims = { (ulong)values.Count };
                    space = H5S.create_simple(1, dims, null);
                    CheckId(space, "H5S.create_simple(float attribute)");
                    attr = H5A.create(target, name, H5T.IEEE_F32LE, space);
                    CheckId(attr, "H5A.create(float attribute)");
                    float[] copy = values.ToArray();
                    handle = GCHandle.Alloc(copy, GCHandleType.Pinned);
                    CheckNonNegative(H5A.write(attr, H5T.NATIVE_FLOAT, handle.AddrOfPinnedObject()), "H5A.write(float attribute)");
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
