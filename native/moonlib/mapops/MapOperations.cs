using moonlib.horizon;
using moonlib.pipeline;
using moonlib.spice;
using moonlib.util;
using OSGeo.GDAL;
using Serilog;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Drawing.Drawing2D;
using System.Security.Cryptography;

namespace moonlib.mapops
{
    public class MapOperations
    {
        const int HorizonSamples = 1440;
        public static void GenerateHorizons(AnalysisContext context)
        {

        }

        /// <summary>
        /// Generate and write a permanent shadow file.  The horizon files need to already exist.
        /// </summary>
        /// <param name="context"></param>
        /// <param name="output_path"></param>
        /// <param name="progress"></param>
        /// <param name="isCancellationRequested"></param>
        /// <returns></returns>
        /// <exception cref="OperationCanceledException"></exception>
        /// <exception cref="Exception"></exception>
        public static async Task GeneratePermanentShadowMap(
            AnalysisContext context,
            string output_path,
            bool overwrite_existing = false,
            IProgress<float>? progress = null,
            Func<bool>? isCancellationRequested = null,
            List<string>? horizon_filenames = null)
        {
            void ThrowIfCancelled()
            {
                if (isCancellationRequested != null && isCancellationRequested())
                    throw new OperationCanceledException("Permanent shadow map generation was cancelled.");
            }

            Debug.Assert(context.DEM_path != null, "DEM_path must be loaded in context for PSR generation.");
            Debug.Assert(context.HorizonDirectory != null, "HorizonDirectory must be set in context for PSR generation.");
            ThrowIfCancelled();
            var dem = new ElevationMap(context.DEM_path);

            ThrowIfCancelled();
            var all_sunvecs_me = ViperDate.GetTimes(ViperDate.New(1970, 1, 1), ViperDate.New(2044, 1, 1), TimeSpan.FromHours(6))
                .Select(t => SpiceManager.SunPosition_meters(t)).ToList();

            var sunvecs_me = GenerateReducedSunVectorListForPermanentShadowCalculation(dem, all_sunvecs_me);
            Console.WriteLine($"Generated reduced sun vector list for permanent shadow calculation. From {all_sunvecs_me.Count} to {sunvecs_me.Count}");

            // Initialize GDAL if not already (assuming calling app does, but safe to check)
            // In a library, we might assume the app has configured GDAL.
            var geotiff_driver = Gdal.GetDriverByName("GTiff") ?? throw new Exception("GDAL GTiff driver not found.");

            if (overwrite_existing & File.Exists(output_path))
                File.Delete(output_path);

            var psr_image = LightmapPipeline.OpenDataset(geotiff_driver, output_path, DataType.GDT_Byte, null, dem.Width, dem.Height, dem.Projection, dem.GeoTransform);
            var validTiles = new ConcurrentDictionary<(int Col, int Row), byte>();
            progress ??= new Progress<float>(percent => Console.WriteLine($"Progress: {100 * percent}%"));

            await ExecuteAsync(context, output_path, progress);

            async Task ExecuteAsync(AnalysisContext context, string output_path, IProgress<float>? progress = null)
            {
                ThrowIfCancelled();
                var pipeline = new Pipeline<HorizonProcessingToken>();

                pipeline.AddStep(LightmapPipeline.ReadHorizons, maxDegreeOfParallelism: 24, boundedCapacity: 40);
                pipeline.AddStep(GeneratePermanentShadow, maxDegreeOfParallelism: 24, boundedCapacity: 40);

                int processedCount = 0;
                int totalCount = 0;

                pipeline.AddTerminalStep(async token =>
                {
                    ThrowIfCancelled();
                    if (progress != null)
                    {
                        int current = Interlocked.Increment(ref processedCount);
                        progress.Report((float)current / totalCount);
                    }
                }, maxDegreeOfParallelism: 4, boundedCapacity: 40);

                Debug.Assert(context.HorizonDirectory != null, "HorizonDirectory must be set in context for PSR generation.");
                List<string> _horizon_filenames;
                if (horizon_filenames != null)
                    _horizon_filenames = horizon_filenames;
                else
                    _horizon_filenames = new HorizonTileStore(context.HorizonDirectory!)
                        .EnumerateFiles(observerElevationMeters: 0f)
                        .ToList();

                totalCount = _horizon_filenames.Count;

                Log.Information($"Found {_horizon_filenames.Count} horizon files for lightmap generation.");

                ThrowIfCancelled();
                await pipeline.ProcessAsync(_horizon_filenames.Select(f => new HorizonProcessingToken { filename = f }));

                FinalizePsrValidityMask(psr_image, dem.Width, dem.Height, validTiles.Keys);
                psr_image.Close();
            }

            unsafe Task<HorizonProcessingToken> GeneratePermanentShadow(HorizonProcessingToken token)
            {
                if (token.horizons == null)
                    return Task.FromResult(token); // Skip if horizons failed to load for this tile
                ThrowIfCancelled();
                var col = token.col;
                var row = token.row;
                int width = 128;
                int height = 128;
                int numPixels = width * height;

                //Console.WriteLine($"row={row}, col={col}, width={width}, height={height}");

                var psr_buffer = new byte[height * width];

                var horizons_for_patch = token.horizons;
                Debug.Assert(horizons_for_patch != null, "Horizons must be loaded in token for permanent shadow generation.");

                for (int y = 0; y < height; y++)
                {
                    ThrowIfCancelled();
                    for (int x = 0; x < width; x++)
                    {
                        int psr_buffer_index = y * width + x;
                        var (line, sample) = (row + y, col + x);
                        var horizon_base = psr_buffer_index * HorizonSamples;  // in units of horizons
                        var mat = dem.GetMoonMEToENU(line, sample);

                        byte val = 255;

                        for (var i = 0; i < sunvecs_me.Count; i++)
                        {
                            var (az_rad, el_rad) = dem.GetAzEl(sunvecs_me[i], mat);
                            float az_deg = az_rad * 57.2957795f;
                            float el_deg = el_rad * 57.2957795f;

                            var upper_limb_elevation = el_deg + 0.545f / 2f; // Add ~0.25 degrees to account for the sun's upper limb
                            var over_horizon = LightmapGenerator.OverHorizon(horizons_for_patch, horizon_base, az_deg, el_deg);
                            if (over_horizon >= 0f)
                                val = 0;

                            //float frac = LightmapGenerator.BuilderSunFraction(horizons_for_patch, horizon_base, az_deg, el_deg);
                            //if (frac > 0f)
                            //   val = 0;

                        }

                        psr_buffer[psr_buffer_index] = val;
                    }
                }

                lock (psr_image)
                {
                    CPLErr writeResult = psr_image.GetRasterBand(1).WriteRaster(
                        token.col, token.row, width, height,
                        psr_buffer, width, height, 0, 0);
                    if (writeResult != CPLErr.CE_None)
                        throw new IOException(
                            $"Failed to write PSR tile at col={token.col}, row={token.row}.");
                    validTiles.TryAdd((token.col, token.row), 0);
                }

                return Task.FromResult(token);
            }
        }

        internal static bool FinalizePsrValidityMask(
            Dataset dataset,
            int width,
            int height,
            IEnumerable<(int Col, int Row)> validTiles,
            int tileSize = 128)
        {
            ArgumentNullException.ThrowIfNull(dataset);
            ArgumentNullException.ThrowIfNull(validTiles);
            if (width <= 0)
                throw new ArgumentOutOfRangeException(nameof(width));
            if (height <= 0)
                throw new ArgumentOutOfRangeException(nameof(height));
            if (tileSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(tileSize));

            var valid = validTiles.ToHashSet();
            bool allValid = true;
            for (int row = 0; row < height && allValid; row += tileSize)
            {
                for (int col = 0; col < width; col += tileSize)
                {
                    if (!valid.Contains((col, row)))
                    {
                        allValid = false;
                        break;
                    }
                }
            }

            var dataBand = dataset.GetRasterBand(1);
            dataBand.DeleteNoDataValue();
            if (allValid)
                return false;

            string? previousInternalMaskOption = Gdal.GetConfigOption(
                "GDAL_TIFF_INTERNAL_MASK", null);
            try
            {
                Gdal.SetConfigOption("GDAL_TIFF_INTERNAL_MASK", "YES");
                CPLErr createMaskResult = dataset.CreateMaskBand(
                    GdalConst.GMF_PER_DATASET);
                if (createMaskResult != CPLErr.CE_None)
                    throw new IOException("Failed to create the internal PSR validity mask.");
            }
            finally
            {
                Gdal.SetConfigOption(
                    "GDAL_TIFF_INTERNAL_MASK", previousInternalMaskOption);
            }
            var maskBand = dataBand.GetMaskBand();
            for (int row = 0; row < height; row += tileSize)
            {
                int blockHeight = Math.Min(tileSize, height - row);
                for (int col = 0; col < width; col += tileSize)
                {
                    int blockWidth = Math.Min(tileSize, width - col);
                    bool isValid = valid.Contains((col, row));
                    var mask = new byte[blockWidth * blockHeight];
                    if (isValid)
                    {
                        Array.Fill(mask, byte.MaxValue);
                    }
                    else
                    {
                        var emptyData = new byte[blockWidth * blockHeight];
                        CPLErr clearResult = dataBand.WriteRaster(
                            col, row, blockWidth, blockHeight,
                            emptyData, blockWidth, blockHeight, 0, 0);
                        if (clearResult != CPLErr.CE_None)
                            throw new IOException(
                                $"Failed to initialize invalid PSR tile at col={col}, row={row}.");
                    }
                    CPLErr maskWriteResult = maskBand.WriteRaster(
                        col, row, blockWidth, blockHeight,
                        mask, blockWidth, blockHeight, 0, 0);
                    if (maskWriteResult != CPLErr.CE_None)
                        throw new IOException(
                            $"Failed to write PSR validity at col={col}, row={row}.");
                }
            }
            maskBand.FlushCache();
            dataBand.FlushCache();
            dataset.FlushCache();
            return true;
        }

        // Filter the long list of sun vectors to a smaller set.  Keep only those that are the highest elevation in their azimuth bin
        // for at least one of 5 points on the map (the 4 corners and the center).  This is a heuristic to reduce the number of sun
        // vectors we need to consider for permanent shadow calculation.
        public static List<math.Vector3d> GenerateReducedSunVectorListForPermanentShadowCalculation(ElevationMap dem, List<math.Vector3d> input_sunvecs_me)
        {
            var matrices = new math.Matrix4d[5]
            {
                    dem.GetMoonMEToENU(0, 0),
                    dem.GetMoonMEToENU(0, dem.Width - 1),
                    dem.GetMoonMEToENU(dem.Height - 1, 0),
                    dem.GetMoonMEToENU(dem.Height - 1, dem.Width - 1),
                    dem.GetMoonMEToENU(dem.Height / 2, dem.Width / 2)
            };

            var maximum_elevation = Enumerable.Range(0, 5).Select(u => Utilities.PreloadArray(1440, float.NegativeInfinity)).ToArray();
            var vector_indices = Enumerable.Range(0, 5).Select(u => Utilities.PreloadArray(1440, (int)-1)).ToArray();

            for (var i = 0; i < input_sunvecs_me.Count; i++)
            {
                var sunvec = input_sunvecs_me[i];
                for (var u = 0; u < 5; u++)
                {
                    var (az_rad, el_rad) = dem.GetAzEl(sunvec, matrices[u]);
                    float az_deg = az_rad * 57.2957795f;
                    float el_deg = el_rad * 57.2957795f;
                    int az_bin = (int)(az_deg / 360f * HorizonSamples) % HorizonSamples;
                    if (el_deg > maximum_elevation[u][az_bin])
                    {
                        maximum_elevation[u][az_bin] = el_deg;
                        vector_indices[u][az_bin] = i;
                    }
                }
            }

            var unique_indices = new HashSet<int>(vector_indices.SelectMany(arr => arr).Where(idx => idx >= 0));
            var sunvecs_me = unique_indices.Select(idx => input_sunvecs_me[idx]).ToList();

            return sunvecs_me;
        }

        /// <summary>
        /// Generate a set of safe haven duration files.
        /// Safe havens are locations where a rover can park safely during communications
        /// outages.  This method considers outages to be periods when the Earth is less than
        /// 
        /// </summary>
        /// <param name="context"></param>
        /// <param name="output_path"></param>
        /// <param name="earth_threshold_deg"></param>
        /// <param name="progress"></param>
        /// <param name="isCancellationRequested"></param>
        /// <returns></returns>
        /// <exception cref="OperationCanceledException"></exception>
        /// <exception cref="Exception"></exception>
        public static async Task GenerateSafeHavenDurations(
            AnalysisContext context,
            string output_directory,
            DateTime start_time,
            DateTime stop_time,
            float time_step_hours = 2f,
            float earth_threshold_deg = 2f,
            float sun_fraction_threshold = 0.2f,
            IProgress<float>? progress = null,
            Func<bool>? isCancellationRequested = null)
        {
            void ThrowIfCancelled()
            {
                if (isCancellationRequested != null && isCancellationRequested())
                    throw new OperationCanceledException("Permanent shadow map generation was cancelled.");
            }

            Debug.Assert(context.DEM_path != null, "DEM_path must be loaded in context for PSR generation.");
            Debug.Assert(context.HorizonDirectory != null, "HorizonDirectory must be set in context for PSR generation.");

            // Initialize GDAL if not already (assuming calling app does, but safe to check)
            // In a library, we might assume the app has configured GDAL.
            var geotiff_driver = Gdal.GetDriverByName("GTiff") ?? throw new Exception("GDAL GTiff driver not found.");

            ThrowIfCancelled();

            var dem = new ElevationMap(context.DEM_path);
            var center_mat = dem.GetMoonMEToENU(dem.Height / 2, dem.Width / 2);

            start_time = ViperDate.New(start_time);
            stop_time = ViperDate.New(stop_time);

            var times = ViperDate.GetTimes(start_time, stop_time, TimeSpan.FromHours(time_step_hours));
            var sunvecs_me = times.Select(t => SpiceManager.SunPosition_meters(t)).ToList();
            var earthvecs_me = times.Select(t => SpiceManager.EarthPosition_meters(t)).ToList();

            ThrowIfCancelled();

            var earth_elevations = GetEarthElevations(earthvecs_me, dem, center_mat);
            var min_earth_indices = GetMinElevationIndices(earth_elevations);
            var earth_below_threshold_regions = GetRegionsWhereEarthIsBelowThreshold(earth_elevations, min_earth_indices, earth_threshold_deg);
            Debug.Assert(min_earth_indices.Count == earth_below_threshold_regions.Count, "Expected the number of Earth elevation minima to match the number of regions where Earth is below the threshold.");

            var output_file_count = min_earth_indices.Count;
            Console.WriteLine($"Identified {output_file_count} safe haven files to generate based on Earth elevation minima.");
            if (output_file_count == 0) throw new Exception("No safe haven files to generate. Check the specified time range and Earth elevation threshold.");

            var min_earth_times = min_earth_indices.Select(idx => times[idx]).ToList();
            var filenames = min_earth_times.Select(t => GetHavenOutputPath(output_directory, t)).ToList();
            var datasets = filenames.Select(f => LightmapPipeline.OpenDataset(geotiff_driver, f, DataType.GDT_Byte, -1, dem.Width, dem.Height, dem.Projection, dem.GeoTransform)).ToList();
            Trace.Assert(datasets.Count == earth_below_threshold_regions.Count, "Expected the number of datasets to match the number of Earth below threshold regions.");

            progress ??= new Progress<float>(percent => Console.WriteLine($"Progress: {100 * percent}%"));

            await ExecuteAsync(context, progress);

            async Task ExecuteAsync(AnalysisContext context, IProgress<float>? progress = null)
            {
                ThrowIfCancelled();
                var pipeline = new Pipeline<HorizonProcessingToken>();

                pipeline.AddStep(LightmapPipeline.ReadHorizons, maxDegreeOfParallelism: 24, boundedCapacity: 40);
                pipeline.AddStep(GenerateSafeHavens, maxDegreeOfParallelism: 24, boundedCapacity: 40);

                int processedCount = 0;
                int totalCount = 0;

                pipeline.AddTerminalStep(async token =>
                {
                    ThrowIfCancelled();
                    if (progress != null)
                    {
                        int current = Interlocked.Increment(ref processedCount);
                        progress.Report((float)current / totalCount);
                    }
                }, maxDegreeOfParallelism: 4, boundedCapacity: 40);

                Debug.Assert(context.HorizonDirectory != null, "HorizonDirectory must be set in context for PSR generation.");
                var horizon_filenames = new HorizonTileStore(context.HorizonDirectory!)
                    .EnumerateFiles(observerElevationMeters: 0f)
                    .ToList();
                totalCount = horizon_filenames.Count;

                Log.Information($"Found {horizon_filenames.Count} horizon files for lightmap generation.");

                ThrowIfCancelled();
                await pipeline.ProcessAsync(horizon_filenames.Select(f => new HorizonProcessingToken { filename = f }));

                datasets.ForEach(ds => { ds.Dispose(); });
            }

            unsafe Task<HorizonProcessingToken> GenerateSafeHavens(HorizonProcessingToken token)
            {
                ThrowIfCancelled();
                var col = token.col;
                var row = token.row;
                int width = 128;
                int height = 128;
                int numPixels = width * height;

                int test_line = 762, test_sample = 1030;

                var horizons_for_patch = token.horizons;
                Debug.Assert(horizons_for_patch != null, "Horizons must be loaded in token for safe haven generation.");
                var buffer = new byte[height * width];

                // debugging
                var shadow_durations = new byte[datasets.Count];

                for (var region_idx = 0; region_idx < datasets.Count; region_idx++)
                {
                    var (low_idx, high_idx) = earth_below_threshold_regions[region_idx];

                    for (int y = 0; y < height; y++)
                    {
                        ThrowIfCancelled();
                        for (int x = 0; x < width; x++)
                        {
                            int buffer_index = y * width + x;
                            var (line, sample) = (row + y, col + x);

                            // debugging
                            //if (line != 379 || sample != 449)
                            //    continue;

                            var horizon_base = buffer_index * HorizonSamples;  // in units of horizons
                            var mat = dem.GetMoonMEToENU(line, sample);

                            var count = 0;
                            var max_count = -1;

                            // Search the region of sun vectors where Earth is below the threshold.
                            // Find the length of the longest contiguous sequence of sun vectors in this region
                            // where the sun fraction is below the sun_fraction_threshold.  This is the longest
                            // period where a rover would need to rely on battery power during a period when controllers
                            // on Earth cannot communicate with it.
                            for (var i = low_idx; i < high_idx; i++)
                            {
                                var (az_rad, el_rad) = dem.GetAzEl(sunvecs_me[i], mat);
                                float az_deg = az_rad * 57.2957795f;
                                float el_deg = el_rad * 57.2957795f;
                                var sun_fraction = LightmapGenerator.BuilderSunFraction(horizons_for_patch, horizon_base, az_deg, el_deg);

                                if (sun_fraction < sun_fraction_threshold)
                                    count++;
                                else
                                {
                                    if (count > max_count)
                                        max_count = count;
                                    count = 0;
                                }
                            }

                            max_count = Math.Max(max_count, count); // In case the longest sequence ends at the end of the region

                            var shadow_duration_hours = max_count * time_step_hours;
                            byte val = (byte)Math.Clamp((int)shadow_duration_hours, 0, 255);

                            buffer[buffer_index] = val;

                            // debugging
                            if (line == test_line && sample == test_sample)
                            {
                                shadow_durations[region_idx] = val;
                                Console.WriteLine($"Debug safe haven duration at line={line}, sample={sample} for region {region_idx} with Earth below threshold from {times[low_idx]} to {times[high_idx]}: {shadow_duration_hours} hours (max_count={max_count})");
                            }
                        }
                    }

                    lock (datasets[region_idx])
                    {
                        datasets[region_idx].GetRasterBand(1).WriteRaster(token.col, token.row, width, height, buffer, width, height, 0, 0);
                        //Console.WriteLine($"  Wrote safe haven durations for region={region_idx} row={token.row}, col={token.col}");
                    }
                }

                // debugging
                if (true)
                {
                    var line = test_line;
                    var sample = test_sample;
                    var token_line = 128 * (line / 128);
                    var token_sample = 128 * (sample / 128);
                    if (token.row == token_line && token.col == token_sample)
                    {
                        // Write light curve for this location
                        var y = line % 128;
                        var x = sample % 128;
                        int buffer_index = y * width + x;
                        var horizon_base = buffer_index * HorizonSamples;  // in units of horizons
                        var mat = dem.GetMoonMEToENU(line, sample);
                        var light_curve_path = Path.Combine(output_directory, $"light_curve_{line}_{sample}.csv");
                        using (var writer = new StreamWriter(light_curve_path))
                        {
                            writer.WriteLine("Time,SunFraction,EarthElevation");
                            for (var i = 0; i < times.Count; i++)
                            {
                                var (az_rad, el_rad) = dem.GetAzEl(sunvecs_me[i], mat);
                                float az_deg = az_rad * 57.2957795f;
                                float el_deg = el_rad * 57.2957795f;
                                var sun_fraction = LightmapGenerator.BuilderSunFraction(horizons_for_patch, horizon_base, az_deg, el_deg);
                                var earth_elevation = earth_elevations[i];
                                writer.WriteLine($"{times[i]},{sun_fraction},{earth_elevation}");
                            }
                        }

                        using (var writer = new StreamWriter(Path.Combine(output_directory, $"debug_durations.csv")))
                        {
                            writer.WriteLine("RegionIndex,EarthBelowThresholdStart,EarthBelowThresholdEnd,SafeHavenDurationHours,DebugDurationHours");
                            for (var region_idx = 0; region_idx < datasets.Count; region_idx++)
                            {
                                var (low_idx, high_idx) = earth_below_threshold_regions[region_idx];
                                var earth_below_threshold_start = times[low_idx];
                                var earth_below_threshold_end = times[high_idx];
                                var shadow_duration_hours = shadow_durations[region_idx];
                                var hibernation_hours = (high_idx - low_idx) * time_step_hours;
                                writer.WriteLine($"{region_idx},{earth_below_threshold_start},{earth_below_threshold_end},{hibernation_hours},{shadow_duration_hours}");
                            }
                        }
                    }
                }

                return Task.FromResult(token);
            }

            string GetHavenOutputPath(string directory, DateTime time)
            {
                string timestamp = time.ToString("yyyy-MM-dd-HH-mm-ss");
                return Path.Combine(directory, $"safe_haven_{timestamp}.tif");
            }

            List<float> GetEarthElevations(List<math.Vector3d> sunvecs_me, ElevationMap dem, math.Matrix4d mat)
            {
                var result = new List<float>(sunvecs_me.Count);
                for (var i = 0; i < sunvecs_me.Count; i++)
                {
                    var vec = sunvecs_me[i];
                    var (_, el_rad) = dem.GetAzEl(vec, mat);
                    result.Add(el_rad * 57.2957795f);
                }
                return result;
            }

            List<int> GetMinElevationIndices(List<float> elevations)
            {
                var minima_indices = new List<int>();
                for (var i = 1; i < elevations.Count - 1; i++)
                    if (elevations[i - 1] > elevations[i] && elevations[i] <= elevations[i + 1])
                        minima_indices.Add(i);

                return minima_indices;
            }

            List<(int low, int high)> GetRegionsWhereEarthIsBelowThreshold(List<float> elevations, List<int> minima_indices, float threshold)
            {
                var regions = new List<(int low, int high)>();
                bool in_region = false;
                int region_start = 0;

                for (var i = 0; i < elevations.Count; i++)
                {
                    if (!in_region && elevations[i] < threshold)
                    {
                        in_region = true;
                        region_start = i;
                    }
                    else if (in_region && elevations[i] >= threshold)
                    {
                        in_region = false;
                        regions.Add((region_start, i - 1));
                    }
                }

                // Handle case where we end while still in a region
                if (in_region)
                    regions.Add((region_start, elevations.Count - 1));

                return regions;
            }
        }

        public static async Task GenerateLightingFunction(
            AnalysisContext context,
            List<string> filenames,
            List<List<DateTime>> times,
            Func<Span<float>, float> reduce_lightcurve,
            IProgress<float>? progress = null,
            Func<bool>? isCancellationRequested = null)
        {
            ThrowIfCancelled(isCancellationRequested);
            ArgumentNullException.ThrowIfNull(context.DEM_path);
            ArgumentNullException.ThrowIfNull(context.HorizonDirectory);
            ArgumentNullException.ThrowIfNull(filenames);
            ArgumentNullException.ThrowIfNull(times);
            ArgumentNullException.ThrowIfNull(reduce_lightcurve);

            var start_time = ViperDate.Now();
            progress ??= new Progress<float>(percent =>
            {
                var log_time = ViperDate.Now();
                var elapsed = (log_time - start_time).TotalSeconds;
                var estimated_total_time = elapsed / percent;
                var estimated_finish_time = start_time + TimeSpan.FromSeconds(estimated_total_time);
                Log.Information($"Progress: {100 * percent}%, ETA: {estimated_finish_time}");
            });

            if (filenames.Count < 1)
                throw new Exception($"filenames.Count ({filenames.Count}) must be at least 1");
            if (filenames.Count != times.Count)
                throw new Exception($"filenames.Count ({filenames.Count}) must equal times.Count ({times.Count})");

            // Initialize GDAL if not already (assuming calling app does, but safe to check)
            // In a library, we might assume the app has configured GDAL.
            var geotiff_driver = Gdal.GetDriverByName("GTiff") ?? throw new Exception("GDAL GTiff driver not found.");

            var dem = new ElevationMap(context.DEM_path);

            var horizon_filenames = new HorizonTileStore(context.HorizonDirectory!)
                .EnumerateFiles(observerElevationMeters: 0f)
                .ToList();
            var totalCount = horizon_filenames.Count;
            int processedCount = 0;

            Log.Information($"Found {horizon_filenames.Count} horizon files for lightmap generation.");

            // Build the sun vectors in MOON_ME frame
            var all_vectors = times
            .Select(list_of_times => list_of_times.Select(SpiceManager.SunPosition_meters).ToList())
            .ToList();

            var sun_fraction_buffer_size = times.Max(list_of_times => list_of_times.Count);

            var datasets = new List<Dataset>(filenames.Count);

            try
            {
                foreach (var filename in filenames)
                    datasets.Add(LightmapPipeline.OpenDataset(geotiff_driver, filename, DataType.GDT_Float32, -1, dem.Width, dem.Height, dem.Projection, dem.GeoTransform));

                var pipeline = new Pipeline<HorizonProcessingToken>();
                pipeline.AddStep(LightmapPipeline.ReadHorizons, maxDegreeOfParallelism: 6, boundedCapacity: 12);
                pipeline.AddStep(Internal, maxDegreeOfParallelism: 24, boundedCapacity: 24);

                pipeline.AddTerminalStep(async token =>
                {
                    ThrowIfCancelled(isCancellationRequested);
                    if (progress != null)
                    {
                        int current = Interlocked.Increment(ref processedCount);
                        progress.Report((float)current / totalCount);
                    }
                }, maxDegreeOfParallelism: 4, boundedCapacity: 40);

                await pipeline.ProcessAsync(horizon_filenames.Select(f => new HorizonProcessingToken { filename = f }));
            }
            finally
            {
                foreach (var dataset in datasets)
                    dataset.Close();
            }

            async Task<HorizonProcessingToken> Internal(HorizonProcessingToken token)
            {
                ThrowIfCancelled(isCancellationRequested);
                var col = token.col;
                var row = token.row;
                int width = 128;
                int height = 128;
                int numPixels = width * height;

                var horizons_for_patch = token.horizons ?? throw new Exception("Horizons in patch are null.");

                var sun_fraction_buffer = new float[sun_fraction_buffer_size];
                var val_buffer = new float[height * width];

                for (var dataset_id = 0; dataset_id < datasets.Count; dataset_id++)
                {
                    var (dataset, sunvecs_me) = (datasets[dataset_id], all_vectors[dataset_id]);
                    for (int y = 0; y < height; y++)
                    {
                        for (int x = 0; x < width; x++)
                        {
                            int val_buffer_index = y * width + x;
                            var (line, sample) = (row + y, col + x);
                            var horizon_base = val_buffer_index * HorizonSamples;  // in units of horizons
                            var mat = dem.GetMoonMEToENU(line, sample);

                            for (var i = 0; i < sunvecs_me.Count; i++)
                            {
                                var (az_rad, el_rad) = dem.GetAzEl(sunvecs_me[i], mat);
                                float az_deg = az_rad * 57.2957795f;
                                float el_deg = el_rad * 57.2957795f;
                                var frac = LightmapGenerator.BuilderSunFraction(horizons_for_patch, horizon_base, az_deg, el_deg);
                                sun_fraction_buffer[i] = frac;
                            }

                            var span = sun_fraction_buffer.AsSpan()[0..sunvecs_me.Count];
                            var val = reduce_lightcurve(span);
                            val_buffer[val_buffer_index] = val;
                        }
                    }

                    lock (dataset)
                    {
                        dataset.GetRasterBand(1).WriteRaster(token.col, token.row, width, height, val_buffer, width, height, 0, 0);
                    }

                    ThrowIfCancelled(isCancellationRequested);
                }

                return token;
            }
        }

        static void ThrowIfCancelled(Func<bool>? isCancellationRequested)
        {
            if (isCancellationRequested != null && isCancellationRequested())
                throw new OperationCanceledException("Permanent shadow map generation was cancelled.");
        }

        public static Func<Span<float>, float> MaxHoursOverThreshold(float threshold, float time_step_hours)
        {
            return span =>
            {
                int count = 0;
                int max_count = 0;
                for (int i = 0; i < span.Length; i++)
                {
                    if (span[i] >= threshold)
                        count++;
                    else
                    {
                        if (count > max_count)
                            max_count = count;
                        count = 0;
                    }
                }
                max_count = Math.Max(max_count, count); // In case the longest sequence ends at the end of the array
                return max_count * time_step_hours;
            };
        }

        public static Progress<float> MakeProgress(Stopwatch? stopwatch = null, int stride = 1)
        {
            int progress_count = 0;
            if (stopwatch == null)
                return new Progress<float>(completion_fraction =>
                    {
                        if (++progress_count % stride == 0)
                            Log.Information($"Progress: {100 * completion_fraction:F2}%");
                    });
            return new Progress<float>(completion_fraction =>
            {
                if (completion_fraction == 0f) return;
                var elapsed_minutes = stopwatch.Elapsed.TotalMinutes;
                var total_minutes = elapsed_minutes * (1f / completion_fraction);
                var remaining_minutes = total_minutes - elapsed_minutes;
                var eta = DateTime.Now.AddMinutes(remaining_minutes);
                if (++progress_count % stride == 0)
                    Log.Information($"Progress: {100 * completion_fraction:F2}% minutes_remaining: {remaining_minutes:F2} eta={eta}");
            });
        }

    }
}
