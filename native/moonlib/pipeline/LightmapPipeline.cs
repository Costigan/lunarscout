using moonlib.horizon;
using moonlib.math;
using moonlib.spice;
using OSGeo.GDAL;
using Serilog;
using System.Diagnostics;

namespace moonlib.pipeline
{
    public class LightmapPipeline
    {
        public const int Width = 128;
        public const int Height = 128;
        public const int HorizonSamples = 1440;
        public const double HapkeSingleScatteringAlbedoHighland = 0.23;
        public const double HapkeAsymmetryHighland = -0.34;
        public const double HapkeOppositionAmplitudeHighland = 1.0;
        public const double HapkeOppositionWidthHighland = 0.06;
        public const double SolarIrradianceScale = 1.0;

        Driver? geotiff_driver;
        string? output_directory;
        string? camera_output_directory;
        public List<string>? horizon_filenames;
        public List<DateTime>? times;
        public List<Vector3d>? sunvecs_me;
        public Dictionary<DateTime, Dataset>? sun_images;
        public Dictionary<DateTime, Dataset>? camera_images;
        public ElevationMap? dem;

        public async Task ExecuteAsync(
            List<DateTime> timestamps,
            string? sunDir,
            string? cameraOutDir,
            ElevationMap elevationMap,
            string horizonDir,
            IProgress<float>? progress = null,
            Func<DateTime, Vector3d>? sunVectorProvider = null)
        {
            output_directory = sunDir;
            camera_output_directory = cameraOutDir;

            if (output_directory != null && !Directory.Exists(output_directory))
            {
                Directory.CreateDirectory(output_directory);
            }
            if (camera_output_directory != null && !Directory.Exists(camera_output_directory))
            {
                Directory.CreateDirectory(camera_output_directory);
            }

            // Initialize GDAL if not already (assuming calling app does, but safe to check)
            // In a library, we might assume the app has configured GDAL.

            geotiff_driver = Gdal.GetDriverByName("GTiff") ?? throw new Exception("GDAL GTiff driver not found.");
            dem = elevationMap;
            sun_images = new Dictionary<DateTime, Dataset>();
            camera_images = new Dictionary<DateTime, Dataset>();

            times = timestamps;
            sunvecs_me = sunVectorProvider is null
                ? times.Select(t => SpiceManager.SunPosition(t) * 1000.0).ToList()
                : times.Select(sunVectorProvider).ToList();

            var pipeline = new Pipeline<HorizonProcessingToken>();

            pipeline.AddStep(ReadHorizons, maxDegreeOfParallelism: 12);
            pipeline.AddStep(GenerateMatrices, maxDegreeOfParallelism: 40);
            if (sunDir != null)
                pipeline.AddStep(GenerateShadows, maxDegreeOfParallelism: 40);
            if (cameraOutDir != null)
                pipeline.AddStep(GenerateSimulatedCameraImages, maxDegreeOfParallelism: 40);
            
            int processedCount = 0;
            int totalCount = 0;

            if (progress == null)
                 progress = new Progress<float>(percent => Console.WriteLine($"Progress: {100 * percent}%"));

            pipeline.AddTerminalStep(async token => 
            {
                await WriteData(token);
                if (progress != null)
                {
                    int current = Interlocked.Increment(ref processedCount);
                    progress.Report((float)current / totalCount);
                }
            }, maxDegreeOfParallelism: 20);

            horizon_filenames = new HorizonTileStore(horizonDir)
                .EnumerateFiles(observerElevationMeters: 0f)
                .ToList();
            totalCount = horizon_filenames.Count;

            Log.Information($"Found {horizon_filenames.Count} horizon files for lightmap generation.");

            await pipeline.ProcessAsync(horizon_filenames.Select(f => new HorizonProcessingToken { filename = f, dem = dem, sunvecs_me = sunvecs_me }));

            foreach (var dataset in sun_images.Values)
            {
                dataset.FlushCache();
                dataset.Dispose();
            }
            foreach (var dataset in camera_images.Values)
            {
                dataset.FlushCache();
                dataset.Dispose();
            }
        }

        public static Task<HorizonProcessingToken> ReadHorizons(HorizonProcessingToken token)
        {
            (token.col, token.row, token.observer_elevation) = QuadTreeHorizonGenerator.ParseHorizonFilename(token.filename);
            try
            {
                token.horizons = HorizonFile.ReadHorizonFile(token.filename);
            }
            catch (Exception ex)
            {
                Log.Error(ex, $"Failed to read horizon file: {token.filename}");
                token.horizons = null;
            }
            return Task.FromResult(token);
        }

        public static Task<HorizonProcessingToken> GenerateMatrices(HorizonProcessingToken token)
        {
            Debug.Assert(token.dem != null);
            token.matrices = new Matrix4d[Width, Height];

            for (int y = 0; y < Height; y++)
            {
                var line = token.row + y;
                for (int x = 0; x < Width; x++)
                {
                    var sample = token.col + x;
                    token.matrices[y, x] = token.dem.GetMoonMEToENU(line, sample);
                }
            }

            return Task.FromResult(token);
        }

        public unsafe Task<HorizonProcessingToken> GenerateShadows(HorizonProcessingToken token)
        {
            if (token.horizons == null)
            {
                Log.Warning($"Skipping shadow generation for {token.filename} due to missing horizon data.");
                return Task.FromResult(token);
            }
            Debug.Assert(times != null && token.sunvecs_me != null && token.matrices != null && token.horizons != null && dem != null);
            int width = Width;
            int height = Height;
            int numPixels = width * height;
            int numTimes = times.Count;
            var sunvecs_me = token.sunvecs_me;

            var resultBuffers = new byte[numTimes][];
            for (int t = 0; t < numTimes; t++)
                resultBuffers[t] = new byte[numPixels];

            var horizons = token.horizons;
            var matrices = token.matrices;
            token.resultBuffers = resultBuffers;

            fixed (float* horizonsPtr = horizons)
            {
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int pixelIdx = y * width + x;
                        float* pixelHorizons = horizonsPtr + pixelIdx * HorizonSamples;
                        var mat = matrices[y, x];

                        for (int t = 0; t < numTimes; t++)
                        {
                            var (az_rad, el_rad) = dem.GetAzEl(sunvecs_me[t], mat);

                            float az_deg = az_rad * 57.2957795f;
                            float el_deg = el_rad * 57.2957795f;

                            float frac = LightmapGenerator.BuilderSunFraction(horizons, pixelIdx * HorizonSamples, az_deg, el_deg);

                            resultBuffers[t][pixelIdx] = (byte)(255f * frac);
                        }
                    }
                }
            }

            for (int t = 0; t < numTimes; t++)
            {
                var dataset = GetDatasetForTime(times[t]);
                lock (dataset)
                {
                    dataset.GetRasterBand(1).WriteRaster(token.col, token.row, width, height, resultBuffers[t], width, height, 0, 0);
                }
            }

            return Task.FromResult(token);
        }

        public Task<HorizonProcessingToken> GenerateSimulatedCameraImages(HorizonProcessingToken token)
        {
            if (token.horizons == null)
            {
                Log.Warning($"Skipping canera image generation for {token.filename} due to missing horizon data.");
                return Task.FromResult(token);
            }
            Debug.Assert(times != null && token.sunvecs_me != null && token.matrices != null && token.resultBuffers != null && dem != null);
            int width = Width;
            int height = Height;
            int numPixels = width * height;
            int numTimes = times.Count;
            var sunvecs = token.sunvecs_me;
            var matrices = token.matrices;
            var shadowBuffers = token.resultBuffers;

            var resultBuffers = new float[numTimes][];
            for (int t = 0; t < numTimes; t++)
                resultBuffers[t] = new float[numPixels];

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int pixelIdx = y * width + x;
                    int line = token.row + y;
                    int sample = token.col + x;
                    var viewDirEnu = GetCameraViewDirectionEnu(y, x);
                    var normalEnu = dem.GetSurfaceNormalEnu(line, sample, matrices[y, x]);

                    for (int t = 0; t < numTimes; t++)
                    {
                        double sunVisibility = shadowBuffers[t][pixelIdx] / 255.0;
                        if (sunVisibility <= 0.0)
                        {
                            resultBuffers[t][pixelIdx] = 0.0f;
                            continue;
                        }

                        var (azRad, elRad) = dem.GetAzEl(sunvecs[t], matrices[y, x]);
                        var sunDirEnu = AzElToEnu(azRad, elRad);

                        double mu0 = Math.Max(0.0, Vector3d.Dot(normalEnu, sunDirEnu));
                        double mu = Math.Max(0.0, Vector3d.Dot(normalEnu, viewDirEnu));
                        if (mu0 <= 0.0 || mu <= 0.0)
                        {
                            resultBuffers[t][pixelIdx] = 0.0f;
                            continue;
                        }

                        double cosPhase = Math.Clamp(Vector3d.Dot(sunDirEnu, viewDirEnu), -1.0, 1.0);
                        double reflectance = HapkeLunarHighlandReflectance(mu0, mu, cosPhase);
                        double radiance = Math.Max(0.0, SolarIrradianceScale * sunVisibility * reflectance);
                        resultBuffers[t][pixelIdx] = (float)radiance;
                    }
                }
            }

            for (int t = 0; t < numTimes; t++)
            {
                    var dataset = GetCameraDatasetForTime(times[t]);
                    lock (dataset)
                    {
                        dataset.GetRasterBand(1).WriteRaster(token.col, token.row, width, height, resultBuffers[t], width, height, 0, 0);
                    }
            }

            return Task.FromResult(token);
        }

        public Task<HorizonProcessingToken> WriteData(HorizonProcessingToken token)
        {
            return Task.FromResult(token);
        }

        public Dataset GetDatasetForTime(DateTime time)
        {
            Debug.Assert(sun_images != null && output_directory != null && geotiff_driver != null && dem != null);
            lock (sun_images)
            {
                if (!sun_images.ContainsKey(time))
                {
                    // Use a safe filename format
                    var filename = Path.Combine(output_directory, $"sun_image_{time:yyyy-MM-ddTHH-mm-ss}.tif");
                    var ds = OpenDataset(geotiff_driver, filename, DataType.GDT_Byte, -1, dem.Width, dem.Height, dem.Projection, dem.GeoTransform);
                    sun_images[time] = ds;
                }
                return sun_images[time];
            }
        }

        public Dataset GetCameraDatasetForTime(DateTime time)
        {
            Debug.Assert(camera_images != null && camera_output_directory != null && geotiff_driver != null && dem != null);
            lock (camera_images)
            {
                if (!camera_images.ContainsKey(time))
                {
                    var filename = Path.Combine(camera_output_directory, $"camera_image_{time:yyyy-MM-ddTHH-mm-ss}.tif");
                    var ds = OpenDataset(geotiff_driver, filename, DataType.GDT_Float32, -9999.0, dem.Width, dem.Height, dem.Projection, dem.GeoTransform);
                    camera_images[time] = ds;
                }
                return camera_images[time];
            }
        }

        private static Vector3d GetCameraViewDirectionEnu(int _y, int _x)
        {
            return new Vector3d(0.0, 0.0, 1.0);
        }

        private static Vector3d AzElToEnu(float azRad, float elRad)
        {
            double cosEl = Math.Cos(elRad);
            return new Vector3d(
                cosEl * Math.Sin(azRad),
                cosEl * Math.Cos(azRad),
                Math.Sin(elRad));
        }

        private static double HapkeLunarHighlandReflectance(double mu0, double mu, double cosPhase)
        {
            // Approximate lunar highland parameters for simplified Hapke quicklook rendering.
            const double w = HapkeSingleScatteringAlbedoHighland;
            const double hg = HapkeAsymmetryHighland;
            const double B0 = HapkeOppositionAmplitudeHighland;
            const double h = HapkeOppositionWidthHighland;

            double p = HenyeyGreenstein1P(cosPhase, hg);
            double tanHalf = Math.Sqrt(Math.Max(0.0, (1.0 - cosPhase) / (1.0 + cosPhase + 1e-12)));
            double opposition = 1.0 + B0 / (1.0 + tanHalf / h);
            double hMu0 = HapkeH(mu0, w);
            double hMu = HapkeH(mu, w);

            double r = (w / (4.0 * Math.PI)) * (mu0 / (mu0 + mu)) * (opposition * p + (hMu0 * hMu - 1.0));
            return double.IsFinite(r) ? Math.Max(0.0, r) : 0.0;
        }

        private static double HapkeH(double mu, double w)
        {
            double gamma = Math.Sqrt(Math.Max(0.0, 1.0 - w));
            return (1.0 + 2.0 * mu) / (1.0 + 2.0 * mu * gamma);
        }

        private static double HenyeyGreenstein1P(double cosPhase, double g)
        {
            double denom = Math.Pow(1.0 + 2.0 * g * cosPhase + g * g, 1.5);
            if (denom <= 0.0)
                return 1.0;
            return (1.0 - g * g) / denom;
        }

        public static Dataset OpenDataset(Driver driver, string filename, DataType data_type, double? no_data_value, int width, int height, string projection, double[] geoTransform)
        {
            if (File.Exists(filename))
            {
                Dataset existing = null;
                try
                {
                    existing = Gdal.Open(filename, Access.GA_Update);
                }
                catch
                {
                    existing = null;
                }

                if (existing != null)
                {
                    bool sizeOk = existing.RasterXSize == width && existing.RasterYSize == height;
                    bool hasBand = existing.RasterCount >= 1;
                    bool typeOk = hasBand && existing.GetRasterBand(1).DataType == data_type;
                    if (sizeOk && typeOk)
                    {
                        if (no_data_value.HasValue)
                            existing.GetRasterBand(1).SetNoDataValue(no_data_value.Value);
                        else
                            existing.GetRasterBand(1).DeleteNoDataValue();
                        return existing;
                    }
                    existing.Dispose();
                }
                try
                {
                     File.Delete(filename);
                } 
                catch { }
            }

            var ds = driver.Create(
                filename,
                width,
                height,
                1,
                data_type,
                new string[] { "TILED=YES", "BLOCKXSIZE=128", "BLOCKYSIZE=128", "COMPRESS=LZW", "BIGTIFF=YES", "SPARSE_OK=TRUE" }
            );
            
            if (ds == null)
                throw new Exception($"Failed to create dataset for {filename}");

            if (no_data_value.HasValue)
                ds.GetRasterBand(1).SetNoDataValue(no_data_value.Value);
            ds.SetProjection(projection);
            ds.SetGeoTransform(geoTransform);

            return ds;
        }
    }

    public class HorizonProcessingToken
    {
        public ElevationMap? dem;
        public required string filename { get; set; }
        public int row { get; set; }
        public int col { get; set; }
        public float observer_elevation { get; set; }
        public float[]? horizons { get; set; }
        public Matrix4d[,]? matrices { get; set; }
        public List<math.Vector3d>? sunvecs_me { get; set; }
        public byte[][]? resultBuffers { get; set; }
    }
}
