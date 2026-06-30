using moonlib.horizon;
using moonlib.mapops;
using moonlib.pipeline.streaming;
using moonlib.spice;
using OSGeo.GDAL;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace moonlib
{
    public sealed record HorizonProgress(
        int ProcessedPatches,
        int TotalPatches,
        double Percent,
        string Stage,
        string Message,
        string? FileName
    );

    public delegate void HorizonProgressCallback(HorizonProgress progress);
    public delegate bool HorizonCancellationCallback();

    public sealed record PsrProgress(
        double Percent,
        string Stage,
        string Message
    );

    public delegate void PsrProgressCallback(PsrProgress progress);
    public delegate bool PsrCancellationCallback();

    public class MoonlibBridge
    {
        private readonly LightmapArrayStreamingBridge _lightmapStreaming = new();
        private static readonly object GdalInitLock = new object();
        private static bool _gdalInitialized = false;
        private static bool _gdalResolverConfigured = false;

        public static void EnsureGdalInitialized()
        {
            if (_gdalInitialized)
                return;

            lock (GdalInitLock)
            {
                if (_gdalInitialized)
                    return;

                var assemblyDir = Path.GetDirectoryName(typeof(MoonlibBridge).Assembly.Location) ?? string.Empty;
                ConfigureGdalNativeResolver(assemblyDir);

                var gdalData = ResolveGdalDataDir(assemblyDir);
                var projData = ResolveProjDataDir(assemblyDir);

                if (string.IsNullOrWhiteSpace(projData))
                {
                    throw new InvalidOperationException($"Could not locate proj.db for moonlib GDAL bootstrap under {assemblyDir}");
                }

                if (!string.IsNullOrWhiteSpace(gdalData))
                {
                    Environment.SetEnvironmentVariable("GDAL_DATA", gdalData, EnvironmentVariableTarget.Process);
                    Gdal.SetConfigOption("GDAL_DATA", gdalData);
                }

                if (!string.IsNullOrWhiteSpace(projData))
                {
                    Environment.SetEnvironmentVariable("PROJ_LIB", projData, EnvironmentVariableTarget.Process);
                    Environment.SetEnvironmentVariable("PROJ_DATA", projData, EnvironmentVariableTarget.Process);
                    Gdal.SetConfigOption("PROJ_LIB", projData);
                    Gdal.SetConfigOption("PROJ_DATA", projData);
                }

                Gdal.AllRegister();
                _gdalInitialized = true;
            }
        }

        public static string GdalSmokeTest()
        {
            EnsureGdalInitialized();
            return Gdal.GetConfigOption("GDAL_DATA", string.Empty) ?? string.Empty;
        }

        public string StartLightmapArrayStreaming(LightmapArrayStreamRequest request)
        {
            EnsureGdalInitialized();
            return _lightmapStreaming.StartLightmapArrayStreaming(request);
        }

        public bool RegisterOutputBuffer(string jobId, int bufferId, long ptr, int byteLength)
            => _lightmapStreaming.RegisterOutputBuffer(jobId, bufferId, ptr, byteLength);

        public TileEnvelope? TryGetNextTile(string jobId, int timeoutMs)
            => _lightmapStreaming.TryGetNextTile(jobId, timeoutMs);

        public string StartLightmapArrayStreamingV2(LightmapArrayStreamRequestV2 request)
        {
            EnsureGdalInitialized();
            return _lightmapStreaming.StartLightmapArrayStreamingV2(request);
        }

        public bool RegisterOutputBufferV2(string jobId, int bufferId, long ptr, int byteLength)
            => _lightmapStreaming.RegisterOutputBufferV2(jobId, bufferId, ptr, byteLength);

        public TileEnvelopeV2? TryGetNextTileV2(string jobId, int timeoutMs)
            => _lightmapStreaming.TryGetNextTileV2(jobId, timeoutMs);

        public NativeReduceRasterResult? GetNativeReduceRasterResult(string jobId)
            => _lightmapStreaming.GetNativeReduceRasterResult(jobId);

        public bool ReleaseBuffer(string jobId, int bufferId)
            => _lightmapStreaming.ReleaseBuffer(jobId, bufferId);

        public LightmapArrayStreamStatus GetJobStatus(string jobId)
            => _lightmapStreaming.GetJobStatus(jobId);

        public bool CancelJob(string jobId)
            => _lightmapStreaming.CancelJob(jobId);

        public bool DisposeJob(string jobId)
            => _lightmapStreaming.DisposeJob(jobId);

        private static void ConfigureGdalNativeResolver(string assemblyDir)
        {
            if (_gdalResolverConfigured)
                return;

            var searchDirs = ResolveGdalNativeSearchDirs(assemblyDir);
            var gdalAssembly = typeof(Gdal).Assembly;
            try
            {
                NativeLibrary.SetDllImportResolver(gdalAssembly, (libraryName, assembly, searchPath) =>
                {
                    foreach (var candidate in ResolveGdalNativeLibraryCandidates(libraryName, searchDirs))
                    {
                        if (NativeLibrary.TryLoad(candidate, out var handle))
                            return handle;
                    }
                    return IntPtr.Zero;
                });
            }
            catch (InvalidOperationException)
            {
                // Another bootstrap path already installed a resolver for this assembly.
            }
            _gdalResolverConfigured = true;
        }

        private static List<string> ResolveGdalNativeSearchDirs(string assemblyDir)
        {
            var candidates = new[]
            {
                assemblyDir,
                Path.Combine(assemblyDir, "linux-x64"),
                Path.Combine(assemblyDir, "runtimes", "linux-x64", "native"),
                Path.Combine(Directory.GetParent(assemblyDir)?.FullName ?? assemblyDir, "linux-x64"),
            };
            return candidates
                .Where(path => !string.IsNullOrWhiteSpace(path) && Directory.Exists(path))
                .Distinct()
                .ToList();
        }

        private static IEnumerable<string> ResolveGdalNativeLibraryCandidates(string libraryName, IReadOnlyList<string> searchDirs)
        {
            var normalized = NormalizeNativeLibraryName(libraryName);
            string[] filenames = normalized switch
            {
                "gdal_wrap" => new[] { "libgdal_wrap.so" },
                "gdalconst_wrap" => new[] { "libgdalconst_wrap.so" },
                "gdal" => new[] { "libgdal.so.37", "libgdal.so" },
                _ => Array.Empty<string>(),
            };

            foreach (var dir in searchDirs)
            {
                foreach (var filename in filenames)
                    yield return Path.Combine(dir, filename);
            }
        }

        private static string NormalizeNativeLibraryName(string libraryName)
        {
            var name = (libraryName ?? string.Empty).Trim().ToLowerInvariant();
            if (name.StartsWith("lib"))
                name = name.Substring(3);
            if (name.EndsWith(".so"))
                name = name.Substring(0, name.Length - 3);
            return name;
        }

        private static string? ResolveGdalDataDir(string assemblyDir)
        {
            var parentDir = Directory.GetParent(assemblyDir)?.FullName ?? assemblyDir;
            var candidates = new[]
            {
                Path.Combine(assemblyDir, "gdal", "data"),
                Path.Combine(assemblyDir, "data"),
                Path.Combine(assemblyDir, "gdal-data"),
                Path.Combine(assemblyDir, "runtimes", "linux-x64", "native", "data"),
                Path.Combine(assemblyDir, "runtimes", "linux-x64", "native", "gdal-data"),
                Path.Combine(parentDir, "gdal", "data"),
                Path.Combine(parentDir, "data"),
                Path.Combine(parentDir, "gdal-data"),
            };
            return candidates.FirstOrDefault(Directory.Exists);
        }

        private static string? ResolveProjDataDir(string assemblyDir)
        {
            var parentDir = Directory.GetParent(assemblyDir)?.FullName ?? assemblyDir;
            var candidates = new[]
            {
                assemblyDir,
                Path.Combine(assemblyDir, "gdal", "share"),
                Path.Combine(assemblyDir, "share"),
                Path.Combine(assemblyDir, "proj-lib"),
                Path.Combine(assemblyDir, "runtimes", "linux-x64", "native", "share"),
                Path.Combine(assemblyDir, "runtimes", "linux-x64", "native", "proj-lib"),
                parentDir,
                Path.Combine(parentDir, "gdal", "share"),
                Path.Combine(parentDir, "share"),
                Path.Combine(parentDir, "proj-lib"),
            };
            return candidates.FirstOrDefault(path =>
                Directory.Exists(path) &&
                (File.Exists(Path.Combine(path, "proj.db")) || Directory.Exists(Path.Combine(path, "proj")))
            );
        }

        public void GenerateHorizons(string scenarioRootDir, string demPath, List<string> surroundingDemPaths, string horizonsDir, float observerElevationMeters, bool overwriteHorizons, bool compressHorizons)
        {
            GenerateHorizons(
                scenarioRootDir,
                demPath,
                surroundingDemPaths,
                horizonsDir,
                observerElevationMeters,
                overwriteHorizons,
                compressHorizons,
                null,
                null);
        }

        public void GenerateHorizons(
            string scenarioRootDir,
            string demPath,
            List<string> surroundingDemPaths,
            string horizonsDir,
            float observerElevationMeters,
            bool overwriteHorizons,
            bool compressHorizons,
            HorizonProgressCallback? progress,
            HorizonCancellationCallback? isCancellationRequested)
        {
            // Placeholder for the actual horizons generation logic.
            // In a real implementation, this method would read the input data,
            // perform the necessary calculations to generate the horizons,
            // and then write the output to the specified path.
            void Report(int processed, int total, double percent, string stage, string message, string? fileName = null)
            {
                progress?.Invoke(new HorizonProgress(processed, total, percent, stage, message, fileName));
            }

            void ThrowIfCancelled()
            {
                if (isCancellationRequested?.Invoke() == true)
                    throw new OperationCanceledException("Horizon generation was canceled.");
            }

            ThrowIfCancelled();
            Report(0, 0, 1.0, "validate_inputs", "Validating horizon generation inputs.");
            Console.WriteLine($"Generating horizons from {demPath} and saving to {horizonsDir}...");
            EnsureGdalInitialized();

            if (!File.Exists(demPath))
            {
                Console.WriteLine($"Error: DEM file not found at {demPath}");
                return;
            }

            ThrowIfCancelled();
            Report(0, 0, 5.0, "load_dem", "Loading primary DEM.");
            var primaryDem = new ElevationMap(demPath);

            // Generate full patch list from primary DEM
            ThrowIfCancelled();
            Report(0, 0, 8.0, "prepare_patches", "Preparing horizon patch list.");
            var allPatches = QuadTreeHorizonGenerator.GeneratePatchList(primaryDem);

            if (!Directory.Exists(horizonsDir))
                Directory.CreateDirectory(horizonsDir);

            if (!overwriteHorizons)
            {
                var horizonStore = new HorizonTileStore(horizonsDir);
                if (horizonStore.EnumerateTiles().Any())
                {
                    var needed_patches = new List<QuadTreeHorizonGenerator.PatchDescriptor>();
                    for (var i = 0; i < allPatches.Count; i++)
                    {
                        var patch = allPatches[i];
                        // Treat either uncompressed or compressed patch file as existing.
                        if (horizonStore.FindExistingPath(patch.TileY, patch.TileX, observerElevationMeters) == null)
                            needed_patches.Add(patch);
                    }
                    allPatches = needed_patches;
                }
            }

            if (allPatches.Count < 1)
            {
                Console.WriteLine(allPatches.Count + " horizon patches need to be generated, skipping generation.");
                Report(0, 0, 100.0, "complete", "No horizon patches need to be generated.");
                return;
            }

#if DEBUG
            Console.WriteLine($"DEBUG: Generating horizons for {allPatches.Count} patches...");
#else
             Console.WriteLine($"RELEASE: Generating horizons for {allPatches.Count} patches...");
#endif

            ThrowIfCancelled();
            Report(0, allPatches.Count, 10.0, "load_surrounding_dems", "Loading surrounding DEMs.");
            var allDEMs = new List<ElevationMap> { primaryDem }.Concat(surroundingDemPaths.Select(path => new ElevationMap(path)).ToList()).ToList();

            // Generate horizons for selected patches
            float observerElevation = observerElevationMeters;
            using (var generator = new QuadTreeHorizonGenerator(
                disableHierarchy: false))
            {
                Task.Run(() => generator.GenerateHorizonsForPatches(
                    horizonsDir,
                    allDEMs,
                    allPatches,
                    observerElevation,
                    compressHorizons,
                    progress,
                    isCancellationRequested)).GetAwaiter().GetResult();
            }

            Console.WriteLine($"Horizon files written to: {Path.GetFullPath(horizonsDir)}");

            Report(allPatches.Count, allPatches.Count, 100.0, "complete", "Horizons generation complete.");
            Console.WriteLine("Horizons generation complete.");
        }

        public void GeneratePermanentShadowMap(
            string scenarioRootDir,
            string demPath,
            List<string> surroundingDemPaths,
            string horizonsDir,
            string outputPath)
        {
            GeneratePermanentShadowMap(
                scenarioRootDir,
                demPath,
                surroundingDemPaths,
                horizonsDir,
                outputPath,
                null,
                null);
        }

        public void GeneratePermanentShadowMap(
            string scenarioRootDir,
            string demPath,
            List<string> surroundingDemPaths,
            string horizonsDir,
            string outputPath,
            PsrProgressCallback? progress,
            PsrCancellationCallback? isCancellationRequested)
        {
            EnsureGdalInitialized();
            var spice = SpiceManager.Singleton;
            _ = spice;

            void ThrowIfCancelled()
            {
                if (isCancellationRequested != null && isCancellationRequested())
                    throw new OperationCanceledException("Permanent shadow map generation was cancelled.");
            }

            void Report(double percent, string stage, string message)
            {
                progress?.Invoke(new PsrProgress(percent, stage, message));
            }

            ThrowIfCancelled();
            Report(1.0, "validate_inputs", "Validating PSR inputs.");
            if (!File.Exists(demPath))
                throw new FileNotFoundException($"DEM file not found at {demPath}", demPath);
            if (!Directory.Exists(horizonsDir))
                throw new DirectoryNotFoundException($"Horizon directory not found at {horizonsDir}");

            var context = new AnalysisContext
            {
                Directory = scenarioRootDir,
                DEM_path = demPath,
                SurroundingDEM_paths = surroundingDemPaths ?? new List<string>(),
                HorizonDirectory = horizonsDir,
            };

            var outputDir = Path.GetDirectoryName(outputPath);
            if (!string.IsNullOrWhiteSpace(outputDir))
                Directory.CreateDirectory(outputDir);

            ThrowIfCancelled();
            Report(10.0, "native_execution", "Generating permanent shadow raster.");
            IProgress<float> nativeProgress = new SynchronousProgress<float>(percent =>
            {
                Report(
                    10.0 + Math.Clamp((double)percent, 0.0, 1.0) * 85.0,
                    "native_execution",
                    "Generating permanent shadow raster.");
            });
            MapOperations.GeneratePermanentShadowMap(
                context,
                outputPath,
                overwrite_existing: true,
                progress: nativeProgress,
                isCancellationRequested: () => isCancellationRequested != null && isCancellationRequested()).GetAwaiter().GetResult();
            ThrowIfCancelled();
            Report(100.0, "complete", "Permanent shadow raster generation complete.");
        }

        public int CompressHorizonsDirectory(
            string horizonsDir,
            bool deleteUncompressed,
            bool useParallel)
        {
            if (string.IsNullOrWhiteSpace(horizonsDir))
                throw new ArgumentException("Horizon directory path is required.", nameof(horizonsDir));
            if (!Directory.Exists(horizonsDir))
                throw new DirectoryNotFoundException($"Horizon directory not found at {horizonsDir}");

            return HorizonFile.CompressDirectory(horizonsDir, deleteUncompressed, useParallel);
        }

        
    }
}
