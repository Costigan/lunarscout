using OSGeo.GDAL;

namespace moonlib
{
    public static class TerrainProducts
    {
        public static void GenerateHillshade(string demPath, string outputPath, bool overwrite = false) =>
            GenerateDemProduct(demPath, outputPath, "hillshade", overwrite);

        public static void GenerateSlope(string demPath, string outputPath, bool overwrite = false) =>
            GenerateDemProduct(demPath, outputPath, "slope", overwrite);

        public static void GenerateAspect(string demPath, string outputPath, bool overwrite = false) =>
            GenerateDemProduct(demPath, outputPath, "aspect", overwrite);

        public static void GenerateRoughness(string demPath, string outputPath, bool overwrite = false) =>
            GenerateDemProduct(demPath, outputPath, "roughness", overwrite);

        private static void GenerateDemProduct(string demPath, string outputPath, string processing, bool overwrite)
        {
            if (string.IsNullOrWhiteSpace(demPath))
                throw new ArgumentException("DEM path must be provided.", nameof(demPath));
            if (string.IsNullOrWhiteSpace(outputPath))
                throw new ArgumentException("Output path must be provided.", nameof(outputPath));
            if (!File.Exists(demPath))
                throw new FileNotFoundException($"DEM file not found: {demPath}", demPath);
            if (File.Exists(outputPath) && !overwrite)
                throw new IOException($"Output file already exists: {outputPath}");

            MoonlibBridge.EnsureGdalInitialized();

            string? outputDirectory = Path.GetDirectoryName(outputPath);
            if (!string.IsNullOrWhiteSpace(outputDirectory))
                Directory.CreateDirectory(outputDirectory);

            string tempPath = Path.Combine(
                outputDirectory ?? Directory.GetCurrentDirectory(),
                $".{Path.GetFileName(outputPath)}.staging-{Guid.NewGuid():N}.tif");

            try
            {
                if (File.Exists(tempPath))
                    File.Delete(tempPath);

                using Dataset input = Gdal.Open(demPath, Access.GA_ReadOnly)
                    ?? throw new InvalidOperationException($"Unable to open DEM: {demPath}");
                using var options = new GDALDEMProcessingOptions(new[]
                {
                    "-of", "GTiff",
                    "-compute_edges",
                    "-co", "TILED=YES",
                    "-co", "BLOCKXSIZE=128",
                    "-co", "BLOCKYSIZE=128",
                    "-co", "COMPRESS=DEFLATE",
                    "-co", "PREDICTOR=2",
                    "-co", "BIGTIFF=IF_SAFER",
                });
                using Dataset output = Gdal.wrapper_GDALDEMProcessing(
                    tempPath,
                    input,
                    processing,
                    string.Empty,
                    options,
                    null,
                    null)
                    ?? throw new InvalidOperationException($"GDAL DEMProcessing failed for {processing}.");
                output.FlushCache();

                File.Move(tempPath, outputPath, overwrite: true);
            }
            catch
            {
                TryDelete(tempPath);
                throw;
            }
        }

        private static void TryDelete(string path)
        {
            try
            {
                if (File.Exists(path))
                    File.Delete(path);
            }
            catch
            {
                // Best-effort cleanup after failed native product generation.
            }
        }
    }
}
