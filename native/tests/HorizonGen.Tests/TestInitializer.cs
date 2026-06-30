using OSGeo.GDAL;
using Serilog;

#nullable disable

namespace moonlib.tests
{
    [TestClass]
    public static class TestInitializer
    {
        [AssemblyInitialize]
        public static void Initialize(TestContext context)
        {
            // Configure Serilog for tests
            var logFilePath = Path.Combine(context.TestRunDirectory, "test_log.txt");
            Log.Logger = new LoggerConfiguration()
                .MinimumLevel.Debug()
                .WriteTo.File(logFilePath, rollingInterval: RollingInterval.Day, outputTemplate: "{Timestamp:HH:mm:ss.fff} [{Level:u3}] {Message:lj}{NewLine}{Exception}")
                .CreateLogger();

            Log.Information("Test Run starting up. Logging to {LogFile}", logFilePath);

            // Configure GDAL environment variables for Linux runtime layouts.
            var gdalData = ResolveGdalDataDir(AppContext.BaseDirectory);
            var projData = ResolveProjDataDir(AppContext.BaseDirectory);
            if (!string.IsNullOrWhiteSpace(gdalData))
            {
                Environment.SetEnvironmentVariable("GDAL_DATA", gdalData);
                Gdal.SetConfigOption("GDAL_DATA", gdalData);
            }
            if (!string.IsNullOrWhiteSpace(projData))
            {
                Environment.SetEnvironmentVariable("PROJ_LIB", projData);
                Environment.SetEnvironmentVariable("PROJ_DATA", projData);
                Gdal.SetConfigOption("PROJ_LIB", projData);
                Gdal.SetConfigOption("PROJ_DATA", projData);
            }

            Gdal.AllRegister();
            Assert.IsTrue(
                Gdal.GetDriverCount() > 0,
                $"GDAL payload missing or not loadable under {AppContext.BaseDirectory}"
            );
        }

        private static string ResolveGdalDataDir(string baseDir)
        {
            var candidates = new[]
            {
                Path.Combine(baseDir, "gdal", "data"),
                Path.Combine(baseDir, "data"),
                Path.Combine(baseDir, "gdal-data"),
                Path.Combine(baseDir, "runtimes", "linux-x64", "native", "data"),
                Path.Combine(baseDir, "runtimes", "linux-x64", "native", "gdal-data"),
            };
            return candidates.FirstOrDefault(Directory.Exists);
        }

        private static string ResolveProjDataDir(string baseDir)
        {
            var candidates = new[]
            {
                Path.Combine(baseDir, "gdal", "share"),
                Path.Combine(baseDir, "share"),
                Path.Combine(baseDir, "proj-lib"),
                Path.Combine(baseDir, "runtimes", "linux-x64", "native", "share"),
                Path.Combine(baseDir, "runtimes", "linux-x64", "native", "proj-lib"),
            };
            return candidates.FirstOrDefault(path =>
                Directory.Exists(path) &&
                (File.Exists(Path.Combine(path, "proj.db")) || Directory.Exists(Path.Combine(path, "proj")))
            );
        }
    }
}
