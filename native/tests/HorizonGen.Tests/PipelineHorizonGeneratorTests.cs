using Microsoft.VisualStudio.TestTools.UnitTesting;
using moonlib.horizon;

namespace moonlib.tests
{
    [TestClass]
    public class PipelineHorizonGeneratorTests
    {
        [TestMethod]
        [TestCategory("Integration")]
        [TestCategory("Pipeline")]
        public async Task GenerateHorizonsForAllPatches_ValidatesDimensions()
        {
            // Create a DEM with invalid dimensions (not a multiple of 128)
            const int INVALID_SIZE = 1000; // Not divisible by 128
            float[,] elevation = new float[INVALID_SIZE, INVALID_SIZE];
            
            const string StereographicProj4 = @"+proj=stere +lat_0=90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +no_defs";
            double[] geoTransform = new double[]
            {
                -(INVALID_SIZE / 2.0) * 100.0,
                100.0,
                0,
                (INVALID_SIZE / 2.0) * 100.0,
                0,
                -100.0
            };
            
            var invalidDem = new ElevationMap(elevation, StereographicProj4, geoTransform);
            var dems = new List<ElevationMap> { invalidDem };
            
            var outputDir = Path.Combine(Path.GetTempPath(), "PipelineTest_Invalid");
            
            using var generator = new QuadTreeHorizonGenerator();
            
            // Should throw ArgumentException for invalid dimensions
            var exception = await Assert.ThrowsExceptionAsync<ArgumentException>(async () =>
            {
                await generator.GenerateHorizonsForAllPatches(outputDir, dems, 2.0f);
            });
            
            Assert.IsTrue(exception.Message.Contains("multiple of 128"), 
                "Exception message should mention multiple of 128 requirement");
            
            Console.WriteLine($"Correctly rejected invalid DEM dimensions: {exception.Message}");
        }

        [TestMethod]
        [TestCategory("Fast")]
        [TestCategory("Pipeline")]
        public async Task GenerateHorizonsForAllPatches_ValidDimensions_SmallDEM()
        {
            // Create a small DEM with valid dimensions (multiple of 128)
            const int DEM_SIZE = 256; // 2x2 patches (256 / 128 = 2)
            float[,] elevation = new float[DEM_SIZE, DEM_SIZE];
            
            // Fill with some test data
            for (int r = 0; r < DEM_SIZE; r++)
            {
                for (int c = 0; c < DEM_SIZE; c++)
                {
                    elevation[r, c] = 0.0f;
                }
            }
            
            const string StereographicProj4 = @"+proj=stere +lat_0=90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +no_defs";
            double[] geoTransform = new double[]
            {
                -(DEM_SIZE / 2.0) * 100.0,
                100.0,
                0,
                (DEM_SIZE / 2.0) * 100.0,
                0,
                -100.0
            };
            
            var dem = new ElevationMap(elevation, StereographicProj4, geoTransform);
            var dems = new List<ElevationMap> { dem };
            
            var outputDir = Path.Combine(Path.GetTempPath(), "PipelineTest_Small");
            if (Directory.Exists(outputDir))
            {
                Directory.Delete(outputDir, recursive: true);
            }
            
            using var generator = new QuadTreeHorizonGenerator();
            await generator.GenerateHorizonsForAllPatches(outputDir, dems, 2.0f);
            
            // Verify 4 files created (2x2 patches)
            var store = new HorizonTileStore(outputDir);
            var files = store.EnumerateFiles(observerElevationMeters: 2.0f).ToArray();
            Assert.AreEqual(4, files.Length, "Should generate 4 horizon files for 256x256 DEM");
            
            // Verify expected filenames exist
            var expectedFiles = new[]
            {
                "horizon_00000_00000_020.bin",
                "horizon_00128_00000_020.bin",
                "horizon_00000_00128_020.bin",
                "horizon_00128_00128_020.bin"
            };
            
            foreach (var expectedFile in expectedFiles)
            {
                Assert.IsTrue(HorizonTileStore.TryParseFileName(expectedFile, out var key));
                var filePath = store.BuildPath(key.TileY, key.TileX, key.ObserverElevationMeters, compress: false);
                Assert.IsTrue(File.Exists(filePath), $"Expected file should exist: {expectedFile}");
            }
            
            Console.WriteLine($"Successfully generated {files.Length} horizon files for small DEM test");
        }
    }
}
