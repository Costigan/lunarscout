using moonlib.horizon;

namespace moonlib.tests
{
    [TestClass]
    public class NativeHorizonScenarioRegressionTests
    {
        private const string StereographicProj = "+proj=stere +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs";
        private const double PixelMeters = 30.0;

        [TestMethod]
        [TestCategory("Fast")]
        public void FlatDem_ReferenceRayProducesFiniteBelowHorizontalHorizon()
        {
            var dem = CreateStereographicDem(121, 121, (_, _) => 0f);
            var observer = CenterOrigin(dem);

            var result = ReferenceRayEmulator.Run(
                dem,
                observer,
                azimuthDeg: 90.0,
                outputPath: string.Empty,
                suppressCsv: true,
                maxDistanceMeters: 1000.0);

            Assert.IsTrue(result.Slopes.Length > 0, "Flat DEM ray should produce samples.");
            Assert.IsTrue(result.Slopes.All(double.IsFinite), "Flat DEM slopes should be finite.");
            Assert.IsTrue(result.ElevationDeg < 0.0f, $"Flat spherical DEM horizon should be below horizontal, got {result.ElevationDeg} degrees.");
            Assert.IsTrue(result.ElevationDeg > -1.0f, $"Flat DEM horizon should stay near horizontal for this small synthetic DEM, got {result.ElevationDeg} degrees.");
        }

        [TestMethod]
        [TestCategory("Fast")]
        public void SingleObstacle_ReferenceRayFindsPeakAtExpectedAzimuth()
        {
            const int width = 121;
            const int height = 121;
            int centerRow = height / 2;
            int centerCol = width / 2;

            var dem = CreateStereographicDem(width, height, (row, col) =>
                row == centerRow && col == centerCol + 20 ? 150f : 0f);

            var observer = CenterOrigin(dem);

            var east = ReferenceRayEmulator.Run(
                dem,
                observer,
                azimuthDeg: 90.0,
                outputPath: string.Empty,
                suppressCsv: true,
                maxDistanceMeters: 1200.0);

            var west = ReferenceRayEmulator.Run(
                dem,
                observer,
                azimuthDeg: 270.0,
                outputPath: string.Empty,
                suppressCsv: true,
                maxDistanceMeters: 1200.0);

            Assert.IsTrue(east.Slopes.Length > 0, "East ray should produce samples.");
            Assert.IsTrue(west.Slopes.Length > 0, "West ray should produce samples.");
            Assert.IsTrue(east.ElevationDeg > 5.0f, $"East obstacle should produce a positive horizon angle, got {east.ElevationDeg} degrees.");
            Assert.IsTrue(east.ElevationDeg > west.ElevationDeg + 5.0f, $"East obstacle should dominate west flat terrain. East={east.ElevationDeg}, West={west.ElevationDeg}.");
        }

        [TestMethod]
        [TestCategory("Fast")]
        public void MultiDem_ReferenceRayIncludesObstacleFromOuterDem()
        {
            var inner = CreateStereographicDem(41, 41, (_, _) => 0f);

            const int outerWidth = 121;
            const int outerHeight = 121;
            int outerCenterRow = outerHeight / 2;
            int outerCenterCol = outerWidth / 2;
            var outer = CreateStereographicDem(outerWidth, outerHeight, (row, col) =>
                row == outerCenterRow && col == outerCenterCol + 45 ? 250f : 0f);

            var observer = CenterOrigin(inner);

            var innerOnly = ReferenceRayEmulator.Run(
                inner,
                observer,
                azimuthDeg: 90.0,
                outputPath: string.Empty,
                suppressCsv: true,
                maxDistanceMeters: 2500.0);

            var combinedResults = ReferenceRayEmulator.RunMultiDem(
                new List<ElevationMap> { inner, outer },
                observer,
                azimuthDeg: 90.0,
                maxDistanceMeters: 2500.0,
                suppressCsv: true);
            var combined = EmulatorResult.Combine(combinedResults);

            Assert.IsTrue(innerOnly.Slopes.Length > 0, "Inner DEM ray should produce samples.");
            Assert.IsTrue(combined.Slopes.Length > innerOnly.Slopes.Length, "Combined multi-DEM ray should include samples from the outer DEM.");
            Assert.IsTrue(combined.ElevationDeg > innerOnly.ElevationDeg + 5.0f, $"Outer DEM obstacle should raise the combined horizon. Combined={combined.ElevationDeg}, InnerOnly={innerOnly.ElevationDeg}.");
            Assert.IsTrue(combined.ElevationDeg > 5.0f, $"Combined horizon should include the positive outer DEM obstacle, got {combined.ElevationDeg} degrees.");
        }

        private static ElevationMap CreateStereographicDem(int width, int height, Func<int, int, float> elevation)
        {
            var data = new float[height, width];
            for (int row = 0; row < height; row++)
            {
                for (int col = 0; col < width; col++)
                {
                    data[row, col] = elevation(row, col);
                }
            }

            var geoTransform = new[]
            {
                -(width / 2) * PixelMeters,
                PixelMeters,
                0.0,
                (height / 2) * PixelMeters,
                0.0,
                -PixelMeters
            };

            return new ElevationMap(data, StereographicProj, geoTransform);
        }

        private static PixelOrigin CenterOrigin(ElevationMap dem) =>
            new()
            {
                X = dem.Width / 2,
                Y = dem.Height / 2,
                Z = 0f
            };
    }
}
