using moonlib.horizon;

namespace moonlib.tests
{
    [TestClass]
    public class QuadTreeProductionRegressionTests
    {
        private const int AzimuthBins = 1440;
        private const string StereographicProj = "+proj=stere +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs";
        private const double PixelMeters = 30.0;

        [TestMethod]
        [TestCategory("Fast")]
        public void GenerateHorizons_FlatSyntheticDem_ReturnsCompleteFiniteSinglePixelHorizon()
        {
            var dem = CreateStereographicDem(129, 129, (_, _) => 0f);
            using var generator = new QuadTreeHorizonGenerator(disableHierarchy: false);

            var horizons = generator.GenerateHorizons(
                new List<ElevationMap> { dem },
                tileX: 64,
                tileY: 64,
                width: 1,
                height: 1,
                observerElevation: 0.0f);

            AssertCompleteFiniteHorizon(horizons, expectedPixels: 1);
            Assert.IsTrue(horizons.Degrees.All(v => v > -90.0f && v < 90.0f), "Flat synthetic horizon values should stay within plausible elevation angle bounds.");
        }

        [TestMethod]
        [TestCategory("Fast")]
        public void GenerateHorizons_SingleEastObstacle_RaisesEastAzimuthOverWestAzimuth()
        {
            const int size = 129;
            const int center = size / 2;
            var dem = CreateStereographicDem(size, size, (row, col) =>
                row == center && col == center + 20 ? 150f : 0f);
            using var generator = new QuadTreeHorizonGenerator(disableHierarchy: false);

            var horizons = generator.GenerateHorizons(
                new List<ElevationMap> { dem },
                tileX: center,
                tileY: center,
                width: 1,
                height: 1,
                observerElevation: 0.0f);

            AssertCompleteFiniteHorizon(horizons, expectedPixels: 1);

            float east = horizons.Degrees[AzimuthIndex(90.0)];
            float west = horizons.Degrees[AzimuthIndex(270.0)];

            Assert.IsTrue(east > 5.0f, $"East obstacle should produce a positive horizon angle, got {east} degrees.");
            Assert.IsTrue(east > west + 5.0f, $"East obstacle should materially exceed west horizon. East={east}, West={west}.");
        }

        [TestMethod]
        [TestCategory("Fast")]
        public void GenerateHorizons_NearTileBoundary_ReturnsCompleteFiniteBlock()
        {
            var dem = CreateStereographicDem(130, 130, (_, _) => 0f);
            using var generator = new QuadTreeHorizonGenerator(disableHierarchy: false);

            var horizons = generator.GenerateHorizons(
                new List<ElevationMap> { dem },
                tileX: 63,
                tileY: 63,
                width: 2,
                height: 2,
                observerElevation: 0.0f);

            AssertCompleteFiniteHorizon(horizons, expectedPixels: 4);
            Assert.IsFalse(horizons.Degrees.Any(v => v <= -1.0e20f), "Boundary block should not contain initialization sentinel values.");
        }

        [TestMethod]
        [TestCategory("Fast")]
        public void GenerateHorizons_MultiDemOuterObstacle_RaisesCombinedEastAzimuth()
        {
            var inner = CreateStereographicDem(41, 41, (_, _) => 0f);

            const int outerSize = 129;
            const int outerCenter = outerSize / 2;
            var outer = CreateStereographicDem(outerSize, outerSize, (row, col) =>
                row == outerCenter && col == outerCenter + 45 ? 250f : 0f);

            int innerCenter = inner.Width / 2;
            using var generator = new QuadTreeHorizonGenerator(disableHierarchy: false);

            var innerOnly = generator.GenerateHorizons(
                new List<ElevationMap> { inner },
                tileX: innerCenter,
                tileY: innerCenter,
                width: 1,
                height: 1,
                observerElevation: 0.0f);

            var combined = generator.GenerateHorizons(
                new List<ElevationMap> { inner, outer },
                tileX: innerCenter,
                tileY: innerCenter,
                width: 1,
                height: 1,
                observerElevation: 0.0f);

            AssertCompleteFiniteHorizon(innerOnly, expectedPixels: 1);
            AssertCompleteFiniteHorizon(combined, expectedPixels: 1);

            float innerEast = innerOnly.Degrees[AzimuthIndex(90.0)];
            float combinedEast = combined.Degrees[AzimuthIndex(90.0)];

            Assert.IsTrue(combinedEast > innerEast + 5.0f, $"Outer DEM obstacle should raise combined east horizon. Combined={combinedEast}, InnerOnly={innerEast}.");
        }

        private static void AssertCompleteFiniteHorizon(HorizonAngles horizons, int expectedPixels)
        {
            Assert.AreEqual(expectedPixels * AzimuthBins, horizons.Length, "Horizon array length should match pixel count times azimuth bins.");
            Assert.IsTrue(horizons.Degrees.All(float.IsFinite), "All horizon values should be finite.");
            Assert.IsFalse(horizons.Degrees.Any(float.IsNaN), "Horizon values should not contain NaN.");
            Assert.IsFalse(horizons.Degrees.Any(float.IsInfinity), "Horizon values should not contain infinities.");
        }

        private static int AzimuthIndex(double azimuthDegrees) =>
            ((int)Math.Round(azimuthDegrees / 0.25)) % AzimuthBins;

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
    }
}
