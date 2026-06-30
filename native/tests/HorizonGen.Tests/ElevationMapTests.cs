using System;
using moonlib.horizon;
using OSGeo.GDAL;

#nullable disable

namespace moonlib.tests
{
    [TestClass]
    /// <summary>
    /// Tests the core functionality of the <see cref="ElevationMap"/> class, including coordinate transformations
    /// (Pixel <-> CRS), bilinear interpolation of elevation data, and constructor behavior.
    /// </summary>
    public class ElevationMapTests
    {


        /// <summary>
        /// Verifies that <see cref="ElevationMap.GetPointInMoonME"/> correctly converts pixel coordinates to
        /// Moon-Centered Moon-Fixed (ME) Cartesian coordinates when using a LongLat projection.
        /// </summary>
        [TestMethod]
        [TestCategory("Fast")]
        public void GetPointInMoonME_LongLatProjection_ReturnsCorrectMoonCoordinates()
        {
            // Arrange: 2x2 DEM, LongLat projection, simple geo transform
            float[,] arr = new float[2, 2] { { 0, 0 }, { 0, 0 } };
            string proj4 = "+proj=longlat +R=1737400 +no_defs";
            double[] geo = new double[6] { 10, 1, 0, 20, 0, 1 }; // x = lon, y = lat, both in degrees
            var map = new ElevationMap(arr, proj4, geo);

            // Act: pixel (0,0) should map to (lon,lat) = (10,20) deg, elevation = 0
            var moonPt = map.GetPointInMoonME(new PixelPoint(0, 0));

            // Assert: convert expected manually
            double lonRad = 10 * Math.PI / 180.0;
            double latRad = 20 * Math.PI / 180.0;
            double r = 1737400.0;
            double x = r * Math.Cos(latRad) * Math.Cos(lonRad);
            double y = r * Math.Cos(latRad) * Math.Sin(lonRad);
            double z = r * Math.Sin(latRad);
            Assert.AreEqual(x, moonPt.X, 1e-6);
            Assert.AreEqual(y, moonPt.Y, 1e-6);
            Assert.AreEqual(z, moonPt.Z, 1e-6);
        }

        /// <summary>
        /// Verifies that <see cref="ElevationMap.GetPointInMoonME"/> correctly handles Stereographic projections
        /// by converting them to LongLat and then to Moon ME coordinates.
        /// </summary>
        [TestMethod]
        [TestCategory("Fast")]
        public void GetPointInMoonME_StereographicProjection_ConvertsToLongLatAndReturnsMoonCoordinates()
        {
            // Arrange: 2x2 DEM, Stereographic projection, simple geo transform
            float[,] arr = new float[2, 2] { { 0, 0 }, { 0, 0 } };
            string proj4 = "+proj=stere +lat_0=20 +lon_0=10 +R=1737400 +k=1 +x_0=0 +y_0=0";
            double[] geo = new double[6] { 0, 1, 0, 0, 0, 1 }; // x/y in meters
            var map = new ElevationMap(arr, proj4, geo);

            // Act: pixel (0,0) is (x,y) = (0,0) in stereographic
            var moonPt = map.GetPointInMoonME(new PixelPoint(0, 0));

            // Assert: stereographic (0,0) should convert to (lon,lat) = (10,20) deg in radians
            double lonRad = 10 * Math.PI / 180.0;
            double latRad = 20 * Math.PI / 180.0;
            double r = 1737400.0;
            double x = r * Math.Cos(latRad) * Math.Cos(lonRad);
            double y = r * Math.Cos(latRad) * Math.Sin(lonRad);
            double z = r * Math.Sin(latRad);
            Assert.AreEqual(x, moonPt.X, 1e-6);
            Assert.AreEqual(y, moonPt.Y, 1e-6);
            Assert.AreEqual(z, moonPt.Z, 1e-6);
        }

        /// <summary>
        /// Verifies that initialization fails with a meaningful error for unsupported projections.
        /// </summary>
        [TestMethod]
        [TestCategory("Fast")]
        public void GetPointInMoonME_UnsupportedProjection_ReturnsZero()
        {
            // Arrange: 2x2 DEM, unsupported projection
            float[,] arr = new float[2, 2] { { 0, 0 }, { 0, 0 } };
            string proj4 = "+proj=foobar +R=1737400";
            double[] geo = new double[6] { 0, 1, 0, 0, 0, 1 };
            Assert.ThrowsException<ApplicationException>(() => new ElevationMap(arr, proj4, geo));
        }

        /// <summary>
        /// Verifies that <see cref="ElevationMap.GetElevationClipped"/> correctly interpolates values bilinearly
        /// between grid points.
        /// </summary>
        [TestMethod]
        [TestCategory("Fast")]
        public void GetElevationClipped_BilinearInterpolation_Works()
        {
            float[,] arr = new float[2, 2] { { 1, 2 }, { 3, 4 } };
            var map = new ElevationMap(arr, "", new double[6]);
            // At (0.5, 0.5) should be average of all four
            double expected = (1 + 2 + 3 + 4) / 4.0;
            double actual = map.GetElevationClipped(0.5, 0.5);
            Assert.AreEqual(expected, actual, 1e-6);
        }

        /// <summary>
        /// Verifies that <see cref="ElevationMap.GetElevationClipped"/> returns NaN for coordinates outside the map bounds.
        /// </summary>
        [TestMethod]
        [TestCategory("Fast")]
        public void GetElevationClipped_OutOfBounds_ReturnsNaN()
        {
            float[,] arr = new float[2, 2] { { 1, 2 }, { 3, 4 } };
            var map = new ElevationMap(arr, "", new double[6]);
            Assert.IsTrue(double.IsNaN(map.GetElevationClipped(-1, 0)));
            Assert.IsTrue(double.IsNaN(map.GetElevationClipped(0, -1)));
            Assert.IsTrue(double.IsNaN(map.GetElevationClipped(2, 0)));
            Assert.IsTrue(double.IsNaN(map.GetElevationClipped(0, 2)));
        }

        /// <summary>
        /// Verifies that <see cref="ElevationMap.GetElevationClipped"/> returns NaN for coordinates on the far edges,
        /// where interpolation neighbors would be out of bounds.
        /// </summary>
        [TestMethod]
        [TestCategory("Fast")]
        public void GetElevationClipped_EdgeCoordinates_ReturnsNaN()
        {
            float[,] arr = new float[2, 2] { { 1, 2 }, { 3, 4 } };
            var map = new ElevationMap(arr, "", new double[6]);
            // At the last row or column, interpolation is not possible
            Assert.IsTrue(double.IsNaN(map.GetElevationClipped(1, 0)));
            Assert.IsTrue(double.IsNaN(map.GetElevationClipped(0, 1)));
            Assert.IsTrue(double.IsNaN(map.GetElevationClipped(1, 1)));
        }

        /// <summary>
        /// Verifies that <see cref="ElevationMap.GetElevationClipped"/> returns the exact grid value for integer coordinates.
        /// </summary>
        [TestMethod]
        [TestCategory("Fast")]
        public void GetElevationClipped_IntegerCoordinates_ReturnsCellValue()
        {
            float[,] arr = new float[2, 2] { { 1, 2 }, { 3, 4 } };
            var map = new ElevationMap(arr, "", new double[6]);
            // At (0,0), should return arr[0,0] if interpolation is valid
            double actual = map.GetElevationClipped(0, 0);
            double expected = arr[0, 0];
            Assert.AreEqual(expected, actual, 1e-6);
        }

        /// <summary>
        /// Verifies the constructor initializes properties correctly from the provided arguments.
        /// </summary>
        //[TestMethod] //works but is slow
        [TestCategory("Fast")]
        public void Constructor_FromArray_SetsPropertiesCorrectly()
        {
            float[,] arr = new float[2, 2] { { 1, 2 }, { 3, 4 } };
            string proj = "";
            double[] geo = new double[6] { 0, 1, 0, 0, 0, -1 };
            var map = new ElevationMap(arr, proj, geo);
            Assert.AreEqual(arr, map.Elevation);
            Assert.AreEqual(proj, map.Projection);
            CollectionAssert.AreEqual(geo, map.GeoTransform);
        }

        /// <summary>
        /// Verifies that <see cref="ElevationMap.GetElevation"/> performs bilinear interpolation correctly (non-clipped version).
        /// </summary>
        [TestMethod]
        [TestCategory("Fast")]
        public void GetElevation_BilinearInterpolation_Works()
        {
            float[,] arr = new float[2, 2] { { 1, 2 }, { 3, 4 } };
            var map = new ElevationMap(arr, "", new double[6]);
            // At (0.5, 0.5) should be average of all four
            double expected = (1 + 2 + 3 + 4) / 4.0;
            double actual = map.GetElevation(0.5, 0.5);
            Assert.AreEqual(expected, actual, 1e-6);
        }

        /// <summary>
        /// Verifies that <see cref="ElevationMap.CRSToPixel"/> correctly transforms CRS coordinates to pixel coordinates.
        /// </summary>
        [TestMethod]
        [TestCategory("Fast")]
        public void WorldToPixel_InvertsNorthUpTransform()
        {
            float[,] arr = new float[1, 1];
            double[] geo = new double[6] { 100, 10, 0, 200, 0, -10 };
            var map = new ElevationMap(arr, "", geo);
            var (col, row) = map.CRSToPixel(new CRSPoint(150, 150)).Destructure();
            Assert.AreEqual(5, col, 1e-9);
            Assert.AreEqual(5, row, 1e-9);
        }

        /*
        [TestMethod]
        public void WorldToPixel_InvalidTransform_Throws()
        {
            float[,] arr = new float[1, 1];
            double[] geo = new double[6] { 0, 0, 0, 0, 0, 0 };
            var map = new ElevationMap(arr, "", geo);
            Assert.ThrowsException<InvalidOperationException>(() => map.CRSToPixel(new CRSPoint(0, 0)));
        }
        */

        /// <summary>
        /// Verifies that <see cref="ElevationMap.PixelToCRS"/> correctly transforms pixel coordinates to CRS coordinates.
        /// </summary>
        [TestMethod]
        [TestCategory("Fast")]
        public void PixelToWorld_AppliesGeoTransform()
        {
            float[,] arr = new float[1, 1];
            double[] geo = new double[6] { 100, 10, 0, 200, 0, -10 };
            var map = new ElevationMap(arr, "", geo);
            var (x, y) = map.PixelToCRS(new PixelPoint(3, 4)).Destructure();
            Assert.AreEqual(130, x, 1e-9);
            Assert.AreEqual(160, y, 1e-9);
        }

        /// <summary>
        /// Verifies that <see cref="ElevationMap.PixelToCRS"/> throws an exception if the GeoTransform is invalid.
        /// </summary>
        [TestMethod]
        [TestCategory("Fast")]
        public void PixelToWorld_GeoTransformNotDefined_Throws()
        {
            float[,] arr = new float[1, 1];
            double[] geo = Array.Empty<double>();
            var map = new ElevationMap(arr, "", geo);
            Assert.ThrowsException<IndexOutOfRangeException>(() => map.PixelToCRS(new PixelPoint(0, 0)));
        }

        /// <summary>
        /// Verifies that the constructor can successfully read a raster file from disk and populate the elevation data.
        /// </summary>
        //[TestMethod] // works but is slow
        [TestCategory("Fast")]
        public void Constructor_FromPath_ReadsRasterCorrectly()
        {
            // Dummy path, replace with a real file before running
            string path = "/d/datasets/viper_v71_2024_medium/other/dem.tif";
            var map = new ElevationMap(path);
            using var ds = Gdal.Open(path, Access.GA_ReadOnly);
            var band = ds.GetRasterBand(1);
            int width = band.XSize;
            int height = band.YSize;
            float[] buffer = new float[width * height];
            band.ReadRaster(0, 0, width, height, buffer, width, height, 0, 0);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    Assert.AreEqual(buffer[y * width + x], map.Elevation[y, x], 1e-6);
                }
            }
        }
    }
}
