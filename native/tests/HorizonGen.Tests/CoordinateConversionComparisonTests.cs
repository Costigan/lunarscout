using moonlib.horizon;
using OSGeo.GDAL;
using OSGeo.OSR;
using System.Globalization;

namespace moonlib.tests;

[TestClass]
/// <summary>
/// Compares the coordinate conversion logic in <see cref="ElevationMap"/> against the reference implementation provided by GDAL.
/// This ensures that our custom math (which avoids GDAL dependencies in the hot path) is consistent with the industry standard.
/// </summary>
public sealed class CoordinateConversionComparisonTests
{
    public static readonly object _gdalLock = new object();
    private const double PixelTolerance = 1e-9;
    private const double ProjectionTolerance = 1e-6;
    private const double MoonRadiusMeters = 1_737_400.0;

    /// <summary>
    /// Verifies that <see cref="ElevationMap.PixelToCRS"/> produces the same output as GDAL's ApplyGeoTransform
    /// for a random set of affine transforms.
    /// </summary>
    [TestMethod]
    [TestCategory("Fast")]
    public void PixelToCrsMatchesGdalApplyGeoTransform()
    {
        var rng = new Random(42);

        for (int i = 0; i < 20; ++i)
        {
            double[] geoTransform = CreateRandomGeoTransform(rng);
            var map = new ElevationMap(new float[1, 1], ElevationMap.LongLatProj, geoTransform);

            double col = rng.NextDouble() * 200.0 - 100.0;
            double row = rng.NextDouble() * 200.0 - 100.0;

            var (expectedX, expectedY) = map.PixelToCRS(new PixelPoint(col, row)).Destructure();

            Gdal.ApplyGeoTransform(geoTransform, col, row, out double gdalX, out double gdalY);

            Assert.AreEqual(gdalX, expectedX, PixelTolerance);
            Assert.AreEqual(gdalY, expectedY, PixelTolerance);
        }
    }

    /// <summary>
    /// Verifies that <see cref="ElevationMap.CRSToPixel"/> produces the same output as GDAL's inverse ApplyGeoTransform
    /// (InvGeoTransform) for a random set of affine transforms.
    /// </summary>
    [TestMethod]
    [TestCategory("Fast")]
    public void CrsToPixelMatchesGdalInverseGeoTransform()
    {
        var rng = new Random(84);

        for (int i = 0; i < 20; ++i)
        {
            double[] geoTransform = CreateRandomGeoTransform(rng);
            var map = new ElevationMap(new float[1, 1], ElevationMap.LongLatProj, geoTransform);

            double[] inverse = InvertGeoTransformWithGdal(geoTransform);

            for (int sample = 0; sample < 5; ++sample)
            {
                double col = rng.NextDouble() * 200.0 - 100.0;
                double row = rng.NextDouble() * 200.0 - 100.0;
                var pt_crs = map.PixelToCRS(new PixelPoint(col, row));
                var (roundTripCol, roundTripRow) = map.CRSToPixel(pt_crs).Destructure();

                Gdal.ApplyGeoTransform(inverse, pt_crs.X, pt_crs.Y, out double gdalCol, out double gdalRow);

                Assert.AreEqual(gdalCol, roundTripCol, PixelTolerance);
                Assert.AreEqual(gdalRow, roundTripRow, PixelTolerance);
            }
        }
    }

    /// <summary>
    /// Verifies that the custom projection lambdas created by <see cref="MoonSrsLambdaFactory"/> match GDAL's
    /// CoordinateTransformation for supported projections (LongLat and Stereographic).
    /// </summary>
    [TestMethod]
    [TestCategory("Fast")]
    public void MakeLambdaMatchesGdalForSupportedPairs()
    {
        var rng = new Random(20240520);
        var projections = new List<string> { ElevationMap.LongLatProj };

        while (projections.Count < 6)
        {
            projections.Add(CreateRandomStereographicProj4(rng));
        }

        foreach (var srcProj in projections)
        {
            foreach (var dstProj in projections)
            {
                var lambda = MoonSrsLambdaFactory.MakeLambda(srcProj, dstProj);

                using var src = ImportProj4(srcProj);
                using var dst = ImportProj4(dstProj);
                using var transform = new CoordinateTransformation(src, dst);

                for (int sample = 0; sample < 20; ++sample)
                {
                    var (inputA, inputB) = SampleCoordinate(rng, srcProj);

                    var our_crs = lambda(new CRSPoint(inputA, inputB));
                    var (gdalX, gdalY) = TransformWithGdal(transform, srcProj, dstProj, inputA, inputB);

                    Assert.AreEqual(gdalX, our_crs.X, ProjectionTolerance, $"Mismatch for {srcProj} → {dstProj}");
                    Assert.AreEqual(gdalY, our_crs.Y, ProjectionTolerance, $"Mismatch for {srcProj} → {dstProj}");
                }
            }
        }
    }

    /// <summary>
    /// End-to-end test verifying that transforming a pixel from one Stereographic map to another
    /// (Pixel -> CRS -> Reproject -> CRS -> Pixel) matches the equivalent GDAL pipeline.
    /// </summary>
    [TestMethod]
    [TestCategory("Fast")]
    public void StereographicPixelToPixelMatchesGdalPipeline()
    {
        var rng = new Random(60617);

        for (int trial = 0; trial < 5; ++trial)
        {
            string srcProj = CreateRandomStereographicProj4(rng);
            string dstProj = CreateRandomStereographicProj4(rng);

            double[] srcGeo = CreateRandomGeoTransform(rng);
            double[] dstGeo = CreateRandomGeoTransform(rng);
            double[] dstInverse = InvertGeoTransformWithGdal(dstGeo);

            var srcMap = new ElevationMap(new float[1, 1], srcProj, srcGeo);
            var dstMap = new ElevationMap(new float[1, 1], dstProj, dstGeo);

            var lambda = MoonSrsLambdaFactory.MakeLambda(srcProj, dstProj);

            using var src = ImportProj4(srcProj);
            using var dst = ImportProj4(dstProj);
            using var transform = new CoordinateTransformation(src, dst);
            double[] gdalPoint = new double[3];

            for (int sample = 0; sample < 10; ++sample)
            {
                double col = rng.NextDouble() * 400.0 - 200.0;
                double row = rng.NextDouble() * 400.0 - 200.0;

                var src_crs = srcMap.PixelToCRS(new PixelPoint(col, row));

                var dst_crs = lambda(src_crs);
                var (ourCol, ourRow) = dstMap.CRSToPixel(dst_crs).Destructure();

                transform.TransformPoint(gdalPoint, src_crs.X, src_crs.Y, 0.0);
                double gdalDstX = gdalPoint[0];
                double gdalDstY = gdalPoint[1];
                Gdal.ApplyGeoTransform(dstInverse, gdalDstX, gdalDstY, out double gdalCol, out double gdalRow);

                Assert.AreEqual(gdalCol, ourCol, ProjectionTolerance);
                Assert.AreEqual(gdalRow, ourRow, ProjectionTolerance);
            }
        }
    }

    private static double[] CreateRandomGeoTransform(Random rng)
    {
        while (true)
        {
            double originX = rng.NextDouble() * 10_000.0 - 5_000.0;
            double pixelWidth = rng.NextDouble() * 50.0 + 0.1;
            double rotationX = rng.NextDouble() * 2.0 - 1.0;
            double originY = rng.NextDouble() * 10_000.0 - 5_000.0;
            double rotationY = rng.NextDouble() * 2.0 - 1.0;
            double pixelHeight = rng.NextDouble() * 50.0 + 0.1;

            double determinant = pixelWidth * pixelHeight - rotationX * rotationY;
            if (Math.Abs(determinant) < 1e-9)
            {
                continue;
            }

            return new[] { originX, pixelWidth, rotationX, originY, rotationY, pixelHeight };
        }
    }

    private static string CreateRandomStereographicProj4(Random rng)
    {
        double lat0 = rng.NextDouble() * 170.0 - 85.0;
        double lon0 = rng.NextDouble() * 360.0 - 180.0;
        double k = rng.NextDouble() * 1.5 + 0.5;
        double x0 = (rng.NextDouble() * 2.0 - 1.0) * 100_000.0;
        double y0 = (rng.NextDouble() * 2.0 - 1.0) * 100_000.0;

        return string.Format(
            CultureInfo.InvariantCulture,
            "+proj=stere +lat_0={0} +lon_0={1} +k={2} +x_0={3} +y_0={4} +R={5} +units=m +no_defs",
            lat0,
            lon0,
            k,
            x0,
            y0,
            MoonRadiusMeters);
    }

    private static double[] InvertGeoTransformWithGdal(double[] geoTransform)
    {
        var inverse = new double[6];
        int success = Gdal.InvGeoTransform(geoTransform, inverse);
        Assert.AreEqual(1, success, "GDAL failed to invert the geotransform.");
        return inverse;
    }

    private static SpatialReference ImportProj4(string proj4)
    {
        var srs = new SpatialReference(null);
        int err = srs.ImportFromProj4(proj4);
        Assert.AreEqual(0, err, $"Failed to import proj4: {proj4}");
        return srs;
    }

    private static (double x, double y) TransformWithGdal(CoordinateTransformation transform, string srcProj, string dstProj, double a, double b)
    {
        double x = a;
        double y = b;

        if (IsLongLat(srcProj))
        {
            x = RadToDeg(a);
            y = RadToDeg(b);
        }

        double[] result = new double[3];
        transform.TransformPoint(result, x, y, 0.0);
        double outX = result[0];
        double outY = result[1];

        if (IsLongLat(dstProj))
        {
            outX = DegToRad(outX);
            outY = DegToRad(outY);
        }

        return (outX, outY);
    }

    private static (double a, double b) SampleCoordinate(Random rng, string proj4)
    {
        if (IsLongLat(proj4))
        {
            double lon = rng.NextDouble() * 2.0 * Math.PI - Math.PI;
            double lat = rng.NextDouble() * Math.PI - Math.PI / 2.0;
            return (lon, lat);
        }

        double x = (rng.NextDouble() * 2.0 - 1.0) * 500_000.0;
        double y = (rng.NextDouble() * 2.0 - 1.0) * 500_000.0;
        return (x, y);
    }

    private static bool IsLongLat(string proj4) =>
        proj4.Contains("+proj=longlat", StringComparison.OrdinalIgnoreCase);

    private static double RadToDeg(double angle) => angle * 180.0 / Math.PI;

    private static double DegToRad(double angle) => angle * Math.PI / 180.0;
}
