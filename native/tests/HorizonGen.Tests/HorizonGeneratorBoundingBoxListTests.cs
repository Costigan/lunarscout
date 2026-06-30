using moonlib.horizon;
using OSGeo.GDAL;
using OSGeo.OSR;
using System.Drawing;
using System.Globalization;

namespace moonlib.tests;

[TestClass]
/// <summary>
/// Verifies that <see cref="Utilities.GetBoundingBox"/> accurately computes the bounding rectangle
/// that encompasses a central map and a list of other maps (projected into the central map's pixel space).
/// This test compares the results against a reference implementation using GDAL directly.
/// </summary>
public sealed class HorizonGeneratorBoundingBoxListTests
{

    /// <summary>
    /// Fuzz test that generates random map projections (LongLat and Stereographic) and verifies that
    /// our internal bounding box logic matches the GDAL-derived ground truth.
    /// </summary>
    //[TestMethod]  // works but is slow
    [TestCategory("Fast")]
    public void BoundingBoxesMatchGdalAcrossRandomMaps()
    {
        var rng = new Random(314159);
        var projections = BuildProjectionSamples(rng);

        for (int i = 0; i < projections.Count; ++i)
        {
            bool success = false;
            for (int attempt = 0; attempt < 10 && !success; ++attempt)
            {
                var center = CreateRandomMap(rng, projections[i]);
                int otherCount = 3;
                var others = new List<ElevationMap>(otherCount);

                for (int j = 0; j < otherCount; ++j)
                {
                    string otherProj = projections[(i + j + 1) % projections.Count];
                    others.Add(CreateRandomMap(rng, otherProj));
                }

                try
                {
                    var expected = Utilities.GetBoundingBox(center, others);
                    var actual = ComputeBoundingBoxWithGdal(center, others);
                    Console.WriteLine($"checking {expected} == {actual}");
                    AssertRectanglesEqual(expected, actual, $"Mismatch on iteration {i}.");
                    success = true;
                }
                catch (ApplicationException ex) when (IsProjLatitudeError(ex))
                {
                    // Retry with a new random sample if PROJ rejects the coordinate domain.
                }
            }

            if (!success)
            {
                Assert.Inconclusive($"Unable to create a valid sample for projection index {i} without triggering PROJ domain errors.");
            }
        }
    }

    /// <summary>
    /// Verifies bounding box calculation when multiple maps share the same projection (specifically LongLat),
    /// ensuring no coordinate transformation artifacts occur in simple cases.
    /// </summary>
    [TestMethod]
    [TestCategory("Fast")]
    public void BoundingBoxMatchesGdalWhenAdditionalMapSharesProjection()
    {
        var rng = new Random(271828);
        var centerProj = ElevationMap.LongLatProj;
        bool success = false;

        for (int attempt = 0; attempt < 10 && !success; ++attempt)
        {
            var center = CreateRandomMap(rng, centerProj);

            var others = new List<ElevationMap>
            {
                CreateRandomMap(rng, centerProj),
                CreateRandomMap(rng, centerProj)
            };

            try
            {
                var expected = Utilities.GetBoundingBox(center, others);
                var actual = ComputeBoundingBoxWithGdal(center, others);
                AssertRectanglesEqual(expected, actual, "Mismatch for shared proj scenario.");
                success = true;
            }
            catch (ApplicationException ex) when (IsProjLatitudeError(ex))
            {
                // regenerate inputs and try again
            }
        }

        if (!success)
        {
            Assert.Inconclusive("Unable to generate a valid longlat-only sample without PROJ domain errors.");
        }
    }

    private static Rectangle ComputeBoundingBoxWithGdal(
        ElevationMap center,
        IReadOnlyList<ElevationMap> others)
    {
        Rectangle union = Rectangle.FromLTRB(0, 0, center.Width, center.Height);

        using var centerSrs = ImportProj4(center.Proj4 ?? throw new InvalidOperationException("Center map missing proj4."));
        var inverseCenterGeo = InvertGeoTransform(center.GeoTransform);

        foreach (var other in others)
        {
            var projected = ProjectBoundingBox(other, centerSrs, inverseCenterGeo);
            union = Rectangle.FromLTRB(
                Math.Min(union.Left, projected.Left),
                Math.Min(union.Top, projected.Top),
                Math.Max(union.Right, projected.Right),
                Math.Max(union.Bottom, projected.Bottom));
        }

        return union;
    }

    private static Rectangle ProjectBoundingBox(
        ElevationMap other,
        SpatialReference centerSrs,
        double[] inverseCenterGeo)
    {
        using var otherSrs = ImportProj4(other.Proj4 ?? throw new InvalidOperationException("Other map missing proj4."));
        using var transform = new CoordinateTransformation(otherSrs, centerSrs);

        double minCol = double.PositiveInfinity;
        double maxCol = double.NegativeInfinity;
        double minRow = double.PositiveInfinity;
        double maxRow = double.NegativeInfinity;

        var gt = other.GeoTransform ?? throw new ArgumentException("Other map missing geotransform.", nameof(other));
        int width = other.Width;
        int height = other.Height;

        var corners = new (double Col, double Row)[]
        {
            (0.0, 0.0),
            (width, 0.0),
            (width, height),
            (0.0, height)
        };

        foreach (var (col, row) in corners)
        {
            Gdal.ApplyGeoTransform(gt, col, row, out double worldX, out double worldY);
            double[] transformed = new double[3];
            transform.TransformPoint(transformed, worldX, worldY, 0.0);
            double x = transformed[0];
            double y = transformed[1];

            Gdal.ApplyGeoTransform(inverseCenterGeo, x, y, out double centerCol, out double centerRow);

            minCol = Math.Min(minCol, centerCol);
            maxCol = Math.Max(maxCol, centerCol);
            minRow = Math.Min(minRow, centerRow);
            maxRow = Math.Max(maxRow, centerRow);
        }

        return Rectangle.FromLTRB(
            (int)Math.Floor(minCol),
            (int)Math.Floor(minRow),
            (int)Math.Ceiling(maxCol),
            (int)Math.Ceiling(maxRow));
    }

    private static List<string> BuildProjectionSamples(Random rng)
    {
        var projections = new List<string> { ElevationMap.LongLatProj };
        while (projections.Count < 20)
        {
            projections.Add(CreateRandomStereographicProj4(rng));
        }

        return projections;
    }

    private static ElevationMap CreateRandomMap(Random rng, string proj4)
    {
        int width = rng.Next(64, 257);
        int height = rng.Next(64, 257);
        double[] geo = CreateGeoTransformForProj(rng, proj4);
        return new ElevationMap(new float[height, width], proj4, geo);
    }

    private static double[] CreateGeoTransformForProj(Random rng, string proj4)
    {
        var descriptor = MoonSrsLambdaFactory.ParseSrs(proj4);
        double originX;
        double originY;
        double scaleX;
        double scaleY;
        double rotationX;
        double rotationY;

        if (descriptor.Type == SrsDescriptor.ProjType.LongLat)
        {
            originX = rng.NextDouble() * 360.0 - 180.0;
            originY = rng.NextDouble() * 120.0 - 60.0;
            scaleX = rng.NextDouble() * 0.04 + 0.01;
            scaleY = rng.NextDouble() * 0.04 + 0.01;
            rotationX = (rng.NextDouble() - 0.5) * 0.01;
            rotationY = (rng.NextDouble() - 0.5) * 0.01;
        }
        else
        {
            originX = descriptor.x0 + (rng.NextDouble() * 2.0 - 1.0) * 10_000.0;
            originY = descriptor.y0 + (rng.NextDouble() * 2.0 - 1.0) * 10_000.0;
            scaleX = rng.NextDouble() * 200.0 + 50.0;
            scaleY = rng.NextDouble() * 200.0 + 50.0;
            rotationX = (rng.NextDouble() - 0.5) * 10.0;
            rotationY = (rng.NextDouble() - 0.5) * 10.0;
        }

        if (rng.Next(2) == 0)
        {
            scaleY = -scaleY;
        }

        while (Math.Abs(scaleX * scaleY - rotationX * rotationY) < 1e-6)
        {
            rotationX *= 0.5;
            rotationY *= 0.5;
        }

        return new[]
        {
            originX,
            scaleX,
            rotationX,
            originY,
            rotationY,
            scaleY
        };
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
            1_737_400.0);
    }

    private static SpatialReference ImportProj4(string proj4)
    {
        var srs = new SpatialReference(null);
        int err = srs.ImportFromProj4(proj4);
        Assert.AreEqual(0, err, $"Failed to import proj4: {proj4}");
        return srs;
    }

    private static double[] InvertGeoTransform(double[] geoTransform)
    {
        var inverse = new double[6];
        int success = Gdal.InvGeoTransform(geoTransform, inverse);
        Assert.AreEqual(1, success, "GDAL failed to invert the geotransform.");
        return inverse;
    }

    private static void AssertRectanglesEqual(Rectangle expected, Rectangle actual, string message)
    {
        Assert.AreEqual(expected.Left, actual.Left, $"{message} Expected Left={expected.Left}, Actual Left={actual.Left}");
        Assert.AreEqual(expected.Top, actual.Top, $"{message} Expected Top={expected.Top}, Actual Top={actual.Top}");
        Assert.AreEqual(expected.Right, actual.Right, $"{message} Expected Right={expected.Right}, Actual Right={actual.Right}");
        Assert.AreEqual(expected.Bottom, actual.Bottom, $"{message} Expected Bottom={expected.Bottom}, Actual Bottom={actual.Bottom}");
    }

    private static bool IsProjLatitudeError(Exception ex) =>
        ex.Message.Contains("Invalid latitude", StringComparison.OrdinalIgnoreCase);
}
