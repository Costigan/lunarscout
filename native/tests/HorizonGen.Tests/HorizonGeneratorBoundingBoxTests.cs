using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using moonlib.horizon;

namespace moonlib.tests;

[TestClass]
/// <summary>
/// Tests for determining the bounding box of a secondary "caster" map when projected onto a primary "observer" map.
/// This ensures that we correctly identify the region of interest when dealing with multiple DEMs.
/// </summary>
public class HorizonGeneratorBoundingBoxTests
{
    private enum ProjectionKind
    {
        LongLat,
        Stereographic
    }

    /// <summary>
    /// Verifies that <see cref="Utilities.GetBoundingBox(ElevationMap, ElevationMap)"/> correctly calculates the
    /// bounds of a 'caster' map in the 'observer' map's pixel coordinates for randomized map pairs.
    /// </summary>
    [TestMethod]
    [TestCategory("Fast")]
    public void GetBoundingBox_RandomizedPairs_MatchProjectedExtents()
    {
        var rng = new Random(20240528);
        for (int iteration = 0; iteration < 40; iteration++)
        {
            var observer = CreateRandomMap(rng);
            var caster = CreateRandomMap(rng);

            Rectangle expected = ComputeProjectedBounds(observer, caster);
            Rectangle actual = Utilities.GetBoundingBox(observer, caster);

            Assert.AreEqual(expected, actual, $"Iteration {iteration} failed.");
        }
    }

    /// <summary>
    /// Verifies that <see cref="Utilities.GetBoundingBox(ElevationMap, IEnumerable{ElevationMap})"/> correctly
    /// calculates the union of the observer's bounds and all projected 'other' maps.
    /// </summary>
    [TestMethod]
    [TestCategory("Fast")]
    public void GetBoundingBox_RandomizedCollection_AccumulatesUnion()
    {
        var rng = new Random(827361);
        var observer = CreateRandomMap(rng);
        var others = new List<ElevationMap>();
        int otherCount = rng.Next(2, 5);
        for (int i = 0; i < otherCount; i++)
            others.Add(CreateRandomMap(rng));

        Rectangle expected = observer.BoundingBox;
        foreach (var map in others)
            expected = Rectangle.Union(expected, ComputeProjectedBounds(observer, map));

        Rectangle actual = Utilities.GetBoundingBox(observer, others);

        Assert.AreEqual(expected, actual);
    }

    private static Rectangle ComputeProjectedBounds(ElevationMap observer, ElevationMap caster)
    {
        var shadowSrs = caster.SrsDescriptor ?? throw new InvalidOperationException("Caster missing SRS descriptor.");
        var observerSrs = observer.SrsDescriptor ?? throw new InvalidOperationException("Observer missing SRS descriptor.");
        var transform = MoonSrsLambdaFactory.MakeLambda(shadowSrs, observerSrs);

        double minCol = double.PositiveInfinity;
        double maxCol = double.NegativeInfinity;
        double minRow = double.PositiveInfinity;
        double maxRow = double.NegativeInfinity;

        foreach (var corner in EnumerateCorners(caster))
        {
            var srcCrs = caster.PixelToCRS(corner);
            srcCrs = MoonSrsLambdaFactory.ToLambdaInputUnits(srcCrs, shadowSrs);
            var dstCrs = transform(srcCrs);
            dstCrs = MoonSrsLambdaFactory.FromLambdaOutputUnits(dstCrs, observerSrs);
            var (col, row) = observer.CRSToPixel(dstCrs).Destructure();

            if (col < minCol) minCol = col;
            if (col > maxCol) maxCol = col;
            if (row < minRow) minRow = row;
            if (row > maxRow) maxRow = row;
        }

        return Rectangle.FromLTRB((int)Math.Floor(minCol),
            (int)Math.Floor(minRow),
            (int)Math.Ceiling(maxCol),
            (int)Math.Ceiling(maxRow));
    }

    private static IEnumerable<PixelPoint> EnumerateCorners(ElevationMap map)
    {
        yield return new PixelPoint(0, 0);
        yield return new PixelPoint(map.Width, 0);
        yield return new PixelPoint(map.Width, map.Height);
        yield return new PixelPoint(0, map.Height);
    }

    private static ElevationMap CreateRandomMap(Random rng)
    {
        var kind = (ProjectionKind)rng.Next(0, 2);
        int width = rng.Next(24, 64);
        int height = rng.Next(24, 64);

        var definition = kind switch
        {
            ProjectionKind.LongLat => CreateLongLatDefinition(rng, width, height),
            ProjectionKind.Stereographic => CreateStereographicDefinition(rng),
            _ => throw new InvalidOperationException("Unknown projection kind.")
        };

        var data = new float[height, width];
        var map = new ElevationMap(data, string.Empty, definition.GeoTransform);
        map.Projection = definition.Proj4;
        map.Proj4 = definition.Proj4;
        map.SrsDescriptor = definition.Descriptor;
        return map;
    }

    private static MapDefinition CreateLongLatDefinition(Random rng, int width, int height)
    {
        const string proj4 = "+proj=longlat +R=1737400 +no_defs";
        for (int attempt = 0; attempt < 128; attempt++)
        {
            double pixelWidth = NextDouble(rng, 0.05, 0.35);
            double pixelHeight = -NextDouble(rng, 0.05, 0.35);
            double rotX = NextDouble(rng, -0.01, 0.01);
            double rotY = NextDouble(rng, -0.01, 0.01);
            double originLon = NextDouble(rng, -170, 170);
            double originLat = NextDouble(rng, -70, 70);

            var geo = new[] { originLon, pixelWidth, rotX, originLat, rotY, pixelHeight };
            if (!IsInvertible(geo))
                continue;

            if (!CornersRespectLatLonLimits(geo, width, height))
                continue;

            return new MapDefinition(proj4, MoonSrsLambdaFactory.ParseSrs(proj4), geo);
        }

        throw new InvalidOperationException("Could not create valid longlat transform.");
    }

    private static MapDefinition CreateStereographicDefinition(Random rng)
    {
        double lat0 = NextDouble(rng, -80, 80);
        double lon0 = NextDouble(rng, -180, 180);
        double k0 = NextDouble(rng, 0.85, 1.15);
        double x0 = NextDouble(rng, -200000, 200000);
        double y0 = NextDouble(rng, -200000, 200000);

        string proj4 = string.Format(
            CultureInfo.InvariantCulture,
            "+proj=stere +lat_0={0:F6} +lon_0={1:F6} +k={2:F6} +x_0={3:F2} +y_0={4:F2} +R=1737400 +no_defs",
            lat0, lon0, k0, x0, y0);

        for (int attempt = 0; attempt < 128; attempt++)
        {
            double pixelWidth = NextDouble(rng, 300, 1500);
            double pixelHeight = -NextDouble(rng, 300, 1500);
            double rotX = NextDouble(rng, -30, 30);
            double rotY = NextDouble(rng, -30, 30);
            double originX = NextDouble(rng, -1000000, 1000000);
            double originY = NextDouble(rng, -1000000, 1000000);

            var geo = new[] { originX, pixelWidth, rotX, originY, rotY, pixelHeight };
            if (!IsInvertible(geo))
                continue;

            return new MapDefinition(proj4, MoonSrsLambdaFactory.ParseSrs(proj4), geo);
        }

        throw new InvalidOperationException("Could not create valid stereographic transform.");
    }

    private static bool CornersRespectLatLonLimits(double[] geo, int width, int height)
    {
        foreach (var (col, row) in new (double col, double row)[]
                 {
                     (0, 0),
                     (width, 0),
                     (width, height),
                     (0, height)
                 })
        {
            double lon = geo[0] + geo[1] * col + geo[2] * row;
            double lat = geo[3] + geo[4] * col + geo[5] * row;
            if (lon < -180 || lon > 180)
                return false;
            if (lat < -90 || lat > 90)
                return false;
        }
        return true;
    }

    private static bool IsInvertible(double[] geo)
    {
        double det = geo[1] * geo[5] - geo[2] * geo[4];
        return Math.Abs(det) > 1e-9;
    }

    private static double NextDouble(Random rng, double min, double max)
    {
        return min + rng.NextDouble() * (max - min);
    }

    private readonly struct MapDefinition
    {
        public MapDefinition(string proj4, SrsDescriptor descriptor, double[] geoTransform)
        {
            Proj4 = proj4;
            Descriptor = descriptor;
            GeoTransform = geoTransform;
        }

        public string Proj4 { get; }
        public SrsDescriptor Descriptor { get; }
        public double[] GeoTransform { get; }
    }
}
