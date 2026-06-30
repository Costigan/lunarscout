using moonlib.horizon;
using System.Globalization;

namespace moonlib.tests;

[TestClass]
/// <summary>
/// Tests the <see cref="MoonSrsLambdaFactory"/> class, verifying that it correctly parses projection strings
/// and creates accurate coordinate transformation lambdas for supported Moon SRS types (LongLat, Stereographic).
/// </summary>
public sealed class MoonSrsLambdaFactoryTests
{
    private const double MoonRadiusMeters = 1_737_400.0;
    private const double Tolerance = 1e-7;

    //string proj1 = "+proj=longlat +R=1737400 +no_defs";
    //string proj2 = "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs";
    //string proj3 = "+proj=stere +lat_0=-85.42088 +lon_0=31.6218 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs";

    private const string LongLatProj = "+proj=longlat +R=1737400";

    // Centered stereographic projection at the lunar north pole with no offsets.
    private const string StereographicProj = "+proj=stere +R=1737400 +lat_0=90 +lon_0=0 +k=1 +x_0=0 +y_0=0";

    private const string ObliqueStereographicProj = "+proj=stere +lat_0=-85.42088 +lon_0=31.6218 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs";

    /// <summary>
    /// Verifies that the North Pole in LongLat (0, 90 deg) projects to the origin (0,0) in a North Pole-centered Stereographic projection.
    /// </summary>
    [TestMethod]
    [TestCategory("Fast")]
    public void ForwardProjectionOfNorthPoleIsAtOrigin()
    {
        var transform = MoonSrsLambdaFactory.MakeLambda(LongLatProj, StereographicProj);
        var (x, y) = transform(new CRSPoint(DegToRad(0), DegToRad(90))).Destructure();

        Assert.AreEqual(0.0, x, Tolerance);
        Assert.AreEqual(0.0, y, Tolerance);
    }

    /// <summary>
    /// Verifies that transforming from LongLat to Stereographic and back returns the original coordinates.
    /// </summary>
    [TestMethod]
    [TestCategory("Fast")]
    public void RoundTripBetweenLongLatAndStereographicReturnsOriginalCoordinates()
    {
        var forward = MoonSrsLambdaFactory.MakeLambda(LongLatProj, StereographicProj);
        var inverse = MoonSrsLambdaFactory.MakeLambda(StereographicProj, LongLatProj);

        var originalLon = DegToRad(12.5);
        var originalLat = DegToRad(82.0);

        var (x, y) = forward(new CRSPoint(originalLon, originalLat)).Destructure();
        var (lon, lat) = inverse(new CRSPoint(x, y)).Destructure();

        Assert.AreEqual(originalLon, lon, Tolerance);
        Assert.AreEqual(originalLat, lat, Tolerance);
    }

    /// <summary>
    /// Verifies round-trip transformation accuracy for an Oblique Stereographic projection (centered away from the poles).
    /// </summary>
    [TestMethod]
    [TestCategory("Fast")]
    public void RoundTripBetweenLongLatAndObliqueStereographicReturnsOriginalCoordinates()
    {
        var forward = MoonSrsLambdaFactory.MakeLambda(LongLatProj, ObliqueStereographicProj);
        var inverse = MoonSrsLambdaFactory.MakeLambda(ObliqueStereographicProj, LongLatProj);

        var originalLon = DegToRad(35.0);
        var originalLat = DegToRad(-82.5);

        var (x, y) = forward(new CRSPoint(originalLon, originalLat)).Destructure();
        var (lon, lat) = inverse(new CRSPoint(x, y)).Destructure();

        Assert.AreEqual(originalLon, lon, Tolerance);
        Assert.AreEqual(originalLat, lat, Tolerance);
    }

    /// <summary>
    /// Verifies that the center point defined in an Oblique Stereographic projection maps exactly to the origin (0,0).
    /// </summary>
    [TestMethod]
    [TestCategory("Fast")]
    public void ObliqueStereographicCenterPointRoundTripsToOrigin()
    {
        var forward = MoonSrsLambdaFactory.MakeLambda(LongLatProj, ObliqueStereographicProj);
        var inverse = MoonSrsLambdaFactory.MakeLambda(ObliqueStereographicProj, LongLatProj);

        var centerLon = DegToRad(31.6218);
        var centerLat = DegToRad(-85.42088);

        var (x, y) = forward(new CRSPoint(centerLon, centerLat)).Destructure();

        Assert.AreEqual(0.0, x, Tolerance);
        Assert.AreEqual(0.0, y, Tolerance);

        var (lon, lat) = inverse(new CRSPoint(x, y)).Destructure();

        Assert.AreEqual(centerLon, lon, Tolerance);
        Assert.AreEqual(centerLat, lat, Tolerance);
    }

    /// <summary>
    /// Verifies that creating a lambda for the same source and destination projection results in an identity transformation.
    /// </summary>
    [TestMethod]
    [TestCategory("Fast")]
    public void LongLatToLongLatReturnsIdentity()
    {
        var transform = MoonSrsLambdaFactory.MakeLambda(LongLatProj, LongLatProj);

        var lon = DegToRad(12.5);
        var lat = DegToRad(-34.2);

        var (resultLon, resultLat) = transform(new CRSPoint(lon, lat)).Destructure();

        Assert.AreEqual(lon, resultLon, Tolerance);
        Assert.AreEqual(lat, resultLat, Tolerance);
    }

    /// <summary>
    /// Verifies that transforming directly between two Stereographic projections matches the result of
    /// transforming Source -> LongLat -> Destination.
    /// </summary>
    [TestMethod]
    [TestCategory("Fast")]
    public void StereographicToStereographicMatchesComposition()
    {
        var sourceSrs =
            $"+proj=stere +R={MoonRadiusMeters.ToString(CultureInfo.InvariantCulture)} +lat_0=75 +lon_0=10 +k=1.2 +x_0=1000 +y_0=-500";
        var destinationSrs =
            $"+proj=stere +R={MoonRadiusMeters.ToString(CultureInfo.InvariantCulture)} +lat_0=45 +lon_0=-30 +k=0.8 +x_0=-200 +y_0=300";

        var toSource = MoonSrsLambdaFactory.MakeLambda(LongLatProj, sourceSrs);
        var toLongLat = MoonSrsLambdaFactory.MakeLambda(sourceSrs, LongLatProj);
        var toDestination = MoonSrsLambdaFactory.MakeLambda(LongLatProj, destinationSrs);
        var direct = MoonSrsLambdaFactory.MakeLambda(sourceSrs, destinationSrs);

        var sampleLon = DegToRad(-12.25);
        var sampleLat = DegToRad(68.4);

        var (srcX, srcY) = toSource(new CRSPoint(sampleLon, sampleLat)).Destructure();

        var (directX, directY) = direct(new CRSPoint(srcX, srcY)).Destructure();
        var (lon, lat) = toLongLat(new CRSPoint(srcX, srcY)).Destructure();
        var (expectedX, expectedY) = toDestination(new CRSPoint(lon, lat)).Destructure();

        Assert.AreEqual(expectedX, directX, Tolerance);
        Assert.AreEqual(expectedY, directY, Tolerance);
    }

    /// <summary>
    /// Verifies that attempting to create a lambda for an unsupported projection (e.g., Mercator) throws a <see cref="NotSupportedException"/>.
    /// </summary>
    [TestMethod]
    [TestCategory("Fast")]
    [ExpectedException(typeof(NotSupportedException))]
    public void UnsupportedProjectionPairThrows()
    {
        const string UnsupportedProj = "+proj=merc +R=1737400";
        MoonSrsLambdaFactory.MakeLambda(UnsupportedProj, LongLatProj);
    }

    private static double DegToRad(double degrees) => degrees * Math.PI / 180.0;

    /// <summary>
    /// Iterates through a grid of coordinates to verify identity transformation for LongLat -> LongLat.
    /// </summary>
    [TestMethod]
    [TestCategory("Fast")]
    public void ConvertsToLongLat1()
    {
        string crs1 = LongLatProj;
        string crs2 = LongLatProj;
        var func = MoonSrsLambdaFactory.MakeLambda(crs1, crs2);
        for (var lon = 0d; lon < 10d; lon += 0.1d)
            for (var lat = 0d; lat < 10d; lat += 0.1d)
            {
                var (outLon, outLat) = func(new CRSPoint(lon, lat)).Destructure();
                Assert.AreEqual(lon, outLon, Tolerance);
                Assert.AreEqual(lat, outLat, Tolerance);
            }
    }

    /// <summary>
    /// Verifies a specific point conversion from Oblique Stereographic to LongLat.
    /// </summary>
    [TestMethod]
    [TestCategory("Fast")]
    public void ConvertsToLongLat2()
    {
        // This demonstrates that oblique stereographic to longlat yields angles in radians
        string crs1 = ObliqueStereographicProj;
        string crs2 = LongLatProj;
        var func = MoonSrsLambdaFactory.MakeLambda(crs1, crs2);
        var (outLon, outLat) = func(new CRSPoint(0.0, 0.0)).Destructure();
        Assert.AreEqual(DegToRad(31.6218), outLon, Tolerance);
        Assert.AreEqual(DegToRad(-85.42088), outLat, Tolerance);
    }

    /// <summary>
    /// Data-driven test verifying Stereographic <-> LongLat conversion for various center points (Poles, Equator, etc.).
    /// </summary>
    [TestMethod]
    [TestCategory("Fast")]
    [DataRow(31.6218, -85.42088)]
    [DataRow(0, -90)]
    [DataRow(0, 90)]
    [DataRow(180, -90)]
    [DataRow(180, 90)]
    [DataRow(10, -89.88)]
    [DataRow(-10, -89.88)]
    public void StereographicToLongLat(double longitude, double latitude)
    {
        var src_proj4 = $"+proj=stere +lat_0={latitude} +lon_0={longitude} +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs";
        var stereographic_to_spherical = MoonSrsLambdaFactory.MakeLambda(src_proj4, LongLatProj);
        var spherical_to_stereographic = MoonSrsLambdaFactory.MakeLambda(LongLatProj, src_proj4);

        var (outLon, outLat) = stereographic_to_spherical(new CRSPoint(0.0, 0.0)).Destructure();
        Assert.AreEqual(DegToRad(longitude), outLon, Tolerance);
        Assert.AreEqual(DegToRad(latitude), outLat, Tolerance);

        var (x, y) = spherical_to_stereographic(new CRSPoint(outLon, outLat)).Destructure();
        Assert.AreEqual(0.0, x, Tolerance);
        Assert.AreEqual(0.0, y, Tolerance);
    }
}

[TestClass]
/// <summary>
/// Tests specifically for the <see cref="MoonSrsLambdaFactory.ParseSrs"/> method, ensuring it correctly
/// extracts projection parameters from PROJ strings.
/// </summary>
public sealed class MoonSrsLambdaFactoryParseSrsTests
{
    private const double MoonRadiusMeters = 1_737_400.0;
    private const double Tolerance = 1e-7;

    /// <summary>
    /// Verifies parsing of a standard LongLat projection string.
    /// </summary>
    [TestMethod]
    public void ParsesLongLatProjectionCorrectly()
    {
        string srs = $"+proj=longlat +R={MoonRadiusMeters.ToString(CultureInfo.InvariantCulture)}";
        var sd = MoonSrsLambdaFactory.ParseSrs(srs);
        Assert.AreEqual(SrsDescriptor.ProjType.LongLat, sd.Type);
        Assert.AreEqual(MoonRadiusMeters, sd.R, Tolerance);
    }

    /// <summary>
    /// Verifies parsing of a Stereographic projection string with all parameters specified.
    /// </summary>
    [TestMethod]
    public void ParsesStereographicProjectionWithAllParameters()
    {
        string srs = $"+proj=stere +lat_0=45 +lon_0=10 +k=2 +x_0=100 +y_0=200 +R={MoonRadiusMeters.ToString(CultureInfo.InvariantCulture)}";
        var sd = MoonSrsLambdaFactory.ParseSrs(srs);
        Assert.AreEqual(SrsDescriptor.ProjType.Stereographic, sd.Type);
        Assert.AreEqual(MoonRadiusMeters, sd.R, Tolerance);
        Assert.AreEqual(45 * Math.PI / 180.0, sd.lat0, Tolerance);
        Assert.AreEqual(10 * Math.PI / 180.0, sd.lon0, Tolerance);
        Assert.AreEqual(2.0, sd.k0, Tolerance);
        Assert.AreEqual(100.0, sd.x0, Tolerance);
        Assert.AreEqual(200.0, sd.y0, Tolerance);
    }

    /// <summary>
    /// Verifies parsing of a Stereographic projection string using default values for omitted parameters.
    /// </summary>
    [TestMethod]
    public void ParsesStereographicProjectionWithDefaults()
    {
        string srs = $"+proj=stere +R={MoonRadiusMeters.ToString(CultureInfo.InvariantCulture)}";
        var sd = MoonSrsLambdaFactory.ParseSrs(srs);
        Assert.AreEqual(SrsDescriptor.ProjType.Stereographic, sd.Type);
        Assert.AreEqual(MoonRadiusMeters, sd.R, Tolerance);
        Assert.AreEqual(0.0, sd.lat0, Tolerance);
        Assert.AreEqual(0.0, sd.lon0, Tolerance);
        Assert.AreEqual(1.0, sd.k0, Tolerance);
        Assert.AreEqual(0.0, sd.x0, Tolerance);
        Assert.AreEqual(0.0, sd.y0, Tolerance);
    }

    /// <summary>
    /// Verifies that the parser defaults to the standard Moon radius if the +R parameter is missing.
    /// </summary>
    [TestMethod]
    public void UsesMoonRadiusIfRIsMissing()
    {
        string srs = "+proj=longlat";
        var sd = MoonSrsLambdaFactory.ParseSrs(srs);
        Assert.AreEqual(SrsDescriptor.ProjType.LongLat, sd.Type);
        Assert.AreEqual(MoonRadiusMeters, sd.R, Tolerance);
    }

    /// <summary>
    /// Verifies that an unsupported projection type is correctly identified.
    /// </summary>
    [TestMethod]
    public void SetsTypeUnsupportedIfProjIsMissing()
    {
        string srs = $"+R={MoonRadiusMeters.ToString(CultureInfo.InvariantCulture)}";
        var sd = MoonSrsLambdaFactory.ParseSrs(srs);
        Assert.AreEqual(SrsDescriptor.ProjType.Unsupported, sd.Type);
    }

    /// <summary>
    /// Verifies that unknown projection names result in an Unsupported type.
    /// </summary>
    [TestMethod]
    public void SetsTypeUnsupportedIfProjIsUnknown()
    {
        string srs = $"+proj=foobar +R={MoonRadiusMeters.ToString(CultureInfo.InvariantCulture)}";
        var sd = MoonSrsLambdaFactory.ParseSrs(srs);
        Assert.AreEqual(SrsDescriptor.ProjType.Unsupported, sd.Type);
    }

    /// <summary>
    /// Verifies that invalid R parameter values throw an <see cref="ArgumentException"/>.
    /// </summary>
    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void ThrowsIfRIsInvalid()
    {
        string srs = "+proj=longlat +R=notanumber";
        MoonSrsLambdaFactory.ParseSrs(srs);
    }
}
