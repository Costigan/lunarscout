using System.Security.Cryptography;
using System.Text.Json;
using moonlib.horizon;

const int PatchSize = 128;
const int AzimuthCount = 1440;
const double Radius = 1_737_400.0;

if (args.Length != 1)
{
    Console.Error.WriteLine("Usage: CSharpPhase6BPsrOracle <output.json>");
    return 2;
}

var transform = new GeoTransformD
{
    T0 = 1000.0,
    T1 = 20.0,
    T2 = 0.0,
    T3 = -1000.0,
    T4 = 0.0,
    T5 = -20.0,
};
var projection = new ProjectionParamsDouble
{
    R = Radius,
    Lat0 = -Math.PI / 2.0,
    Lon0 = 0.0,
    K0 = 1.0,
    FalseEasting = 0.0,
    FalseNorthing = 0.0,
};
var dem = new float[PatchSize * PatchSize];
for (int y = 0; y < PatchSize; y++)
for (int x = 0; x < PatchSize; x++)
    dem[y * PatchSize + x] = (float)((x - y) * 0.1);

float[] PositionForLocalAngle(double azimuthDeg, double elevationDeg)
{
    const int column = 64;
    const int row = 64;
    double crsX = transform.T0 + transform.T1 * column + transform.T2 * row;
    double crsY = transform.T3 + transform.T4 * column + transform.T5 * row;
    double rho = Math.Sqrt(crsX * crsX + crsY * crsY);
    double c = 2.0 * Math.Atan2(rho, 2.0 * projection.K0 * projection.R);
    double sinC = Math.Sin(c);
    double cosC = Math.Cos(c);
    double cosLat0 = Math.Cos(projection.Lat0);
    double sinLat0 = Math.Sin(projection.Lat0);
    double latitude = Math.Asin(cosC * sinLat0 + crsY * sinC * cosLat0 / rho);
    double longitude = projection.Lon0 + Math.Atan2(
        crsX * sinC,
        rho * cosLat0 * cosC - crsY * sinLat0 * sinC);
    double cosLat = Math.Cos(latitude);
    double sinLat = Math.Sin(latitude);
    double cosLon = Math.Cos(longitude);
    double sinLon = Math.Sin(longitude);
    var up = new[] { cosLat * cosLon, cosLat * sinLon, sinLat };
    var east = new[] { -sinLon, cosLon, 0.0 };
    var north = new[] { -sinLat * cosLon, -sinLat * sinLon, cosLat };
    double observerRadius = Radius + dem[row * PatchSize + column];
    double azimuth = azimuthDeg * Math.PI / 180.0;
    double elevation = elevationDeg * Math.PI / 180.0;
    double localEast = Math.Sin(azimuth) * Math.Cos(elevation);
    double localNorth = Math.Cos(azimuth) * Math.Cos(elevation);
    double localUp = Math.Sin(elevation);
    const double distance = 150_000_000_000.0;
    var result = new float[3];
    for (int axis = 0; axis < 3; axis++)
    {
        double observer = observerRadius * up[axis];
        double direction = localEast * east[axis] + localNorth * north[axis] + localUp * up[axis];
        result[axis] = (float)(observer + distance * direction);
    }
    return result;
}

var cases = new List<object>();
using var lightmaps = new Lightmaps(1);

void Capture(string name, Func<int, int, float> horizonValue, params float[][] vectors)
{
    var horizons = new float[PatchSize * PatchSize * AzimuthCount];
    for (int pixel = 0; pixel < PatchSize * PatchSize; pixel++)
    for (int azimuth = 0; azimuth < AzimuthCount; azimuth++)
        horizons[pixel * AzimuthCount + azimuth] = horizonValue(pixel, azimuth);
    var flatVectors = vectors.SelectMany(value => value).ToArray();
    var output = lightmaps.ComputePSRPatchForDiagnostics(
        dem, transform, projection, flatVectors, horizons);
    cases.Add(new
    {
        name,
        sun_vectors_m = flatVectors,
        output_base64 = Convert.ToBase64String(output),
        output_sha256 = Convert.ToHexString(SHA256.HashData(output)).ToLowerInvariant(),
        psr_count = output.Count(value => value == byte.MaxValue),
        non_psr_count = output.Count(value => value == 0),
    });
}

Capture(
    "constant_shadow",
    (_, _) => 0.3f,
    PositionForLocalAngle(0.0, 0.0));
Capture(
    "interpolated_mixed",
    (pixel, azimuth) => 0.45f + 0.01f * (pixel % 23) + 0.0001f * (azimuth % 17),
    PositionForLocalAngle(10.12, 0.25),
    PositionForLocalAngle(200.2, -1.0));
float QuantizedElevation(float elevation)
{
    float clamped = Math.Clamp(elevation, -50f, 50f);
    int quantized = (int)MathF.Round(
        clamped * (32767f / 50f), MidpointRounding.AwayFromZero);
    quantized = Math.Clamp(quantized, -32767, 32767);
    return quantized * (50f / 32767f);
}
Capture(
    "compressed_quantized_mixed",
    (pixel, azimuth) => QuantizedElevation(
        0.45f + 0.01f * (pixel % 23) + 0.0001f * (azimuth % 17)),
    PositionForLocalAngle(10.12, 0.25),
    PositionForLocalAngle(200.2, -1.0));

var artifact = new
{
    schema = "lunarscout-numba-phase6b-psr-csharp-v1",
    source = "Lightmaps.ComputePSRKernel via ComputePSRPatchForDiagnostics",
    patch_size = PatchSize,
    azimuth_count = AzimuthCount,
    geotransform = new[] { transform.T0, transform.T1, transform.T2, transform.T3, transform.T4, transform.T5 },
    projection = new
    {
        radius_m = projection.R,
        latitude_origin_rad = projection.Lat0,
        longitude_origin_rad = projection.Lon0,
        scale = projection.K0,
        false_easting_m = projection.FalseEasting,
        false_northing_m = projection.FalseNorthing,
    },
    dem_formula = "float32((x-y)*0.1)",
    horizon_formulas = new Dictionary<string, string>
    {
        ["constant_shadow"] = "float32(0.3)",
        ["interpolated_mixed"] = "float32(0.45 + 0.01*(pixel%23) + 0.0001*(azimuth%17))",
        ["compressed_quantized_mixed"] = "HorizonCompressor float32 quantization of interpolated_mixed",
    },
    cases,
};
var options = new JsonSerializerOptions { WriteIndented = true };
File.WriteAllText(args[0], JsonSerializer.Serialize(artifact, options) + Environment.NewLine);
return 0;
