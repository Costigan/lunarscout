using System.Runtime.InteropServices;
using System.Text.Json;
using moonlib;

const int AzimuthCount = 1440;

if (args.Length != 1)
{
    Console.Error.WriteLine("Usage: CSharpPhase6BLightmapOracle <output.json>");
    return 2;
}

var sampleAngles = new (float Azimuth, float Elevation)[]
{
    (0f, 1f),
    (0f, 0f),
    (0f, -1f),
    (0.02f, 0.18f),
    (359.98f, 0.18f),
    (10.12f, 0.45f),
    (123.456f, 0.61f),
    (275.249f, 0.52f),
};

var cases = new List<object>();

void Capture(string name, Func<int, float> horizonValue)
{
    var horizons = new float[AzimuthCount];
    for (int azimuth = 0; azimuth < AzimuthCount; azimuth++)
        horizons[azimuth] = horizonValue(azimuth);
    var samples = new List<object>();
    foreach (var (azimuth, elevation) in sampleAngles)
    {
        float fraction = LightmapGenerator.BuilderSunFraction(
            horizons, 0, azimuth, elevation);
        samples.Add(new
        {
            azimuth_deg = azimuth,
            elevation_deg = elevation,
            fraction,
            fraction_float32_bits = BitConverter.SingleToInt32Bits(fraction),
            encoded_byte = (byte)(255f * fraction),
        });
    }
    cases.Add(new
    {
        name,
        horizons_float32_base64 = Convert.ToBase64String(
            MemoryMarshal.AsBytes(horizons.AsSpan())),
        samples,
    });
}

Capture("constant_zero", _ => 0f);
Capture(
    "azimuth_wrap",
    azimuth => azimuth switch
    {
        1438 => 0.10f,
        1439 => 0.20f,
        0 => 0.30f,
        1 => 0.40f,
        _ => 0.25f,
    });
Capture(
    "interpolated_ripple",
    azimuth => 0.35f
        + 0.0125f * (azimuth % 23)
        + 0.0001f * (azimuth % 7));

var artifact = new
{
    schema = "lunarscout-numba-phase6b-lightmap-csharp-v1",
    source = "LightmapGenerator.BuilderSunFraction",
    azimuth_count = AzimuthCount,
    sun_half_angle_deg = 0.27f,
    solar_disk_slices = 16,
    encoding = "unchecked C# byte cast of float32(255 * fraction)",
    cases,
};
var options = new JsonSerializerOptions { WriteIndented = true };
File.WriteAllText(args[0], JsonSerializer.Serialize(artifact, options) + Environment.NewLine);
return 0;
