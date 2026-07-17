using System.Text.Json;
using moonlib.spice;

if (args.Length != 1)
{
    Console.Error.WriteLine("Usage: CSharpPhase6BSpiceOracle <output.json>");
    return 2;
}

_ = new SpiceManager();
var times = new[]
{
    new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc),
    new DateTime(2000, 1, 1, 12, 0, 0, DateTimeKind.Utc),
    new DateTime(2023, 12, 1, 0, 0, 0, DateTimeKind.Utc),
    new DateTime(2027, 1, 1, 0, 0, 0, DateTimeKind.Utc),
    new DateTime(2044, 1, 1, 0, 0, 0, DateTimeKind.Utc),
};

var samples = times.Select(time =>
{
    var sun = SpiceManager.SunPosition_meters(time);
    var earth = SpiceManager.EarthPosition_meters(time);
    return new
    {
        timestamp_utc = time.ToString("yyyy-MM-ddTHH:mm:ss.ffffff'Z'"),
        csharp_et = SpiceMethods.DateTimeToET(time),
        sun_m = new[] { sun.X, sun.Y, sun.Z },
        earth_m = new[] { earth.X, earth.Y, earth.Z },
    };
}).ToArray();

var artifact = new
{
    schema = "lunarscout-numba-phase6b-csharp-spice-v1",
    frame = "MOON_ME",
    observer = "MOON",
    correction = "NONE",
    units = "meters",
    csharp_conversion = "SpiceMethods.DateTimeToET",
    local_epoch_utc = SpiceMethods.LocalEpoch.ToString("yyyy-MM-ddTHH:mm:ss.ffffff'Z'"),
    local_epoch_et = SpiceMethods.LocalEpochEpochTime,
    samples,
};
File.WriteAllText(
    args[0],
    JsonSerializer.Serialize(artifact, new JsonSerializerOptions { WriteIndented = true })
        + Environment.NewLine);
return 0;
