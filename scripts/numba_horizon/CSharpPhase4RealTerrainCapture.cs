using System.Security.Cryptography;
using System.Text.Json;
using moonlib;
using moonlib.horizon;

if (args.Length is not (2 or 4))
{
    Console.Error.WriteLine(
        "Usage: CSharpPhase4RealTerrainCapture <input-dem.tif[|outer-dem.tif...]> <output-prefix> [trace-pixel trace-azimuth]");
    return 2;
}

string[] inputPaths = args[0].Split('|').Select(Path.GetFullPath).ToArray();
string inputPath = inputPaths[0];
string outputPrefix = Path.GetFullPath(args[1]);
string slopePath = outputPrefix + ".slopes.f32le";
string degreePath = outputPrefix + ".degrees.f32le";
string segmentPath = outputPrefix + ".segments.f32le";
string metadataPath = outputPrefix + ".json";
int tracePixel = args.Length == 4 ? int.Parse(args[2]) : 104;
int traceAzimuth = args.Length == 4 ? int.Parse(args[3]) : 319;
Directory.CreateDirectory(Path.GetDirectoryName(outputPrefix)!);
MoonlibBridge.EnsureGdalInitialized();
var dems = inputPaths.Select(path => new ElevationMap(path)).ToList();
using var generator = new QuadTreeHorizonGenerator(
    disableHierarchy: false,
    maxConcurrentGpuOps: 1,
    maxSegmentQueueSize: 1);

const int tileColumn = 240;
const int tileRow = 240;
const int tileWidth = 16;
const int tileHeight = 16;
const int subpatchSize = 8;
var segmentSnapshot = generator.CalculateSubpatchRaySegmentsForDiagnostics(
    dems,
    tileColumn,
    tileRow,
    tileWidth,
    tileHeight,
    numAzimuths: 1440,
    maxDistanceMeters: 1_000_000f,
    observerElevationMeters: 0f,
    subpatchSize);
var started = System.Diagnostics.Stopwatch.StartNew();
var snapshot = generator.CaptureSubpatchBuffersForDiagnostics(
    dems,
    tileColumn,
    tileRow,
    tileWidth,
    tileHeight,
    observerElevationMeters: 0f,
    subpatchSize,
    traversalTraceDemPass: dems.Count - 1,
    traversalTraceAzimuthIndex: traceAzimuth,
    traversalTracePixelIndex: tracePixel);
started.Stop();

WriteFloats(slopePath, snapshot.FinalSlopes);
WriteFloats(degreePath, snapshot.FinalDegrees);
WriteSegments(segmentPath, segmentSnapshot.Segments);
string[] passPaths = snapshot.PerDemSlopes.Select(
    (_, index) => $"{outputPrefix}.pass{index}.slopes.f32le").ToArray();
for (int index = 0; index < passPaths.Length; index++)
    WriteFloats(passPaths[index], snapshot.PerDemSlopes[index]);
var report = new
{
    schema_version = 1,
    input_path = inputPath,
    input_paths = inputPaths,
    selected_accelerator_name = generator.SelectedAcceleratorName,
    selected_accelerator_type = generator.SelectedAcceleratorType.ToString(),
    configuration = new
    {
        tile_column = tileColumn,
        tile_row = tileRow,
        tile_width = tileWidth,
        tile_height = tileHeight,
        azimuth_count = 1440,
        subpatch_size = subpatchSize,
        observer_elevation_m = 0f,
        hierarchy_enabled = true,
    },
    output = new
    {
        shape = new[] { tileWidth * tileHeight, 1440 },
        dtype = "float32",
        byte_order = "little",
        slope_path = slopePath,
        slope_sha256 = Hash(slopePath),
        degree_path = degreePath,
        degree_sha256 = Hash(degreePath),
        segment_path = segmentPath,
        segment_shape = new[] { 1440, 16, dems.Count, 18 },
        segment_sha256 = Hash(segmentPath),
        per_dem_slope_paths = passPaths,
        per_dem_slope_sha256 = passPaths.Select(Hash).ToArray(),
    },
    elapsed_seconds = started.Elapsed.TotalSeconds,
    traversal_trace = new
    {
        dem_pass = dems.Count - 1,
        pixel_index = tracePixel,
        azimuth_index = traceAzimuth,
        rows = snapshot.TraversalTrace.Select(step => new
        {
            s_km = step.ParameterDistanceKm,
            true_distance_m = step.TrueDistanceMeters,
            level = step.Level,
            cell_x = step.CellX,
            cell_y = step.CellY,
            pixel_x = step.PixelX,
            pixel_y = step.PixelY,
            maximum_elevation_m = Finite(step.MaximumElevationMeters),
            sample_elevation_m = Finite(step.SampleElevationMeters),
            sample_slope = Finite(step.SampleSlope),
            advance_km = step.AdvanceKm,
            action = step.Action,
        }),
    },
    selected_interpolated_segment = snapshot.TraversalInterpolatedSegment,
};
File.WriteAllText(
    metadataPath,
    JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true })
    + Environment.NewLine);
Console.WriteLine(metadataPath);
return 0;

static void WriteFloats(string path, float[] values)
{
    using var stream = File.Create(path);
    using var writer = new BinaryWriter(stream);
    foreach (float value in values)
        writer.Write(value);
}

static void WriteSegments(string path, RaySegment[] segments)
{
    using var stream = File.Create(path);
    using var writer = new BinaryWriter(stream);
    foreach (RaySegment segment in segments)
    {
        foreach (float value in Values(segment))
            writer.Write(value);
    }
}

static float[] Values(RaySegment segment) => new[]
{
    segment.StartPixel.X, segment.StartPixel.Y, segment.X0, segment.Y0,
    segment.A1, segment.A2, segment.A3, segment.A4,
    segment.B1, segment.B2, segment.B3, segment.B4,
    segment.SStart, segment.SEnd, segment.SStartChord,
    segment.PlanarToChordC1, segment.PlanarToChordC2, segment.PlanarToChordC3,
};

static string Hash(string path) =>
    Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant();

static float? Finite(float value) => float.IsFinite(value) ? value : null;
