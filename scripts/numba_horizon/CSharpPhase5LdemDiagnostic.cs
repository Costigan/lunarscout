using System.Diagnostics;
using System.Text.RegularExpressions;
using System.Text.Json;
using moonlib;
using moonlib.horizon;

if (args is ["--index-linearization"])
{
    var backendType = typeof(ILGPU.Index2D).Assembly.GetType(
        "ILGPU.Backends.IL.ILBackend", throwOnError: true)!;
    var reconstruct = backendType.GetMethod(
        "Reconstruct2DIndex",
        System.Reflection.BindingFlags.Static |
        System.Reflection.BindingFlags.NonPublic)!;
    var extent = new ILGPU.Index2D(16384, 1440);
    foreach (int linearIndex in new[] { 0, 1, 31, 32, 767, 768, 1439, 1440 })
    {
        var index = (ILGPU.Index2D)reconstruct.Invoke(
            null, new object[] { extent, linearIndex })!;
        Console.WriteLine($"{linearIndex}: ({index.X}, {index.Y})");
    }
    return 0;
}

if (args is ["--kernel-ptx-info"])
{
    using var ptxGenerator = new QuadTreeHorizonGenerator(
        disableHierarchy: false,
        maxConcurrentGpuOps: 1,
        maxSegmentQueueSize: 1);
    string ptx = ptxGenerator.SelectedSubpatchKernelPtx;
    Console.WriteLine(JsonSerializer.Serialize(new
    {
        accelerator = ptxGenerator.SelectedAcceleratorName,
        ptx_characters = ptx.Length,
        ptx_lines = ptx.Count(character => character == '\n'),
        f64_instruction_lines = ptx.Split('\n').Count(
            line => Regex.IsMatch(line, @"\.(f64|rn\.f64|f64\.)")),
        local_declarations = ptx.Split('\n').Count(line => line.Contains(".local ")),
        register_declarations = ptx.Split('\n')
            .Where(line => line.Contains(".reg "))
            .Select(line => line.Trim())
            .ToArray(),
    }, new JsonSerializerOptions { WriteIndented = true }));
    return 0;
}

if (args.Length != 5)
{
    Console.Error.WriteLine(
        "Usage: CSharpPhase5LdemDiagnostic <output-json> <primary-dem> <dem-2> <dem-3> <ldem>");
    return 2;
}

string outputPath = Path.GetFullPath(args[0]);
string[] inputPaths = args[1..].Select(Path.GetFullPath).ToArray();
const int selectedAzimuth = 763;
const int selectedDemPass = 3;

MoonlibBridge.EnsureGdalInitialized();
var dems = inputPaths.Select(path => new ElevationMap(path)).ToList();
using var generator = new QuadTreeHorizonGenerator(
    disableHierarchy: false,
    maxConcurrentGpuOps: 1,
    maxSegmentQueueSize: 1);

Console.WriteLine("PHASE csharp_selected_ray_start");
var stopwatch = Stopwatch.StartNew();
var snapshot = generator.CaptureSubpatchBuffersForDiagnostics(
    dems,
    tileColumn: 0,
    tileRow: 0,
    tileWidth: 128,
    tileHeight: 128,
    observerElevationMeters: 0f,
    subpatchSize: 8,
    traversalTraceDemPass: selectedDemPass,
    traversalTraceAzimuthIndex: selectedAzimuth,
    traversalTracePixelIndex: 0);
stopwatch.Stop();
Console.WriteLine("PHASE csharp_selected_ray_end");

var trace = snapshot.TraversalTrace;
var report = new
{
    schema_version = 1,
    scope = "full production LDEM selected-ray traversal diagnostic",
    input_paths = inputPaths,
    selected_accelerator_name = generator.SelectedAcceleratorName,
    selected_accelerator_type = generator.SelectedAcceleratorType.ToString(),
    configuration = new
    {
        tile_column = 0,
        tile_row = 0,
        tile_width = 128,
        tile_height = 128,
        subpatch_size = 8,
        azimuth_index = selectedAzimuth,
        azimuth_degrees = selectedAzimuth * 0.25,
        dem_pass = selectedDemPass,
    },
    elapsed_seconds_all_four_passes_all_azimuths = stopwatch.Elapsed.TotalSeconds,
    selected_output_slope = snapshot.PerDemSlopes[selectedDemPass][selectedAzimuth],
    selected_interpolated_segment = snapshot.TraversalInterpolatedSegment,
    counters = new
    {
        trace_rows = trace.Length,
        maximum_level = trace.Length == 0 ? -1 : trace.Max(step => step.Level),
        descents = trace.Count(step => step.Action == 0),
        culled_blocks = trace.Count(step => step.Action == 1),
        nodata_skips = trace.Count(step => step.Action == 2),
        out_of_bounds = trace.Count(step => step.Action == 3),
        level0_samples = trace.Count(step => step.Action == 4),
    },
    trace = trace.Select(step => new
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
};

Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
File.WriteAllText(
    outputPath,
    JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true })
    + Environment.NewLine);
Console.WriteLine(outputPath);
return 0;

static float? Finite(float value) => float.IsFinite(value) ? value : null;
