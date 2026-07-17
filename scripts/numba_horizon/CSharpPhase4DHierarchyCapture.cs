using System.Text.Json;
using moonlib;
using moonlib.horizon;

if (args.Length != 1)
{
    Console.Error.WriteLine("Usage: CSharpPhase4DHierarchyCapture <output-json>");
    return 2;
}

MoonlibBridge.EnsureGdalInitialized();
var inner = CreateDem(41, 41, (_, _) => 0f);
var outer = CreateDem(
    1025,
    1025,
    (row, column) => row == inner.Height / 2 && column == 990 ? 5000f : 0f,
    anchorColumn: inner.Width / 2,
    anchorRow: inner.Height / 2);
using var generator = new QuadTreeHorizonGenerator(
    disableHierarchy: false, maxConcurrentGpuOps: 1, maxSegmentQueueSize: 1);
var snapshot = generator.CalculateSubpatchRaySegmentsForDiagnostics(
    new List<ElevationMap> { inner, outer },
    tileColumn: 20,
    tileRow: 20,
    tileWidth: 1,
    tileHeight: 1,
    numAzimuths: 1440,
    maxDistanceMeters: 1_000_000f,
    observerElevationMeters: 0f,
    subpatchSize: 8);

var selected = new float[4][];
for (int center = 0; center < 4; center++)
{
    int index = ((360 * 4 + center) * 2) + 1;
    selected[center] = Values(snapshot.Segments[index]);
}
var report = new
{
    schema_version = 1,
    source = "QuadTreeHorizonGenerator.CalculateSubpatchRaySegmentsForDiagnostics",
    selected_accelerator_name = generator.SelectedAcceleratorName,
    selected_accelerator_type = generator.SelectedAcceleratorType.ToString(),
    azimuth_index = 360,
    dem_index = 1,
    center_order = new[] { "top_left", "top_right", "bottom_left", "bottom_right" },
    segment_fields = new[]
    {
        "start_pixel_x", "start_pixel_y", "x0", "y0", "a1", "a2", "a3", "a4",
        "b1", "b2", "b3", "b4", "s_start_km", "s_end_km", "s_start_chord_km",
        "planar_to_chord_c1", "planar_to_chord_c2", "planar_to_chord_c3",
    },
    segments = selected,
    production_hierarchy_case = CaptureProductionHierarchyCase(generator, inner, outer),
    bilinear_boundary_case = CaptureBilinearBoundaryCase(generator),
};
string output = Path.GetFullPath(args[0]);
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
File.WriteAllText(
    output,
    JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true })
    + Environment.NewLine);
return 0;

static object CaptureProductionHierarchyCase(
    QuadTreeHorizonGenerator generator, ElevationMap inner, ElevationMap outer)
{
    var snapshot = generator.CaptureSubpatchBuffersForDiagnostics(
        new List<ElevationMap> { inner, outer },
        tileColumn: 20,
        tileRow: 20,
        tileWidth: 1,
        tileHeight: 1,
        observerElevationMeters: 0f,
        subpatchSize: 8,
        traversalTraceDemPass: 1,
        traversalTraceAzimuthIndex: 360);
    return new
    {
        description = "Corrected four-cell bilinear hierarchy bound on the Phase 1 production fixture.",
        csharp_hierarchy_maximum_slope = snapshot.PerDemSlopes[1][360],
        per_dem_slopes = snapshot.PerDemSlopes,
        final_slopes = snapshot.FinalSlopes,
        final_degrees = snapshot.FinalDegrees,
        trace = TraceValues(snapshot.TraversalTrace),
    };
}

static object CaptureBilinearBoundaryCase(QuadTreeHorizonGenerator generator)
{
    const int size = 121;
    const int center = size / 2;
    var dem = CreateDem(
        size,
        size,
        (row, column) => row == center - 20 && column == center ? 150f : 0f);
    var segmentSnapshot = generator.CalculateSubpatchRaySegmentsForDiagnostics(
        new List<ElevationMap> { dem },
        tileColumn: center,
        tileRow: center,
        tileWidth: 1,
        tileHeight: 1,
        numAzimuths: 1440,
        maxDistanceMeters: 1_000_000f,
        observerElevationMeters: 0f,
        subpatchSize: 8);
    var segments = Enumerable.Range(0, 4)
        .Select(index => Values(segmentSnapshot.Segments[index]))
        .ToArray();
    var snapshot = generator.CaptureSubpatchBuffersForDiagnostics(
        new List<ElevationMap> { dem },
        tileColumn: center,
        tileRow: center,
        tileWidth: 1,
        tileHeight: 1,
        observerElevationMeters: 0f,
        subpatchSize: 8,
        traversalTraceDemPass: 0,
        traversalTraceAzimuthIndex: 0);
    return new
    {
        description = "One-pixel obstacle immediately across a bilinear cell boundary.",
        csharp_hierarchy_maximum_slope = snapshot.PerDemSlopes[0][0],
        segments,
        trace = TraceValues(snapshot.TraversalTrace),
    };
}

static object[] TraceValues(
    IEnumerable<QuadTreeHorizonGenerator.TraversalStepDiagnostic> steps) =>
    steps.Select(step => (object)new
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
    }).ToArray();

static float? Finite(float value) => float.IsFinite(value) ? value : null;

static float[] Values(RaySegment segment) => new[]
{
    segment.StartPixel.X, segment.StartPixel.Y, segment.X0, segment.Y0,
    segment.A1, segment.A2, segment.A3, segment.A4,
    segment.B1, segment.B2, segment.B3, segment.B4,
    segment.SStart, segment.SEnd, segment.SStartChord,
    segment.PlanarToChordC1, segment.PlanarToChordC2, segment.PlanarToChordC3,
};

static ElevationMap CreateDem(
    int width,
    int height,
    Func<int, int, float> elevation,
    int? anchorColumn = null,
    int? anchorRow = null)
{
    const string projection =
        "+proj=stere +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs";
    var data = new float[height, width];
    for (int row = 0; row < height; row++)
        for (int column = 0; column < width; column++)
            data[row, column] = elevation(row, column);
    var transform = new[]
    {
        -(anchorColumn ?? width / 2) * 30.0,
        30.0,
        0.0,
        (anchorRow ?? height / 2) * 30.0,
        0.0,
        -30.0,
    };
    return new ElevationMap(data, projection, transform);
}
