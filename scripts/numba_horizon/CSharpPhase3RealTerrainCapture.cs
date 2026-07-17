using System.Text.Json;
using moonlib;
using moonlib.horizon;

if (args.Length != 2)
{
    Console.Error.WriteLine(
        "Usage: CSharpPhase3RealTerrainCapture <input-dem.tif> <output-json>");
    return 2;
}

string inputPath = Path.GetFullPath(args[0]);
string outputPath = Path.GetFullPath(args[1]);
MoonlibBridge.EnsureGdalInitialized();
var dem = new ElevationMap(inputPath);
using var generator = new QuadTreeHorizonGenerator(
    disableHierarchy: false,
    maxConcurrentGpuOps: 1,
    maxSegmentQueueSize: 1);

const int tileColumn = 240;
const int tileRow = 240;
const int tileWidth = 16;
const int tileHeight = 16;
const int azimuthCount = 64;
const int subpatchSize = 8;
const float maximumDistanceMeters = 5000f;
const float observerElevationMeters = 0f;

var snapshot = generator.CalculateSubpatchRaySegmentsForDiagnostics(
    new List<ElevationMap> { dem },
    tileColumn,
    tileRow,
    tileWidth,
    tileHeight,
    azimuthCount,
    maximumDistanceMeters,
    observerElevationMeters,
    subpatchSize);

var report = new
{
    schema_version = 1,
    input_path = inputPath,
    input_width = dem.Width,
    input_height = dem.Height,
    projection = dem.Proj4,
    geo_transform = dem.GeoTransform,
    selected_accelerator_name = generator.SelectedAcceleratorName,
    selected_accelerator_type = generator.SelectedAcceleratorType.ToString(),
    configuration = new
    {
        tile_column = tileColumn,
        tile_row = tileRow,
        tile_width = tileWidth,
        tile_height = tileHeight,
        azimuth_count = azimuthCount,
        subpatch_size = subpatchSize,
        maximum_distance_m = maximumDistanceMeters,
        observer_elevation_m = observerElevationMeters,
    },
    grid_convergence = new[]
    {
        snapshot.GridConvergence.GammaCenter,
        snapshot.GridConvergence.DGammaDx,
        snapshot.GridConvergence.DGammaDy,
    },
    centers = snapshot.Centers.Select(center => new[]
    {
        center.Index,
        center.GridRow,
        center.GridColumn,
        center.RequestedCenterColumn,
        center.RequestedCenterRow,
        center.SegmentCenterColumn,
        center.SegmentCenterRow,
    }),
    segments = snapshot.Segments.Select(segment => new[]
    {
        segment.StartPixel.X,
        segment.StartPixel.Y,
        segment.X0,
        segment.Y0,
        segment.A1,
        segment.A2,
        segment.A3,
        segment.A4,
        segment.B1,
        segment.B2,
        segment.B3,
        segment.B4,
        segment.SStart,
        segment.SEnd,
        segment.SStartChord,
        segment.PlanarToChordC1,
        segment.PlanarToChordC2,
        segment.PlanarToChordC3,
    }),
};

Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
File.WriteAllText(
    outputPath,
    JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true })
    + Environment.NewLine);
return 0;
