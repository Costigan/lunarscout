using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using moonlib;
using moonlib.horizon;

var totalStopwatch = Stopwatch.StartNew();

if (args.Length < 5)
{
    Console.Error.WriteLine(
        "Usage: CSharpBaselineCapture <work-dir> <tile-x> <tile-y> " +
        "<observer-elevation-m> <primary-dem> [surrounding-dem ...]");
    return 2;
}

string workDirectory = Path.GetFullPath(args[0]);
int tileX = int.Parse(args[1]);
int tileY = int.Parse(args[2]);
float observerElevation = float.Parse(
    args[3], System.Globalization.CultureInfo.InvariantCulture);
string[] sourceDemPaths = args[4..].Select(Path.GetFullPath).ToArray();

if (Directory.Exists(workDirectory) && Directory.EnumerateFileSystemEntries(workDirectory).Any())
    throw new InvalidOperationException($"Capture work directory is not empty: {workDirectory}");

Directory.CreateDirectory(workDirectory);
string inputDirectory = Path.Combine(workDirectory, "inputs");
string outputDirectory = Path.Combine(workDirectory, "outputs");
Directory.CreateDirectory(inputDirectory);
Directory.CreateDirectory(outputDirectory);

var inputRecords = new List<object>();
var copiedDemPaths = new List<string>();
for (int index = 0; index < sourceDemPaths.Length; index++)
{
    string source = sourceDemPaths[index];
    if (!File.Exists(source))
        throw new FileNotFoundException("DEM does not exist.", source);

    string destination = Path.Combine(inputDirectory, $"dem-{index:D2}.tif");
    string sourceSha256 = Sha256(source);
    File.Copy(source, destination, overwrite: false);
    string copiedSha256 = Sha256(destination);
    if (!string.Equals(sourceSha256, copiedSha256, StringComparison.Ordinal))
        throw new InvalidOperationException($"Copied DEM hash differs from source: {source}");

    copiedDemPaths.Add(destination);
    inputRecords.Add(new
    {
        index,
        source_path = source,
        source_size_bytes = new FileInfo(source).Length,
        sha256 = sourceSha256,
        copied_path = destination,
    });
}

MoonlibBridge.EnsureGdalInitialized();
var dems = copiedDemPaths.Select(path => new ElevationMap(path)).ToList();
var patch = QuadTreeHorizonGenerator.GeneratePatchList(dems[0]).SingleOrDefault(
    candidate => candidate.TileX == tileX && candidate.TileY == tileY)
    ?? throw new ArgumentException(
        $"Primary DEM has no 128x128 patch at tile ({tileX}, {tileY}).");

Environment.SetEnvironmentVariable("QUADTREE_PIPELINE_PROFILE", "1");
var process = Process.GetCurrentProcess();
var generationStopwatch = Stopwatch.StartNew();

using (var generator = new QuadTreeHorizonGenerator(
    disableHierarchy: false,
    maxConcurrentGpuOps: 1,
    maxSegmentQueueSize: 1))
{
    await generator.GenerateHorizonsForPatches(
        outputDirectory,
        dems,
        new List<QuadTreeHorizonGenerator.PatchDescriptor> { patch },
        observerElevation,
        compressHorizons: true);
}

generationStopwatch.Stop();
process.Refresh();

var store = new HorizonTileStore(outputDirectory);
string outputPath = store.FindExistingPath(tileY, tileX, observerElevation)
    ?? throw new FileNotFoundException("Generator did not produce the expected horizon tile.");
float[] degrees = HorizonFile.ReadHorizonFile(outputPath);

int finiteCount = 0;
int nonFiniteCount = 0;
float minimum = float.PositiveInfinity;
float maximum = float.NegativeInfinity;
double sum = 0.0;
foreach (float value in degrees)
{
    if (!float.IsFinite(value))
    {
        nonFiniteCount++;
        continue;
    }
    finiteCount++;
    minimum = Math.Min(minimum, value);
    maximum = Math.Max(maximum, value);
    sum += value;
}

var pyramidRecords = copiedDemPaths.Select((demPath, index) =>
{
    string pyramidPath = Path.ChangeExtension(demPath, ".pyr.bin");
    return new
    {
        dem_index = index,
        path = pyramidPath,
        exists = File.Exists(pyramidPath),
        size_bytes = File.Exists(pyramidPath) ? new FileInfo(pyramidPath).Length : 0,
        sha256 = File.Exists(pyramidPath) ? Sha256(pyramidPath) : null,
    };
}).ToArray();

totalStopwatch.Stop();

var report = new
{
    schema_version = 1,
    captured_at_utc = DateTime.UtcNow.ToString("O"),
    baseline_commit = Environment.GetEnvironmentVariable("LUNARSCOUT_BASELINE_COMMIT"),
    configuration = new
    {
        tile_x = tileX,
        tile_y = tileY,
        patch_width = 128,
        patch_height = 128,
        azimuth_bins = 1440,
        observer_elevation_m = observerElevation,
        hierarchy_enabled = true,
        compression_requested = true,
        concurrent_gpu_operations = 1,
        segment_queue_size = 1,
    },
    inputs = inputRecords,
    pyramids = pyramidRecords,
    output = new
    {
        path = outputPath,
        size_bytes = new FileInfo(outputPath).Length,
        sha256 = Sha256(outputPath),
        value_count = degrees.Length,
        finite_count = finiteCount,
        non_finite_count = nonFiniteCount,
        minimum_degrees = finiteCount > 0 ? minimum : null as float?,
        maximum_degrees = finiteCount > 0 ? maximum : null as float?,
        mean_degrees = finiteCount > 0 ? sum / finiteCount : null as double?,
    },
    runtime = new
    {
        generation_elapsed_seconds = generationStopwatch.Elapsed.TotalSeconds,
        total_process_elapsed_seconds = totalStopwatch.Elapsed.TotalSeconds,
        process_peak_working_set_bytes = process.PeakWorkingSet64,
        generation_timing_scope =
            "Generator construction, fresh pyramid construction, segment generation, " +
            "GPU traversal, compression, and output write. Excludes input copying, " +
            "input hashing, DEM loading, output hashing, and output statistics.",
    },
    limitations = new[]
    {
        "The capture records process peak host working set but not peak GPU memory.",
        "One patch does not establish sustained multi-patch throughput.",
        "The copied DEMs force pyramid reconstruction and do not use external pyramid caches.",
    },
};

string reportPath = Path.Combine(workDirectory, "baseline-report.json");
File.WriteAllText(
    reportPath,
    JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }) + "\n");
Console.WriteLine(reportPath);
return 0;

static string Sha256(string path)
{
    using var stream = File.OpenRead(path);
    return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
}
