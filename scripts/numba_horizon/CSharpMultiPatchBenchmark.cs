using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using moonlib;
using moonlib.horizon;
using Serilog;
using Serilog.Core;
using Serilog.Events;

var totalStopwatch = Stopwatch.StartNew();

if (args.Length < 6)
{
    Console.Error.WriteLine(
        "Usage: CSharpMultiPatchBenchmark <work-dir> <patch-count> " +
        "<observer-elevation-m> <gpu-concurrency> <segment-queue-size> " +
        "<primary-dem> [surrounding-dem ...]");
    return 2;
}

string workDirectory = Path.GetFullPath(args[0]);
int patchCount = int.Parse(args[1], CultureInfo.InvariantCulture);
float observerElevation = float.Parse(args[2], CultureInfo.InvariantCulture);
int gpuConcurrency = int.Parse(args[3], CultureInfo.InvariantCulture);
int segmentQueueSize = int.Parse(args[4], CultureInfo.InvariantCulture);
string[] sourceDemPaths = args[5..].Select(Path.GetFullPath).ToArray();

if (patchCount <= 1)
    throw new ArgumentOutOfRangeException(nameof(patchCount), "Use at least two patches for throughput measurement.");
if (gpuConcurrency <= 0)
    throw new ArgumentOutOfRangeException(nameof(gpuConcurrency));
if (segmentQueueSize <= 0)
    throw new ArgumentOutOfRangeException(nameof(segmentQueueSize));
if (Directory.Exists(workDirectory) && Directory.EnumerateFileSystemEntries(workDirectory).Any())
    throw new InvalidOperationException($"Benchmark work directory is not empty: {workDirectory}");

Directory.CreateDirectory(workDirectory);
string inputDirectory = Path.Combine(workDirectory, "inputs");
Directory.CreateDirectory(inputDirectory);

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
var availablePatches = QuadTreeHorizonGenerator.GeneratePatchList(dems[0]);
if (availablePatches.Count < patchCount)
    throw new ArgumentException(
        $"Primary DEM provides {availablePatches.Count} patches, fewer than requested {patchCount}.");
var patches = availablePatches.Take(patchCount).ToList();

var logSink = new CollectingSink();
Log.Logger = new LoggerConfiguration()
    .MinimumLevel.Information()
    .WriteTo.Sink(logSink)
    .CreateLogger();
Environment.SetEnvironmentVariable("QUADTREE_PIPELINE_PROFILE", "1");

using var memorySampler = new HostMemorySampler();
Console.WriteLine($"BENCHMARK_PID {Environment.ProcessId}");

memorySampler.Phase = "generator_initialization";
var generatorStopwatch = Stopwatch.StartNew();
using var generator = new QuadTreeHorizonGenerator(
    disableHierarchy: false,
    maxConcurrentGpuOps: gpuConcurrency,
    maxSegmentQueueSize: segmentQueueSize);
generatorStopwatch.Stop();

var accelerator = logSink.FindAccelerator();
var runs = new List<object>();
var outputHashesByRun = new List<string[]>();
foreach (string runName in new[] { "cold_fresh_pyramids", "warm_cached_pyramids" })
{
    string outputDirectory = Path.Combine(workDirectory, $"outputs-{runName}");
    Directory.CreateDirectory(outputDirectory);
    bool pyramidCachePresentBefore = copiedDemPaths.All(PyramidExists);
    var pyramidStateBefore = CapturePyramids(copiedDemPaths);
    int logStart = logSink.Count;

    memorySampler.Phase = runName;
    Console.WriteLine($"BENCHMARK_PHASE {runName}_start");
    var runStopwatch = Stopwatch.StartNew();
    await generator.GenerateHorizonsForPatches(
        outputDirectory,
        dems,
        patches,
        observerElevation,
        compressHorizons: true);
    runStopwatch.Stop();
    Console.WriteLine($"BENCHMARK_PHASE {runName}_end");
    memorySampler.Phase = "between_runs";

    var outputRecords = CaptureOutputs(outputDirectory, patches, observerElevation);
    string[] outputHashes = outputRecords.Select(record => record.sha256).ToArray();
    outputHashesByRun.Add(outputHashes);
    var patchProfiles = logSink.CapturePatchProfiles(logStart);
    var selectedTraversalProfiles = logSink.CaptureSelectedTraversalProfiles(logStart);
    if (patchProfiles.Count != patchCount)
        throw new InvalidOperationException(
            $"Expected {patchCount} pipeline profiles for {runName}, found {patchProfiles.Count}.");

    runs.Add(new
    {
        name = runName,
        timing_class = runName == "cold_fresh_pyramids"
            ? "Cold benchmark scope with copied DEMs and no pyramid cache files."
            : "Warm benchmark scope in the same process and generator, with pyramid cache files and reusable pipeline resources retained from the cold run.",
        pyramid_cache_present_before = pyramidCachePresentBefore,
        pyramids_before = pyramidStateBefore,
        pyramids_after = CapturePyramids(copiedDemPaths),
        elapsed_seconds = runStopwatch.Elapsed.TotalSeconds,
        patches_per_second = patchCount / runStopwatch.Elapsed.TotalSeconds,
        host_peak_working_set_bytes = memorySampler.Peak(runName),
        per_patch_profiles = patchProfiles,
        selected_traversal_profiles = selectedTraversalProfiles,
        stage_aggregates = AggregateProfiles(patchProfiles),
        outputs = outputRecords,
        combined_output_sha256 = CombinedOutputSha256(outputRecords),
    });
}

memorySampler.Phase = "reporting";
totalStopwatch.Stop();
bool outputsMatchBetweenRuns = outputHashesByRun.Count == 2 &&
    outputHashesByRun[0].SequenceEqual(outputHashesByRun[1], StringComparer.Ordinal);

var report = new
{
    schema_version = 1,
    report_kind = "csharp_phase_0_multi_patch_benchmark",
    captured_at_utc = DateTime.UtcNow.ToString("O"),
    baseline_commit = Environment.GetEnvironmentVariable("LUNARSCOUT_BASELINE_COMMIT"),
    process_id = Environment.ProcessId,
    configuration = new
    {
        patch_count = patchCount,
        patch_selection = "First N patches in GeneratePatchList row-major order.",
        patches = patches.Select(patch => new
        {
            patch.Index,
            tile_x = patch.TileX,
            tile_y = patch.TileY,
            patch_x = patch.PatchX,
            patch_y = patch.PatchY,
            width = 128,
            height = 128,
        }),
        azimuth_bins = 1440,
        observer_elevation_m = observerElevation,
        hierarchy_enabled = true,
        compression_requested = true,
        concurrent_gpu_operations = gpuConcurrency,
        segment_queue_size = segmentQueueSize,
        shared_subpatch_segment_cache = false,
        dem_pass_count = dems.Count,
    },
    accelerator,
    inputs = inputRecords,
    runtime = new
    {
        generator_initialization_seconds = generatorStopwatch.Elapsed.TotalSeconds,
        process_peak_working_set_bytes = Process.GetCurrentProcess().PeakWorkingSet64,
        sampled_host_peak_working_set_bytes = memorySampler.OverallPeak,
        total_process_elapsed_seconds = totalStopwatch.Elapsed.TotalSeconds,
        host_memory_sample_interval_ms = HostMemorySampler.SampleIntervalMilliseconds,
    },
    runs,
    repeatability = new
    {
        cold_and_warm_output_hashes_equal = outputsMatchBetweenRuns,
    },
    timing_notes = new[]
    {
        "Each run elapsed scope is GenerateHorizonsForPatches, including pyramid build/load, segment generation, GPU work, compression, and writes.",
        "The cold run excludes input copying, input hashing, DEM loading, generator construction/kernel compilation, and output hashing.",
        "The warm run uses the same process and generator after the cold run; pyramid files and reusable pipeline buffers/streams may be reused.",
        "Stage totals overlap across producer and GPU workers and must not be summed to estimate wall time.",
        "kernel_launch_sec measures asynchronous enqueue time; stream_sync_sec includes waiting for queued CUDA work to complete.",
    },
    limitations = new[]
    {
        "The production shared SubpatchSegmentCache is hard-coded off, so neighboring patches recompute segment centers.",
        "The first-N selection is a bounded contiguous row-major batch, not a whole-region throughput claim.",
        "GPU memory is added by the external Python sampler; this raw C# report does not measure device memory itself.",
    },
};

string reportPath = Path.Combine(workDirectory, "multi-patch-benchmark-report.json");
File.WriteAllText(
    reportPath,
    JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }) + "\n");
Console.WriteLine($"BENCHMARK_REPORT {reportPath}");
await Log.CloseAndFlushAsync();
return 0;

static bool PyramidExists(string demPath) => File.Exists(Path.ChangeExtension(demPath, ".pyr.bin"));

static object[] CapturePyramids(IReadOnlyList<string> demPaths) =>
    demPaths.Select((demPath, index) =>
    {
        string path = Path.ChangeExtension(demPath, ".pyr.bin");
        return (object)new
        {
            dem_index = index,
            path,
            exists = File.Exists(path),
            size_bytes = File.Exists(path) ? new FileInfo(path).Length : 0,
            sha256 = File.Exists(path) ? Sha256(path) : null,
            last_write_utc = File.Exists(path) ? File.GetLastWriteTimeUtc(path).ToString("O") : null,
        };
    }).ToArray();

static List<OutputRecord> CaptureOutputs(
    string outputDirectory,
    IReadOnlyList<QuadTreeHorizonGenerator.PatchDescriptor> patches,
    float observerElevation)
{
    var store = new HorizonTileStore(outputDirectory);
    return patches.Select(patch =>
    {
        string path = store.FindExistingPath(patch.TileY, patch.TileX, observerElevation)
            ?? throw new FileNotFoundException(
                $"Missing horizon output for tile ({patch.TileX}, {patch.TileY}).");
        return new OutputRecord(
            patch.Index,
            patch.TileX,
            patch.TileY,
            path,
            new FileInfo(path).Length,
            Sha256(path));
    }).OrderBy(record => record.patch_index).ToList();
}

static string CombinedOutputSha256(IReadOnlyList<OutputRecord> outputs)
{
    string manifest = string.Join(
        "\n",
        outputs.Select(record =>
            $"{record.patch_index}:{record.tile_x}:{record.tile_y}:{record.sha256}")) + "\n";
    return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(manifest))).ToLowerInvariant();
}

static Dictionary<string, object> AggregateProfiles(IReadOnlyList<Dictionary<string, object>> profiles)
{
    string[] excluded =
    {
        "patch_index", "tile_x", "tile_y", "queue_depth_after_enqueue",
        "queue_depth_on_dequeue", "active_gpu_workers", "active_streams",
    };
    return profiles[0].Keys
        .Where(key => !excluded.Contains(key, StringComparer.Ordinal))
        .OrderBy(key => key, StringComparer.Ordinal)
        .ToDictionary(
            key => key,
            key =>
            {
                double[] values = profiles.Select(profile => Convert.ToDouble(profile[key], CultureInfo.InvariantCulture)).ToArray();
                return (object)new
                {
                    samples = values.Length,
                    total_seconds = values.Sum(),
                    average_seconds = values.Average(),
                    minimum_seconds = values.Min(),
                    maximum_seconds = values.Max(),
                };
            },
            StringComparer.Ordinal);
}

static string Sha256(string path)
{
    using var stream = File.OpenRead(path);
    return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
}

sealed record OutputRecord(
    int patch_index,
    int tile_x,
    int tile_y,
    string path,
    long size_bytes,
    string sha256);

sealed class HostMemorySampler : IDisposable
{
    public const int SampleIntervalMilliseconds = 25;
    private readonly ConcurrentDictionary<string, long> _peaks = new(StringComparer.Ordinal);
    private readonly CancellationTokenSource _cancellation = new();
    private readonly Task _task;
    private string _phase = "startup";
    private long _overallPeak;

    public HostMemorySampler()
    {
        _task = Task.Run(SampleAsync);
    }

    public string Phase
    {
        get => Volatile.Read(ref _phase);
        set => Volatile.Write(ref _phase, value);
    }

    public long OverallPeak => Volatile.Read(ref _overallPeak);
    public long Peak(string phase) => _peaks.TryGetValue(phase, out long peak) ? peak : 0;

    private async Task SampleAsync()
    {
        using var process = Process.GetCurrentProcess();
        while (!_cancellation.IsCancellationRequested)
        {
            process.Refresh();
            long workingSet = process.WorkingSet64;
            InterlockedExtensions.Max(ref _overallPeak, workingSet);
            _peaks.AddOrUpdate(Phase, workingSet, (_, current) => Math.Max(current, workingSet));
            try
            {
                await Task.Delay(SampleIntervalMilliseconds, _cancellation.Token);
            }
            catch (OperationCanceledException)
            {
                break;
            }
        }
    }

    public void Dispose()
    {
        _cancellation.Cancel();
        try { _task.GetAwaiter().GetResult(); } catch (OperationCanceledException) { }
        _cancellation.Dispose();
    }
}

static class InterlockedExtensions
{
    public static void Max(ref long target, long value)
    {
        long current;
        do
        {
            current = Volatile.Read(ref target);
            if (value <= current)
                return;
        } while (Interlocked.CompareExchange(ref target, value, current) != current);
    }
}

sealed class CollectingSink : ILogEventSink
{
    private readonly object _lock = new();
    private readonly List<LogEvent> _events = new();

    public int Count
    {
        get { lock (_lock) return _events.Count; }
    }

    public void Emit(LogEvent logEvent)
    {
        lock (_lock) _events.Add(logEvent);
    }

    public object FindAccelerator()
    {
        LogEvent logEvent;
        lock (_lock)
        {
            logEvent = _events.LastOrDefault(item =>
                item.MessageTemplate.Text.StartsWith("QuadTreeHorizonGenerator using device:", StringComparison.Ordinal))
                ?? throw new InvalidOperationException("Accelerator selection log event was not captured.");
        }
        return new
        {
            name = ScalarString(logEvent, "DeviceName"),
            type = ScalarString(logEvent, "AcceleratorType"),
        };
    }

    public List<Dictionary<string, object>> CapturePatchProfiles(int startIndex)
    {
        LogEvent[] events;
        lock (_lock) events = _events.Skip(startIndex).ToArray();
        return events
            .Where(item => item.MessageTemplate.Text.StartsWith("PipelineProfile patch_index=", StringComparison.Ordinal))
            .Select(item => new Dictionary<string, object>(StringComparer.Ordinal)
            {
                ["patch_index"] = ScalarInt(item, "PatchIndex"),
                ["tile_x"] = ScalarInt(item, "TileX"),
                ["tile_y"] = ScalarInt(item, "TileY"),
                ["queue_depth_after_enqueue"] = ScalarInt(item, "QueueDepthAfterEnqueue"),
                ["queue_depth_on_dequeue"] = ScalarInt(item, "QueueDepthOnDequeue"),
                ["active_gpu_workers"] = ScalarInt(item, "ActiveGpuWorkers"),
                ["active_streams"] = ScalarInt(item, "ActiveStreams"),
                ["segment_generation_seconds"] = ScalarDouble(item, "SegmentGenerationSec"),
                ["patch_enqueue_wait_seconds"] = ScalarDouble(item, "PatchEnqueueWaitSec"),
                ["buffer_wait_seconds"] = ScalarDouble(item, "WaitBufferSec"),
                ["stream_wait_seconds"] = ScalarDouble(item, "WaitStreamSec"),
                ["buffer_reset_seconds"] = ScalarDouble(item, "BufferResetSec"),
                ["segment_upload_seconds"] = ScalarDouble(item, "SegmentUploadSec"),
                ["kernel_launch_seconds"] = ScalarDouble(item, "KernelLaunchSec"),
                ["stream_sync_seconds"] = ScalarDouble(item, "StreamSyncSec"),
                ["copy_back_seconds"] = ScalarDouble(item, "CopyBackSec"),
                ["degree_conversion_seconds"] = ScalarDouble(item, "ConvertSec"),
                ["compression_and_write_seconds"] = ScalarDouble(item, "WriteSec"),
                ["gpu_worker_total_seconds"] = ScalarDouble(item, "GpuWorkerTotalSec"),
            })
            .OrderBy(profile => Convert.ToInt32(profile["patch_index"], CultureInfo.InvariantCulture))
            .ToList();
    }

    public List<Dictionary<string, object>> CaptureSelectedTraversalProfiles(int startIndex)
    {
        LogEvent[] events;
        lock (_lock) events = _events.Skip(startIndex).ToArray();
        return events
            .Where(item => item.MessageTemplate.Text.StartsWith("SelectedTraversalProfile ", StringComparison.Ordinal))
            .Select(item => new Dictionary<string, object>(StringComparer.Ordinal)
            {
                ["patch_index"] = ScalarInt(item, "PatchIndex"),
                ["tile_x"] = ScalarInt(item, "TileX"),
                ["tile_y"] = ScalarInt(item, "TileY"),
                ["pixel"] = ScalarInt(item, "Pixel"),
                ["azimuth"] = ScalarInt(item, "Azimuth"),
                ["dem_pass"] = ScalarInt(item, "DemPass"),
                ["iterations"] = ScalarLong(item, "Iterations"),
                ["descents"] = ScalarLong(item, "Descents"),
                ["culled_blocks"] = ScalarLong(item, "CulledBlocks"),
                ["nodata_skips"] = ScalarLong(item, "NodataSkips"),
                ["out_of_bounds"] = ScalarLong(item, "OutOfBounds"),
                ["level0_samples"] = ScalarLong(item, "Level0Samples"),
                ["maximum_level"] = ScalarInt(item, "MaximumLevel"),
                ["group_x"] = ScalarInt(item, "GroupX"),
                ["group_y"] = ScalarInt(item, "GroupY"),
                ["group_z"] = ScalarInt(item, "GroupZ"),
            })
            .OrderBy(profile => Convert.ToInt32(profile["patch_index"], CultureInfo.InvariantCulture))
            .ThenBy(profile => Convert.ToInt32(profile["dem_pass"], CultureInfo.InvariantCulture))
            .ToList();
    }

    private static object Scalar(LogEvent logEvent, string propertyName) =>
        logEvent.Properties.TryGetValue(propertyName, out LogEventPropertyValue? value) && value is ScalarValue scalar
            ? scalar.Value ?? throw new InvalidOperationException($"Log property {propertyName} is null.")
            : throw new InvalidOperationException($"Log property {propertyName} is missing or not scalar.");

    private static string ScalarString(LogEvent logEvent, string propertyName) =>
        Convert.ToString(Scalar(logEvent, propertyName), CultureInfo.InvariantCulture)
        ?? throw new InvalidOperationException($"Log property {propertyName} is not a string.");
    private static int ScalarInt(LogEvent logEvent, string propertyName) =>
        Convert.ToInt32(Scalar(logEvent, propertyName), CultureInfo.InvariantCulture);
    private static long ScalarLong(LogEvent logEvent, string propertyName) =>
        Convert.ToInt64(Scalar(logEvent, propertyName), CultureInfo.InvariantCulture);
    private static double ScalarDouble(LogEvent logEvent, string propertyName) =>
        Convert.ToDouble(Scalar(logEvent, propertyName), CultureInfo.InvariantCulture);
}
