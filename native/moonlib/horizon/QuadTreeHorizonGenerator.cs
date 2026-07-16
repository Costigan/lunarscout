using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using moonlib.math;
using OSGeo.GDAL;
using Serilog;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Channels;
using System.Linq;

namespace moonlib.horizon
{
    /// <summary>
    /// Delegate for reporting diagnostic horizon buffers.
    /// </summary>
    /// <param name="bufferType">The type of buffer being reported (see HorizonDiagnosticsBuffer).</param>
    /// <param name="angles">A read-only span of horizon angles (in degrees).</param>
    public delegate void HorizonDiagnosticsCallback(HorizonBufferType bufferType, HorizonAngles angles);
    /// <summary>
    /// Enumerates the types of diagnostic buffers that can be reported via the diagnostics callback.
    /// </summary>
    public enum HorizonBufferType
    {
        FullHorizon,// Buffer containing the full, merged horizon angles.
        NearField,  // Buffer containing the near-field (merged, close-range) horizon angles.
        FarField,   // Buffer containing the far-field (quadtree-only) horizon angles.
        DEM1,       // Buffer containing the horizon angles for DEM 1 (first DEM pass).
        DEM2,       // Buffer containing the horizon angles for DEM 2 (second DEM pass).
        DEM3,       // Buffer containing the horizon angles for DEM 3 (third DEM pass).
        DEM4,       // Buffer containing the horizon angles for DEM 4 (fourth DEM pass).
        DEM5,       // Buffer containing the horizon angles for DEM 5 (fifth DEM pass).
        DEMN        // Buffer containing the horizon angles for DEM N (any DEM beyond 5).
    }

    /// <summary>
    /// Contains multiple sets of polynomial coefficients for subpatches within a 128x128 tile.
    /// Each subpatch has its own fitted polynomials to reduce approximation error.
    /// </summary>
    public struct SubpatchPolynomials
    {
        public int SubpatchSize;     // Size of each subpatch (8, 16, 32, 64 or 128)
        public int NumSubpatches;    // Total number of subpatches (e.g., 16 for 32x32 subpatches)
        public ArrayView<RaySegment> Segments; // All segments for all subpatches, layout: [Azimuth][Subpatch][DEM]
        
        public SubpatchPolynomials(int subpatchSize, int numSubpatches, ArrayView<RaySegment> segments)
        {
            if (subpatchSize != 2 && subpatchSize != 4 && subpatchSize != 8 && subpatchSize != 16 &&
                subpatchSize != 32 && subpatchSize != 64 && subpatchSize != 128)
                throw new ArgumentException("Subpatch size must be 2, 4, 8, 16, 32, 64 or 128", nameof(subpatchSize));
            if (128 % subpatchSize != 0)
                throw new ArgumentException("128 must be evenly divisible by subpatch size", nameof(subpatchSize));
                
            SubpatchSize = subpatchSize;
            NumSubpatches = numSubpatches;
            Segments = segments;
        }
    }

    public struct HorizonAngles
    {
        private const float RadToDeg = 180f / MathF.PI;

        public static HorizonAngles FromDegrees(float[] degrees) => new HorizonAngles(degrees);
        public static HorizonAngles FromRadians(float[] radians)
        {
            ArgumentNullException.ThrowIfNull(radians);
            var buffer = radians;
            for (int i = 0; i < buffer.Length; i++)
                buffer[i] = ConvertValue(buffer[i]);
            return new HorizonAngles(buffer);
        }

        public static HorizonAngles FromSlopes(float[] slopes)
        {
            ArgumentNullException.ThrowIfNull(slopes);
            var buffer = slopes;
            for (int i = 0; i < buffer.Length; i++)
                buffer[i] = ConvertSlopeValue(buffer[i]);
            return new HorizonAngles(buffer);
        }

        private HorizonAngles(float[] degrees) => Degrees = degrees ?? throw new ArgumentNullException(nameof(degrees));

        public float[] Degrees { get; }

        public int Length => Degrees.Length;

        public HorizonAngles Clone() => new HorizonAngles((float[])Degrees.Clone());

        public static void ConvertRadiansToDegreesInPlace(float[] buffer)
        {
            ArgumentNullException.ThrowIfNull(buffer);
            for (int i = 0; i < buffer.Length; i++)
                buffer[i] = ConvertValue(buffer[i]);
        }

        private static float ConvertValue(float value)
        {
            if (float.IsNaN(value) || float.IsPositiveInfinity(value) || float.IsNegativeInfinity(value))
                return value;
            return value * RadToDeg;
        }

        private static float ConvertSlopeValue(float value)
        {
            if (float.IsNaN(value) || float.IsPositiveInfinity(value) || float.IsNegativeInfinity(value))
                return value;
            return MathF.Atan(value) * RadToDeg;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct RaySegment
    {
        public Vector2 StartPixel;
        public int DemId; // Index of the DEM this segment belongs to

        // Cubic ray representation in pixel space: x(s) = x0 + a1*s + a2*s^2 + a3*s^3; y(s) = y0 + b1*s + b2*s^2 + b3*s^3
        // s is expressed in kilometers along the ray. When unset, HasCubic == 0 and linear fields are used.
        public float X0;
        public float Y0;
        public float A1;
        public float A2;
        public float A3;
        public float A4;
        public float B1;
        public float B2;
        public float B3;
        public float B4;
        public float SStart; // kilometers where this segment starts (usually == StartGroundDist)
        public float SEnd;   // kilometers where this segment ends
        public float SStartChord; // true chord length at SStart (kilometers)
        public float PlanarToChordC1;
        public float PlanarToChordC2;
        public float PlanarToChordC3;
    }

    public struct ProjectionParams
    {
        public float R;
        public float Lat0;
        public float Lon0;
        public float K0;
        public float FalseEasting;
        public float FalseNorthing;
    }

    public readonly struct MapParams
    {
        public readonly float R;
        public readonly float K0;
        public readonly float Fe;
        public readonly float Fn;
        public readonly float InvDet;
        public readonly float T0;
        public readonly float T1;
        public readonly float T2;
        public readonly float T3;
        public readonly float T4;
        public readonly float T5;

        public float X0 => T0;
        public float Y0 => T3;

        public MapParams(float r, float k0, float fe, float fn, float invDet, float t0, float t1, float t2, float t3, float t4, float t5)
        {
            R = r;
            K0 = k0;
            Fe = fe;
            Fn = fn;
            InvDet = invDet;
            T0 = t0;
            T1 = t1;
            T2 = t2;
            T3 = t3;
            T4 = t4;
            T5 = t5;
        }
        public (float px, float py) CRSToPixel(float cX, float cY)
        {
            float det = T1 * T5 - T2 * T4;
            float dx = cX - T0;
            float dy = cY - T3;
            float col = (T5 * dx - T2 * dy) / det;
            float row = (-T4 * dx + T1 * dy) / det;
            return (col, row);
        }

        public (float cx, float cy) PixelToCRS(float pX, float pY)
        {
            float x = T0 + T1 * pX + T2 * pY;
            float y = T3 + T4 * pX + T5 * pY;
            return (x, y);
        }
    }

    public struct LevelInfo
    {
        public int Offset;
        public int Width;
        public int Height;
        public float CellSizeX; // Needed for heuristic level selection? Or derived?
        public float CellSizeY;

        public override string ToString() => $"LevelInfo: [Offset: {Offset}, Width: {Width}, Height: {Height}, CellSizeX: {CellSizeX}, CellSizeY: {CellSizeY}]";
    }

    public struct PyramidView
    {
        public ArrayView<float> DataLevel0;
        public ArrayView<float> DataMips;
        public ArrayView<LevelInfo> Infos;
        public MapParams Map;
        public ProjectionParams Proj;
        public int Levels;
    }

    public struct PixelBounds
    {
        public int Width;
        public int Height;
        public float MinX => 0;
        public float MinY => 0;
        public float MaxX => Width;
        public float MaxY => Height;
    }

    public readonly struct KernelParams
    {
        public readonly float ObserverElevation;
        public readonly float MinTraverseDistanceKm;
        public readonly int DebugAzimuthIndex;

        // Grid Convergence compensation for Compact Mode
        // GammaCenter: Grid Convergence at tile center (radians)
        // DGammaDx/DGammaDy: Gradient of gamma per pixel in X and Y directions (radians/pixel)
        public readonly float GammaCenter;
        public readonly float DGammaDx;
        public readonly float DGammaDy;
        public readonly int DebugFlags;
        public readonly int PrimaryWidth;
        public readonly int PrimaryHeight;

        public KernelParams(float observerElevation, float minTraverseDistanceKm, int debugAzimuthIndex)
            : this(observerElevation, minTraverseDistanceKm, debugAzimuthIndex, 0f, 0f, 0f) { }

        public KernelParams(float observerElevation, float minTraverseDistanceKm, int debugAzimuthIndex,
            float gammaCenter, float dGammaDx, float dGammaDy)
            : this(observerElevation, minTraverseDistanceKm, debugAzimuthIndex, gammaCenter, dGammaDx, dGammaDy, 0) { }

        public KernelParams(float observerElevation, float minTraverseDistanceKm, int debugAzimuthIndex,
            float gammaCenter, float dGammaDx, float dGammaDy, int debugFlags)
            : this(observerElevation, minTraverseDistanceKm, debugAzimuthIndex, gammaCenter, dGammaDx, dGammaDy, debugFlags, 0, 0) { }

        public KernelParams(float observerElevation, float minTraverseDistanceKm, int debugAzimuthIndex,
            float gammaCenter, float dGammaDx, float dGammaDy, int debugFlags, int primaryWidth, int primaryHeight)
        {
            ObserverElevation = observerElevation;
            MinTraverseDistanceKm = minTraverseDistanceKm;
            DebugAzimuthIndex = debugAzimuthIndex;
            GammaCenter = gammaCenter;
            DGammaDx = dGammaDx;
            DGammaDy = dGammaDy;
            DebugFlags = debugFlags;
            PrimaryWidth = primaryWidth;
            PrimaryHeight = primaryHeight;
        }
    }

    public class Pyramid : IDisposable
    {
        public MemoryBuffer1D<float, Stride1D.Dense>? DataLevel0;
        public MemoryBuffer1D<float, Stride1D.Dense>? DataMips;
        public MemoryBuffer1D<LevelInfo, Stride1D.Dense>? Infos;
        public LevelInfo[]? CpuInfos;
        public MapParams Map;
        public ProjectionParams Proj;
        public ElevationMap? SourceDem;

        public void Dispose()
        {
            DataLevel0?.Dispose();
            DataMips?.Dispose();
            Infos?.Dispose();
        }
    }

    public struct ProjectionParamsDouble
    {
        public double R;
        public double Lat0;
        public double Lon0;
        public double K0;
        public double FalseEasting;
        public double FalseNorthing;
    }

    public class QuadTreeHorizonGenerator : IDisposable
    {
        public const int DefaultMaxConcurrentGpuOps = 4; // Default concurrent GPU operations used for stream pool sizing and pipeline worker count.
        public const int DefaultSegmentQueueSize = 6;
        private static readonly bool UseDemElevationChordCorrection = false;
        private readonly Context _context;
        private readonly Accelerator _accelerator;
        private readonly int _maxConcurrentGpuOps;
        private readonly int _maxSegmentQueueSize;

        // Set to true to disable hierarchical filtering (always use Level 0)
        private readonly bool _disableHierarchy;
        private readonly bool _enableNearFieldReferenceMerge;
        private readonly float _nearFieldClampMeters;
        private readonly bool _forceFixedStepDebug;
        private readonly bool _enablePipelineProfiling;
        private readonly PipelineProfiler _pipelineProfiler = new();
        private readonly int _pipelineSubpatchSize;
        private const float DEBUG_FIXED_STEP_METERS = 1.2f;
        private const float NEARFIELD_BORDER_MARGIN_METERS = 5f;
        internal const float METERS_TO_KILOMETERS = 1f / 1000f;
        internal const float KILOMETERS_TO_METERS = 1000f;
        internal const double METERS_TO_KILOMETERS_D = 1.0 / 1000.0;
        internal const double KILOMETERS_TO_METERS_D = 1000.0;
        private const float DEBUG_FIXED_STEP_KM = DEBUG_FIXED_STEP_METERS * METERS_TO_KILOMETERS;

        /// <summary>
        /// Optional callback for reporting diagnostic horizon buffers. If set, will be called when diagnostic buffers are available.
        /// </summary>
        public HorizonDiagnosticsCallback? DiagnosticsCallback { get; set; }

        internal AcceleratorType SelectedAcceleratorType => _accelerator.AcceleratorType;
        internal string SelectedAcceleratorName => _accelerator.Name;
        
        /// <summary>
        /// Buffer pool for managing GPU memory reuse in pipeline processing.
        /// </summary>
        private BufferPool? _bufferPool;
        
        /// <summary>
        /// Pre-compiled GPU subpatch kernel for ray casting - loaded once and reused for all calls.
        /// Note: This is the stream-based version, first parameter is AcceleratorStream
        /// </summary>
#if QUADTREE_TRAVERSAL_PROFILE
        private Action<AcceleratorStream, Index2D, PyramidView, PyramidView, ArrayView<float>,
                      ArrayView<RaySegment>, int, int, int, int, int, int, int, KernelParams, ArrayView<float>, ArrayView<long>>? _subpatchKernel;
#else
        private Action<AcceleratorStream, Index2D, PyramidView, PyramidView, ArrayView<float>,
                      ArrayView<RaySegment>, int, int, int, int, int, int, int, KernelParams, ArrayView<float>>? _subpatchKernel;
#endif
                      
        /// <summary>
        /// Pool of GPU streams to enable truly concurrent kernel launches.
        /// Each stream provides an independent GPU command queue.
        /// </summary>
        private ConcurrentStack<AcceleratorStream>? _streamPool;

        private static class PipelineStageNames
        {
            public const string SegmentGeneration = "segment_generation";
            public const string PatchEnqueueWait = "patch_enqueue_wait";
            public const string BufferAcquire = "buffer_acquire";
            public const string StreamAcquire = "stream_acquire";
            public const string BufferResetHorizonsAccum = "buffer_reset_horizons_accum";
            public const string SegmentUpload = "segment_upload";
            public const string KernelLaunchTotal = "kernel_launch_total";
            public const string StreamSync = "stream_sync";
            public const string OutputCopyToHost = "output_copy_to_host";
            public const string RadiansToDegrees = "radians_to_degrees";
            public const string CompressAndWrite = "compress_and_write";
            public const string TotalGpuWorkerPatchWall = "total_gpu_worker_patch_wall";
        }

        private sealed class StageAggregate
        {
            private long _totalTicks;
            private long _count;
            private long _maxTicks;

            public void Add(long elapsedTicks)
            {
                Interlocked.Add(ref _totalTicks, elapsedTicks);
                Interlocked.Increment(ref _count);

                long currentMax;
                do
                {
                    currentMax = Volatile.Read(ref _maxTicks);
                    if (elapsedTicks <= currentMax)
                        break;
                } while (Interlocked.CompareExchange(ref _maxTicks, elapsedTicks, currentMax) != currentMax);
            }

            public long TotalTicks => Volatile.Read(ref _totalTicks);
            public long Count => Volatile.Read(ref _count);
            public long MaxTicks => Volatile.Read(ref _maxTicks);
        }

        private sealed class PipelineProfiler
        {
            private readonly ConcurrentDictionary<string, StageAggregate> _stages = new();

            public void Record(string stageName, long elapsedTicks)
            {
                _stages.GetOrAdd(stageName, static _ => new StageAggregate()).Add(elapsedTicks);
            }

            public IReadOnlyList<KeyValuePair<string, StageAggregate>> Snapshot() =>
                _stages.OrderBy(kvp => kvp.Key, StringComparer.Ordinal).ToArray();
        }

        private readonly struct LaunchPatchProfile
        {
            public long WaitStreamTicks { get; init; }
            public long BufferResetTicks { get; init; }
            public long SegmentUploadTicks { get; init; }
            public long KernelLaunchTicks { get; init; }
            public long StreamSyncTicks { get; init; }
            public long OutputCopyTicks { get; init; }
            public long ConvertTicks { get; init; }
#if QUADTREE_TRAVERSAL_PROFILE
            public long[]? TraversalCounters { get; init; }
#endif
        }

        private readonly struct PatchExecutionResult
        {
            public HorizonAngles HorizonData { get; init; }
            public LaunchPatchProfile Profile { get; init; }
        }

        public QuadTreeHorizonGenerator(
            bool disableHierarchy = true,
            bool enableNearFieldReferenceMerge = false,
            float nearFieldClampMeters = 250f,
            HorizonDiagnosticsCallback? diagnosticsCallback = null,
            int maxConcurrentGpuOps = DefaultMaxConcurrentGpuOps,
            int maxSegmentQueueSize = DefaultSegmentQueueSize)
        {
            _disableHierarchy = disableHierarchy;
            _enableNearFieldReferenceMerge = enableNearFieldReferenceMerge;
            _nearFieldClampMeters = Math.Max(0f, nearFieldClampMeters);
            if (maxConcurrentGpuOps <= 0)
                throw new ArgumentOutOfRangeException(nameof(maxConcurrentGpuOps), maxConcurrentGpuOps, "GPU concurrency must be greater than zero.");
            _maxConcurrentGpuOps = maxConcurrentGpuOps;
            if (maxSegmentQueueSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(maxSegmentQueueSize), maxSegmentQueueSize, "The segment queue size must be greater than zero.");
            _maxSegmentQueueSize = maxSegmentQueueSize;
            DiagnosticsCallback = diagnosticsCallback;
            var dbgEnv = Environment.GetEnvironmentVariable("QUADTREE_FORCE_FIXED_STEPS");
            _forceFixedStepDebug = dbgEnv == "1" || dbgEnv?.Equals("true", StringComparison.OrdinalIgnoreCase) == true;
            var profileEnv = Environment.GetEnvironmentVariable("QUADTREE_PIPELINE_PROFILE");
            _enablePipelineProfiling = profileEnv == "1" || profileEnv?.Equals("true", StringComparison.OrdinalIgnoreCase) == true;
            _pipelineSubpatchSize = ReadPipelineSubpatchSizeFromEnvironment();
            _context = Context.Create(builder => builder.Default().DebugSymbols(DebugSymbolsMode.Kernel).EnableAlgorithms());
            // Prefer CUDA; then NVIDIA OpenCL; then any OpenCL; then CPU
            var cudaDevice = _context.Devices.FirstOrDefault(d => d.AcceleratorType == AcceleratorType.Cuda);
            var oclNvidiaDevice = _context.Devices.FirstOrDefault(d => d.AcceleratorType == AcceleratorType.OpenCL && d.Name.IndexOf("NVIDIA", StringComparison.OrdinalIgnoreCase) >= 0);
            var oclAnyDevice = _context.Devices.FirstOrDefault(d => d.AcceleratorType == AcceleratorType.OpenCL);
            var chosenDevice = cudaDevice ?? oclNvidiaDevice ?? oclAnyDevice ?? _context.GetPreferredDevice(preferCPU: true);

            Log.Information("QuadTreeHorizonGenerator using device: {DeviceName} ({AcceleratorType})", chosenDevice.Name, chosenDevice.AcceleratorType);
            _accelerator = chosenDevice.CreateAccelerator(_context);
            _bufferPool = new BufferPool(_accelerator);
            
            // Pre-compile the subpatch GPU kernel once during initialization  
            // Use LoadAutoGroupedKernel which returns a stream-based kernel
#if QUADTREE_TRAVERSAL_PROFILE
            _subpatchKernel = _accelerator.LoadAutoGroupedKernel<
                Index2D,
                PyramidView, PyramidView,
                ArrayView<float>,
                ArrayView<RaySegment>,
                int, int,
                int, int, int, int, int, KernelParams,
                ArrayView<float>, ArrayView<long>>(QuadTreeSubpatchRayCastKernel);
#else
            _subpatchKernel = _accelerator.LoadAutoGroupedKernel<
                Index2D,
                PyramidView, PyramidView,
                ArrayView<float>,
                ArrayView<RaySegment>,
                int, int,
                int, int, int, int, int, KernelParams,
                ArrayView<float>>(QuadTreeSubpatchRayCastKernel);
#endif

            //var kernelInfo = _subpatchKernel.GetKernelInfo();
            //kernelInfo?.DumpToConsole();

            // Initialize stream pool with one stream per GPU worker.
            _streamPool = new ConcurrentStack<AcceleratorStream>();
            for (int i = 0; i < _maxConcurrentGpuOps; i++)
            {
                _streamPool.Push(_accelerator.CreateStream());
            }
            
            Log.Debug("Initialized {StreamCount} GPU streams and pre-compiled kernel", _maxConcurrentGpuOps);
        }

        public void Dispose()
        {
            // Dispose all streams in the pool
            if (_streamPool != null)
            {
                while (_streamPool.TryPop(out var stream))
                {
                    stream.Dispose();
                }
            }
            
            _bufferPool?.Dispose();
            _accelerator.Dispose();
            _context.Dispose();
        }

        // --- Constants for tuning ---
        public const int PYR_DOWNSAMPLE_FACTOR = 4; // pyramid downsample factor per level
        // TODO: This was intended to handle float precision issues in slope comparisons.
        // It now is negative to make the comparison more conservative, avoiding premature skipping.
        public const float COMPARISON_EPSILON = 0.0f; // Small epsilon for float comparisons in slope
        // GPT5 tuning constants
        public const float GUARD_BAND_PIXELS = 0.5f; // inflate AABB by half a pixel side
        public const float ADAPTIVE_EPSILON_C0 = 0.0f;
        public const float ADAPTIVE_EPSILON_C1 = 0.0f; // scaled to produce small slope margins
        public const float BEAM_WIDTH_RAD = (2.0f * (float)Math.PI) / 1440.0f; // azimuth bin width
        
        // Subpatch polynomial approximation constants
        public const int DEFAULT_SUBPATCH_SIZE = 8; // Subpatch size in pixels for polynomial approximation (8, 16, 32, or 64) - used by both pipeline and standalone methods
        private const int DEBUG_FLAG_DISABLE_HIERARCHY = 0x1;
        private const int DEBUG_FLAG_FORCE_FIXED_STEPS = 0x2;
#if QUADTREE_TRAVERSAL_PROFILE
        private const int DEBUG_FLAG_PROFILE_TRAVERSAL = 0x4;
        private const int TRAVERSAL_COUNTERS_PER_DEM = 5;
        private const int TRAVERSAL_COUNTER_ITERATIONS = 0;
        private const int TRAVERSAL_COUNTER_LEVEL0_SAMPLES = 1;
        private const int TRAVERSAL_COUNTER_CULLED_BLOCKS = 2;
        private const int TRAVERSAL_COUNTER_OUT_OF_BOUNDS = 3;
        private const int TRAVERSAL_COUNTER_NODATA_SKIPS = 4;
#endif
        private const float PRIMARY_DEM_FAR_MIN_STEP_DISTANCE_METERS = 100.0f;
        private const float MIN_ADAPTIVE_STEP_RESOLUTION_FACTOR = 0.5f;
        private const float PRIMARY_DEM_FAR_MIN_STEP_RESOLUTION_FACTOR = 0.8f;
        
        // Adaptive stepping constants
        // Based on 99th percentile slope of 30 degrees: 1/tan(30°) ≈ 1.732
        public const float INV_TAN_MAX_SLOPE = 1.732f;
        // Angular error budget: 0.05° in radians, divided by tan(30°) for step factor
        // 0.05° = 0.000873 rad; 0.000873 / 0.577 ≈ 0.00151
        public const float ANGULAR_STEP_FACTOR = 0.00151f;
        
        private static float AdaptiveEpsilon(float levelMapResOverR)
        {
            return ADAPTIVE_EPSILON_C0 + ADAPTIVE_EPSILON_C1 * levelMapResOverR;
        }

        private static bool IsValidSubpatchSize(int subpatchSize) =>
            subpatchSize == 2 || subpatchSize == 4 || subpatchSize == 8 || subpatchSize == 16 ||
            subpatchSize == 32 || subpatchSize == 64 || subpatchSize == 128;

        private static int ReadPipelineSubpatchSizeFromEnvironment()
        {
            var raw = Environment.GetEnvironmentVariable("QUADTREE_PIPELINE_SUBPATCH_SIZE");
            if (string.IsNullOrWhiteSpace(raw))
                return DEFAULT_SUBPATCH_SIZE;
            if (int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed) &&
                IsValidSubpatchSize(parsed) &&
                128 % parsed == 0)
            {
                return parsed;
            }
            Log.Warning("Ignoring invalid QUADTREE_PIPELINE_SUBPATCH_SIZE={SubpatchSize}; using {DefaultSubpatchSize}.", raw, DEFAULT_SUBPATCH_SIZE);
            return DEFAULT_SUBPATCH_SIZE;
        }

        private int GetSubpatchDebugFlags()
        {
            int debugFlags = 0;
            if (_disableHierarchy) debugFlags |= DEBUG_FLAG_DISABLE_HIERARCHY;
            if (_forceFixedStepDebug) debugFlags |= DEBUG_FLAG_FORCE_FIXED_STEPS;
#if QUADTREE_TRAVERSAL_PROFILE
            if (_enablePipelineProfiling) debugFlags |= DEBUG_FLAG_PROFILE_TRAVERSAL;
#endif
            return debugFlags;
        }
        public const float COMPARISON_MIN_DIST = 2.0f; // Minimum distance for calculations, prevents division by zero or negative distances

        // --- Pipeline Management Classes ---
        
        /// <summary>
        /// Represents a set of GPU buffers for processing a single patch.
        /// Used in the pipeline to overlap CPU preparation with GPU execution.
        /// </summary>
        private class PipelineBuffers : IDisposable
        {
            public MemoryBuffer1D<float, Stride1D.Dense> HorizonsAccum { get; set; } = null!;
            public MemoryBuffer1D<float, Stride1D.Dense> HorizonsPass { get; set; } = null!;
            public MemoryBuffer1D<RaySegment, Stride1D.Dense> GpuSegments { get; set; } = null!;
            public MemoryBuffer1D<float, Stride1D.Dense> Debug { get; set; } = null!;
#if QUADTREE_TRAVERSAL_PROFILE
            public MemoryBuffer1D<long, Stride1D.Dense> TraversalCounters { get; set; } = null!;
#endif
            public float[] CpuOutput { get; set; } = null!;
            public bool InUse { get; set; } = false;

            public void Dispose()
            {
                HorizonsAccum?.Dispose();
                HorizonsPass?.Dispose();
                GpuSegments?.Dispose();
                Debug?.Dispose();
#if QUADTREE_TRAVERSAL_PROFILE
                TraversalCounters?.Dispose();
#endif
            }
        }

        /// <summary>
        /// Manages a pool of reusable GPU buffers for pipeline processing.
        /// </summary>
        private class BufferPool : IDisposable
        {
            private readonly List<PipelineBuffers> _buffers = new();
            private readonly Accelerator _accelerator;
            private readonly object _lock = new();

            public BufferPool(Accelerator accelerator)
            {
                _accelerator = accelerator;
            }

            public PipelineBuffers GetAvailableBuffers(
                int numPixels,
                int numAzimuths,
                int maxSegments
#if QUADTREE_TRAVERSAL_PROFILE
                , int traversalCounterCount
#endif
                )
            {
                lock (_lock)
                {
                    // Look for available buffer set with correct sizes
                    var availableBuffer = _buffers.FirstOrDefault(b => !b.InUse && 
                        b.HorizonsAccum.Length >= numPixels * numAzimuths &&
                        b.HorizonsPass.Length >= numPixels * numAzimuths &&
                        b.GpuSegments.Length >= maxSegments
#if QUADTREE_TRAVERSAL_PROFILE
                        && b.TraversalCounters.Length >= traversalCounterCount
#endif
                        );
                    
                    if (availableBuffer != null)
                    {
                        availableBuffer.InUse = true;
                        return availableBuffer;
                    }

                    // Create new buffer set if none available with correct size
                    var newBuffer = new PipelineBuffers
                    {
                        HorizonsAccum = _accelerator.Allocate1D<float>(numPixels * numAzimuths),
                        HorizonsPass = _accelerator.Allocate1D<float>(numPixels * numAzimuths),
                        GpuSegments = _accelerator.Allocate1D<RaySegment>(maxSegments),
                        Debug = _accelerator.Allocate1D<float>(1024),
#if QUADTREE_TRAVERSAL_PROFILE
                        TraversalCounters = _accelerator.Allocate1D<long>(traversalCounterCount),
#endif
                        CpuOutput = new float[numPixels * numAzimuths],
                        InUse = true
                    };
                    
                    _buffers.Add(newBuffer);
                    return newBuffer;
                }
            }

            public void ReturnBuffers(PipelineBuffers buffers)
            {
                lock (_lock)
                {
                    buffers.InUse = false;
                }
            }

            public void Dispose()
            {
                foreach (var buffer in _buffers)
                    buffer.Dispose();
                _buffers.Clear();
            }
        }

        /// <summary>
        /// Work item representing a patch for GPU processing (all DEMs together like original).
        /// </summary>
        private class PatchWorkItem
        {
            public PatchDescriptor Patch { get; set; } = null!;
            public RaySegment[] Segments { get; set; } = null!;
            public GridConvergenceInfo GCInfo { get; set; }
            public long SegmentGenerationTicks { get; set; }
            public long PatchEnqueueWaitTicks { get; set; }
            public int PatchQueueDepthAfterEnqueue { get; set; }
        }

        public void GenerateHorizons(string outputDirectory, List<string> demPaths, float observerElevationMeters = 0.0f)
        {
            GenerateHorizons(outputDirectory, null, demPaths, 0, 0, 128, 128, observerElevationMeters);
        }

        public void GenerateHorizons(string outputDirectory, List<string> demPaths, int tileX, int tileY, int width, int height, float observerElevationMeters = 0.0f)
        {
            GenerateHorizons(outputDirectory, null, demPaths, tileX, tileY, width, height, observerElevationMeters);
        }

        public void GenerateHorizons(string outputDirectory, string? fileName, List<string> demPaths, int tileX, int tileY, int width, int height, float observerElevation = 0.0f)
        {
            if (demPaths is null) throw new ArgumentNullException(nameof(demPaths));
            var dems = demPaths.Select(p => new ElevationMap(p)).ToList();
            Log.Information("DEM loading completed.");
            GenerateHorizons(outputDirectory, fileName, dems, tileX, tileY, width, height, observerElevation);
        }

        public HorizonAngles GenerateHorizons(List<string> demPaths, int tileX, int tileY, int width, int height, float observerElevation = 0.0f)
        {
            if (demPaths is null) throw new ArgumentNullException(nameof(demPaths));
            var dems = demPaths.Select(p => new ElevationMap(p)).ToList();
            return GenerateHorizons(dems, tileX, tileY, width, height, observerElevation);
        }

        public HorizonAngles GenerateHorizons(List<ElevationMap> dems, int tileX, int tileY, int width, int height, float observerElevation = 0.0f)
        {
            return GenerateHorizonsInternal(dems, tileX, tileY, width, height, observerElevation, captureIntermediate: true);
        }

        /// <summary>
        /// Generates horizons using subpatch approach - single GPU kernel with multiple polynomials
        /// per azimuth to improve polynomial approximation accuracy at patch edges.
        /// </summary>
        /// <param name="subpatchSize">Size of each subpatch in pixels (8, 16, 32, or 64)</param>
        public HorizonAngles GenerateHorizonsWithSubpatches(List<ElevationMap> dems, int tileX, int tileY, int width, int height, 
                                                           float observerElevation = 0.0f, int subpatchSize = DEFAULT_SUBPATCH_SIZE)
        {
            if (!IsValidSubpatchSize(subpatchSize))
                throw new ArgumentException("Subpatch size must be 2, 4, 8, 16, 32, 64, or 128", nameof(subpatchSize));

            if (width != 128 || height != 128)
                throw new ArgumentException("Subpatch approach currently only supports 128x128 patches", nameof(width));

            if (dems == null || dems.Count == 0)
                throw new ArgumentException("At least one DEM is required.", nameof(dems));

            // Build pyramids 
            var pyramids = new List<Pyramid>();
            foreach (var dem in dems)
                pyramids.Add(BuildOrLoadPyramid(dem));

            Log.Debug("Generating horizons with {SubpatchSize}×{SubpatchSize} subpatches using single GPU kernel", subpatchSize, subpatchSize);

            // Launch subpatch-based ray casting
            var result = LaunchSubpatchRayCasting(pyramids, dems, tileX, tileY, width, height, observerElevation, subpatchSize);

            // Clean up pyramids
            foreach (var p in pyramids)
                p?.Dispose();

            return result;
        }

        /// <summary>
        /// Launches the subpatch-based GPU ray casting kernel
        /// </summary>
        private HorizonAngles LaunchSubpatchRayCasting(List<Pyramid> pyramids, List<ElevationMap> dems, 
                                                      int tileColBase, int tileRowBase, int tileW, int tileH, 
                                                      float observerElevation, int subpatchSize)
        {
            int numPixels = tileW * tileH;
            int numAzimuths = 1440;
            int numDems = pyramids.Count;
            float maxDist = 1000000.0f; // 1000 km

            var primaryPV = new PyramidView
            {
                DataLevel0 = pyramids[0].DataLevel0!.View,
                DataMips = pyramids[0].DataMips!.View,
                Infos = pyramids[0].Infos!.View,
                Map = pyramids[0].Map,
                Proj = pyramids[0].Proj,
                Levels = pyramids[0].CpuInfos!.Length
            };

            var sw = Stopwatch.StartNew();
            // Calculate subpatch ray segments using existing method
            var (segments, gcInfo) = CalculateSubpatchRaySegments(
                pyramids, primaryPV, tileColBase, tileRowBase, tileW, tileH,
                numAzimuths, maxDist, observerElevation, subpatchSize);

            Log.Information("Calculated {SegmentCount} subpatch ray segments in {Duration:F3} sec", segments.Length, sw.Elapsed.TotalSeconds);

            // Initialize output
            var outputBuffer = new float[numPixels * numAzimuths];
            for (int i = 0; i < outputBuffer.Length; i++)
                outputBuffer[i] = float.NegativeInfinity;

            using var gpuSegments = _accelerator.Allocate1D(segments);
            using var gpuOutput = _accelerator.Allocate1D(outputBuffer);
            using var debugBuffer = _accelerator.Allocate1D<float>(50);
#if QUADTREE_TRAVERSAL_PROFILE
            using var traversalCounters = _accelerator.Allocate1D<long>(numDems * TRAVERSAL_COUNTERS_PER_DEM);
            if (_enablePipelineProfiling)
                traversalCounters.CopyFromCPU(new long[numDems * TRAVERSAL_COUNTERS_PER_DEM]);
#endif

            var kernelParams = new KernelParams(
                observerElevation,
                0.001f, // MinTraverseDistanceKm
                0, // DebugAzimuthIndex  
                gcInfo.GammaCenter,
                gcInfo.DGammaDx,
                gcInfo.DGammaDy,
                GetSubpatchDebugFlags(),
                pyramids[0].CpuInfos![0].Width,
                pyramids[0].CpuInfos![0].Height);

            sw.Restart();

            // Launch kernel for each DEM pass
            for (int demPass = 0; demPass < numDems; demPass++)
            {
                var activePV = new PyramidView
                {
                    DataLevel0 = pyramids[demPass].DataLevel0!.View,
                    DataMips = pyramids[demPass].DataMips!.View,
                    Infos = pyramids[demPass].Infos!.View,
                    Map = pyramids[demPass].Map,
                    Proj = pyramids[demPass].Proj,
                    Levels = pyramids[demPass].CpuInfos!.Length
                };
                
#if QUADTREE_TRAVERSAL_PROFILE
                var kernel = _accelerator.LoadAutoGroupedStreamKernel<
                    Index2D, PyramidView, PyramidView, ArrayView<float>, ArrayView<RaySegment>,
                    int, int, int, int, int, int, int, KernelParams, ArrayView<float>, ArrayView<long>>(
                    QuadTreeSubpatchRayCastKernel);
#else
                var kernel = _accelerator.LoadAutoGroupedStreamKernel<
                    Index2D, PyramidView, PyramidView, ArrayView<float>, ArrayView<RaySegment>,
                    int, int, int, int, int, int, int, KernelParams, ArrayView<float>>(
                    QuadTreeSubpatchRayCastKernel);
#endif

                // Launch with (pixel, azimuth) threading
                var extent = new Index2D(numPixels, numAzimuths);
#if QUADTREE_TRAVERSAL_PROFILE
                kernel(extent, primaryPV, activePV, gpuOutput.View, gpuSegments.View,
                      demPass, numDems,
                      tileColBase, tileRowBase, tileW, tileH, subpatchSize,
                      kernelParams, debugBuffer.View, traversalCounters.View);
#else
                kernel(extent, primaryPV, activePV, gpuOutput.View, gpuSegments.View,
                      demPass, numDems,
                      tileColBase, tileRowBase, tileW, tileH, subpatchSize,
                      kernelParams, debugBuffer.View);
#endif

                _accelerator.Synchronize();
            }

            Log.Information("Completed GPU Processing                        in {Duration:F3} sec", segments.Length, sw.Elapsed.TotalSeconds);

            // Copy results back
            gpuOutput.CopyToCPU(outputBuffer);
            
            return HorizonAngles.FromSlopes(outputBuffer);
        }

        public void GenerateHorizons(string outputDirectory, List<ElevationMap> dems, int tileX, int tileY, int width, int height, float observerElevation = 0.0f)
        {
            GenerateHorizons(outputDirectory, null, dems, tileX, tileY, width, height, observerElevation);
        }

        public void GenerateHorizons(string outputDirectory, string? fileName, List<ElevationMap> dems, int tileX, int tileY, int width, int height, float observerElevation = 0.0f)
        {
            if (string.IsNullOrWhiteSpace(outputDirectory))
                throw new ArgumentException("Output directory must be provided.", nameof(outputDirectory));

            Directory.CreateDirectory(outputDirectory);

            var horizonData = GenerateHorizonsInternal(dems, tileX, tileY, width, height, observerElevation, captureIntermediate: true, outputDirectory);

            DiagnosticsCallback?.Invoke(HorizonBufferType.FullHorizon, horizonData);

            // What is this for?
            string resolvedName = string.IsNullOrWhiteSpace(fileName)
                ? BuildHorizonFilename(tileX, tileY, observerElevation)
                : fileName!;

            string path;
            if (string.IsNullOrWhiteSpace(fileName))
            {
                var store = new HorizonTileStore(outputDirectory, HorizonTileLayout.PartitionedByY);
                path = store.BuildPath(tileY, tileX, observerElevation, compress: false);
                Directory.CreateDirectory(Path.GetDirectoryName(path)!);
                Utilities.WriteBinaryArray(path, horizonData.Degrees);
            }
            else
            {
                path = Path.Combine(outputDirectory, resolvedName);
                Utilities.WriteBinaryArray(path, horizonData.Degrees);
            }
            Log.Information($"Written horizons (deg) to {path}");
        }

        /// <summary>
        /// Represents a 128x128 pixel patch within a DEM for horizon generation.
        /// </summary>
        public class PatchDescriptor
        {
            /// <summary>X coordinate (column) of the top-left corner of the patch in the DEM</summary>
            public int TileX { get; set; }
            
            /// <summary>Y coordinate (row) of the top-left corner of the patch in the DEM</summary>
            public int TileY { get; set; }
            
            /// <summary>Patch index in row-major order (0-based)</summary>
            public int Index { get; set; }
            
            /// <summary>Patch X index (column in the patch grid)</summary>
            public int PatchX { get; set; }
            
            /// <summary>Patch Y index (row in the patch grid)</summary>
            public int PatchY { get; set; }

            public override string ToString() => $"Patch[{Index}] at ({TileX}, {TileY}) grid=({PatchX}, {PatchY})";
        }

        /// <summary>
        /// Generates a list of all 128x128 pixel patches within the primary DEM.
        /// Patches are ordered in row-major order (left-to-right, top-to-bottom).
        /// </summary>
        /// <param name="primaryDem">The primary (inner) DEM to generate patches for</param>
        /// <returns>List of patch descriptors</returns>
        /// <exception cref="ArgumentException">If DEM dimensions are not even multiples of 128</exception>
        public static List<PatchDescriptor> GeneratePatchList(ElevationMap primaryDem)
        {
            const int PATCH_SIZE = 128;
            
            if (primaryDem == null)
                throw new ArgumentNullException(nameof(primaryDem));
            
            if (primaryDem.Width % PATCH_SIZE != 0)
                throw new ArgumentException($"DEM width ({primaryDem.Width}) must be an even multiple of {PATCH_SIZE}.", nameof(primaryDem));
            if (primaryDem.Height % PATCH_SIZE != 0)
                throw new ArgumentException($"DEM height ({primaryDem.Height}) must be an even multiple of {PATCH_SIZE}.", nameof(primaryDem));
            
            int numPatchesX = primaryDem.Width / PATCH_SIZE;
            int numPatchesY = primaryDem.Height / PATCH_SIZE;
            int totalPatches = numPatchesX * numPatchesY;
            
            var patches = new List<PatchDescriptor>(totalPatches);
            
            for (int patchIndex = 0; patchIndex < totalPatches; patchIndex++)
            {
                int patchY = patchIndex / numPatchesX;
                int patchX = patchIndex % numPatchesX;
                
                patches.Add(new PatchDescriptor
                {
                    Index = patchIndex,
                    TileX = patchX * PATCH_SIZE,
                    TileY = patchY * PATCH_SIZE,
                    PatchX = patchX,
                    PatchY = patchY
                });
            }
            
            return patches;
        }

        public static List<PatchDescriptor> FilterPatchesByRegion(List<PatchDescriptor> patches, int tileXMin, int tileYMin, int tileXMax, int tileYMax) =>
            patches.Where(p => p.TileX >= tileXMin && p.TileX < tileXMax && p.TileY >= tileYMin && p.TileY < tileYMax).ToList();

        public static List<PatchDescriptor> RemoveCompletedPatches(List<PatchDescriptor> patches, string directory, float observerElevation_meters)
        {
            if (!Directory.Exists(directory))
                return patches; // If directory doesn't exist, no patches are completed, return full list
            var store = new HorizonTileStore(directory);
            var filteredPatches = patches
                .Where(p => store.FindExistingPath(p.TileY, p.TileX, observerElevation_meters) == null)
                .ToList();
            return filteredPatches;
        }

        /// <summary>
        /// Generates horizon files for a specified list of patches using an optimized multi-stage GPU pipeline.
        /// This method implements a streamlined pipeline that maximizes GPU utilization through concurrent streams
        /// while maintaining the proven multi-DEM processing logic for correctness.
        /// 
        /// Streamlined Pipeline Architecture (2-Stage):
        /// 1. **Producer Stage** (CPU): Creates patch work items containing all DEMs per patch (maintains original working logic)
        /// 2. **GPU Worker Stage** (8 concurrent streams): Processes complete patches using dedicated GPU streams for true parallelism
        /// 
        /// Key Optimizations:
        /// - **Pre-compiled Kernel**: GPU kernel loaded once during constructor (eliminates ~24 seconds of recompilation overhead)
        /// - **Stream Pool**: 8 independent AcceleratorStream instances enable true concurrent GPU execution (no serialization)
        /// - **Buffer Pool**: Reusable GPU memory buffers eliminate allocation/deallocation overhead
        /// - **Patch-Level Processing**: Complete patches (all DEMs together) maintain proven kernel logic while enabling parallelism
        /// - **Direct File Writing**: Results written directly to disk without complex merging stages
        /// 
        /// Performance Characteristics:
        /// - **GPU Utilization**: Smooth 80%+ usage (eliminates previous spiky 20% patterns)
        /// - **Throughput**: Up to 8x improvement from concurrent stream execution
        /// - **Memory Efficiency**: Buffer reuse and bounded channels (32-item capacity) prevent memory bloat
        /// - **Correctness**: Uses original working multi-DEM kernel logic to prevent lightmap artifacts
        /// 
        /// Prerequisites:
        /// - DEMs must be loaded (input parameter) - pyramids are built/cached automatically as .pyr.bin files
        /// - Output directory will be created if it doesn't exist
        /// - GPU streams and kernel are pre-initialized during QuadTreeHorizonGenerator construction
        /// </summary>
        /// <param name="outputDirectory">Directory where horizon files will be written</param>
        /// <param name="dems">List of elevation maps (nested, with primary DEM first)</param>
        /// <param name="patches">List of patches to process (use GeneratePatchList() and LINQ to filter)</param>
        /// <param name="observerElevation">Observer height above terrain in meters</param>
        /// <param name="compressHorizons">If true, write compressed .cbin horizon tiles directly</param>
        public async Task GenerateHorizonsForPatches(
            string outputDirectory,
            List<ElevationMap> dems,
            List<PatchDescriptor> patches,
            float observerElevation = 0.0f,
            bool compressHorizons = false,
            moonlib.HorizonProgressCallback? progress = null,
            moonlib.HorizonCancellationCallback? isCancellationRequested = null)
        {
            if (string.IsNullOrWhiteSpace(outputDirectory))
                throw new ArgumentException("Output directory must be provided.", nameof(outputDirectory));
            if (dems == null || dems.Count == 0)
                throw new ArgumentException("At least one DEM is required.", nameof(dems));
            if (patches == null || patches.Count == 0)
                throw new ArgumentException("At least one patch is required.", nameof(patches));

            void ThrowIfCancelled()
            {
                if (isCancellationRequested?.Invoke() == true)
                    throw new OperationCanceledException("Horizon generation was canceled.");
            }

            void ReportProgress(int processed, int total, double percent, string stage, string message, string? fileName = null)
            {
                progress?.Invoke(new moonlib.HorizonProgress(processed, total, percent, stage, message, fileName));
            }

            const int PATCH_SIZE = 128;
            var primaryDem = dems[0];
            
            Directory.CreateDirectory(outputDirectory);
            var horizonStore = new HorizonTileStore(outputDirectory, HorizonTileLayout.PartitionedByY);

            int patchCount = patches.Count;
            ThrowIfCancelled();
            ReportProgress(0, patchCount, 10.0, "prepare_pyramids", "Building or loading DEM pyramids.");
            Log.Information($"Starting pipelined horizon generation for {patchCount} patches");
            Log.Information($"Primary DEM dimensions: {primaryDem.Width}x{primaryDem.Height}");
            Log.Information($"Output directory: {Path.GetFullPath(outputDirectory)}");

            var totalStopwatch = Stopwatch.StartNew();

            // Stage 1: Build or load pyramids for all DEMs in parallel (done once, cached to filesystem)
            Log.Debug("Stage 1: Building/loading pyramids for {DemCount} DEMs in parallel...", dems.Count);
            var pyramids = new Pyramid[dems.Count];
            Parallel.For(0, dems.Count, i =>
            {
                pyramids[i] = BuildOrLoadPyramid(dems[i]);
            });
            Log.Information("Pyramids ready in {Elapsed:F2}s", totalStopwatch.Elapsed.TotalSeconds);
            ThrowIfCancelled();
            ReportProgress(0, patchCount, 15.0, "process_patches", "Starting horizon patch generation.");

            // Create primary pyramid view (needed for ray segment calculation)
            var primaryPV = new PyramidView
            {
                DataLevel0 = pyramids[0].DataLevel0!.View,
                DataMips = pyramids[0].DataMips!.View,
                Infos = pyramids[0].Infos!.View,
                Map = pyramids[0].Map,
                Proj = pyramids[0].Proj,
                Levels = pyramids[0].CpuInfos!.Length
            };

            int numAzimuths = 1440;
            float maxDist = 1000000.0f; // 1000 km
            int numPixels = PATCH_SIZE * PATCH_SIZE;
            var pyramidList = pyramids.ToList();
            const bool EnableSharedSegmentCache = false;
            SubpatchSegmentCache? sharedSegmentCache = EnableSharedSegmentCache
                ? new SubpatchSegmentCache(
                    pyramidList,
                    numAzimuths,
                    maxDist,
                    observerElevation,
                    _pipelineSubpatchSize)
                : null;
            
            // Since Full Mode is disabled (always use Compact Mode), we only need: numAzimuths * dems.Count
            int maxSegments = numAzimuths * dems.Count;  // Compact Mode only

            // Stage 2: Pipeline processing with producer-consumer pattern
            Log.Debug("Stage 2: Processing {PatchCount} patches with CPU/GPU pipeline...", patchCount);
            var pipelineStopwatch = Stopwatch.StartNew();

            // Create channels for 2-stage pipeline (back to patch-level processing)
            var channelOptions = new BoundedChannelOptions(_maxSegmentQueueSize)
            {
                FullMode = BoundedChannelFullMode.Wait, // Block producer if queue is full
                SingleReader = false,  // Multiple GPU workers can read
                SingleWriter = true
            };
            
            var patchWorkChannel = Channel.CreateBounded<PatchWorkItem>(channelOptions);

            // No semaphore needed - stream pool provides natural backpressure
            // Initialize progress tracking (needed by GPU workers)
            int processedCount = 0;
            int queuedPatchCount = 0;
            int activeGpuWorkers = 0;
            int activeStreamsInUse = 0;

            // Stage 1: Producer - Generate patch work items (all DEMs together like original)
            var producer = Task.Run(async () =>
            {
                try
                {
                    foreach (var patch in patches)
                    {
                        ThrowIfCancelled();
                        Log.Debug("Calculating subpatch ray segments for patch ({TileX}, {TileY}) with {SubpatchSize}×{SubpatchSize} subpatches...", 
                            patch.TileX, patch.TileY, _pipelineSubpatchSize, _pipelineSubpatchSize);
                        long segmentStart = Stopwatch.GetTimestamp();
                        var (segments, gcInfo) = CalculateSubpatchRaySegments(
                            pyramidList, primaryPV,
                            patch.TileX, patch.TileY, PATCH_SIZE, PATCH_SIZE,
                            numAzimuths, maxDist, observerElevation, _pipelineSubpatchSize, sharedSegmentCache);
                        long segmentElapsed = Stopwatch.GetTimestamp() - segmentStart;
                        RecordPipelineStage(PipelineStageNames.SegmentGeneration, segmentElapsed);
                        Log.Information("Patch={index:D4} ray_count={SegmentCount} sec={Duration:F3}", patch.Index, segments.Length, StopwatchTicksToSeconds(segmentElapsed));

                        var patchWorkItem = new PatchWorkItem
                        {
                            Patch = patch,
                            Segments = segments,  // Subpatch segments (64× more than compact)
                            GCInfo = gcInfo,
                            SegmentGenerationTicks = segmentElapsed
                        };

                        ThrowIfCancelled();
                        patchWorkItem.PatchQueueDepthAfterEnqueue = Interlocked.Increment(ref queuedPatchCount);
                        long enqueueStart = Stopwatch.GetTimestamp();
                        await patchWorkChannel.Writer.WriteAsync(patchWorkItem);
                        long enqueueElapsed = Stopwatch.GetTimestamp() - enqueueStart;
                        RecordPipelineStage(PipelineStageNames.PatchEnqueueWait, enqueueElapsed);
                        patchWorkItem.PatchEnqueueWaitTicks = enqueueElapsed;
                    }
                    patchWorkChannel.Writer.Complete();
                }
                catch (Exception ex)
                {
                    Log.Error(ex, "Producer failed: {Message}", ex.Message);
                    patchWorkChannel.Writer.TryComplete(ex);
                }
            });

            // Stage 2: GPU Workers - Process complete patches (all DEMs together)
            var gpuWorkerTasks = new Task[_maxConcurrentGpuOps];
            for (int workerId = 0; workerId < _maxConcurrentGpuOps; workerId++)
            {
                int capturedWorkerId = workerId;
                gpuWorkerTasks[workerId] = Task.Run(async () =>
                {
                    try
                    {
                        await foreach (var patchWorkItem in patchWorkChannel.Reader.ReadAllAsync())
                        {
                            ThrowIfCancelled();
                            int queueDepthOnDequeue = Interlocked.Decrement(ref queuedPatchCount);
                            int activeWorkersNow = Interlocked.Increment(ref activeGpuWorkers);
                            // Get buffer for this GPU worker
                            var actualSegmentCount = patchWorkItem.Segments.Length;
#if QUADTREE_TRAVERSAL_PROFILE
                            int traversalCounterCount = pyramidList.Count * TRAVERSAL_COUNTERS_PER_DEM;
#endif
                            long bufferAcquireStart = Stopwatch.GetTimestamp();
#if QUADTREE_TRAVERSAL_PROFILE
                            var buffers = _bufferPool!.GetAvailableBuffers(numPixels, numAzimuths, actualSegmentCount, traversalCounterCount);
#else
                            var buffers = _bufferPool!.GetAvailableBuffers(numPixels, numAzimuths, actualSegmentCount);
#endif
                            long bufferAcquireElapsed = Stopwatch.GetTimestamp() - bufferAcquireStart;
                            RecordPipelineStage(PipelineStageNames.BufferAcquire, bufferAcquireElapsed);

                            try
                            {
                                long workerStart = Stopwatch.GetTimestamp();
                                // Process complete patch on GPU using stream pool (all DEMs together like original)
                                var result = await LaunchPatchAsync(
                                    buffers,
                                    pyramidList, dems,
                                    patchWorkItem.Segments,
                                    patchWorkItem.Patch.TileX, patchWorkItem.Patch.TileY, PATCH_SIZE, PATCH_SIZE,
                                    observerElevation,
                                    patchWorkItem.GCInfo,
                                    () => Interlocked.Increment(ref activeStreamsInUse),
                                    () => Interlocked.Decrement(ref activeStreamsInUse));
                                long workerElapsed = Stopwatch.GetTimestamp() - workerStart;
                                RecordPipelineStage(PipelineStageNames.TotalGpuWorkerPatchWall, workerElapsed);
                                Log.Information("GPU processed rays                        sec={Duration:F3}", StopwatchTicksToSeconds(workerElapsed));

                                // Write to file immediately (no merging needed)
                                string fileName = horizonStore.BuildFileName(
                                    patchWorkItem.Patch.TileY,
                                    patchWorkItem.Patch.TileX,
                                    observerElevation,
                                    compressHorizons);
                                var filePath = horizonStore.BuildPath(
                                    patchWorkItem.Patch.TileY,
                                    patchWorkItem.Patch.TileX,
                                    observerElevation,
                                    compressHorizons);
                                long writeStart = Stopwatch.GetTimestamp();
                                if (compressHorizons)
                                {
                                    try
                                    {
                                        horizonStore.Write(
                                            patchWorkItem.Patch.TileY,
                                            patchWorkItem.Patch.TileX,
                                            observerElevation,
                                            result.HorizonData.Degrees,
                                            compress: true);
                                    }
                                    catch (Exception ex)
                                    {
                                        var fallbackPath = horizonStore.BuildPath(
                                            patchWorkItem.Patch.TileY,
                                            patchWorkItem.Patch.TileX,
                                            observerElevation,
                                            compress: false);
                                        Log.Error(
                                            ex,
                                            "Compressed horizon write failed for {CompressedFile}; falling back to uncompressed {FallbackFile}.",
                                            filePath,
                                            fallbackPath);
                                        horizonStore.Write(
                                            patchWorkItem.Patch.TileY,
                                            patchWorkItem.Patch.TileX,
                                            observerElevation,
                                            result.HorizonData.Degrees,
                                            compress: false);
                                        fileName = horizonStore.BuildFileName(
                                            patchWorkItem.Patch.TileY,
                                            patchWorkItem.Patch.TileX,
                                            observerElevation,
                                            compress: false);
                                        filePath = fallbackPath;
                                    }
                                }
                                else
                                {
                                    horizonStore.Write(
                                        patchWorkItem.Patch.TileY,
                                        patchWorkItem.Patch.TileX,
                                        observerElevation,
                                        result.HorizonData.Degrees,
                                        compress: false);
                                    Console.WriteLine($"Wrote horizon file: {filePath}");
                                    Log.Debug($"Wrote horizon file: {filePath}");
                                }
                                long writeElapsed = Stopwatch.GetTimestamp() - writeStart;
                                RecordPipelineStage(PipelineStageNames.CompressAndWrite, writeElapsed);
                                LogPatchProfiling(
                                    patchWorkItem.Patch,
                                    queueDepthOnDequeue,
                                    patchWorkItem.PatchQueueDepthAfterEnqueue,
                                    activeWorkersNow,
                                    Volatile.Read(ref activeStreamsInUse),
                                    patchWorkItem.SegmentGenerationTicks,
                                    patchWorkItem.PatchEnqueueWaitTicks,
                                    bufferAcquireElapsed,
                                    workerElapsed,
                                    writeElapsed,
                                    result.Profile);

                                // Report progress
                                lock (patches) // Use patches list as lock object
                                {
                                    processedCount++;
                                    double percent = processedCount * 100.0 / patchCount;
                                    double elapsed = pipelineStopwatch.Elapsed.TotalSeconds;
                                    double avgTimePerPatch = elapsed / processedCount;
                                    int remaining = patchCount - processedCount;
                                    double estimatedRemainingTime = remaining * avgTimePerPatch;
                                    DateTime eta = DateTime.Now.AddSeconds(estimatedRemainingTime);
                                    string etaStr = eta.ToString("HH:mm:ss").PadRight(10);
                                    string remainStr = FormatTimeSpan(TimeSpan.FromSeconds(estimatedRemainingTime)).PadRight(10);

                                    Log.Information("Progress: {Current}/{Total} ({Percent,5:F1}%) | Avg: {AvgTime,7:F2}s/patch | ETA: {ETA} | Remain: {Remain} | File: {FileName}",
                                        processedCount, patchCount,
                                        percent,
                                        avgTimePerPatch,
                                        etaStr,
                                        remainStr,
                                        fileName);
                                    ReportProgress(
                                        processedCount,
                                        patchCount,
                                        percent,
                                        "process_patches",
                                        $"Generated {processedCount}/{patchCount} horizon patches.",
                                        fileName);
                                }
                            }
                            finally
                            {
                                _bufferPool.ReturnBuffers(buffers);
                                Interlocked.Decrement(ref activeGpuWorkers);
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        Log.Error(ex, "GPU Worker {WorkerId} failed: {Message}", capturedWorkerId, ex.Message);
                        throw;
                    }
                });
            }

            // Wait for producer and all GPU workers to complete
            await Task.WhenAll(producer, Task.WhenAll(gpuWorkerTasks));

            Log.Information("Pipeline processing completed in {Elapsed:F2}s", pipelineStopwatch.Elapsed.TotalSeconds);
            LogPipelineAggregateSummary(patchCount, pipelineStopwatch.Elapsed);

            // Cleanup
            foreach (var p in pyramids)
                p.Dispose();

            totalStopwatch.Stop();
            Log.Information("Total pipeline time: {Elapsed:F2}s for {Count} patches ({AvgTime:F2}s per patch)", 
                totalStopwatch.Elapsed.TotalSeconds, patchCount, totalStopwatch.Elapsed.TotalSeconds / patchCount);
        }



        /// <summary>
        /// Processes a complete patch (all DEMs together) asynchronously on GPU using stream pool for true concurrency.
        /// This method follows the original working approach but uses streams for parallelism.
        /// </summary>
        private async Task<PatchExecutionResult> LaunchPatchAsync(
            PipelineBuffers buffers,
            List<Pyramid> pyramids, List<ElevationMap> dems,
            RaySegment[] allSegments,
            int tileCol, int tileRow, int tileWidth, int tileHeight,
            float observerElevation,
            GridConvergenceInfo gcInfo,
            Action onStreamAcquire,
            Action onStreamRelease)
        {
            int numPixels = tileWidth * tileHeight;
            int numAzimuths = 1440;
            int numDems = pyramids.Count;

            // 1. Initialize buffers
            long bufferResetStart = Stopwatch.GetTimestamp();
            var cpuHorizons = new float[numPixels * numAzimuths];
            for (int i = 0; i < cpuHorizons.Length; i++) cpuHorizons[i] = float.NegativeInfinity;
            buffers.HorizonsAccum.CopyFromCPU(cpuHorizons);
            long bufferResetElapsed = Stopwatch.GetTimestamp() - bufferResetStart;
            RecordPipelineStage(PipelineStageNames.BufferResetHorizonsAccum, bufferResetElapsed);

            long segmentUploadStart = Stopwatch.GetTimestamp();
            buffers.GpuSegments.CopyFromCPU(allSegments);
            long segmentUploadElapsed = Stopwatch.GetTimestamp() - segmentUploadStart;
            RecordPipelineStage(PipelineStageNames.SegmentUpload, segmentUploadElapsed);

#if QUADTREE_TRAVERSAL_PROFILE
            long[]? cpuTraversalCounters = null;
            if (_enablePipelineProfiling)
            {
                cpuTraversalCounters = new long[numDems * TRAVERSAL_COUNTERS_PER_DEM];
                buffers.TraversalCounters.CopyFromCPU(cpuTraversalCounters);
            }
#endif

            // 2. Setup kernel parameters (kernel is pre-loaded)
            float minTraverseDistanceMeters = (_enableNearFieldReferenceMerge && _nearFieldClampMeters > 0f)
                ? _nearFieldClampMeters
                : 0f;
            var kernelParams = new KernelParams(
                observerElevation,
                minTraverseDistanceMeters * METERS_TO_KILOMETERS,
                -1, // debugTargetAzIdx - disabled for pipeline
                gcInfo.GammaCenter,
                gcInfo.DGammaDx,
                gcInfo.DGammaDy,
                GetSubpatchDebugFlags(),
                pyramids[0].CpuInfos![0].Width,
                pyramids[0].CpuInfos![0].Height);

            // 3. Create pyramid views for ALL DEMs (like original working version)
            var primaryPV = new PyramidView
            {
                DataLevel0 = pyramids[0].DataLevel0!.View,
                DataMips = pyramids[0].DataMips!.View,
                Infos = pyramids[0].Infos!.View,
                Map = pyramids[0].Map,
                Proj = pyramids[0].Proj,
                Levels = pyramids[0].CpuInfos!.Length
            };

            // 4. Get stream from pool (blocks if none available)
            AcceleratorStream? stream = null;
            long streamWaitStart = Stopwatch.GetTimestamp();
            while (stream == null)
            {
                if (!_streamPool!.TryPop(out stream))
                {
                    // Wait briefly and retry - provides backpressure
                    await Task.Delay(1);
                }
            }
            long streamWaitElapsed = Stopwatch.GetTimestamp() - streamWaitStart;
            RecordPipelineStage(PipelineStageNames.StreamAcquire, streamWaitElapsed);
            onStreamAcquire();

            try
            {
                // 5. Process all DEMs together using original multi-DEM kernel approach
                long kernelLaunchStart = Stopwatch.GetTimestamp();
                for (int demPass = 0; demPass < numDems; demPass++)
                {
                    var activePV = new PyramidView
                    {
                        DataLevel0 = pyramids[demPass].DataLevel0!.View,
                        DataMips = pyramids[demPass].DataMips!.View,
                        Infos = pyramids[demPass].Infos!.View,
                        Map = pyramids[demPass].Map,
                        Proj = pyramids[demPass].Proj,
                        Levels = pyramids[demPass].CpuInfos!.Length
                    };

                    // Launch subpatch kernel on dedicated stream (true concurrency!)
                    _subpatchKernel!(stream, new Index2D(numPixels, numAzimuths),
                        primaryPV, activePV,
                        buffers.HorizonsAccum.View,
                        buffers.GpuSegments.View,
                        demPass, numDems, // Original multi-DEM parameters
                        tileCol, tileRow, tileWidth, tileHeight, _pipelineSubpatchSize, kernelParams, // Subpatch size instead of isCompact
                        buffers.Debug.View
#if QUADTREE_TRAVERSAL_PROFILE
                        , buffers.TraversalCounters.View
#endif
                        );
                }
                long kernelLaunchElapsed = Stopwatch.GetTimestamp() - kernelLaunchStart;
                RecordPipelineStage(PipelineStageNames.KernelLaunchTotal, kernelLaunchElapsed);

                // 6. Return result asynchronously with stream-specific synchronization
                return await Task.Run(() =>
                {
                    long syncStart = Stopwatch.GetTimestamp();
                    stream.Synchronize(); // Only synchronize THIS stream, not all GPU work
                    long syncElapsed = Stopwatch.GetTimestamp() - syncStart;
                    RecordPipelineStage(PipelineStageNames.StreamSync, syncElapsed);

                    long copyStart = Stopwatch.GetTimestamp();
                    var output = buffers.HorizonsAccum.GetAsArray1D();
#if QUADTREE_TRAVERSAL_PROFILE
                    long[]? traversalCounters = _enablePipelineProfiling
                        ? buffers.TraversalCounters.GetAsArray1D()
                        : null;
#endif
                    long copyElapsed = Stopwatch.GetTimestamp() - copyStart;
                    RecordPipelineStage(PipelineStageNames.OutputCopyToHost, copyElapsed);

                    long convertStart = Stopwatch.GetTimestamp();
                    var horizonData = HorizonAngles.FromSlopes(output);
                    long convertElapsed = Stopwatch.GetTimestamp() - convertStart;
                    RecordPipelineStage(PipelineStageNames.RadiansToDegrees, convertElapsed);

                    return new PatchExecutionResult
                    {
                        HorizonData = horizonData,
                        Profile = new LaunchPatchProfile
                        {
                            WaitStreamTicks = streamWaitElapsed,
                            BufferResetTicks = bufferResetElapsed,
                            SegmentUploadTicks = segmentUploadElapsed,
                            KernelLaunchTicks = kernelLaunchElapsed,
                            StreamSyncTicks = syncElapsed,
                            OutputCopyTicks = copyElapsed,
                            ConvertTicks = convertElapsed,
#if QUADTREE_TRAVERSAL_PROFILE
                            TraversalCounters = traversalCounters
#endif
                        }
                    };
                });
            }
            finally
            {
                // 7. Return stream to pool for reuse
                _streamPool!.Push(stream);
                onStreamRelease();
            }
        }

        private static string FormatTimeSpan(TimeSpan ts)
        {
            if (ts.TotalHours >= 1)
                return $"{ts.Hours}h {ts.Minutes}m {ts.Seconds}s";
            else if (ts.TotalMinutes >= 1)
                return $"{ts.Minutes}m {ts.Seconds}s";
            else
                return $"{ts.Seconds}s";
        }

        private void RecordPipelineStage(string stageName, long elapsedTicks)
        {
            if (_enablePipelineProfiling)
                _pipelineProfiler.Record(stageName, elapsedTicks);
        }

        private void LogPatchProfiling(
            PatchDescriptor patch,
            int queueDepthOnDequeue,
            int queueDepthAfterEnqueue,
            int activeWorkers,
            int activeStreams,
            long segmentGenerationTicks,
            long patchEnqueueWaitTicks,
            long bufferAcquireTicks,
            long totalGpuWorkerPatchTicks,
            long fileWriteTicks,
            LaunchPatchProfile launchProfile)
        {
            if (!_enablePipelineProfiling)
                return;

            Log.Information(
                "PipelineProfile patch_index={PatchIndex} tile_x={TileX} tile_y={TileY} queue_depth_after_enqueue={QueueDepthAfterEnqueue} queue_depth_on_dequeue={QueueDepthOnDequeue} active_gpu_workers={ActiveGpuWorkers} active_streams={ActiveStreams} segment_generation_sec={SegmentGenerationSec:F4} patch_enqueue_wait_sec={PatchEnqueueWaitSec:F4} wait_buffer_sec={WaitBufferSec:F4} wait_stream_sec={WaitStreamSec:F4} buffer_reset_sec={BufferResetSec:F4} segment_upload_sec={SegmentUploadSec:F4} kernel_launch_sec={KernelLaunchSec:F4} stream_sync_sec={StreamSyncSec:F4} copy_back_sec={CopyBackSec:F4} convert_sec={ConvertSec:F4} write_sec={WriteSec:F4} gpu_worker_total_sec={GpuWorkerTotalSec:F4}",
                patch.Index,
                patch.TileX,
                patch.TileY,
                queueDepthAfterEnqueue,
                queueDepthOnDequeue,
                activeWorkers,
                activeStreams,
                StopwatchTicksToSeconds(segmentGenerationTicks),
                StopwatchTicksToSeconds(patchEnqueueWaitTicks),
                StopwatchTicksToSeconds(bufferAcquireTicks),
                StopwatchTicksToSeconds(launchProfile.WaitStreamTicks),
                StopwatchTicksToSeconds(launchProfile.BufferResetTicks),
                StopwatchTicksToSeconds(launchProfile.SegmentUploadTicks),
                StopwatchTicksToSeconds(launchProfile.KernelLaunchTicks),
                StopwatchTicksToSeconds(launchProfile.StreamSyncTicks),
                StopwatchTicksToSeconds(launchProfile.OutputCopyTicks),
                StopwatchTicksToSeconds(launchProfile.ConvertTicks),
                StopwatchTicksToSeconds(fileWriteTicks),
                StopwatchTicksToSeconds(totalGpuWorkerPatchTicks));

#if QUADTREE_TRAVERSAL_PROFILE
            LogTraversalProfiling(patch, launchProfile.TraversalCounters);
#endif
        }

#if QUADTREE_TRAVERSAL_PROFILE
        private static void LogTraversalProfiling(PatchDescriptor patch, long[]? counters)
        {
            if (counters == null || counters.Length == 0)
                return;

            int numDems = counters.Length / TRAVERSAL_COUNTERS_PER_DEM;
            for (int demPass = 0; demPass < numDems; demPass++)
            {
                int offset = demPass * TRAVERSAL_COUNTERS_PER_DEM;
                Log.Information(
                    "TraversalProfile patch_index={PatchIndex} tile_x={TileX} tile_y={TileY} dem_pass={DemPass} iterations={Iterations} level0_samples={Level0Samples} culled_blocks={CulledBlocks} out_of_bounds={OutOfBounds} nodata_skips={NoDataSkips}",
                    patch.Index,
                    patch.TileX,
                    patch.TileY,
                    demPass,
                    counters[offset + TRAVERSAL_COUNTER_ITERATIONS],
                    counters[offset + TRAVERSAL_COUNTER_LEVEL0_SAMPLES],
                    counters[offset + TRAVERSAL_COUNTER_CULLED_BLOCKS],
                    counters[offset + TRAVERSAL_COUNTER_OUT_OF_BOUNDS],
                    counters[offset + TRAVERSAL_COUNTER_NODATA_SKIPS]);
            }
        }
#endif

        private void LogPipelineAggregateSummary(int patchCount, TimeSpan pipelineElapsed)
        {
            if (!_enablePipelineProfiling)
                return;

            foreach (var entry in _pipelineProfiler.Snapshot())
            {
                var totalSec = StopwatchTicksToSeconds(entry.Value.TotalTicks);
                var avgSec = entry.Value.Count > 0 ? totalSec / entry.Value.Count : 0.0;
                var maxSec = StopwatchTicksToSeconds(entry.Value.MaxTicks);
                Log.Information(
                    "PipelineProfileSummary stage={Stage} samples={Samples} total_sec={TotalSec:F4} avg_sec={AvgSec:F4} max_sec={MaxSec:F4}",
                    entry.Key,
                    entry.Value.Count,
                    totalSec,
                    avgSec,
                    maxSec);
            }

            Log.Information(
                "PipelineProfileRun patch_count={PatchCount} pipeline_elapsed_sec={PipelineElapsedSec:F4} effective_patches_per_sec={PatchesPerSec:F4}",
                patchCount,
                pipelineElapsed.TotalSeconds,
                pipelineElapsed.TotalSeconds > 0 ? patchCount / pipelineElapsed.TotalSeconds : 0.0);
        }

        private static double StopwatchTicksToSeconds(long ticks) => ticks / (double)Stopwatch.Frequency;

        /// <summary>
        /// Generates horizon files for all 128x128 pixel patches within the inner (primary) DEM.
        /// This is a convenience method that generates all patches and calls GenerateHorizonsForPatches.
        /// For more control, use GeneratePatchList() with LINQ filtering and GenerateHorizonsForPatches().
        /// </summary>
        /// <param name="outputDirectory">Directory where horizon files will be written</param>
        /// <param name="dems">List of elevation maps (nested, with primary DEM first)</param>
        /// <param name="observerElevation">Observer height above terrain in meters</param>
        /// <param name="compressHorizons">If true, write compressed .cbin horizon tiles directly</param>
        /// <exception cref="ArgumentException">If DEM dimensions are not even multiples of 128</exception>
        public async Task GenerateHorizonsForAllPatches(string outputDirectory, List<ElevationMap> dems, float observerElevation = 0.0f, bool compressHorizons = false)
        {
            if (dems == null || dems.Count == 0)
                throw new ArgumentException("At least one DEM is required.", nameof(dems));
            
            var patches = GeneratePatchList(dems[0]);
            await GenerateHorizonsForPatches(outputDirectory, dems, patches, observerElevation, compressHorizons);
        }

        /// <summary>
        /// Helper struct to store pre-calculated segment data for a patch
        /// </summary>
        /// <summary>
        /// Pre-calculates ray segments.
        /// Hybrid Approach:
        /// - If far from pole (singularity), computes only Center Ray and assumes translation invariance (Compact Mode).
        /// Memory Layout: [Azimuth][Pixel][DEM] (Full) or [Azimuth][DEM] (Compact).
        /// </summary>
        /// <summary>
        /// Grid Convergence data for Compact Mode rotation compensation.
        /// </summary>
        public readonly struct GridConvergenceInfo
        {
            public readonly float GammaCenter;  // Grid Convergence at tile center (radians)
            public readonly float DGammaDx;     // Gradient of gamma per pixel in X direction (radians/pixel)
            public readonly float DGammaDy;     // Gradient of gamma per pixel in Y direction (radians/pixel)

            public GridConvergenceInfo(float gammaCenter, float dGammaDx, float dGammaDy)
            {
                GammaCenter = gammaCenter;
                DGammaDx = dGammaDx;
                DGammaDy = dGammaDy;
            }

            public static GridConvergenceInfo Zero => new(0f, 0f, 0f);
        }

        internal readonly record struct SubpatchCenterDiagnostic(
            int Index,
            int GridRow,
            int GridColumn,
            int RequestedCenterColumn,
            int RequestedCenterRow,
            int SegmentCenterColumn,
            int SegmentCenterRow);

        internal readonly record struct TraversalStepDiagnostic(
            int Sequence,
            float ParameterDistanceKm,
            float TrueDistanceMeters,
            int Level,
            int CellX,
            int CellY,
            float PixelX,
            float PixelY,
            float MaximumElevationMeters,
            float SampleElevationMeters,
            float SampleSlope,
            float AdvanceKm,
            int Action);

        private readonly struct DemSegmentContext
        {
            public ElevationMap Dem { get; init; }
            public double MapRes { get; init; }
            public double RayLimitMeters { get; init; }
        }

        internal readonly struct RaySample
        {
            public double DistanceMeters { get; init; }
            public double PixelX { get; init; }
            public double PixelY { get; init; }
            public double LatRad { get; init; }
            public double LonRad { get; init; }
            public double Row { get; init; }
            public double Col { get; init; }
            public double TerrainHeightMeters { get; init; }
        }

        private (RaySegment[], bool, GridConvergenceInfo) CalculateRaySegments(
            List<Pyramid> pyramids,
            PyramidView primaryPV,
            int tileColBase, int tileRowBase, int tileW, int tileH,
            int numAzimuths, float maxDist,
            float observerElevation)
        {
            int numPixels = tileW * tileH;
            int numDems = pyramids.Count;
            double beamStep = (2.0 * Math.PI) / numAzimuths;
            var primaryDem = pyramids[0].SourceDem ?? throw new InvalidOperationException("Primary DEM missing from pyramid.");

            // Check Proximity to Pole (Singularity)
            // Grid Convergence changes rapidly near the Geographic Poles.
            // Regardless of Projection type (Polar vs Oblique), we must use Full Mode near poles.
            double centerCol = tileColBase + tileW / 2.0;
            double centerRow = tileRowBase + tileH / 2.0;

            // Build Double Projection Params for Primary
            var primaryProjD = BuildProjectionParamsDouble(primaryDem);
            var centerPixel = new PixelPoint(centerCol, centerRow);
            var centerCrs = primaryDem.PixelToCRS(centerPixel);
            var (centerLat, centerLon) = InverseProjectDouble(centerCrs.X, centerCrs.Y, primaryProjD);
            double centerLatRad = centerLat;
            double centerLonRad = centerLon;
            double centerTerrain = primaryDem.GetElevation(centerCol, centerRow);
            double observerHeightMeters = centerTerrain + observerElevation;
            var observerVecCenter = LatLonToVectorMeters(centerLatRad, centerLonRad, primaryProjD.R + observerHeightMeters);
            var obsToMeCenter = GetRotationMatrixd(centerLatRad, centerLonRad);

            // Compute Grid Convergence at tile center and gradients for Compact Mode rotation compensation
            var srs = primaryDem.SrsDescriptor;
            var gcInfo = GridConvergenceInfo.Zero;
            if (srs != null && srs.Type == SrsDescriptor.ProjType.Stereographic)
            {
                // Grid Convergence at center
                var (_, gammaCenter) = MoonSrsLambdaFactory.GetDistortion(new CRSPoint(centerLonRad, centerLatRad), srs);

                // Compute gradients by sampling Grid Convergence at offset points (±1 pixel)
                // Use the same approach as GetDistortion but sample at offset locations
                double offsetPixels = 1.0;

                // Sample at (+1, 0) in pixel space
                var rightPixel = new PixelPoint(centerCol + offsetPixels, centerRow);
                var rightCrs = primaryDem.PixelToCRS(rightPixel);
                var (rightLat, rightLon) = InverseProjectDouble(rightCrs.X, rightCrs.Y, primaryProjD);
                var (_, gammaRight) = MoonSrsLambdaFactory.GetDistortion(new CRSPoint(rightLon, rightLat), srs);

                // Sample at (0, +1) in pixel space (note: +Y in pixel space is typically South)
                var downPixel = new PixelPoint(centerCol, centerRow + offsetPixels);
                var downCrs = primaryDem.PixelToCRS(downPixel);
                var (downLat, downLon) = InverseProjectDouble(downCrs.X, downCrs.Y, primaryProjD);
                var (_, gammaDown) = MoonSrsLambdaFactory.GetDistortion(new CRSPoint(downLon, downLat), srs);

                double dGammaDx = (gammaRight - gammaCenter) / offsetPixels;
                double dGammaDy = (gammaDown - gammaCenter) / offsetPixels;

                gcInfo = new GridConvergenceInfo((float)gammaCenter, (float)dGammaDx, (float)dGammaDy);
                Log.Debug("Grid Convergence: center={GammaCenter:F6}rad, dGamma/dx={DGammaDx:F8}rad/px, dGamma/dy={DGammaDy:F8}rad/px",
                    gammaCenter, dGammaDx, dGammaDy);
            }

            // Always use Compact Mode - Grid Convergence rotation compensation handles all latitudes

            int sampleDumpAzIdx = -1;
            int sampleDumpDemIdx = -1;
            string? sampleAzEnv = Environment.GetEnvironmentVariable("QUADTREE_DEBUG_AZ");
            if (!string.IsNullOrEmpty(sampleAzEnv) && int.TryParse(sampleAzEnv, out int parsedAz))
                sampleDumpAzIdx = parsedAz;
            string? sampleDemEnv = Environment.GetEnvironmentVariable("QUADTREE_DEBUG_DEM");
            if (!string.IsNullOrEmpty(sampleDemEnv) && int.TryParse(sampleDemEnv, out int parsedDem))
                sampleDumpDemIdx = parsedDem;
            string? sampleDumpPath = null;
            if (sampleDumpAzIdx >= 0 && sampleDumpDemIdx >= 0)
                sampleDumpPath = Path.Combine(Directory.GetCurrentDirectory(), $"quadtree_samples_dem{sampleDumpDemIdx}.txt");

            var compactStopwatch = Stopwatch.StartNew();
            // COMPACT MODE: Center Ray
            var segments = new RaySegment[numAzimuths * numDems];
            double R = primaryProjD.R;
            var demContexts = BuildDemSegmentContexts(pyramids, maxDist);

            Parallel.For(0, numAzimuths, azIdx =>
            {
                double az = azIdx * beamStep;
                double demStartDistMeters = 1.0;  // Start at 1m to match reference behavior
                if (azIdx == 0) Log.Debug("Compact mode: demStartDistMeters = {Distance}m", demStartDistMeters);
                int baseIdx = azIdx * numDems;
                var dirMe = ComputeDirectionVector(obsToMeCenter, az);
                Span<RaySample> sampleBuffer = stackalloc RaySample[MAX_RAY_SAMPLE_CAPACITY];

                for (int i = 0; i < numDems; i++)
                {
                    var demContext = demContexts[i];

                    int sampleCount = BuildRaySamples(
                        observerVecCenter,
                        dirMe,
                        demStartDistMeters, demContext.RayLimitMeters,
                        demContext.Dem, demContext.MapRes, sampleBuffer);

                    var samples = sampleBuffer[..sampleCount];

                    if (sampleCount < 3)
                    {
                        float fallbackX = sampleCount > 0 ? (float)samples[0].PixelX : 0f;
                        float fallbackY = sampleCount > 0 ? (float)samples[0].PixelY : 0f;
                        float fallbackS = sampleCount > 0 ? (float)(samples[0].DistanceMeters * METERS_TO_KILOMETERS_D) : (float)(demStartDistMeters * METERS_TO_KILOMETERS_D);
                        segments[baseIdx + i] = new RaySegment
                        {
                            StartPixel = new Vector2(fallbackX, fallbackY),
                            DemId = i,
                            X0 = fallbackX,
                            Y0 = fallbackY,
                            A1 = 0,
                            A2 = 0,
                            A3 = 0,
                            A4 = 0,
                            B1 = 0,
                            B2 = 0,
                            B3 = 0,
                            B4 = 0,
                            SStart = fallbackS,
                            SEnd = fallbackS,
                            SStartChord = fallbackS,
                            PlanarToChordC1 = 1f,
                            PlanarToChordC2 = 0f,
                            PlanarToChordC3 = 0f
                        };
                        if (sampleDumpPath != null && sampleDumpDemIdx == i && sampleDumpAzIdx == azIdx)
                        {
                            try
                            {
                                File.WriteAllLines(sampleDumpPath, FormatSampleLines(samples));
                            }
                            catch (Exception ex)
                            {
                                Log.Error("Failed to write sample dump: {ErrorMessage}", ex.Message);
                            }
                        }
                    }
                    else
                    {
                        segments[baseIdx + i] = FitRaySegment(samples, demContext.MapRes, observerVecCenter, R + centerTerrain, demContext.Dem, demStartDistMeters, i);
                        if (sampleDumpPath != null && sampleDumpDemIdx == i && sampleDumpAzIdx == azIdx)
                        {
                            try
                            {
                                File.WriteAllLines(sampleDumpPath, FormatSampleLines(samples));
                            }
                            catch (Exception ex)
                            {
                                Log.Error("Failed to write sample dump: {ErrorMessage}", ex.Message);
                            }
                        }
                        demStartDistMeters = Math.Min(demContext.RayLimitMeters, samples[sampleCount - 1].DistanceMeters);
                    }
                    if (demStartDistMeters >= maxDist) break;
                }
            });
            Log.Information($"Segment creation (compact) took {compactStopwatch.Elapsed.TotalSeconds:F2} sec");
            return (segments, true, gcInfo);
        }

        private sealed class SubpatchSegmentCache
        {
            private readonly List<Pyramid> _pyramids;
            private readonly ElevationMap _primaryDem;
            private readonly ProjectionParamsDouble _primaryProjD;
            private readonly int _numAzimuths;
            private readonly float _maxDist;
            private readonly float _observerElevation;
            private readonly int _subpatchSize;
            private readonly int _numDems;
            private readonly double _beamStep;
            private readonly double _radius;
            private readonly DemSegmentContext[] _demContexts;
            private readonly Dictionary<(int CenterCol, int CenterRow), RaySegment[]> _segmentsByCenter = new();

            public SubpatchSegmentCache(
                List<Pyramid> pyramids,
                int numAzimuths,
                float maxDist,
                float observerElevation,
                int subpatchSize)
            {
                _pyramids = pyramids;
                _primaryDem = pyramids[0].SourceDem ?? throw new InvalidOperationException("Primary DEM missing from pyramid.");
                _primaryProjD = BuildProjectionParamsDouble(_primaryDem);
                _numAzimuths = numAzimuths;
                _maxDist = maxDist;
                _observerElevation = observerElevation;
                _subpatchSize = subpatchSize;
                _numDems = pyramids.Count;
                _beamStep = (2.0 * Math.PI) / numAzimuths;
                _radius = _primaryProjD.R;
                _demContexts = BuildDemSegmentContexts(pyramids, maxDist);
            }

            public RaySegment[] GetCenterSegments(int centerCol, int centerRow)
            {
                var key = (centerCol, centerRow);
                if (_segmentsByCenter.TryGetValue(key, out var cached))
                    return cached;

                var computed = ComputeCenterSegments(centerCol, centerRow);
                _segmentsByCenter.Add(key, computed);
                return computed;
            }

            private RaySegment[] ComputeCenterSegments(int centerCol, int centerRow)
            {
                var centerPixel = new PixelPoint(centerCol, centerRow);
                var centerCrs = _primaryDem.PixelToCRS(centerPixel);
                var (centerLat, centerLon) = InverseProjectDouble(centerCrs.X, centerCrs.Y, _primaryProjD);
                double terrainCol = ClampDouble(centerCol, 0.0, _primaryDem.Width - 1.001);
                double terrainRow = ClampDouble(centerRow, 0.0, _primaryDem.Height - 1.001);
                double centerTerrain = _primaryDem.GetElevation(terrainCol, terrainRow);
                double observerHeightMeters = centerTerrain + _observerElevation;
                var observerVec = LatLonToVectorMeters(centerLat, centerLon, _primaryProjD.R + observerHeightMeters);
                var obsToMe = GetRotationMatrixd(centerLat, centerLon);
                var centerSegments = new RaySegment[_numAzimuths * _numDems];

                Parallel.For(0, _numAzimuths, azIdx =>
                {
                    double az = azIdx * _beamStep;
                    var dirMe = ComputeDirectionVector(obsToMe, az);
                    double demStartDistMeters = 1.0;
                    int baseIdx = azIdx * _numDems;
                    Span<RaySample> sampleBuffer = stackalloc RaySample[MAX_RAY_SAMPLE_CAPACITY];

                    for (int i = 0; i < _numDems; i++)
                    {
                        var demContext = _demContexts[i];
                        int sampleCount = BuildRaySamples(
                            observerVec,
                            dirMe,
                            demStartDistMeters, demContext.RayLimitMeters,
                            demContext.Dem, demContext.MapRes, sampleBuffer);

                        centerSegments[baseIdx + i] = FitRaySegment(sampleBuffer[..sampleCount], demContext.MapRes, observerVec, _radius + centerTerrain, demContext.Dem, demStartDistMeters, i);

                        double lastS = sampleCount > 0 ? sampleBuffer[sampleCount - 1].DistanceMeters : demStartDistMeters;
                        demStartDistMeters = Math.Min(demContext.RayLimitMeters, lastS);
                        if (demStartDistMeters >= _maxDist) break;
                    }
                });

                return centerSegments;
            }
        }

        private static DemSegmentContext[] BuildDemSegmentContexts(List<Pyramid> pyramids, float maxDist)
        {
            var contexts = new DemSegmentContext[pyramids.Count];
            for (int i = 0; i < pyramids.Count; i++)
            {
                var pyramid = pyramids[i];
                var dem = pyramid.SourceDem ?? throw new InvalidOperationException("Pyramid missing DEM reference.");
                double pixCol = Math.Sqrt(dem.GeoTransform[1] * dem.GeoTransform[1] + dem.GeoTransform[4] * dem.GeoTransform[4]);
                double pixRow = Math.Sqrt(dem.GeoTransform[2] * dem.GeoTransform[2] + dem.GeoTransform[5] * dem.GeoTransform[5]);
                double mapRes = (pixCol + pixRow) * 0.5;
                double demWidthM = pyramid.CpuInfos![0].Width * mapRes;
                double demHeightM = pyramid.CpuInfos![0].Height * mapRes;
                double demSizeM = Math.Min(demWidthM, demHeightM);

                contexts[i] = new DemSegmentContext
                {
                    Dem = dem,
                    MapRes = mapRes,
                    RayLimitMeters = Math.Min(maxDist, demSizeM * 1.2)
                };
            }

            return contexts;
        }

        private static string[] FormatSampleLines(ReadOnlySpan<RaySample> samples)
        {
            if (samples.Length == 0)
                return new[] { "EMPTY" };

            var lines = new string[samples.Length];
            for (int i = 0; i < samples.Length; i++)
                lines[i] = $"{samples[i].DistanceMeters * METERS_TO_KILOMETERS_D:F6}:{samples[i].PixelX:F6}:{samples[i].PixelY:F6}";
            return lines;
        }

        private static RaySegment FitRaySegment(
            ReadOnlySpan<RaySample> samples,
            double mapRes,
            Vector3d observerVec,
            double correctionSphereRadius,
            ElevationMap dem,
            double fallbackStartMeters,
            int demId)
        {
            if (samples.Length < 3)
            {
                float fallbackX = samples.Length > 0 ? (float)samples[0].PixelX : 0f;
                float fallbackY = samples.Length > 0 ? (float)samples[0].PixelY : 0f;
                float fallbackS = samples.Length > 0 ? (float)(samples[0].DistanceMeters * METERS_TO_KILOMETERS_D) : (float)(fallbackStartMeters * METERS_TO_KILOMETERS_D);
                return new RaySegment
                {
                    StartPixel = new Vector2(fallbackX, fallbackY),
                    DemId = demId,
                    X0 = fallbackX,
                    Y0 = fallbackY,
                    A1 = 0,
                    A2 = 0,
                    A3 = 0,
                    A4 = 0,
                    B1 = 0,
                    B2 = 0,
                    B3 = 0,
                    B4 = 0,
                    SStart = fallbackS,
                    SEnd = fallbackS,
                    SStartChord = fallbackS,
                    PlanarToChordC1 = 1f,
                    PlanarToChordC2 = 0f,
                    PlanarToChordC3 = 0f
                };
            }

            double x0 = samples[0].PixelX;
            double y0 = samples[0].PixelY;
            double sAnchorKm = samples[0].DistanceMeters * METERS_TO_KILOMETERS_D;
            int n = samples.Length;
            double sEndKmDouble = samples[n - 1].DistanceMeters * METERS_TO_KILOMETERS_D;
            double spanKm = Math.Max(0.001, sEndKmDouble - sAnchorKm);
            float sStart = (float)sAnchorKm;
            float sEnd = (float)sEndKmDouble;

            Span<double> sArr = stackalloc double[MAX_RAY_SAMPLE_CAPACITY];
            Span<double> vx = stackalloc double[MAX_RAY_SAMPLE_CAPACITY];
            Span<double> vy = stackalloc double[MAX_RAY_SAMPLE_CAPACITY];
            for (int k = 0; k < n; k++)
            {
                double sampleKm = samples[k].DistanceMeters * METERS_TO_KILOMETERS_D;
                double ds = (sampleKm - sAnchorKm) / spanKm;
                sArr[k] = ds;
                vx[k] = samples[k].PixelX - x0;
                vy[k] = samples[k].PixelY - y0;
            }

            FitQuartic4TermsDouble(sArr[..n], vx[..n], out double a1, out double a2, out double a3, out double a4);
            FitQuartic4TermsDouble(sArr[..n], vy[..n], out double b1, out double b2, out double b3, out double b4);
            double chordC1;
            double chordC2;
            double chordC3;
            if (UseDemElevationChordCorrection)
            {
                FitPlanarToChordCubicWithTerrain(samples, mapRes, observerVec, correctionSphereRadius, dem, out double chordC1Dem, out double chordC2Dem, out double chordC3Dem);
                chordC1 = chordC1Dem;
                chordC2 = chordC2Dem;
                chordC3 = chordC3Dem;
            }
            else
            {
                FitPlanarToChordCubicWithTerrain(samples, mapRes, observerVec, correctionSphereRadius, out double chordC1Sphere, out double chordC2Sphere, out double chordC3Sphere);
                chordC1 = chordC1Sphere;
                chordC2 = chordC2Sphere;
                chordC3 = chordC3Sphere;
            }

            double inv = 1.0 / spanKm;
            double inv2 = inv * inv;
            double inv3 = inv2 * inv;
            double inv4 = inv2 * inv2;
            a1 *= inv;
            a2 *= inv2;
            a3 *= inv3;
            a4 *= inv4;
            b1 *= inv;
            b2 *= inv2;
            b3 *= inv3;
            b4 *= inv4;

            return new RaySegment
            {
                StartPixel = new Vector2((float)x0, (float)y0),
                DemId = demId,
                X0 = (float)x0,
                Y0 = (float)y0,
                A1 = (float)a1,
                A2 = (float)a2,
                A3 = (float)a3,
                A4 = (float)a4,
                B1 = (float)b1,
                B2 = (float)b2,
                B3 = (float)b3,
                B4 = (float)b4,
                SStart = sStart,
                SEnd = sEnd,
                SStartChord = sStart,
                PlanarToChordC1 = (float)chordC1,
                PlanarToChordC2 = (float)chordC2,
                PlanarToChordC3 = (float)chordC3
            };
        }

        internal static RaySegment FitRaySegmentForDiagnostics(
            ReadOnlySpan<RaySample> samples,
            double mapRes,
            Vector3d observerVec,
            double correctionSphereRadius,
            ElevationMap dem,
            double fallbackStartMeters,
            int demId) => FitRaySegment(
                samples,
                mapRes,
                observerVec,
                correctionSphereRadius,
                dem,
                fallbackStartMeters,
                demId);

        /// <summary>
        /// Calculates subpatch ray segments for improved polynomial approximation accuracy.
        /// Instead of fitting one polynomial per azimuth at the 128x128 patch center,
        /// this method fits polynomials at multiple subpatch centers to reduce approximation errors.
        /// </summary>
        /// <param name="subpatchSize">Size of each subpatch (8, 16, 32, or 64 pixels)</param>
        /// <returns>Array of segments with layout [Azimuth][Subpatch][DEM]</returns>
        private (RaySegment[], GridConvergenceInfo) CalculateSubpatchRaySegments(
            List<Pyramid> pyramids,
            PyramidView primaryPV,
            int tileColBase, int tileRowBase, int tileW, int tileH,
            int numAzimuths, float maxDist,
            float observerElevation, int subpatchSize = DEFAULT_SUBPATCH_SIZE,
            SubpatchSegmentCache? segmentCache = null)
        {
            if (!IsValidSubpatchSize(subpatchSize))
                throw new ArgumentException("Subpatch size must be 2, 4, 8, 16, 32, 64 or 128", nameof(subpatchSize));
            if (128 % subpatchSize != 0)
                throw new ArgumentException("128 must be evenly divisible by subpatch size", nameof(subpatchSize));

            int interiorSubpatchesPerDim = tileW / subpatchSize;
            int numSubpatchesPerDim = interiorSubpatchesPerDim + 2;
            int numSubpatches = numSubpatchesPerDim * numSubpatchesPerDim;
            int numDems = pyramids.Count;
            double beamStep = (2.0 * Math.PI) / numAzimuths;

            Log.Debug("Calculating subpatch ray segments: {SubpatchSize}x{SubpatchSize} subpatches, {NumSubpatches} total subpatches with interpolation halo",
                subpatchSize, subpatchSize, numSubpatches);

            var primaryDem = pyramids[0].SourceDem ?? throw new InvalidOperationException("Primary DEM missing from pyramid.");

            // Build Double Projection Params for Primary
            var primaryProjD = BuildProjectionParamsDouble(primaryDem);

            // Calculate Grid Convergence at tile center
            double centerCol = tileColBase + tileW / 2.0;
            double centerRow = tileRowBase + tileH / 2.0;
            var centerPixel = new PixelPoint(centerCol, centerRow);
            var centerCrs = primaryDem.PixelToCRS(centerPixel);
            var (centerLat, centerLon) = InverseProjectDouble(centerCrs.X, centerCrs.Y, primaryProjD);

            var srs = primaryDem.SrsDescriptor;
            var gcInfo = GridConvergenceInfo.Zero;
            if (srs != null && srs.Type == SrsDescriptor.ProjType.Stereographic)
            {
                // Grid Convergence at center  
                var (_, gammaCenter) = MoonSrsLambdaFactory.GetDistortion(new CRSPoint(centerLon, centerLat), srs);

                // Compute gradients
                double offsetPixels = 1.0;
                var rightPixel = new PixelPoint(centerCol + offsetPixels, centerRow);
                var rightCrs = primaryDem.PixelToCRS(rightPixel);
                var (rightLat, rightLon) = InverseProjectDouble(rightCrs.X, rightCrs.Y, primaryProjD);
                var (_, gammaRight) = MoonSrsLambdaFactory.GetDistortion(new CRSPoint(rightLon, rightLat), srs);

                var downPixel = new PixelPoint(centerCol, centerRow + offsetPixels);
                var downCrs = primaryDem.PixelToCRS(downPixel);
                var (downLat, downLon) = InverseProjectDouble(downCrs.X, downCrs.Y, primaryProjD);
                var (_, gammaDown) = MoonSrsLambdaFactory.GetDistortion(new CRSPoint(downLon, downLat), srs);

                double dGammaDx = (gammaRight - gammaCenter) / offsetPixels;
                double dGammaDy = (gammaDown - gammaCenter) / offsetPixels;

                gcInfo = new GridConvergenceInfo((float)gammaCenter, (float)dGammaDx, (float)dGammaDy);
            }

            // Segments layout: [Azimuth][Subpatch][DEM]
            var segments = new RaySegment[numAzimuths * numSubpatches * numDems];
            var compactStopwatch = Stopwatch.StartNew();
            segmentCache ??= new SubpatchSegmentCache(pyramids, numAzimuths, maxDist, observerElevation, subpatchSize);

            for (int subpatchIdx = 0; subpatchIdx < numSubpatches; subpatchIdx++)
            {
                int subpatchRow = subpatchIdx / numSubpatchesPerDim;
                int subpatchCol = subpatchIdx % numSubpatchesPerDim;
                int requestedCenterCol = tileColBase + (subpatchCol - 1) * subpatchSize + subpatchSize / 2;
                int requestedCenterRow = tileRowBase + (subpatchRow - 1) * subpatchSize + subpatchSize / 2;
                int segmentCenterCol = ClampSubpatchCenter(requestedCenterCol, primaryDem.Width, subpatchSize);
                int segmentCenterRow = ClampSubpatchCenter(requestedCenterRow, primaryDem.Height, subpatchSize);
                var centerSegments = segmentCache.GetCenterSegments(segmentCenterCol, segmentCenterRow);

                for (int azIdx = 0; azIdx < numAzimuths; azIdx++)
                {
                    Array.Copy(
                        centerSegments,
                        azIdx * numDems,
                        segments,
                        (azIdx * numSubpatches + subpatchIdx) * numDems,
                        numDems);
                }
            }

            Log.Debug("Subpatch segment creation took {ElapsedSeconds:F2} sec", compactStopwatch.Elapsed.TotalSeconds);
            return (segments, gcInfo);
        }

        internal (
            RaySegment[] Segments,
            GridConvergenceInfo GridConvergence,
            SubpatchCenterDiagnostic[] Centers) CalculateSubpatchRaySegmentsForDiagnostics(
                List<ElevationMap> dems,
                int tileColumn,
                int tileRow,
                int tileWidth,
                int tileHeight,
                int numAzimuths,
                float maxDistanceMeters,
                float observerElevationMeters,
                int subpatchSize)
        {
            var pyramids = dems.Select(BuildOrLoadPyramid).ToList();
            try
            {
                var primary = pyramids[0];
                var primaryView = new PyramidView
                {
                    DataLevel0 = primary.DataLevel0!.View,
                    DataMips = primary.DataMips!.View,
                    Infos = primary.Infos!.View,
                    Map = primary.Map,
                    Proj = primary.Proj,
                    Levels = primary.CpuInfos!.Length,
                };
                var (segments, gridConvergence) = CalculateSubpatchRaySegments(
                    pyramids,
                    primaryView,
                    tileColumn,
                    tileRow,
                    tileWidth,
                    tileHeight,
                    numAzimuths,
                    maxDistanceMeters,
                    observerElevationMeters,
                    subpatchSize);

                int interiorPerDimension = tileWidth / subpatchSize;
                int centersPerDimension = interiorPerDimension + 2;
                var centers = new SubpatchCenterDiagnostic[
                    centersPerDimension * centersPerDimension];
                for (int index = 0; index < centers.Length; index++)
                {
                    int gridRow = index / centersPerDimension;
                    int gridColumn = index % centersPerDimension;
                    int requestedColumn = tileColumn +
                        (gridColumn - 1) * subpatchSize + subpatchSize / 2;
                    int requestedRow = tileRow +
                        (gridRow - 1) * subpatchSize + subpatchSize / 2;
                    centers[index] = new SubpatchCenterDiagnostic(
                        index,
                        gridRow,
                        gridColumn,
                        requestedColumn,
                        requestedRow,
                        ClampSubpatchCenter(
                            requestedColumn, dems[0].Width, subpatchSize),
                        ClampSubpatchCenter(
                            requestedRow, dems[0].Height, subpatchSize));
                }
                return (segments, gridConvergence, centers);
            }
            finally
            {
                foreach (var pyramid in pyramids)
                    pyramid.Dispose();
            }
        }

        internal (
            float[][] PerDemSlopes,
            float[] FinalSlopes,
            float[] FinalDegrees,
            GridConvergenceInfo GridConvergence,
            int TraversalTraceDemPass,
            int TraversalTraceAzimuthIndex,
            TraversalStepDiagnostic[] TraversalTrace) CaptureSubpatchBuffersForDiagnostics(
                List<ElevationMap> dems,
                int tileColumn,
                int tileRow,
                int tileWidth,
                int tileHeight,
                float observerElevationMeters,
                int subpatchSize)
        {
            const int numAzimuths = 1440;
            const float maxDistanceMeters = 1000000f;
            const int traceDemPass = 1;
            const int traceAzimuthIndex = 360;
            const int traceFieldCount = 12;
            const int traceCapacity = 16384;
            var pyramids = dems.Select(BuildOrLoadPyramid).ToList();
            try
            {
                PyramidView View(Pyramid pyramid) => new()
                {
                    DataLevel0 = pyramid.DataLevel0!.View,
                    DataMips = pyramid.DataMips!.View,
                    Infos = pyramid.Infos!.View,
                    Map = pyramid.Map,
                    Proj = pyramid.Proj,
                    Levels = pyramid.CpuInfos!.Length,
                };

                var primaryView = View(pyramids[0]);
                var (segments, gridConvergence) = CalculateSubpatchRaySegments(
                    pyramids,
                    primaryView,
                    tileColumn,
                    tileRow,
                    tileWidth,
                    tileHeight,
                    numAzimuths,
                    maxDistanceMeters,
                    observerElevationMeters,
                    subpatchSize);
                int outputLength = tileWidth * tileHeight * numAzimuths;
                var reset = Enumerable.Repeat(float.NegativeInfinity, outputLength).ToArray();
                var perDemSlopes = new float[dems.Count][];
                using var gpuSegments = _accelerator.Allocate1D(segments);
                using var gpuOutput = _accelerator.Allocate1D(reset);
                using var debugBuffer = _accelerator.Allocate1D<float>(
                    1 + traceCapacity * traceFieldCount);
                using var stream = _accelerator.CreateStream();
#if QUADTREE_TRAVERSAL_PROFILE
                using var traversalCounters = _accelerator.Allocate1D<long>(
                    dems.Count * TRAVERSAL_COUNTERS_PER_DEM);
#endif
                float[]? traceBuffer = null;

                for (int demPass = 0; demPass < dems.Count; demPass++)
                {
                    gpuOutput.CopyFromCPU(reset);
                    debugBuffer.MemSetToZero();
#if QUADTREE_TRAVERSAL_PROFILE
                    traversalCounters.CopyFromCPU(
                        new long[dems.Count * TRAVERSAL_COUNTERS_PER_DEM]);
#endif
                    var kernelParameters = new KernelParams(
                        observerElevationMeters,
                        0f,
                        demPass == traceDemPass ? traceAzimuthIndex : -1,
                        gridConvergence.GammaCenter,
                        gridConvergence.DGammaDx,
                        gridConvergence.DGammaDy,
                        GetSubpatchDebugFlags(),
                        pyramids[0].CpuInfos![0].Width,
                        pyramids[0].CpuInfos![0].Height);
                    _subpatchKernel!(
                        stream,
                        new Index2D(tileWidth * tileHeight, numAzimuths),
                        primaryView,
                        View(pyramids[demPass]),
                        gpuOutput.View,
                        gpuSegments.View,
                        demPass,
                        dems.Count,
                        tileColumn,
                        tileRow,
                        tileWidth,
                        tileHeight,
                        subpatchSize,
                        kernelParameters,
                        debugBuffer.View
#if QUADTREE_TRAVERSAL_PROFILE
                        , traversalCounters.View
#endif
                        );
                    stream.Synchronize();
                    perDemSlopes[demPass] = gpuOutput.GetAsArray1D();
                    if (demPass == traceDemPass)
                        traceBuffer = debugBuffer.GetAsArray1D();
                }

                if (traceBuffer is null)
                    throw new InvalidOperationException(
                        "Traversal trace DEM pass was not executed.");
                int traceCount = Math.Min((int)traceBuffer[0], traceCapacity);
                var traversalTrace = new TraversalStepDiagnostic[traceCount];
                for (int index = 0; index < traceCount; index++)
                {
                    int offset = 1 + index * traceFieldCount;
                    traversalTrace[index] = new TraversalStepDiagnostic(
                        index,
                        traceBuffer[offset],
                        traceBuffer[offset + 1],
                        (int)traceBuffer[offset + 2],
                        (int)traceBuffer[offset + 3],
                        (int)traceBuffer[offset + 4],
                        traceBuffer[offset + 5],
                        traceBuffer[offset + 6],
                        traceBuffer[offset + 7],
                        traceBuffer[offset + 8],
                        traceBuffer[offset + 9],
                        traceBuffer[offset + 10],
                        (int)traceBuffer[offset + 11]);
                }

                var finalSlopes = Enumerable.Repeat(
                    float.NegativeInfinity, outputLength).ToArray();
                foreach (var pass in perDemSlopes)
                    for (int index = 0; index < outputLength; index++)
                        finalSlopes[index] = Math.Max(finalSlopes[index], pass[index]);
                return (
                    perDemSlopes,
                    finalSlopes,
                    HorizonAngles.FromSlopes((float[])finalSlopes.Clone()).Degrees,
                    gridConvergence,
                    traceDemPass,
                    traceAzimuthIndex,
                    traversalTrace);
            }
            finally
            {
                foreach (var pyramid in pyramids)
                    pyramid.Dispose();
            }
        }

        private static double ClampDouble(double value, double min, double max)
        {
            if (value < min) return min;
            if (value > max) return max;
            return value;
        }

        private static int ClampSubpatchCenter(int requestedCenter, int demSize, int subpatchSize)
        {
            int half = subpatchSize / 2;
            int minCenter = half;
            int maxCenter = Math.Max(minCenter, demSize - half);
            if (requestedCenter < minCenter) return minCenter;
            if (requestedCenter > maxCenter) return maxCenter;
            return requestedCenter;
        }

        static void IntersectRayBounds(Vector2 origin, Vector2 dir, PixelBounds bounds, out float tEnter, out float tExit)
        {
            float divX = 1.0f / (Math.Abs(dir.X) > 1e-8f ? dir.X : 1e30f);
            float divY = 1.0f / (Math.Abs(dir.Y) > 1e-8f ? dir.Y : 1e30f);

            float t1 = (bounds.MinX - origin.X) * divX;
            float t2 = (bounds.MaxX - origin.X) * divX;
            float t3 = (bounds.MinY - origin.Y) * divY;
            float t4 = (bounds.MaxY - origin.Y) * divY;

            float tMin = Math.Max(Math.Min(t1, t2), Math.Min(t3, t4));
            float tMax = Math.Min(Math.Max(t1, t2), Math.Max(t3, t4));

            tEnter = tMin;
            tExit = tMax;
        }

        private Pyramid BuildOrLoadPyramid(ElevationMap dem)
        {
            // Calculate total size and levels
            int w = dem.Width;
            int h = dem.Height;
            var levels = new List<LevelInfo>();

            // Level 0
            // Offset 0 relative to DataLevel0
            levels.Add(new LevelInfo { Offset = 0, Width = w, Height = h });

            int mipsOffset = 0;
            // Subsequent levels (downsample by PYR_DOWNSAMPLE_FACTOR per level)
            while (w > 1 || h > 1)
            {
                w = (w + (PYR_DOWNSAMPLE_FACTOR - 1)) / PYR_DOWNSAMPLE_FACTOR;
                h = (h + (PYR_DOWNSAMPLE_FACTOR - 1)) / PYR_DOWNSAMPLE_FACTOR;
                levels.Add(new LevelInfo { Offset = mipsOffset, Width = w, Height = h });
                mipsOffset += w * h;
            }

            string? cachePath = null;
            if (!string.IsNullOrEmpty(dem.Path))
                cachePath = Path.ChangeExtension(dem.Path!, ".pyr.bin");

            float[] mipsHost = new float[mipsOffset]; // Initialize here to guarantee assignment

            bool cacheExists = cachePath != null && File.Exists(cachePath);
            if (cacheExists)
            {
                try {
                    float[]? cachedData = Utilities.LoadBinaryArray<float>(cachePath!);
                    if (cachedData != null && cachedData.Length == mipsOffset)
                    {
                        mipsHost = cachedData;
                    }
                    else
                    {
                        Log.Warning("Cache size mismatch or invalid. Rebuilding.");
                        cacheExists = false;
                    }
                } catch (Exception ex) {
                    Log.Error("Error loading cache: {ErrorMessage}. Rebuilding.", ex.Message);
                    cacheExists = false;
                }
            }

            // Flatten Level 0 from ElevationMap
            float[] level0Host = FlattenElevation(dem.Elevation);

            // Upload Level 0 to GPU
            var pyramid = new Pyramid();
            pyramid.DataLevel0 = _accelerator.Allocate1D<float>(level0Host.Length);
            pyramid.DataLevel0.CopyFromCPU(level0Host);

            if (!cacheExists) // Rebuild logic for Mips
            {
                if (cachePath != null)
                    Log.Information("Building pyramid for {DemPath}...", Path.GetFileName(dem.Path!));

                // Allocate Mips on GPU
                pyramid.DataMips = _accelerator.Allocate1D<float>(mipsOffset);

                var downsampleKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<float>, ArrayView<float>, int, int, int>(DownsampleKernel);

                for (int i = 0; i < levels.Count - 1; i++)
                {
                    var src = levels[i];
                    var dst = levels[i + 1];

                    ArrayView<float> srcView;
                    ArrayView<float> dstView;

                    // Source View
                    if (i == 0)
                    {
                        // Level 0 -> Level 1
                        srcView = pyramid.DataLevel0.View.SubView(src.Offset, src.Width * src.Height);
                    }
                    else
                    {
                        // Level N -> Level N+1
                        srcView = pyramid.DataMips.View.SubView(src.Offset, src.Width * src.Height);
                    }

                    // Dest View (always in Mips)
                    dstView = pyramid.DataMips.View.SubView(dst.Offset, dst.Width * dst.Height);

                    downsampleKernel(new Index2D(dst.Width, dst.Height), srcView, dstView, src.Width, src.Height, dst.Width);
                    _accelerator.Synchronize();
                }

                // Download Mips back to cache
                pyramid.DataMips.CopyToCPU(mipsHost);

                if (cachePath != null)
                    Utilities.WriteBinaryArray(cachePath, mipsHost);
            }
            else
            {
                // Upload loaded Mips
                pyramid.DataMips = _accelerator.Allocate1D<float>(mipsHost.Length);
                pyramid.DataMips.CopyFromCPU(mipsHost);
            }

            pyramid.CpuInfos = levels.ToArray();
            pyramid.Infos = _accelerator.Allocate1D<LevelInfo>(pyramid.CpuInfos.Length);
            pyramid.Infos.CopyFromCPU(pyramid.CpuInfos);

            pyramid.Map = BuildMapParams(dem);
            pyramid.Proj = BuildProjectionParams(dem);
            pyramid.SourceDem = dem;

            return pyramid;
        }

        internal (
            float[] Level0,
            float[] Mips,
            LevelInfo[] Levels,
            MapParams Map,
            ProjectionParams Projection) BuildPyramidForDiagnostics(ElevationMap dem)
        {
            using var pyramid = BuildOrLoadPyramid(dem);
            return (
                pyramid.DataLevel0!.GetAsArray1D(),
                pyramid.DataMips!.GetAsArray1D(),
                (LevelInfo[])pyramid.CpuInfos!.Clone(),
                pyramid.Map,
                pyramid.Proj);
        }

        static ProjectionParams BuildProjectionParams(ElevationMap dem)
        {
            var srs = dem.SrsDescriptor;
            return new ProjectionParams
            {
                R = (float)srs.R,
                Lat0 = (float)srs.lat0,
                Lon0 = (float)srs.lon0,
                K0 = (float)srs.k0,
                FalseEasting = (float)srs.FalseEasting,
                FalseNorthing = (float)srs.FalseNorthing
            };
        }

        static bool IsValid(float h) => !float.IsNaN(h) && !float.IsInfinity(h) && h > -20000.0f;

        private static void DownsampleKernel(Index2D index, ArrayView<float> src, ArrayView<float> dst, int srcW, int srcH, int dstW)
        {
            int c = index.X; // col
            int r = index.Y; // row

            if (c >= dstW) return; // Bounds check (implicit in grid, but good for safety)

            // Map to source coordinates
            int srcC = c * PYR_DOWNSAMPLE_FACTOR;
            int srcR = r * PYR_DOWNSAMPLE_FACTOR;

            float maxVal = -32000.0f; // Sentinel for NoData

            // Sample factor x factor
            for (int dy = 0; dy < PYR_DOWNSAMPLE_FACTOR; dy++)
            {
                for (int dx = 0; dx < PYR_DOWNSAMPLE_FACTOR; dx++)
                {
                    int sc = srcC + dx;
                    int sr = srcR + dy;

                    if (sc < srcW && sr < srcH)
                    {
                        float val = src[sr * srcW + sc];
                        if (IsValid(val))
                        {
                            maxVal = XMath.Max(maxVal, val);
                        }
                    }
                }
            }

            dst[r * dstW + c] = maxVal;
        }

        public static string BuildHorizonFilename(int tileCol, int tileRow, float observerElevation)
        {
            return new HorizonTileStore(".", HorizonTileLayout.Flat)
                .BuildFileName(tileRow, tileCol, observerElevation, compress: false);
        }

        public static (int col, int row, float observerElevation) ParseHorizonFilename(string filePath)
        {
            return HorizonTileStore.TryParseFileName(filePath, out var key)
                ? (key.TileX, key.TileY, key.ObserverElevationMeters)
                : (-1, -1, float.NaN);
        }

        private HorizonAngles GenerateHorizonsInternal(List<ElevationMap> dems, int tileX, int tileY, int width, int height, float observerElevation, bool captureIntermediate, string? diagnosticsDirectory = null)
        {
            if (dems == null || dems.Count == 0)
                throw new ArgumentException("At least one DEM is required.", nameof(dems));

            // Build or Load Pyramids. Do this even if filtering is disabled so that loading data
            // for the kernel is consistent.
            var pyramids = new List<Pyramid>();
            foreach (var dem in dems)
                pyramids.Add(BuildOrLoadPyramid(dem));

            var qtResult = LaunchRayCasting(pyramids, dems, tileX, tileY, width, height, observerElevation, diagnosticsDirectory);

            if (_enableNearFieldReferenceMerge && _nearFieldClampMeters > 0f)
            {
                var nearField = ComputeNearFieldBlock(dems, tileX, tileY, width, height, observerElevation);
                Debug.Assert(nearField != null);

                var qtDegrees = qtResult.Degrees;
                var nearDegrees = nearField.Value.Degrees;

#if DEBUG
                Debug.Assert(qtDegrees != null && nearDegrees != null && qtDegrees.Length == nearDegrees.Length);
                Debug.Assert(qtDegrees.All(v => IsValid(v)));
                Debug.Assert(nearDegrees.All(v => IsValid(v)));
#endif

                for (int i = 0; i < qtDegrees.Length; i++)
                    qtDegrees[i] = Math.Max(qtDegrees[i], nearDegrees[i]);

                DiagnosticsCallback?.Invoke(HorizonBufferType.NearField, nearField.Value);

                if (captureIntermediate)
                    DiagnosticsCallback?.Invoke(HorizonBufferType.FarField, qtResult.Clone());
            }

            foreach (var p in pyramids)
                p.Dispose();

            return qtResult;
        }

        private HorizonAngles LaunchRayCasting(List<Pyramid> pyramids, List<ElevationMap> dems, int tileCol, int tileRow, int tileWidth, int tileHeight, float observerElevation, string? diagnosticsDirectory)
        {
            int numPixels = tileWidth * tileHeight;
            int numAzimuths = 1440;
            int numDems = pyramids.Count;
            float maxDist = 1000000.0f; // 1000 km

            // 1. Create Primary View (for Observer Coordinate Reference)
            var primaryPV = new PyramidView
            {
                DataLevel0 = pyramids[0].DataLevel0!.View,
                DataMips = pyramids[0].DataMips!.View,
                Infos = pyramids[0].Infos!.View,
                Map = pyramids[0].Map,
                Proj = pyramids[0].Proj,
                Levels = pyramids[0].CpuInfos!.Length
            };

            // 2. Calculate Segments on CPU
            Log.Debug("Calculating Ray Segments on CPU...");
            // Calculate ray segments using standard approach (no subpatch complexity)
            var (cpuSegments, isCompact, gcInfo) = CalculateRaySegments(pyramids, primaryPV, tileCol, tileRow, tileWidth, tileHeight, numAzimuths, maxDist, observerElevation);
            Log.Debug("Generated {SegmentCount} segments", cpuSegments.Length);
            int debugTargetAzIdx = 1121;
            string? debugAzEnv = Environment.GetEnvironmentVariable("QUADTREE_DEBUG_AZ");
            if (!string.IsNullOrEmpty(debugAzEnv) && int.TryParse(debugAzEnv, out int envAz))
            {
                debugTargetAzIdx = envAz;
            }
            int debugTargetDemIdx = -1;
            string? debugDemEnv = Environment.GetEnvironmentVariable("QUADTREE_DEBUG_DEM");
            if (!string.IsNullOrEmpty(debugDemEnv) && int.TryParse(debugDemEnv, out int envDem))
            {
                debugTargetDemIdx = envDem;
            }
            string? sampleDumpPath = null;
            if (debugTargetAzIdx >= 0 && debugTargetDemIdx >= 0)
            {
                sampleDumpPath = Path.Combine(Directory.GetCurrentDirectory(), $"quadtree_samples_dem{debugTargetDemIdx}.txt");
            }
            int targetSegIndex = isCompact
                ? debugTargetAzIdx * numDems
                : ((debugTargetAzIdx * numPixels) + 0) * numDems;
            if (targetSegIndex >= 0 && targetSegIndex < cpuSegments.Length)
            {
                var segDebug = cpuSegments[targetSegIndex];
                Log.Debug("Segment debug (az={AzimuthIndex}): SStart={SStart:F3}, SEnd={SEnd:F3}, Dem={DemId}, StartPixel=({StartPixelX:F3},{StartPixelY:F3})", debugTargetAzIdx, segDebug.SStart, segDebug.SEnd, segDebug.DemId, segDebug.StartPixel.X, segDebug.StartPixel.Y);
            }

            // 3. Allocate GPU Memory
            // Accumulated output buffer: [Pixel * Azimuth]
            var cpuHorizonsAccum = new float[numPixels * numAzimuths];
            for (int i = 0; i < cpuHorizonsAccum.Length; i++) cpuHorizonsAccum[i] = float.NegativeInfinity;
            using var horizonsAccum = _accelerator.Allocate1D<float>(cpuHorizonsAccum.Length);
            horizonsAccum.CopyFromCPU(cpuHorizonsAccum);

            // Per-pass buffer to capture individual DEM contributions before accumulation
            var cpuHorizonsPass = new float[numPixels * numAzimuths];
            for (int i = 0; i < cpuHorizonsPass.Length; i++) cpuHorizonsPass[i] = float.NegativeInfinity;
            using var horizonsPass = _accelerator.Allocate1D<float>(cpuHorizonsPass.Length);
            horizonsPass.CopyFromCPU(cpuHorizonsPass);

            // Segments Buffer
            using var gpuSegments = _accelerator.Allocate1D<RaySegment>(cpuSegments.Length);
            gpuSegments.CopyFromCPU(cpuSegments);

            // Debug buffer
            using var debug = _accelerator.Allocate1D<float>(1024);

            var kernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                PyramidView, PyramidView,
                ArrayView<float>,
                ArrayView<RaySegment>,
                int, int, int,
                int, int, int, int, int, KernelParams,
                ArrayView<float>>(QuadTreeRayCastKernel);

            // Merge kernel to combine per-pass horizons into accumulated result
            var mergeKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(MergeMaxKernel);
            float minTraverseDistanceMeters = (_enableNearFieldReferenceMerge && _nearFieldClampMeters > 0f)
                ? _nearFieldClampMeters
                : 0f;
            var kernelParams = new KernelParams(
                observerElevation,
                minTraverseDistanceMeters * METERS_TO_KILOMETERS,
                debugTargetAzIdx,
                gcInfo.GammaCenter,
                gcInfo.DGammaDx,
                gcInfo.DGammaDy);
            bool capturePerPassDebug = debugTargetAzIdx >= 0;

            // 5. Multi-Pass Launch
            for (int i = 0; i < numDems; i++)
            {
                var pyramid = pyramids[i];
                var pv = new PyramidView
                {
                    DataLevel0 = pyramid.DataLevel0!.View,
                    DataMips = pyramid.DataMips!.View,
                    Infos = pyramid.Infos!.View,
                    Map = pyramid.Map,
                    Proj = pyramid.Proj,
                    Levels = pyramid.CpuInfos!.Length
                };

                Log.Debug("Launching Pass {PassIndex} (DEM {DemIndex})...", i, i);

                // Launch for all pixels/azimuths
                int debugFlags = (_disableHierarchy ? 1 : 0) | (_forceFixedStepDebug ? 2 : 0);
                // Reset per-pass buffer to -inf
                horizonsPass.CopyFromCPU(cpuHorizonsPass);

                kernel(new Index2D(numPixels, numAzimuths),
                    primaryPV, pv,
                    horizonsPass.View,
                    gpuSegments.View,
                    i, numDems, debugFlags,
                    tileCol, tileRow, tileWidth, tileHeight, isCompact ? 1 : 0, kernelParams,
                    debug.View);

                _accelerator.Synchronize();

                // Diagnostics callback for per-DEM pass horizon buffer
                if (DiagnosticsCallback != null)
                {
                    var demEnum = i switch {
                        0 => HorizonBufferType.DEM1,
                        1 => HorizonBufferType.DEM2,
                        2 => HorizonBufferType.DEM3,
                        3 => HorizonBufferType.DEM4,
                        4 => HorizonBufferType.DEM5,
                        _ => HorizonBufferType.DEMN
                    };
                    DiagnosticsCallback.Invoke(demEnum, HorizonAngles.FromSlopes(horizonsPass.GetAsArray1D()));
                }

                // Merge per-pass into accumulated horizon (max)
                mergeKernel(new Index1D((int)horizonsAccum.Length), horizonsAccum.View, horizonsPass.View);
                _accelerator.Synchronize();
            }

            // Retrieve and log debug buffer
            var debugValues = debug.GetAsArray1D();
            Log.Debug(
                "Debug Buffer: sStart={sStart:F4}, seg.SStart={segSStart:F4}, seg.X0={segX0:F4}, startPx={startPx:F4}, " +
                "initialPx={initialPx:F4}, initialPy={initialPy:F4}, finalSlope={finalSlope:F6}, sampleSlope={sampleSlope:F6}, " +
                "sampleDist={sampleDist:F3}, samplePx={samplePx:F3}, samplePy={samplePy:F3}, sampleElev={sampleElev:F3}, " +
                "loggedAz={loggedAz:F0}, loggedPixel={loggedPixel:F0}, loggedSamples={loggedSamples:F0}, " +
                "obsTerrain={obsTerrain:F3}, obsZ={obsZ:F3}, obsCol={obsCol:F3}, obsRow={obsRow:F3}, " +
                "tileColBase={tileColBase:F0}, tileRowBase={tileRowBase:F0}",
                debugValues[0], debugValues[1], debugValues[2], debugValues[3],
                debugValues[4], debugValues[5], debugValues[6], debugValues[7],
                debugValues[8], debugValues[9], debugValues[10], debugValues[11],
                debugValues[12], debugValues[13], debugValues[14],
                debugValues[15], debugValues[16], debugValues[17], debugValues[18],
                debugValues[19], debugValues[20]);
            // Optionally, diagnostics callback for debug buffer (not angles, but for completeness)
            // DiagnosticsCallback?.Invoke(HorizonDiagnosticsBuffer.Debug, debugValues); // Uncomment if you want debug buffer reporting

            // 6. Read Results
            return HorizonAngles.FromSlopes(horizonsAccum.GetAsArray1D());
        }

        /// <summary>
        /// Asynchronous version of LaunchRayCasting for pipeline processing.
        /// Launches GPU kernels without blocking synchronization, allowing CPU/GPU overlap.
        /// </summary>
        private async Task<HorizonAngles> LaunchRayCastingAsync(
            PipelineBuffers buffers,
            List<Pyramid> pyramids, 
            List<ElevationMap> dems,
            RaySegment[] cpuSegments,
            int tileCol, int tileRow, int tileWidth, int tileHeight,
            float observerElevation,
            GridConvergenceInfo gcInfo)
        {
            int numPixels = tileWidth * tileHeight;
            int numAzimuths = 1440;
            int numDems = pyramids.Count;

            // 1. Create Primary View
            var primaryPV = new PyramidView
            {
                DataLevel0 = pyramids[0].DataLevel0!.View,
                DataMips = pyramids[0].DataMips!.View,
                Infos = pyramids[0].Infos!.View,
                Map = pyramids[0].Map,
                Proj = pyramids[0].Proj,
                Levels = pyramids[0].CpuInfos!.Length
            };

            // 2. Initialize buffers
            var cpuHorizonsAccum = new float[numPixels * numAzimuths];
            for (int i = 0; i < cpuHorizonsAccum.Length; i++) cpuHorizonsAccum[i] = float.NegativeInfinity;
            buffers.HorizonsAccum.CopyFromCPU(cpuHorizonsAccum);

            var cpuHorizonsPass = new float[numPixels * numAzimuths];
            for (int i = 0; i < cpuHorizonsPass.Length; i++) cpuHorizonsPass[i] = float.NegativeInfinity;
            buffers.HorizonsPass.CopyFromCPU(cpuHorizonsPass);

            buffers.GpuSegments.CopyFromCPU(cpuSegments);

            // 3. Setup kernels
            var kernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                PyramidView, PyramidView,
                ArrayView<float>,
                ArrayView<RaySegment>,
                int, int, int,
                int, int, int, int, int, KernelParams,
                ArrayView<float>>(QuadTreeRayCastKernel);

            var mergeKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(MergeMaxKernel);

            float minTraverseDistanceMeters = (_enableNearFieldReferenceMerge && _nearFieldClampMeters > 0f)
                ? _nearFieldClampMeters
                : 0f;
            var kernelParams = new KernelParams(
                observerElevation,
                minTraverseDistanceMeters * METERS_TO_KILOMETERS,
                -1, // debugTargetAzIdx - disabled for pipeline
                gcInfo.GammaCenter,
                gcInfo.DGammaDx,
                gcInfo.DGammaDy);

            // 4. Launch multi-pass kernels (async)
            for (int i = 0; i < numDems; i++)
            {
                var pyramid = pyramids[i];
                var pv = new PyramidView
                {
                    DataLevel0 = pyramid.DataLevel0!.View,
                    DataMips = pyramid.DataMips!.View,
                    Infos = pyramid.Infos!.View,
                    Map = pyramid.Map,
                    Proj = pyramid.Proj,
                    Levels = pyramid.CpuInfos!.Length
                };

                int debugFlags = (_disableHierarchy ? 1 : 0) | (_forceFixedStepDebug ? 2 : 0);
                buffers.HorizonsPass.CopyFromCPU(cpuHorizonsPass);

                kernel(new Index2D(numPixels, numAzimuths),
                    primaryPV, pv,
                    buffers.HorizonsPass.View,
                    buffers.GpuSegments.View,
                    i, numDems, debugFlags,
                    tileCol, tileRow, tileWidth, tileHeight, 1, kernelParams, // Always use Compact Mode (1)
                    buffers.Debug.View);

                // NO synchronization here - let kernels overlap

                // Merge per-pass into accumulated horizon (max)
                mergeKernel(new Index1D((int)buffers.HorizonsAccum.Length), buffers.HorizonsAccum.View, buffers.HorizonsPass.View);
            }

            // 5. Queue result retrieval without blocking
            return await Task.Run(() =>
            {
                // This is the only place we synchronize - to read final results
                // All kernel launches happened above without blocking
                _accelerator.Synchronize();
                return HorizonAngles.FromSlopes(buffers.HorizonsAccum.GetAsArray1D());
            });
        }

        // Element-wise max: accum = max(accum, pass)
        static void MergeMaxKernel(Index1D index, ArrayView<float> accum, ArrayView<float> pass)
        {
            accum[index] = XMath.Max(accum[index], pass[index]);
        }

        private HorizonAngles? ComputeNearFieldBlock(List<ElevationMap> dems, int tileCol, int tileRow, int tileWidth, int tileHeight, float observerElevation)
        {
            if (!_enableNearFieldReferenceMerge || _nearFieldClampMeters <= 0f || dems.Count == 0)
                return null;

            if (dems.Any(d => string.IsNullOrWhiteSpace(d.Path)))
            {
                Log.Warning("Near-field GPU merge skipped: DEM path missing for one or more inputs.");
                return null;
            }

            Gdal.AllRegister();

            const int numAzimuths = 1440;
            int numPixels = tileWidth * tileHeight;

            var innerPath = dems[0].Path!;
            using var innerDs = Gdal.Open(innerPath, Access.GA_ReadOnly);
            if (innerDs == null)
            {
                Log.Error("Failed to open inner DEM at {InnerPath}", innerPath);
                return null;
            }

            double[] innerGt = new double[6];
            innerDs.GetGeoTransform(innerGt);

            // Compute a bordered grid around the requested tile so every ray stays inside the temp DEM.
            double borderMeters = _nearFieldClampMeters + NEARFIELD_BORDER_MARGIN_METERS;
            var (tempGt, tempCols, tempRows, borderPxX, borderPxY) = ComputeBorderedGrid(innerGt, tileCol, tileRow, tileWidth, tileHeight, borderMeters);
            var bounds = GeoTransformBounds(tempGt, tempCols, tempRows);

            var innerBand = innerDs.GetRasterBand(1);
            var targetType = innerBand.DataType;
            innerBand.GetNoDataValue(out double targetNoData, out int hasNoData);
            if (hasNoData == 0)
                targetNoData = -9999.0;

            var tempBuffer = new float[tempRows * tempCols];
            for (int i = 0; i < tempBuffer.Length; i++)
                tempBuffer[i] = (float)targetNoData;

            string targetSrs = innerDs.GetProjectionRef();
            foreach (var dem in Enumerable.Reverse(dems))
            {
                using var ds = Gdal.Open(dem.Path!, Access.GA_ReadOnly);
                if (ds == null)
                {
                    Log.Warning("Skipping DEM {Path} because it could not be opened.", dem.Path);
                    continue;
                }

                var band = ds.GetRasterBand(1);
                band.GetNoDataValue(out double srcNoData, out int srcHasNoData);
                if (srcHasNoData == 0)
                    srcNoData = targetNoData;

                var warpArgs = BuildWarpArgs(targetSrs, bounds, tempCols, tempRows, srcNoData, targetNoData, targetType);
                using var warpOpts = new GDALWarpAppOptions(warpArgs);
                using var warped = Gdal.Warp(string.Empty, new Dataset[] { ds }, warpOpts, null, null);
                if (warped == null)
                {
                    Log.Warning("Warp failed for DEM {Path}", dem.Path);
                    continue;
                }

                var warpedBand = warped.GetRasterBand(1);
                var warpedBuf = new float[tempBuffer.Length];
                warpedBand.ReadRaster(0, 0, tempCols, tempRows, warpedBuf, tempCols, tempRows, 0, 0);

                MergeWarp(tempBuffer, warpedBuf, (float)targetNoData);
            }

            // Launch a lightweight GPU raycast over the inner tile using the flattened temp DEM.
            float pixelSizeMeters = (float)((Math.Abs(innerGt[1]) + Math.Abs(innerGt[5])) * 0.5);
            pixelSizeMeters = Math.Max(0.01f, pixelSizeMeters);
            float maxDistMeters = _nearFieldClampMeters;

            using var demBuf = _accelerator.Allocate1D(tempBuffer);
            var outputBuffer = new float[numPixels * numAzimuths];
            for (int i = 0; i < outputBuffer.Length; i++)
                outputBuffer[i] = float.NegativeInfinity;
            using var gpuOut = _accelerator.Allocate1D(outputBuffer);

            var kernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<float>, int, int, int, int, int, int, float, float, float, float, ArrayView<float>>(NearFieldRayKernel);
            var launchExtent = new Index2D(numPixels, numAzimuths);
            kernel(
                launchExtent,
                demBuf.View,
                tempCols,
                tempRows,
                tileWidth,
                tileHeight,
                borderPxX,
                borderPxY,
                pixelSizeMeters,
                maxDistMeters,
                observerElevation,
                (float)targetNoData,
                gpuOut.View);

            return HorizonAngles.FromRadians(gpuOut.GetAsArray1D());
        }

        internal static (double[] gt, int cols, int rows, int borderPxX, int borderPxY) ComputeBorderedGrid(double[] innerGt, int tileCol, int tileRow, int tileWidth, int tileHeight, double borderMeters)
        {
            double pixelWidth = innerGt[1];
            double pixelHeight = innerGt[5];
            if (pixelWidth <= 0 || pixelHeight >= 0)
                throw new InvalidOperationException("Unexpected GeoTransform orientation for near-field warp.");

            int borderPxX = (int)Math.Ceiling(borderMeters / pixelWidth);
            int borderPxY = (int)Math.Ceiling(borderMeters / Math.Abs(pixelHeight));

            int cols = tileWidth + 2 * borderPxX;
            int rows = tileHeight + 2 * borderPxY;

            double originX = innerGt[0] + tileCol * pixelWidth + tileRow * innerGt[2] - borderPxX * pixelWidth;
            double originY = innerGt[3] + tileCol * innerGt[4] + tileRow * pixelHeight + borderPxY * Math.Abs(pixelHeight);

            var gt = new double[]
            {
                originX,
                pixelWidth,
                0.0,
                originY,
                0.0,
                pixelHeight
            };
            return (gt, cols, rows, borderPxX, borderPxY);
        }

        internal static (double minX, double minY, double maxX, double maxY) GeoTransformBounds(double[] gt, int cols, int rows)
        {
            double minX = gt[0];
            double maxY = gt[3];
            double maxX = gt[0] + gt[1] * cols;
            double minY = gt[3] + gt[5] * rows;
            return (Math.Min(minX, maxX), Math.Min(minY, maxY), Math.Max(minX, maxX), Math.Max(minY, maxY));
        }

        internal static string[] BuildWarpArgs(string dstSrs, (double minX, double minY, double maxX, double maxY) bounds, int cols, int rows, double srcNoData, double dstNoData, DataType targetType)
        {
            string typeName = targetType.ToString();
            if (typeName.StartsWith("GDT_", StringComparison.OrdinalIgnoreCase))
                typeName = typeName.Substring(4);

            return new[]
            {
                "-t_srs", dstSrs,
                "-te",
                bounds.minX.ToString(CultureInfo.InvariantCulture),
                bounds.minY.ToString(CultureInfo.InvariantCulture),
                bounds.maxX.ToString(CultureInfo.InvariantCulture),
                bounds.maxY.ToString(CultureInfo.InvariantCulture),
                "-ts", cols.ToString(CultureInfo.InvariantCulture), rows.ToString(CultureInfo.InvariantCulture),
                "-r", "bilinear",
                "-srcnodata", srcNoData.ToString(CultureInfo.InvariantCulture),
                "-dstnodata", dstNoData.ToString(CultureInfo.InvariantCulture),
                "-ot", typeName,
                "-of", "MEM"
            };
        }

        internal static void MergeWarp(float[] target, float[] warped, float nodata)
        {
            float eps = Math.Abs(nodata) * 1e-5f + 1e-3f;
            for (int i = 0; i < target.Length && i < warped.Length; i++)
            {
                float v = warped[i];
                if (!IsNoDataValue(v, nodata, eps))
                    target[i] = v;
            }
        }

        static void NearFieldRayKernel(
            Index2D index,
            ArrayView<float> dem,
            int demWidth,
            int demHeight,
            int tileWidth,
            int tileHeight,
            int tileOffsetX,
            int tileOffsetY,
            float pixelSizeMeters,
            float maxDistanceMeters,
            float observerElevation,
            float noDataValue,
            ArrayView<float> output)
        {
            int numAz = 1440;
            int pixelIdx = index.X;
            int azIdx = index.Y;
            if (pixelIdx >= tileWidth * tileHeight || azIdx >= numAz)
                return;

            int rowInTile = pixelIdx / tileWidth;
            int colInTile = pixelIdx % tileWidth;
            int demCol = tileOffsetX + colInTile;
            int demRow = tileOffsetY + rowInTile;

            float obsH = SampleBilinearFlat(dem, demWidth, demHeight, (float)demCol, (float)demRow, noDataValue);
            float eps = XMath.Abs(noDataValue) * 1e-5f + 1e-3f;
            if (IsNoDataValue(obsH, noDataValue, eps))
            {
                output[pixelIdx * numAz + azIdx] = float.NegativeInfinity;
                return;
            }

            float obsZ = obsH + observerElevation;
            float angleRad = (float)(azIdx * (2.0 * XMath.PI / numAz) - XMath.PI / 2.0);
            float stepMeters = pixelSizeMeters;
            float stepPx = stepMeters / pixelSizeMeters;
            float dx = XMath.Cos(angleRad) * stepPx;
            float dy = XMath.Sin(angleRad) * stepPx;
            float px = (float)demCol;
            float py = (float)demRow;
            float traveled = 0f;
            float maxSlope = float.NegativeInfinity;

            while (traveled <= maxDistanceMeters)
            {
                px += dx;
                py += dy;
                traveled += stepMeters;

                if (px < 1f || py < 1f || px >= demWidth - 1 || py >= demHeight - 1)
                    break;

                float h = SampleBilinearFlat(dem, demWidth, demHeight, px, py, noDataValue);
                if (IsNoDataValue(h, noDataValue, eps))
                    continue;

                float slope = (h - obsZ) / XMath.Max(traveled, 0.01f);
                if (slope > maxSlope)
                    maxSlope = slope;
            }

            output[pixelIdx * numAz + azIdx] = maxSlope <= -1e20f ? float.NegativeInfinity : XMath.Atan(maxSlope);
        }

        internal static float SampleBilinearFlat(ArrayView<float> data, int width, int height, float col, float row, float nodata)
        {
            int x0 = (int)XMath.Floor(col);
            int y0 = (int)XMath.Floor(row);
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            if (x0 < 0 || y0 < 0 || x1 >= width || y1 >= height)
                return nodata;

            float q00 = data[y0 * width + x0];
            float q10 = data[y0 * width + x1];
            float q01 = data[y1 * width + x0];
            float q11 = data[y1 * width + x1];

            float eps = XMath.Abs(nodata) * 1e-5f + 1e-3f;
            if (IsNoDataValue(q00, nodata, eps) || IsNoDataValue(q10, nodata, eps) || IsNoDataValue(q01, nodata, eps) || IsNoDataValue(q11, nodata, eps))
                return nodata;

            float tx = col - x0;
            float ty = row - y0;
            float a = q00 + tx * (q10 - q00);
            float b = q01 + tx * (q11 - q01);
            return a + ty * (b - a);
        }

        // Overload for emulator using simple array
        internal static float SampleBilinearFlat(float[] data, int width, int height, float col, float row, float nodata)
        {
            int x0 = (int)Math.Floor(col);
            int y0 = (int)Math.Floor(row);
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            if (x0 < 0 || y0 < 0 || x1 >= width || y1 >= height)
                return nodata;

            float q00 = data[y0 * width + x0];
            float q10 = data[y0 * width + x1];
            float q01 = data[y1 * width + x0];
            float q11 = data[y1 * width + x1];

            float eps = Math.Abs(nodata) * 1e-5f + 1e-3f;
            if (IsNoDataValue(q00, nodata, eps) || IsNoDataValue(q10, nodata, eps) || IsNoDataValue(q01, nodata, eps) || IsNoDataValue(q11, nodata, eps))
                return nodata;

            float tx = col - x0;
            float ty = row - y0;
            float a = q00 + tx * (q10 - q00);
            float b = q01 + tx * (q11 - q01);
            return a + ty * (b - a);
        }

        internal static bool IsNoDataValue(float value, float nodata, float eps) => Math.Abs(value - nodata) <= eps;

        static float CalculateScaleFactor(float lat, float lon, ProjectionParams p)
        {
            float sinPhi = XMath.Sin(lat);
            float cosPhi = XMath.Cos(lat);
            float sinPhi0 = XMath.Sin(p.Lat0);
            float cosPhi0 = XMath.Cos(p.Lat0);
            float dLam = lon - p.Lon0;
            float cosDLam = XMath.Cos(dLam);

            float denom = 1.0f + sinPhi0 * sinPhi + cosPhi0 * cosPhi * cosDLam;
            if (XMath.Abs(denom) < 1e-10f) denom = 1e-10f;

            return 2.0f * p.K0 / denom;
        }

        // -----------------------------------------------------------------------
        // SEGMENTED RAY CAST KERNEL
        // -----------------------------------------------------------------------

        /// <summary>
        /// GPU kernel that evaluates segmented horizon rays against a DEM min-max pyramid.
        /// It consumes the pre-fit cubic ray segments described in `docs/description.md`,
        /// optionally shifts them for compact tiles, performs hierarchical culling with the
        /// quadtree, and writes the max slope per pixel/azimuth along with diagnostics.
        /// </summary>
        /// <param name="index">2D launch index: `X`=pixel within tile, `Y`=azimuth bin.</param>
        /// <param name="primaryPV">Primary DEM pyramid providing observer height sampling.</param>
        /// <param name="activePV">Current DEM pyramid for this pass (min-max hierarchy).</param>
        /// <param name="output">Per-pixel/azimuth slope buffer in radians (max reduced later).</param>
        /// <param name="segments">Precomputed ray segments laid out as described in the docs.</param>
        /// <param name="passIndex">DEM pass index, used to pick the segment and diagnostics type.</param>
        /// <param name="numDems">Total number of DEM passes encoded in `segments`.</param>
        /// <param name="debugFlag">Bit0 disables hierarchy, Bit1 forces fixed marching steps.</param>
        /// <param name="tileColBase">Column of the tile origin in primary DEM pixel space.</param>
        /// <param name="tileRowBase">Row of the tile origin in primary DEM pixel space.</param>
        /// <param name="tileW">Tile width in pixels.</param>
        /// <param name="tileH">Tile height in pixels.</param>
        /// <param name="isCompact">Non-zero if the launch uses compact (per-azimuth) segments.</param>
        /// <param name="kernelParams">Observer elevation, traverse clamp, and debug azimuth.</param>
        /// <param name="debugBuffer">Scratch buffer for emitting single-ray diagnostics.</param>
        static void QuadTreeRayCastKernel(
            Index2D index,
            PyramidView primaryPV,
            PyramidView activePV,
            ArrayView<float> output,
            ArrayView<RaySegment> segments,
            int passIndex, int numDems, int debugFlag, // lower bit = disableHierarchy, bit1 = force fixed steps
            int tileColBase, int tileRowBase, int tileW, int tileH, int isCompact, KernelParams kernelParams,
            ArrayView<float> debugBuffer)
        {
            int pixelIdx = index.X;
            int azIdx = index.Y;
            int numPixels = tileW * tileH;

            // Calculate pixel position relative to tile
            int rowInTile = pixelIdx / tileW;
            int colInTile = pixelIdx % tileW;

            RaySegment seg;
            float startPx, startPy;
            float sStart = 0f;

            if (isCompact != 0)
            {
                // COMPACT MODE: One segment per Azimuth (at tile center)
                // Layout: [Azimuth][DEM]
                long segmentIdx = (long)azIdx * numDems + passIndex;
                seg = segments[segmentIdx];

                // Shift StartPixel based on pixel position relative to center
                // Center of tile is at (tileW/2, tileH/2) in tile coords
                float dCol = (float)colInTile - (tileW / 2.0f);
                float dRow = (float)rowInTile - (tileH / 2.0f);

                float primaryRes = XMath.Sqrt(primaryPV.Map.T1 * primaryPV.Map.T1 + primaryPV.Map.T4 * primaryPV.Map.T4);
                float activeRes = XMath.Sqrt(activePV.Map.T1 * activePV.Map.T1 + activePV.Map.T4 * activePV.Map.T4);
                float scaleRatio = primaryRes / activeRes;

                startPx = seg.StartPixel.X + dCol * scaleRatio;
                startPy = seg.StartPixel.Y + dRow * scaleRatio;
                sStart = seg.SStart;
            }
            else
            {
                // FULL MODE: Unique segment per Pixel
                // Layout: [Azimuth][Pixel][DEM]
                // Note: CalculateRaySegments writes [Azimuth][Pixel][DEM] layout.
                long segmentIdx = ((long)azIdx * numPixels + pixelIdx) * numDems + passIndex;
                seg = segments[segmentIdx];



                startPx = seg.StartPixel.X;
                startPy = seg.StartPixel.Y;
                sStart = seg.SStart;
            }

            // Apply Grid Convergence correction to output bin
            // The ray was computed at tile center with Grid Convergence γ_center
            // At this pixel, local Grid Convergence creates an azimuth offset
            int correctedAzIdx = azIdx;
            if (isCompact != 0)
            {
                // Calculate Grid Convergence offset for this pixel position
                float dCol = (float)colInTile - (tileW / 2.0f);
                float dRow = (float)rowInTile - (tileH / 2.0f);
                
                // Grid Convergence difference at this pixel: γ_pixel - γ_center
                float deltaGamma = kernelParams.DGammaDx * dCol + kernelParams.DGammaDy * dRow;
                
                // Convert to bin offset: 1440 bins span 2π radians
                float binOffset = deltaGamma * (1440.0f / (2.0f * 3.14159265f));
                int binOffsetInt = (int)XMath.Round(binOffset);
                
                // Apply offset with wraparound
                correctedAzIdx = azIdx + binOffsetInt;
                if (correctedAzIdx < 0) correctedAzIdx += 1440;
                if (correctedAzIdx >= 1440) correctedAzIdx -= 1440;
            }

            // The output buffer stores horizon slope across DEM passes. Convert to
            // elevation angle only after all passes have completed.
            long outIdx = (long)pixelIdx * 1440 + correctedAzIdx;
            float storedSlope = output[outIdx];
            float currentHorizonSlope = float.IsNegativeInfinity(storedSlope) ? -1e30f : storedSlope;

            // Get Observer Z
            int globalRow = tileRowBase + rowInTile;
            int globalCol = tileColBase + colInTile;
            float obsTerrain = SampleBilinear(primaryPV.DataLevel0, primaryPV.Infos, 0, (float)globalCol, (float)globalRow);
            float obsZ = obsTerrain + kernelParams.ObserverElevation;

            // Write debug info for the first pixel and azimuth
            if (pixelIdx == 0 && azIdx == 0)
            {
                debugBuffer[0] = sStart;
                debugBuffer[1] = seg.SStart;
                debugBuffer[2] = seg.X0;
                debugBuffer[3] = startPx;

                // Evaluate cubic for s=sStart to get initial px, py
                float initialPx = EvalCubic(startPx, seg.A1, seg.A2, seg.A3, seg.A4, sStart - sStart);
                float initialPy = EvalCubic(startPy, seg.B1, seg.B2, seg.B3, seg.B4, sStart - sStart);
                debugBuffer[4] = initialPx;
                debugBuffer[5] = initialPy;
                debugBuffer[15] = obsTerrain;
                debugBuffer[16] = obsZ;
                debugBuffer[17] = (float)globalCol;
                debugBuffer[18] = (float)globalRow;
                debugBuffer[19] = tileColBase;
                debugBuffer[20] = tileRowBase;
            }

            // Setup Marcher (cubic-only)
            float sEnd = seg.SEnd;
            float runtimeStart = (kernelParams.MinTraverseDistanceKm > 0f)
                ? XMath.Max(sStart, kernelParams.MinTraverseDistanceKm)
                : XMath.Max(sStart, 1.0f * METERS_TO_KILOMETERS);
            float s = runtimeStart;
            bool disableHierarchy = (debugFlag & 0x1) != 0;
            bool useFixedSteps = (debugFlag & DEBUG_FLAG_FORCE_FIXED_STEPS) != 0;
            float fixedStep = DEBUG_FIXED_STEP_KM;
            if (useFixedSteps)
            {
                s = runtimeStart + fixedStep;
            }
            float R = activePV.Proj.R;
            float pixCol = XMath.Sqrt(activePV.Map.T1 * activePV.Map.T1 + activePV.Map.T4 * activePV.Map.T4);
            float pixRow = XMath.Sqrt(activePV.Map.T2 * activePV.Map.T2 + activePV.Map.T5 * activePV.Map.T5);
            float activeMapRes = (pixCol + pixRow) * 0.5f;
            float minAdaptiveStepKm = MIN_ADAPTIVE_STEP_RESOLUTION_FACTOR * activeMapRes * METERS_TO_KILOMETERS;
            float primaryDemFarMinAdaptiveStepKm = PRIMARY_DEM_FAR_MIN_STEP_RESOLUTION_FACTOR * activeMapRes * METERS_TO_KILOMETERS;
            bool logTarget = kernelParams.DebugAzimuthIndex >= 0 && azIdx == kernelParams.DebugAzimuthIndex;
            float loggedSlope = -1e30f;
            float loggedDist = 0f;
            float loggedPx = 0f;
            float loggedPy = 0f;
            float loggedElev = 0f;
            float loggedSampleCount = 0f;
            bool rayIsOutOfBounds = false;

            while (s <= sEnd)
            {
                float px, py;
                float trueDistMeters = s * KILOMETERS_TO_METERS;
                // Evaluate cubic at s (kilometers -> convert inside)
                px = EvalCubic(startPx, seg.A1, seg.A2, seg.A3, seg.A4, s - sStart);
                py = EvalCubic(startPy, seg.B1, seg.B2, seg.B3, seg.B4, s - sStart);

                // Bounds check against the full DEM extents, also checking for NaN
                if (XMath.IsNaN(px) || XMath.IsNaN(py) || px < 0f || py < 0f || px >= activePV.Infos[0].Width - 1f || py >= activePV.Infos[0].Height - 1f)
                {
                    break;
                }

                float planarDx = (px - startPx) * activeMapRes;
                float planarDy = (py - startPy) * activeMapRes;
                float planarMeters = XMath.Sqrt(planarDx * planarDx + planarDy * planarDy);
                
                // For close distances, use s directly; for far distances, use polynomial-corrected chordDist
                // The polynomial is fit to capture curvature at larger scales and may be inaccurate at short range
                const float CLOSE_DISTANCE_THRESHOLD_KM = 0.5f; // 500 meters
                float trueDist;
                if (s < CLOSE_DISTANCE_THRESHOLD_KM)
                {
                    // Use parameterized distance directly - at short range, s ≈ true distance
                    trueDist = s * KILOMETERS_TO_METERS;
                }
                else
                {
                    // Use polynomial-corrected chord distance for larger distances
                    trueDist = (seg.SStartChord * KILOMETERS_TO_METERS) + EvalPlanarChord(seg, planarMeters);
                }

                if (disableHierarchy)
                {
                    // Always sample Level 0, no culling; use adaptive stepping
                    float bilinearH = SampleBilinear(activePV.DataLevel0, activePV.Infos, 0, px, py);
                    float sampleSlope = -1e30f;
                    if (IsValid(bilinearH))
                    {
                        if (s < CLOSE_DISTANCE_THRESHOLD_KM)
                        {
                            // Flat-earth approximation for short range: slope = dH / distance
                            float dH = bilinearH - obsZ;
                            sampleSlope = (trueDist > 1e-6f) ? (dH / trueDist) : -1e30f;
                        }
                        else
                        {
                            // Exact spherical calculation using Law of Cosines for far field
                            float r_o = R + obsZ;
                            float s_sq = trueDist * trueDist;
                            // Precision fix: (r_p - r_o) replaced with (bilinearH - obsZ)
                            float z_local_bi = ((bilinearH - obsZ) * (2.0f * R + bilinearH + obsZ) - s_sq) / (2.0f * r_o);
                            float x_sq = s_sq - z_local_bi * z_local_bi;
                            float x_local_bi = (x_sq > 0f) ? XMath.Sqrt(x_sq) : 1e-6f;
                            sampleSlope = (x_local_bi == 0.0f) ? -1e30f : (z_local_bi / x_local_bi);
                        }
                        currentHorizonSlope = XMath.Max(currentHorizonSlope, sampleSlope);
                        if (logTarget && sampleSlope > loggedSlope)
                        {
                            loggedSlope = sampleSlope;
                            loggedDist = trueDistMeters;
                            loggedPx = px;
                            loggedPy = py;
                            loggedElev = bilinearH;
                            loggedSampleCount += 1f;
                        }
                    }
                    
                    // Adaptive stepping based on margin below horizon
                    var tanNoCull = EvalCubicTangent(seg.A1, seg.A2, seg.A3, seg.A4, seg.B1, seg.B2, seg.B3, seg.B4, s - sStart);
                    float magNoCull = XMath.Sqrt(tanNoCull.dxds * tanNoCull.dxds + tanNoCull.dyds * tanNoCull.dyds);
                    float dsPixel = (magNoCull > 1e-6f) ? (1.0f / magNoCull) : 0.001f;

                    float dsNoCull;
                    if (useFixedSteps)
                    {
                        dsNoCull = fixedStep;
                    }
                    else
                    {
                        // Compute margin-based step: larger steps when well below horizon
                        float margin = currentHorizonSlope - sampleSlope;
                        float dsMargin = (margin > 0f) ? (margin * trueDistMeters * INV_TAN_MAX_SLOPE * METERS_TO_KILOMETERS) : 0f;
                        
                        // Angular error budget cap: step proportional to distance
                        float dsAngular = trueDistMeters * ANGULAR_STEP_FACTOR * METERS_TO_KILOMETERS;
                        
                        // Use max of pixel step and margin step, capped by angular budget
                        dsNoCull = XMath.Max(dsPixel, XMath.Min(dsMargin, dsAngular));
                        
                        // Increase sampling frequency 4x for close distances (under 500m)
                        if (s < CLOSE_DISTANCE_THRESHOLD_KM)
                        {
                            dsNoCull *= 0.25f;
                        }
                        float stepFloorKm = (passIndex == 0 && trueDistMeters >= PRIMARY_DEM_FAR_MIN_STEP_DISTANCE_METERS)
                            ? primaryDemFarMinAdaptiveStepKm
                            : minAdaptiveStepKm;
                        dsNoCull = XMath.Max(dsNoCull, stepFloorKm);
                    }

                    s += dsNoCull;
                    continue;
                }

                // Original hierarchical path (unchanged) ...
                int level = ComputeStartLevel(trueDistMeters, activeMapRes, activePV.Levels);
                while (level >= 0)
                {
                    var info = activePV.Infos[level];
                    int lW = info.Width;
                    int lH = info.Height;
                    int shift = level * 2;
                    int scale = 1 << shift;
                    int lx = (int)px >> shift;
                    int ly = (int)py >> shift;

                    if (lx < 0 || ly < 0 || lx >= lW || ly >= lH) {
                        float stepKm = (scale * activeMapRes) * METERS_TO_KILOMETERS;
                        s += XMath.Max(0.001f, stepKm);
                        rayIsOutOfBounds = true;
                        break;
                    }

                    float maxH = (level == 0)
                        ? activePV.DataLevel0[info.Offset + ly * lW + lx]
                        : activePV.DataMips[info.Offset + ly * lW + lx];
                        
                    float minX = lx * scale;
                    float minY = ly * scale;
                    float maxX = minX + scale;
                    float maxY = minY + scale;

                    // Tangent-linear approximation for exit: use local tangent to estimate sExit
                    var tan = EvalCubicTangent(seg.A1, seg.A2, seg.A3, seg.A4, seg.B1, seg.B2, seg.B3, seg.B4, s - sStart);
                    float invDx = (XMath.Abs(tan.dxds) > 1e-8f) ? (1.0f / tan.dxds) : 1e30f;
                    float invDy = (XMath.Abs(tan.dyds) > 1e-8f) ? (1.0f / tan.dyds) : 1e30f;
                    float t1 = (minX - px) * invDx;
                    float t2 = (maxX - px) * invDx;
                    float t3 = (minY - py) * invDy;
                    float t4 = (maxY - py) * invDy;
                    float tminX = XMath.Min(t1, t2);
                    float tmaxX = XMath.Max(t1, t2);
                    float tminY = XMath.Min(t3, t4);
                    float tmaxY = XMath.Max(t3, t4);
                    float tEnter = XMath.Max(tminX, tminY);
                    float tExit = XMath.Min(tmaxX, tmaxY);
                    // We want distance to exit from current s; ensure positive
                    float fallbackDist = (scale * activeMapRes * 0.5f) * METERS_TO_KILOMETERS;
                    float distToExit = (tExit > 0f) ? tExit : fallbackDist; // fallback small step if degenerate

                    if (maxH < -20000.0f)
                    {
                        float advance = (distToExit > 0) ? distToExit + 0.0001f : fallbackDist;
                        s += advance;
                        break;
                    }

                    // Compute chord distance to the NEAREST point in the block (entry point)
                    // This gives a conservative (higher) max slope estimate for the block
                    float blockSizeMeters = scale * activeMapRes;
                    float trueDistNear = XMath.Max(trueDist - blockSizeMeters, 1.0f);
                    
                    float possibleSlope;
                    float r_o = R + obsZ;
                    if (s < CLOSE_DISTANCE_THRESHOLD_KM)
                    {
                        // Flat-earth approximation for short range: slope = dH / distance
                        float dH = maxH - obsZ;
                        possibleSlope = (trueDistNear > 1e-6f) ? (dH / trueDistNear) : -1e30f;
                    }
                    else
                    {
                        // Exact spherical calculation using Law of Cosines at nearest point
                        float s_sq_near = trueDistNear * trueDistNear;
                        // Precision fix: (r_p - r_o) replaced with (maxH - obsZ)
                        float z_local = ((maxH - obsZ) * (2.0f * R + maxH + obsZ) - s_sq_near) / (2.0f * r_o);
                        float x_sq = s_sq_near - z_local * z_local;
                        float x_local = (x_sq > 0f) ? XMath.Sqrt(x_sq) : 1e-6f;
                        possibleSlope = z_local / x_local;
                    }

                    float levelMapResOverR = ((scale * activeMapRes) / R);
                    float eps = AdaptiveEpsilon(levelMapResOverR);

                    if (possibleSlope <= (currentHorizonSlope + COMPARISON_EPSILON + eps))
                    {
                        // Advance s by a conservative fraction of tangent exit estimate
                        float advance = useFixedSteps ? fixedStep : ((distToExit > 0) ? (distToExit + 0.0001f) : fallbackDist);
                        s += advance;
                        break;
                    }
                    else
                    {
                        if (level == 0)
                        {
                            float bilinearH = SampleBilinear(activePV.DataLevel0, activePV.Infos, 0, px, py);
                            float sampleSlope = -1e30f;
                            if (IsValid(bilinearH))
                            {
                                if (s < CLOSE_DISTANCE_THRESHOLD_KM)
                                {
                                    // Flat-earth approximation for short range: slope = dH / distance
                                    float dH = bilinearH - obsZ;
                                    sampleSlope = (trueDist > 1e-6f) ? (dH / trueDist) : -1e30f;
                                }
                                else
                                {
                                    // Exact spherical calculation using Law of Cosines at current sample point
                                    float s_sq = trueDist * trueDist;
                                    float r_p_bi = R + bilinearH;
                                    float z_local_bi = ((r_p_bi - r_o) * (r_p_bi + r_o) - s_sq) / (2.0f * r_o);
                                    float x_sq_bi = s_sq - z_local_bi * z_local_bi;
                                    float x_local_bi = (x_sq_bi > 0f) ? XMath.Sqrt(x_sq_bi) : 1e-6f;
                                    sampleSlope = (x_local_bi == 0.0f) ? -1e30f : (z_local_bi / x_local_bi);
                                }
                                currentHorizonSlope = XMath.Max(currentHorizonSlope, sampleSlope);
                                if (logTarget && sampleSlope > loggedSlope)
                                {
                                    loggedSlope = sampleSlope;
                                    loggedDist = trueDistMeters;
                                    loggedPx = px;
                                    loggedPy = py;
                                    loggedElev = bilinearH;
                                    loggedSampleCount += 1f;
                                }
                            }
                            
                            // Adaptive stepping based on margin below horizon
                            var tan2 = EvalCubicTangent(seg.A1, seg.A2, seg.A3, seg.A4, seg.B1, seg.B2, seg.B3, seg.B4, s - sStart);
                            float mag = XMath.Sqrt(tan2.dxds * tan2.dxds + tan2.dyds * tan2.dyds);
                            float dsPixel = (mag > 1e-6f) ? (1.0f / mag) : 0.0005f; // minimum: 1 pixel step
                            
                            float ds;
                            if (useFixedSteps)
                            {
                                ds = fixedStep;
                            }
                            else
                            {
                                // Compute margin-based step: larger steps when well below horizon
                                float margin = currentHorizonSlope - sampleSlope;
                                float dsMargin = (margin > 0f) ? (margin * trueDistMeters * INV_TAN_MAX_SLOPE * METERS_TO_KILOMETERS) : 0f;
                                
                                // Angular error budget cap: step proportional to distance
                                float dsAngular = trueDistMeters * ANGULAR_STEP_FACTOR * METERS_TO_KILOMETERS;
                                
                                // Use max of pixel step and margin step, capped by angular budget
                                ds = XMath.Max(dsPixel, XMath.Min(dsMargin, dsAngular));
                                
                                // Increase sampling frequency 4x for close distances (under 500m)
                                if (s < CLOSE_DISTANCE_THRESHOLD_KM)
                                {
                                    ds *= 0.25f;
                                }
                                float stepFloorKm = (passIndex == 0 && trueDistMeters >= PRIMARY_DEM_FAR_MIN_STEP_DISTANCE_METERS)
                                    ? primaryDemFarMinAdaptiveStepKm
                                    : minAdaptiveStepKm;
                                ds = XMath.Max(ds, stepFloorKm);
                            }
                            s += ds;
                            break;
                        }
                        else
                        {
                            level--;
                        }
                    }
                }
                if (rayIsOutOfBounds) break;

                // s march handled above
            }

            if (logTarget)
            {
                debugBuffer[6] = currentHorizonSlope;
                debugBuffer[7] = loggedSlope;
                debugBuffer[8] = loggedDist;
                debugBuffer[9] = loggedPx;
                debugBuffer[10] = loggedPy;
                debugBuffer[11] = loggedElev;
                debugBuffer[12] = azIdx;
                debugBuffer[13] = pixelIdx;
                debugBuffer[14] = loggedSampleCount;
            }

            output[outIdx] = (currentHorizonSlope <= -1e20f)
                ? float.NegativeInfinity
                : currentHorizonSlope;
        }

        // -----------------------------------------------------------------------
        // DEVICE HELPERS
        // -----------------------------------------------------------------------

        // Returns (Lat, Lon) in Radians
        public static (double, double) InverseProjectDouble(double x, double y, ProjectionParamsDouble p)
        {
            // Stereographic Inverse
            double xp = x - p.FalseEasting;
            double yp = y - p.FalseNorthing;
            double rho = Math.Sqrt(xp * xp + yp * yp);

            if (rho < 1e-9) return (p.Lat0, p.Lon0);

            double c = 2.0 * Math.Atan2(rho, 2.0 * p.K0 * p.R);
            double sinc = Math.Sin(c);
            double cosc = Math.Cos(c);
            double sinPhi0 = Math.Sin(p.Lat0);
            double cosPhi0 = Math.Cos(p.Lat0);

            double lat = Math.Asin(cosc * sinPhi0 + (yp * sinc * cosPhi0) / rho);

            // Atan2(y, x) order
            double term1 = xp * sinc;
            double term2 = rho * cosPhi0 * cosc - yp * sinPhi0 * sinc;
            double lon = p.Lon0 + Math.Atan2(term1, term2);

            return (lat, lon);
        }

        // Returns (MapX, MapY) in Meters
        static (double, double) ProjectToMapDouble(double lat, double lon, ProjectionParamsDouble p)
        {
            // Stereographic Forward
            double sinPhi = Math.Sin(lat);
            double cosPhi = Math.Cos(lat);
            double sinPhi0 = Math.Sin(p.Lat0);
            double cosPhi0 = Math.Cos(p.Lat0);
            double dLam = lon - p.Lon0;
            double cosDLam = Math.Cos(dLam);
            double sinDLam = Math.Sin(dLam);

            double denom = 1.0 + sinPhi0 * sinPhi + cosPhi0 * cosPhi * cosDLam;
            if (Math.Abs(denom) < 1e-10) denom = 1e-10;

            double k = 2.0 * p.K0 * p.R / denom;

            double x = k * cosPhi * sinDLam + p.FalseEasting;
            double y = k * (cosPhi0 * sinPhi - sinPhi0 * cosPhi * cosDLam) + p.FalseNorthing;

            return (x, y);
        }

        internal const int MIN_RAY_SAMPLE_COUNT = 4;
        internal const double MIN_RAY_SAMPLE_SPAN_METERS = 100.0;
        internal const int MAX_RAY_SAMPLE_CAPACITY = 16;
        internal const double MIN_PLANAR_SPAN_FOR_CHORD_FIT_METERS = 500.0;

        // Great Circle Destination
        internal static int BuildRaySamples(
            Vector3d observerVec,
            Vector3d dirMeNormalized,
            double startDist, double maxDist,
            ElevationMap dem, double mapRes,
            Span<RaySample> samples)
        {
            if (samples.Length < MAX_RAY_SAMPLE_CAPACITY)
                throw new ArgumentException($"Sample buffer must have at least {MAX_RAY_SAMPLE_CAPACITY} entries.", nameof(samples));

            int sampleCount = 0;
            if (!TrySampleChord(observerVec, dirMeNormalized, startDist, dem, out var startSample))
                return 0;

            samples[sampleCount++] = startSample;
            RaySample finalInsideSample = startSample;

            if (TrySampleChord(observerVec, dirMeNormalized, maxDist, dem, out var endSample))
            {
                finalInsideSample = endSample;
            }
            else
            {
                double lo = startDist;
                double hi = maxDist;
                RaySample bestInside = startSample;
                for (int iter = 0; iter < 24; iter++)
                {
                    double mid = 0.5 * (lo + hi);
                    if (TrySampleChord(observerVec, dirMeNormalized, mid, dem, out var midSample))
                    {
                        lo = mid;
                        bestInside = midSample;
                    }
                    else
                    {
                        hi = mid;
                    }
                }
                finalInsideSample = bestInside;
            }

            double span = Math.Max(0.0, finalInsideSample.DistanceMeters - startDist);
            if (span > 1e-3)
            {
                const int numSamples = 10;
                for (int i = 1; i <= numSamples && sampleCount < samples.Length; i++)
                {
                    double fraction = (double)i / numSamples;
                    double targetMeters = startDist + span * fraction;
                    if (targetMeters > finalInsideSample.DistanceMeters - 0.5)
                        break;
                    if (targetMeters - samples[sampleCount - 1].DistanceMeters < 0.5)
                        continue;
                    if (TrySampleChord(observerVec, dirMeNormalized, targetMeters, dem, out var targetSample))
                        samples[sampleCount++] = targetSample;
                }

                if (sampleCount < samples.Length && finalInsideSample.DistanceMeters - samples[sampleCount - 1].DistanceMeters > 0.5)
                    samples[sampleCount++] = finalInsideSample;
            }

            return EnsureMinimumSamples(samples, sampleCount, observerVec, dirMeNormalized, dem, mapRes, finalInsideSample.DistanceMeters);
        }

        internal static int EnsureMinimumSamples(
            Span<RaySample> samples,
            int sampleCount,
            Vector3d observerVec,
            Vector3d dirMeNormalized,
            ElevationMap dem,
            double mapRes,
            double finalInsideMeters)
        {
            if (sampleCount == 0)
                return 0;

            double requiredEndMeters = samples[0].DistanceMeters + MIN_RAY_SAMPLE_SPAN_METERS;
            if (sampleCount >= MIN_RAY_SAMPLE_COUNT && samples[sampleCount - 1].DistanceMeters >= requiredEndMeters)
                return sampleCount;

            double step = Math.Max(10.0, mapRes * 2.0);
            double currentMeters = samples[sampleCount - 1].DistanceMeters;
            double extendLimit = Math.Max(finalInsideMeters, requiredEndMeters);

            while ((sampleCount < MIN_RAY_SAMPLE_COUNT || samples[sampleCount - 1].DistanceMeters < requiredEndMeters) &&
                   currentMeters < extendLimit &&
                   sampleCount < samples.Length)
            {
                currentMeters = Math.Min(extendLimit, currentMeters + step);
                if (!TrySampleChord(observerVec, dirMeNormalized, currentMeters, dem, out var sample))
                    break;
                if (currentMeters - samples[sampleCount - 1].DistanceMeters < 0.5)
                    break;
                samples[sampleCount++] = sample;
            }

            return sampleCount;
        }

        internal static Vector3d LatLonToVectorMeters(double latRad, double lonRad, double radiusMeters)
        {
            double cosLat = Math.Cos(latRad);
            double sinLat = Math.Sin(latRad);
            double cosLon = Math.Cos(lonRad);
            double sinLon = Math.Sin(lonRad);
            return new Vector3d(
                radiusMeters * cosLat * cosLon,
                radiusMeters * cosLat * sinLon,
                radiusMeters * sinLat);
        }

        /// <summary>
        /// GPU kernel that evaluates subpatch-based segmented horizon rays.
        /// Each pixel selects the polynomial from its closest subpatch center.
        /// Memory layout: segments[azimuth * numSubpatches * numDems + subpatchIndex * numDems + demIdx]
        /// </summary>
        static void QuadTreeSubpatchRayCastKernel(
            Index2D index,
            PyramidView primaryPV,
            PyramidView activePV,
            ArrayView<float> output,
            ArrayView<RaySegment> segments,
            int passIndex, int numDems,
            int tileColBase, int tileRowBase, int tileW, int tileH, 
            int subpatchSize, KernelParams kernelParams,
            ArrayView<float> debugBuffer
#if QUADTREE_TRAVERSAL_PROFILE
            , ArrayView<long> traversalCounters
#endif
            )
        {
            int pixelIdx = index.X;
            int azIdx = index.Y;

            if (pixelIdx >= tileW * tileH) return;

            // Calculate pixel position relative to tile
            int rowInTile = pixelIdx / tileW;
            int colInTile = pixelIdx % tileW;

            // Calculate which subpatch this pixel belongs to
            int interiorSubpatchesPerDim = tileW / subpatchSize;
            int numSubpatchesPerDim = interiorSubpatchesPerDim + 2;

            float primaryRes = XMath.Sqrt(primaryPV.Map.T1 * primaryPV.Map.T1 + primaryPV.Map.T4 * primaryPV.Map.T4);
            float activeRes = XMath.Sqrt(activePV.Map.T1 * activePV.Map.T1 + activePV.Map.T4 * activePV.Map.T4);
            float scaleRatio = primaryRes / activeRes;

            int numSubpatches = numSubpatchesPerDim * numSubpatchesPerDim;
            float gx = ((float)colInTile - subpatchSize / 2.0f) / subpatchSize + 1.0f;
            float gy = ((float)rowInTile - subpatchSize / 2.0f) / subpatchSize + 1.0f;
            int leftCol = (int)gx;
            int topRow = (int)gy;
            float tx = gx - leftCol;
            float ty = gy - topRow;

            if (leftCol < 0) { leftCol = 0; tx = 0.0f; }
            if (topRow < 0) { topRow = 0; ty = 0.0f; }
            if (leftCol > numSubpatchesPerDim - 2) { leftCol = numSubpatchesPerDim - 2; tx = 1.0f; }
            if (topRow > numSubpatchesPerDim - 2) { topRow = numSubpatchesPerDim - 2; ty = 1.0f; }

            int rightCol = leftCol + 1;
            int bottomRow = topRow + 1;

            long idx00 = ((long)azIdx * numSubpatches + (topRow * numSubpatchesPerDim + leftCol)) * numDems + passIndex;
            long idx10 = ((long)azIdx * numSubpatches + (topRow * numSubpatchesPerDim + rightCol)) * numDems + passIndex;
            long idx01 = ((long)azIdx * numSubpatches + (bottomRow * numSubpatchesPerDim + leftCol)) * numDems + passIndex;
            long idx11 = ((long)azIdx * numSubpatches + (bottomRow * numSubpatchesPerDim + rightCol)) * numDems + passIndex;

            int requestedCenterCol00 = tileColBase + (leftCol - 1) * subpatchSize + subpatchSize / 2;
            int requestedCenterCol10 = tileColBase + (rightCol - 1) * subpatchSize + subpatchSize / 2;
            int requestedCenterRow00 = tileRowBase + (topRow - 1) * subpatchSize + subpatchSize / 2;
            int requestedCenterRow01 = tileRowBase + (bottomRow - 1) * subpatchSize + subpatchSize / 2;

            float centerCol00 = (float)(ClampSubpatchCenterDevice(requestedCenterCol00, kernelParams.PrimaryWidth, subpatchSize) - tileColBase);
            float centerCol10 = (float)(ClampSubpatchCenterDevice(requestedCenterCol10, kernelParams.PrimaryWidth, subpatchSize) - tileColBase);
            float centerRow00 = (float)(ClampSubpatchCenterDevice(requestedCenterRow00, kernelParams.PrimaryHeight, subpatchSize) - tileRowBase);
            float centerRow01 = (float)(ClampSubpatchCenterDevice(requestedCenterRow01, kernelParams.PrimaryHeight, subpatchSize) - tileRowBase);

            var seg00 = ShiftRaySegmentToPixel(segments[idx00], (float)colInTile - centerCol00, (float)rowInTile - centerRow00, scaleRatio);
            var seg10 = ShiftRaySegmentToPixel(segments[idx10], (float)colInTile - centerCol10, (float)rowInTile - centerRow00, scaleRatio);
            var seg01 = ShiftRaySegmentToPixel(segments[idx01], (float)colInTile - centerCol00, (float)rowInTile - centerRow01, scaleRatio);
            var seg11 = ShiftRaySegmentToPixel(segments[idx11], (float)colInTile - centerCol10, (float)rowInTile - centerRow01, scaleRatio);

            var seg = LerpRaySegment(
                LerpRaySegment(seg00, seg10, tx),
                LerpRaySegment(seg01, seg11, tx),
                ty);

            float startPx = seg.StartPixel.X;
            float startPy = seg.StartPixel.Y;
            float sStart = seg.SStart;

            // Apply Grid Convergence correction to output bin.
            // Subpatch-mode rays are fitted at the selected subpatch center, so only correct
            // the local frame delta between that center and the current pixel. Using tile-
            // relative coordinates here makes the correction jump at 128x128 tile seams.
            int correctedAzIdx = azIdx;

            // The output buffer stores horizon slope across DEM passes. Convert to
            // elevation angle only after all passes have completed.
            long outIdx = (long)pixelIdx * 1440 + correctedAzIdx;
            float storedSlope = output[outIdx];
            float currentHorizonSlope = float.IsNegativeInfinity(storedSlope) ? -1e30f : storedSlope;

            // Get Observer Z
            int globalRow = tileRowBase + rowInTile;
            int globalCol = tileColBase + colInTile;
            float obsTerrain = SampleBilinear(primaryPV.DataLevel0, primaryPV.Infos, 0, (float)globalCol, (float)globalRow);
            float obsZ = obsTerrain + kernelParams.ObserverElevation;

            // Ray marching setup (similar to main kernel but using subpatch polynomial)
            float sEnd = seg.SEnd;
            float runtimeStart = (kernelParams.MinTraverseDistanceKm > 0f) ? XMath.Max(sStart, kernelParams.MinTraverseDistanceKm) 
                                                                          : XMath.Max(sStart, 1.0f * METERS_TO_KILOMETERS);
            float s = runtimeStart;
            bool useFixedSteps = (kernelParams.DebugFlags & DEBUG_FLAG_FORCE_FIXED_STEPS) != 0;
            if (useFixedSteps) s = runtimeStart + DEBUG_FIXED_STEP_KM;

            // March the ray using the subpatch polynomial
            float px, py;
            
            // Calculate map resolution for this DEM level (needed for traversal)
            float pixCol = XMath.Sqrt(activePV.Map.T1 * activePV.Map.T1 + activePV.Map.T4 * activePV.Map.T4);
            float pixRow = XMath.Sqrt(activePV.Map.T2 * activePV.Map.T2 + activePV.Map.T5 * activePV.Map.T5);
            float activeMapRes = (pixCol + pixRow) * 0.5f;
            float minAdaptiveStepKm = MIN_ADAPTIVE_STEP_RESOLUTION_FACTOR * activeMapRes * METERS_TO_KILOMETERS;
            float primaryDemFarMinAdaptiveStepKm = PRIMARY_DEM_FAR_MIN_STEP_RESOLUTION_FACTOR * activeMapRes * METERS_TO_KILOMETERS;
            float R = activePV.Proj.R;

            bool disableHierarchy = (kernelParams.DebugFlags & DEBUG_FLAG_DISABLE_HIERARCHY) != 0;
            const float CLOSE_DISTANCE_THRESHOLD_KM = 0.5f; // 500 meters
            bool traceTraversal = pixelIdx == 0 &&
                kernelParams.DebugAzimuthIndex >= 0 &&
                azIdx == kernelParams.DebugAzimuthIndex;
            int traceCount = 0;
#if QUADTREE_TRAVERSAL_PROFILE
            bool profileTraversal = (kernelParams.DebugFlags & DEBUG_FLAG_PROFILE_TRAVERSAL) != 0;
#endif
            bool rayIsOutOfBounds = false;

            while (s <= sEnd)
            {
#if QUADTREE_TRAVERSAL_PROFILE
                if (profileTraversal)
                    AddTraversalCounter(traversalCounters, passIndex, TRAVERSAL_COUNTER_ITERATIONS);
#endif

                // Evaluate cubic polynomial at current s
                px = EvalCubic(startPx, seg.A1, seg.A2, seg.A3, seg.A4, s - sStart);
                py = EvalCubic(startPy, seg.B1, seg.B2, seg.B3, seg.B4, s - sStart);

                // Bounds check
                if (XMath.IsNaN(px) || XMath.IsNaN(py) || px < 0f || py < 0f || 
                    px >= activePV.Infos[0].Width - 1f || py >= activePV.Infos[0].Height - 1f)
                {
#if QUADTREE_TRAVERSAL_PROFILE
                    if (profileTraversal)
                        AddTraversalCounter(traversalCounters, passIndex, TRAVERSAL_COUNTER_OUT_OF_BOUNDS);
#endif
                    break;
                }

                // Calculate true distance (chord)
                float planarDx = (px - startPx) * activeMapRes;
                float planarDy = (py - startPy) * activeMapRes;
                float planarMeters = XMath.Sqrt(planarDx * planarDx + planarDy * planarDy);
                
                float trueDist;
                float trueDistMeters;
                
                if (s < CLOSE_DISTANCE_THRESHOLD_KM)
                {
                    // Use parameterized distance directly - at short range, s ≈ true distance
                    trueDist = s; // km
                    trueDistMeters = s * KILOMETERS_TO_METERS;
                }
                else
                {
                    // Use polynomial-corrected chord distance for larger distances
                    trueDistMeters = (seg.SStartChord * KILOMETERS_TO_METERS) + EvalPlanarChord(seg, planarMeters);
                    trueDist = trueDistMeters * METERS_TO_KILOMETERS;
                }

                if (disableHierarchy)
                {
                    // Always sample Level 0, no culling; use adaptive stepping
#if QUADTREE_TRAVERSAL_PROFILE
                    if (profileTraversal)
                        AddTraversalCounter(traversalCounters, passIndex, TRAVERSAL_COUNTER_LEVEL0_SAMPLES);
#endif
                    float bilinearH = SampleBilinear(activePV.DataLevel0, activePV.Infos, 0, px, py);
                    float sampleSlope = -1e30f;
                    if (IsValid(bilinearH))
                    {
                        if (s < CLOSE_DISTANCE_THRESHOLD_KM)
                        {
                            // Flat-earth approximation for short range: slope = dH / distance
                            float dH = bilinearH - obsZ;
                            sampleSlope = (trueDist > 1e-6f) ? (dH / (trueDist * KILOMETERS_TO_METERS)) : -1e30f;
                        }
                        else
                        {
                            // Exact spherical calculation using Law of Cosines for far field
                            float r_o = R + obsZ;
                            float s_sq = trueDist * trueDist; // s in km, R in m? No, keep units consistent.
                            // R is in meters. trueDist is in km. Need to convert trueDist to meters for R interaction.
                            // Wait, R is meters (Proj.R). obsZ is meters.
                            // Formula: z_local = ((h - z_obs) * (2R + h + z_obs) - s^2) / (2(R + z_obs))
                            // All units must be meters.
                            float s_sq_m = trueDistMeters * trueDistMeters;
                            float z_local_bi = ((bilinearH - obsZ) * (2.0f * R + bilinearH + obsZ) - s_sq_m) / (2.0f * r_o);
                            float x_sq = s_sq_m - z_local_bi * z_local_bi;
                            float x_local_bi = (x_sq > 0f) ? XMath.Sqrt(x_sq) : 1e-6f;
                            sampleSlope = (x_local_bi == 0.0f) ? -1e30f : (z_local_bi / x_local_bi);
                        }
                        
                        if (sampleSlope > currentHorizonSlope)
                            currentHorizonSlope = sampleSlope;
                    }
                    
                    // Adaptive stepping
                    var tanNoCull = EvalCubicTangent(seg.A1, seg.A2, seg.A3, seg.A4, seg.B1, seg.B2, seg.B3, seg.B4, s - sStart);
                    float magNoCull = XMath.Sqrt(tanNoCull.dxds * tanNoCull.dxds + tanNoCull.dyds * tanNoCull.dyds);
                    float dsPixel = (magNoCull > 1e-6f) ? (1.0f / magNoCull) : 0.001f;

                    float dsNoCull;
                    if (useFixedSteps)
                    {
                        dsNoCull = DEBUG_FIXED_STEP_KM;
                    }
                    else
                    {
                        float margin = currentHorizonSlope - sampleSlope;
                        float dsMargin = (margin > 0f) ? (margin * trueDistMeters * INV_TAN_MAX_SLOPE * METERS_TO_KILOMETERS) : 0f;
                        float dsAngular = trueDistMeters * ANGULAR_STEP_FACTOR * METERS_TO_KILOMETERS;
                        dsNoCull = XMath.Max(dsPixel, XMath.Min(dsMargin, dsAngular));
                        
                        if (s < CLOSE_DISTANCE_THRESHOLD_KM) dsNoCull *= 0.25f;
                        float stepFloorKm = (passIndex == 0 && trueDistMeters >= PRIMARY_DEM_FAR_MIN_STEP_DISTANCE_METERS)
                            ? primaryDemFarMinAdaptiveStepKm
                            : minAdaptiveStepKm;
                        dsNoCull = XMath.Max(dsNoCull, stepFloorKm);
                    }

                    s += dsNoCull;
                    continue;
                }

                // Hierarchical Path
                int level = ComputeStartLevel(trueDistMeters, activeMapRes, activePV.Levels);
                while (level >= 0)
                {
                    var info = activePV.Infos[level];
                    int lW = info.Width;
                    int lH = info.Height;
                    int shift = level * 2;
                    int scale = 1 << shift;
                    int lx = (int)px >> shift;
                    int ly = (int)py >> shift;
                    int traceOffset = -1;
                    if (traceTraversal)
                    {
                        traceOffset = 1 + traceCount * 12;
                        if (traceOffset + 11 < debugBuffer.Length)
                        {
                            debugBuffer[traceOffset] = s;
                            debugBuffer[traceOffset + 1] = trueDistMeters;
                            debugBuffer[traceOffset + 2] = level;
                            debugBuffer[traceOffset + 3] = lx;
                            debugBuffer[traceOffset + 4] = ly;
                            debugBuffer[traceOffset + 5] = px;
                            debugBuffer[traceOffset + 6] = py;
                            debugBuffer[traceOffset + 7] = float.NaN;
                            debugBuffer[traceOffset + 8] = float.NaN;
                            debugBuffer[traceOffset + 9] = float.NaN;
                            debugBuffer[traceOffset + 10] = 0f;
                            debugBuffer[traceOffset + 11] = -1f;
                            traceCount++;
                            debugBuffer[0] = traceCount;
                        }
                        else
                        {
                            traceOffset = -1;
                        }
                    }

                    if (lx < 0 || ly < 0 || lx >= lW || ly >= lH) {
#if QUADTREE_TRAVERSAL_PROFILE
                        if (profileTraversal)
                            AddTraversalCounter(traversalCounters, passIndex, TRAVERSAL_COUNTER_OUT_OF_BOUNDS);
#endif
                        float stepKm = (scale * activeMapRes) * METERS_TO_KILOMETERS;
                        float advance = XMath.Max(0.001f, stepKm);
                        if (traceOffset >= 0)
                        {
                            debugBuffer[traceOffset + 10] = advance;
                            debugBuffer[traceOffset + 11] = 3f;
                        }
                        s += advance;
                        rayIsOutOfBounds = true;
                        break;
                    }

                    float maxH = (level == 0)
                        ? activePV.DataLevel0[info.Offset + ly * lW + lx]
                        : activePV.DataMips[info.Offset + ly * lW + lx];
                    if (traceOffset >= 0)
                        debugBuffer[traceOffset + 7] = maxH;
                        
                    float minX = lx * scale;
                    float minY = ly * scale;
                    float maxX = minX + scale;
                    float maxY = minY + scale;

                    // Tangent-linear approximation for exit
                    var tan = EvalCubicTangent(seg.A1, seg.A2, seg.A3, seg.A4, seg.B1, seg.B2, seg.B3, seg.B4, s - sStart);
                    float invDx = (XMath.Abs(tan.dxds) > 1e-8f) ? (1.0f / tan.dxds) : 1e30f;
                    float invDy = (XMath.Abs(tan.dyds) > 1e-8f) ? (1.0f / tan.dyds) : 1e30f;
                    float t1 = (minX - px) * invDx;
                    float t2 = (maxX - px) * invDx;
                    float t3 = (minY - py) * invDy;
                    float t4 = (maxY - py) * invDy;
                    float tEnter = XMath.Max(XMath.Min(t1, t2), XMath.Min(t3, t4));
                    float tExit = XMath.Min(XMath.Max(t1, t2), XMath.Max(t3, t4));
                    
                    float fallbackDist = (scale * activeMapRes * 0.5f) * METERS_TO_KILOMETERS;
                    float distToExit = (tExit > 0f) ? tExit : fallbackDist;

                    if (maxH < -20000.0f)
                    {
#if QUADTREE_TRAVERSAL_PROFILE
                        if (profileTraversal)
                            AddTraversalCounter(traversalCounters, passIndex, TRAVERSAL_COUNTER_NODATA_SKIPS);
#endif
                        float advance = (distToExit > 0) ? distToExit + 0.0001f : fallbackDist;
                        if (traceOffset >= 0)
                        {
                            debugBuffer[traceOffset + 10] = advance;
                            debugBuffer[traceOffset + 11] = 2f;
                        }
                        s += advance;
                        break;
                    }

                    float blockSizeMeters = scale * activeMapRes;
                    float trueDistNear = XMath.Max(trueDistMeters - blockSizeMeters, 1.0f);
                    
                    float possibleSlope;
                    float r_o = R + obsZ;
                    if (s < CLOSE_DISTANCE_THRESHOLD_KM)
                    {
                        float dH = maxH - obsZ;
                        possibleSlope = (trueDistNear > 1e-6f) ? (dH / trueDistNear) : -1e30f;
                    }
                    else
                    {
                        float s_sq_near = trueDistNear * trueDistNear;
                        float z_local = ((maxH - obsZ) * (2.0f * R + maxH + obsZ) - s_sq_near) / (2.0f * r_o);
                        float x_sq = s_sq_near - z_local * z_local;
                        float x_local = (x_sq > 0f) ? XMath.Sqrt(x_sq) : 1e-6f;
                        possibleSlope = z_local / x_local;
                    }

                    float levelMapResOverR = ((scale * activeMapRes) / R);
                    float eps = AdaptiveEpsilon(levelMapResOverR);

                    if (possibleSlope <= (currentHorizonSlope + COMPARISON_EPSILON + eps))
                    {
#if QUADTREE_TRAVERSAL_PROFILE
                        if (profileTraversal)
                            AddTraversalCounter(traversalCounters, passIndex, TRAVERSAL_COUNTER_CULLED_BLOCKS);
#endif
                        float advance = useFixedSteps ? DEBUG_FIXED_STEP_KM : ((distToExit > 0) ? (distToExit + 0.0001f) : fallbackDist);
                        if (traceOffset >= 0)
                        {
                            debugBuffer[traceOffset + 10] = advance;
                            debugBuffer[traceOffset + 11] = 1f;
                        }
                        s += advance;
                        break;
                    }
                    else
                    {
                        if (level == 0)
                        {
#if QUADTREE_TRAVERSAL_PROFILE
                            if (profileTraversal)
                                AddTraversalCounter(traversalCounters, passIndex, TRAVERSAL_COUNTER_LEVEL0_SAMPLES);
#endif
                            float bilinearH = SampleBilinear(activePV.DataLevel0, activePV.Infos, 0, px, py);
                            float sampleSlope = -1e30f;
                            if (IsValid(bilinearH))
                            {
                                if (s < CLOSE_DISTANCE_THRESHOLD_KM)
                                {
                                    float dH = bilinearH - obsZ;
                                    sampleSlope = (trueDistMeters > 1e-6f) ? (dH / trueDistMeters) : -1e30f;
                                }
                                else
                                {
                                    float s_sq = trueDistMeters * trueDistMeters;
                                    float r_p_bi = R + bilinearH;
                                    float z_local_bi = ((r_p_bi - r_o) * (r_p_bi + r_o) - s_sq) / (2.0f * r_o);
                                    float x_sq_bi = s_sq - z_local_bi * z_local_bi;
                                    float x_local_bi = (x_sq_bi > 0f) ? XMath.Sqrt(x_sq_bi) : 1e-6f;
                                    sampleSlope = (x_local_bi == 0.0f) ? -1e30f : (z_local_bi / x_local_bi);
                                }
                                
                                if (sampleSlope > currentHorizonSlope)
                                    currentHorizonSlope = sampleSlope;
                            }
                            if (traceOffset >= 0)
                            {
                                debugBuffer[traceOffset + 8] = bilinearH;
                                debugBuffer[traceOffset + 9] = sampleSlope;
                            }
                            
                            var tan2 = EvalCubicTangent(seg.A1, seg.A2, seg.A3, seg.A4, seg.B1, seg.B2, seg.B3, seg.B4, s - sStart);
                            float mag = XMath.Sqrt(tan2.dxds * tan2.dxds + tan2.dyds * tan2.dyds);
                            float dsPixel = (mag > 1e-6f) ? (1.0f / mag) : 0.0005f;
                            
                            float ds;
                            if (useFixedSteps)
                            {
                                ds = DEBUG_FIXED_STEP_KM;
                            }
                            else
                            {
                                float margin = currentHorizonSlope - sampleSlope;
                                float dsMargin = (margin > 0f) ? (margin * trueDistMeters * INV_TAN_MAX_SLOPE * METERS_TO_KILOMETERS) : 0f;
                                float dsAngular = trueDistMeters * ANGULAR_STEP_FACTOR * METERS_TO_KILOMETERS;
                                ds = XMath.Max(dsPixel, XMath.Min(dsMargin, dsAngular));
                                
                                if (s < CLOSE_DISTANCE_THRESHOLD_KM) ds *= 0.25f;
                                float stepFloorKm = (passIndex == 0 && trueDistMeters >= PRIMARY_DEM_FAR_MIN_STEP_DISTANCE_METERS)
                                    ? primaryDemFarMinAdaptiveStepKm
                                    : minAdaptiveStepKm;
                                ds = XMath.Max(ds, stepFloorKm);
                            }
                            if (traceOffset >= 0)
                            {
                                debugBuffer[traceOffset + 10] = ds;
                                debugBuffer[traceOffset + 11] = 4f;
                            }
                            s += ds;
                            break;
                        }
                        else
                        {
                            if (traceOffset >= 0)
                                debugBuffer[traceOffset + 11] = 0f;
                            level--;
                        }
                    }
                }
                if (rayIsOutOfBounds) break;
            }

            output[outIdx] = (currentHorizonSlope > -1e29f)
                ? currentHorizonSlope
                : float.NegativeInfinity;
        }

        static RaySegment ShiftRaySegmentToPixel(RaySegment seg, float dCol, float dRow, float scaleRatio)
        {
            float dx = dCol * scaleRatio;
            float dy = dRow * scaleRatio;
            seg.StartPixel = new Vector2(seg.StartPixel.X + dx, seg.StartPixel.Y + dy);
            seg.X0 += dx;
            seg.Y0 += dy;
            return seg;
        }

#if QUADTREE_TRAVERSAL_PROFILE
        static void AddTraversalCounter(ArrayView<long> counters, int demPass, int counterOffset)
        {
            long index = (long)demPass * TRAVERSAL_COUNTERS_PER_DEM + counterOffset;
            Atomic.Add(ref counters[index], 1L);
        }
#endif

        static int ClampSubpatchCenterDevice(int requestedCenter, int demSize, int subpatchSize)
        {
            int half = subpatchSize / 2;
            int minCenter = half;
            int maxCenter = demSize - half;
            if (maxCenter < minCenter) maxCenter = minCenter;
            if (requestedCenter < minCenter) return minCenter;
            if (requestedCenter > maxCenter) return maxCenter;
            return requestedCenter;
        }

        static RaySegment LerpRaySegment(RaySegment a, RaySegment b, float t)
        {
            return new RaySegment
            {
                StartPixel = new Vector2(Lerp(a.StartPixel.X, b.StartPixel.X, t), Lerp(a.StartPixel.Y, b.StartPixel.Y, t)),
                DemId = a.DemId,
                X0 = Lerp(a.X0, b.X0, t),
                Y0 = Lerp(a.Y0, b.Y0, t),
                A1 = Lerp(a.A1, b.A1, t),
                A2 = Lerp(a.A2, b.A2, t),
                A3 = Lerp(a.A3, b.A3, t),
                A4 = Lerp(a.A4, b.A4, t),
                B1 = Lerp(a.B1, b.B1, t),
                B2 = Lerp(a.B2, b.B2, t),
                B3 = Lerp(a.B3, b.B3, t),
                B4 = Lerp(a.B4, b.B4, t),
                SStart = Lerp(a.SStart, b.SStart, t),
                SEnd = Lerp(a.SEnd, b.SEnd, t),
                SStartChord = Lerp(a.SStartChord, b.SStartChord, t),
                PlanarToChordC1 = Lerp(a.PlanarToChordC1, b.PlanarToChordC1, t),
                PlanarToChordC2 = Lerp(a.PlanarToChordC2, b.PlanarToChordC2, t),
                PlanarToChordC3 = Lerp(a.PlanarToChordC3, b.PlanarToChordC3, t)
            };
        }

        static float Lerp(float a, float b, float t)
        {
            return a + (b - a) * t;
        }

        internal static (double lat, double lon) VecME2LatLon(Vector3d vec)
        {
            double lon = Math.Atan2(vec.Y, vec.X);
            if (lon < 0d) lon += Math.PI * 2d;
            double alen = Math.Sqrt((vec.X * vec.X) + (vec.Y * vec.Y));
            double lat = Math.Atan2(vec.Z, alen);
            return (lat, lon);
        }

        internal static Matrix4d GetRotationMatrixd(double lat_rad, double lon_rad)
        {
            double cosLat = Math.Cos(lat_rad);
            double sinLat = Math.Sin(lat_rad);
            double cosLon = Math.Cos(lon_rad);
            double sinLon = Math.Sin(lon_rad);

            // Standard ENU basis vectors in MME frame
            Vector3d up = new Vector3d(cosLat * cosLon, cosLat * sinLon, sinLat);
            Vector3d east = new Vector3d(-sinLon, cosLon, 0);
            Vector3d north = new Vector3d(-sinLat * cosLon, -sinLat * sinLon, cosLat);

            // Row-major matrix for V_me = V_enu * M
            // where M rows are basis vectors.
            return new Matrix4d(
                east.X, east.Y, east.Z, 0,
                north.X, north.Y, north.Z, 0,
                up.X, up.Y, up.Z, 0,
                0, 0, 0, 1
            );
        }

        internal static Vector3d ComputeDirectionVector(Matrix4d obsToMe, double az)
        {
            double angle = (Math.PI / 2.0) - az;
            var dirObs = new Vector3d(Math.Cos(angle), Math.Sin(angle), 0.0);
            var dirMe = Vector3d.Transform(dirObs, obsToMe);
            return Vector3d.Normalize(dirMe);
        }

        internal static bool TrySampleChord(
            Vector3d observerVec,
            Vector3d dirMeNormalized,
            double dist,
            ElevationMap dem,
            out RaySample sample)
        {
            var sampleVec = observerVec + dirMeNormalized * dist;
            var (lat, lon) = VecME2LatLon(sampleVec);
            var (row, col) = dem.LonLatRad2RowCol(lon, lat);
            bool inBounds = col >= 0 && col < dem.Width && row >= 0 && row < dem.Height;
            double terrainHeight = inBounds ? dem.GetElevation(col, row) : 0.0;
            sample = new RaySample
            {
                DistanceMeters = dist,
                PixelX = col,
                PixelY = row,
                LatRad = lat,
                LonRad = lon,
                Row = row,
                Col = col,
                TerrainHeightMeters = terrainHeight
            };
            return inBounds;
        }

        internal static void FitQuartic4TermsDouble(ReadOnlySpan<double> s, ReadOnlySpan<double> v, out double a1, out double a2, out double a3, out double a4)
        {
            double m11 = 0, m12 = 0, m13 = 0, m14 = 0;
            double m22 = 0, m23 = 0, m24 = 0;
            double m33 = 0, m34 = 0;
            double m44 = 0;
            double b1 = 0, b2 = 0, b3 = 0, b4 = 0;

            for (int i = 0; i < s.Length; i++)
            {
                double si = s[i];
                double s2 = si * si;
                double s3 = s2 * si;
                double s4 = s2 * s2;
                double s5 = s4 * si;
                double s6 = s5 * si;
                double s7 = s6 * si;
                double s8 = s7 * si;

                m11 += s2;
                m12 += s3;
                m13 += s4;
                m14 += s5;
                m22 += s4;
                m23 += s5;
                m24 += s6;
                m33 += s6;
                m34 += s7;
                m44 += s8;

                double vi = v[i];
                b1 += si * vi;
                b2 += s2 * vi;
                b3 += s3 * vi;
                b4 += s4 * vi;
            }

            double[,] mat = new double[4, 4]
            {
                { m11, m12, m13, m14 },
                { m12, m22, m23, m24 },
                { m13, m23, m33, m34 },
                { m14, m24, m34, m44 }
            };
            double[] rhs = new double[] { b1, b2, b3, b4 };
            double[] solution = new double[4];
            if (!SolveLinearSystem4(mat, rhs, solution))
            {
                a1 = a2 = a3 = a4 = 0;
                return;
            }
            a1 = solution[0];
            a2 = solution[1];
            a3 = solution[2];
            a4 = solution[3];
        }

        internal static void FitQuartic4TermsDouble(double[] s, double[] v, out double a1, out double a2, out double a3, out double a4) =>
            FitQuartic4TermsDouble(s.AsSpan(), v.AsSpan(), out a1, out a2, out a3, out a4);

        internal static void FitPlanarToChordCubic(List<(double s, double x, double y)> samples, double mapRes, out double c1, out double c2, out double c3)
        {
            // Legacy overload for backward compatibility - uses tangent distance as approximation
            // This is only accurate for flat terrain; real terrain should use the overload with observer/DEM info
            c1 = 1.0;
            c2 = 0.0;
            c3 = 0.0;
            if (samples.Count < 2 || mapRes <= 0.0)
                return;

            double x0 = samples[0].x;
            double y0 = samples[0].y;
            double s0 = samples[0].s;
            double maxPlanar = 0.0;
            double[] planar = new double[samples.Count];
            double[] delta = new double[samples.Count];
            planar[0] = 0.0;
            delta[0] = 0.0;

            for (int i = 1; i < samples.Count; i++)
            {
                double dx = (samples[i].x - x0) * mapRes;
                double dy = (samples[i].y - y0) * mapRes;
                double p = Math.Sqrt(dx * dx + dy * dy);
                planar[i] = p;
                delta[i] = (samples[i].s - s0) * KILOMETERS_TO_METERS_D;
                if (p > maxPlanar)
                    maxPlanar = p;
            }

            if (maxPlanar < 1e-4)
                return;

            FitCubicNoIntercept(planar, delta, out c1, out c2, out c3);
        }

        /// <summary>
        /// Fits a cubic polynomial mapping planar distance to true 3D chord distance.
        /// This overload is used by the main horizon path and models the surface as a
        /// fixed-radius sphere using the observer-pixel elevation.
        /// </summary>
        internal static void FitPlanarToChordCubicWithTerrain(
            ReadOnlySpan<RaySample> samples,
            double mapRes,
            Vector3d observerVec,
            double R,
            ElevationMap dem,
            out double c1, out double c2, out double c3,
            bool verbose = false)
        {
            c1 = 1.0;
            c2 = 0.0;
            c3 = 0.0;
            if (samples.Length < 2 || mapRes <= 0.0)
                return;

            double x0 = samples[0].PixelX;
            double y0 = samples[0].PixelY;
            double chord0 = ComputeChordDistanceOnSphere(observerVec, samples[0].LatRad, samples[0].LonRad, R);
            Span<double> planarArr = stackalloc double[MAX_RAY_SAMPLE_CAPACITY];
            Span<double> chordArr = stackalloc double[MAX_RAY_SAMPLE_CAPACITY];
            int validCount = 1;
            planarArr[0] = 0.0;
            chordArr[0] = 0.0;

            if (verbose)
            {
                Console.WriteLine($"FitPlanarToChordCubicWithTerrain: mapRes={mapRes:F6}, R={R:F0}");
                Console.WriteLine($"  Sample 0: s={samples[0].DistanceMeters * METERS_TO_KILOMETERS_D:F6}km, tangent={samples[0].DistanceMeters:F2}m, chord0={chord0:F2}m");
            }

            double maxPlanar = 0.0;
            for (int i = 1; i < samples.Length && validCount < planarArr.Length; i++)
            {
                double sampleX = samples[i].PixelX;
                double sampleY = samples[i].PixelY;
                bool inBounds = sampleX >= 0 && sampleX < dem.Width - 1 && sampleY >= 0 && sampleY < dem.Height - 1;

                if (!inBounds)
                {
                    if (verbose)
                    {
                        Console.WriteLine($"  Sample {i}: SKIPPED (at DEM edge) px=({sampleX:F2},{sampleY:F2}), DEM=({dem.Width},{dem.Height})");
                    }
                    continue;
                }

                double chord = ComputeChordDistanceOnSphere(observerVec, samples[i].LatRad, samples[i].LonRad, R);
                double dx = (sampleX - x0) * mapRes;
                double dy = (sampleY - y0) * mapRes;
                double p = Math.Sqrt(dx * dx + dy * dy);
                double chordDelta = chord - chord0;
                planarArr[validCount] = p;
                chordArr[validCount] = chordDelta;
                validCount++;

                if (verbose)
                {
                    Console.WriteLine($"  Sample {i}: s={samples[i].DistanceMeters * METERS_TO_KILOMETERS_D:F6}km, planar={p:F2}m, chord={chord:F2}m, chordDelta={chordDelta:F2}m, px=({sampleX:F2},{sampleY:F2})");
                }

                if (p > maxPlanar)
                    maxPlanar = p;
            }

            if (verbose)
            {
                Console.WriteLine($"  Valid samples: {validCount}/{samples.Length}, maxPlanar={maxPlanar:F2}m");
            }

            if (validCount < 2 || maxPlanar < 1e-4)
                return;

            FitCubicNoIntercept(planarArr[..validCount], chordArr[..validCount], out c1, out c2, out c3);

            if (verbose)
            {
                Console.WriteLine($"  Fitted: c1={c1:E9}, c2={c2:E9}, c3={c3:E9}");
            }

            const double C1_MIN = 0.5;
            const double C1_MAX = 2.0;
            if (c1 < C1_MIN || c1 > C1_MAX)
            {
                double originalC1 = c1;
                c1 = Math.Clamp(c1, C1_MIN, C1_MAX);
                c2 = 0.0;
                c3 = 0.0;
                Log.Warning(
                    "FitPlanarToChordCubicWithTerrain: invalid C1={OriginalC1:F6} outside [{C1Min:F2},{C1Max:F2}]; " +
                    "using fallback c1={FallbackC1:F6}, c2=0, c3=0. validSamples={ValidSamples} maxPlanar={MaxPlanar:F2}m",
                    originalC1,
                    C1_MIN,
                    C1_MAX,
                    c1,
                    validCount,
                    maxPlanar);
            }
        }

        internal static void FitPlanarToChordCubicWithTerrain(
            ReadOnlySpan<RaySample> samples,
            double mapRes,
            Vector3d observerVec,
            double R,
            out double c1, out double c2, out double c3,
            bool verbose = false)
        {
            c1 = 1.0;
            c2 = 0.0;
            c3 = 0.0;
            if (samples.Length < 2 || mapRes <= 0.0)
                return;

            double x0 = samples[0].PixelX;
            double y0 = samples[0].PixelY;
            double chord0 = ComputeChordDistanceOnSphere(observerVec, samples[0].LatRad, samples[0].LonRad, R);
            Span<double> planarArr = stackalloc double[MAX_RAY_SAMPLE_CAPACITY];
            Span<double> chordArr = stackalloc double[MAX_RAY_SAMPLE_CAPACITY];
            int validCount = 1;
            planarArr[0] = 0.0;
            chordArr[0] = 0.0;

            if (verbose)
            {
                Console.WriteLine($"FitPlanarToChordCubicWithTerrain: mapRes={mapRes:F6}, R={R:F0}");
                Console.WriteLine($"  Sample 0: s={samples[0].DistanceMeters * METERS_TO_KILOMETERS_D:F6}km, tangent={samples[0].DistanceMeters:F2}m, chord0={chord0:F2}m");
            }

            double maxPlanar = 0.0;
            for (int i = 1; i < samples.Length && validCount < planarArr.Length; i++)
            {
                double sampleX = samples[i].PixelX;
                double sampleY = samples[i].PixelY;
                bool inBounds = sampleX >= 0 && sampleY >= 0;
                
                if (!inBounds)
                {
                    if (verbose)
                    {
                        Console.WriteLine($"  Sample {i}: SKIPPED (invalid sample) px=({sampleX:F2},{sampleY:F2})");
                    }
                    continue;  // Skip edge samples
                }
                
                double chord = ComputeChordDistanceOnSphere(observerVec, samples[i].LatRad, samples[i].LonRad, R);
                
                // Planar distance from start pixel
                double dx = (sampleX - x0) * mapRes;
                double dy = (sampleY - y0) * mapRes;
                double p = Math.Sqrt(dx * dx + dy * dy);
                double chordDelta = chord - chord0;
                planarArr[validCount] = p;
                chordArr[validCount] = chordDelta;
                validCount++;

                if (verbose)
                {
                    Console.WriteLine($"  Sample {i}: s={samples[i].DistanceMeters * METERS_TO_KILOMETERS_D:F6}km, planar={p:F2}m, chord={chord:F2}m, chordDelta={chordDelta:F2}m, px=({sampleX:F2},{sampleY:F2})");
                }

                if (p > maxPlanar)
                    maxPlanar = p;
            }

            if (verbose)
            {
                Console.WriteLine($"  Valid samples: {validCount}/{samples.Length}, maxPlanar={maxPlanar:F2}m");
            }

            if (validCount < 2 || maxPlanar < 1e-4)
                return;

            if (maxPlanar < MIN_PLANAR_SPAN_FOR_CHORD_FIT_METERS)
            {
                if (verbose)
                {
                    Console.WriteLine(
                        $"  Short-span fallback: maxPlanar={maxPlanar:F2}m < {MIN_PLANAR_SPAN_FOR_CHORD_FIT_METERS:F2}m, using c1=1,c2=0,c3=0");
                }

                Log.Debug(
                    "FitPlanarToChordCubicWithTerrain: short-span fallback maxPlanar={MaxPlanar:F2}m validSamples={ValidSamples}; using c1=1, c2=0, c3=0.",
                    maxPlanar,
                    validCount);
                return;
            }
            
            FitCubicNoIntercept(planarArr[..validCount], chordArr[..validCount], out c1, out c2, out c3);
            
            if (verbose)
            {
                Console.WriteLine($"  Fitted: c1={c1:E9}, c2={c2:E9}, c3={c3:E9}");
            }
            
            // Validate polynomial coefficients
            // For chord vs planar distance, C1 should be close to 1.0 (chord ≈ planar at short range)
            // Allow reasonable range for spherical effects and terrain variation
            const double C1_MIN = 0.5;
            const double C1_MAX = 2.0;
            if (c1 < C1_MIN || c1 > C1_MAX)
            {
                double originalC1 = c1;
                // Degrade gracefully instead of aborting the entire run.
                // A linear identity mapping is safer than propagating unstable cubic coefficients.
                c1 = 1.0;
                c2 = 0.0;
                c3 = 0.0;
                Log.Warning(
                    "FitPlanarToChordCubicWithTerrain: invalid C1={OriginalC1:F6} outside [{C1Min:F2},{C1Max:F2}]; " +
                    "using fallback c1={FallbackC1:F6}, c2=0, c3=0. validSamples={ValidSamples} maxPlanar={MaxPlanar:F2}m",
                    originalC1,
                    C1_MIN,
                    C1_MAX,
                    c1,
                    validCount,
                    maxPlanar);
            }
        }

        internal static double ComputeChordDistance(
            Vector3d observerVec,
            double lat,
            double lon,
            double terrainHeight,
            double R)
        {
            var terrainPoint = LatLonToVectorMeters(lat, lon, R + terrainHeight);
            return (terrainPoint - observerVec).Length;
        }

        internal static double ComputeChordDistanceOnSphere(
            Vector3d observerVec,
            double lat,
            double lon,
            double sphereRadius)
        {
            var surfacePoint = LatLonToVectorMeters(lat, lon, sphereRadius);
            return (surfacePoint - observerVec).Length;
        }

        internal static void FitCubicNoIntercept(ReadOnlySpan<double> x, ReadOnlySpan<double> y, out double c1, out double c2, out double c3)
        {
            double m11 = 0, m12 = 0, m13 = 0;
            double m22 = 0, m23 = 0, m33 = 0;
            double b1 = 0, b2 = 0, b3 = 0;

            for (int i = 0; i < x.Length; i++)
            {
                double xi = x[i];
                double yi = y[i];
                double x2 = xi * xi;
                double x3 = x2 * xi;
                double x4 = x2 * x2;
                double x5 = x4 * xi;
                double x6 = x3 * x3;

                m11 += x2;
                m12 += x3;
                m13 += x4;
                m22 += x4;
                m23 += x5;
                m33 += x6;

                b1 += xi * yi;
                b2 += x2 * yi;
                b3 += x3 * yi;
            }

            double[,] mat = new double[3, 3]
            {
                { m11, m12, m13 },
                { m12, m22, m23 },
                { m13, m23, m33 },
            };

            double[] rhs = new double[] { b1, b2, b3 };
            double[] sol = new double[3];
            if (!SolveLinearSystem3(mat, rhs, sol))
            {
                c1 = 1.0;
                c2 = 0.0;
                c3 = 0.0;
                return;
            }

            c1 = sol[0];
            c2 = sol[1];
            c3 = sol[2];
        }

        internal static void FitCubicNoIntercept(double[] x, double[] y, out double c1, out double c2, out double c3) =>
            FitCubicNoIntercept(x.AsSpan(), y.AsSpan(), out c1, out c2, out c3);

        internal static bool SolveLinearSystem3(double[,] matrix, double[] rhs, double[] solution)
        {
            double[,] aug = new double[3, 4];
            for (int r = 0; r < 3; r++)
            {
                for (int c = 0; c < 3; c++)
                    aug[r, c] = matrix[r, c];
                aug[r, 3] = rhs[r];
            }

            for (int i = 0; i < 3; i++)
            {
                int pivot = i;
                double max = Math.Abs(aug[i, i]);
                for (int r = i + 1; r < 3; r++)
                {
                    double val = Math.Abs(aug[r, i]);
                    if (val > max)
                    {
                        max = val;
                        pivot = r;
                    }
                }
                if (max < 1e-12)
                    return false;
                if (pivot != i)
                {
                    for (int c = i; c <= 3; c++)
                    {
                        double temp = aug[i, c];
                        aug[i, c] = aug[pivot, c];
                        aug[pivot, c] = temp;
                    }
                }
                double pivotVal = aug[i, i];
                for (int c = i; c <= 3; c++)
                    aug[i, c] /= pivotVal;
                for (int r = 0; r < 3; r++)
                {
                    if (r == i) continue;
                    double factor = aug[r, i];
                    if (Math.Abs(factor) < 1e-18) continue;
                    for (int c = i; c <= 3; c++)
                        aug[r, c] -= factor * aug[i, c];
                }
            }

            for (int i = 0; i < 3; i++)
                solution[i] = aug[i, 3];
            return true;
        }

        internal static bool SolveLinearSystem4(double[,] matrix, double[] rhs, double[] solution)
        {
            double[,] aug = new double[4, 5];
            for (int r = 0; r < 4; r++)
            {
                for (int c = 0; c < 4; c++)
                    aug[r, c] = matrix[r, c];
                aug[r, 4] = rhs[r];
            }

            for (int i = 0; i < 4; i++)
            {
                int pivot = i;
                double max = Math.Abs(aug[i, i]);
                for (int r = i + 1; r < 4; r++)
                {
                    double val = Math.Abs(aug[r, i]);
                    if (val > max)
                    {
                        max = val;
                        pivot = r;
                    }
                }
                if (max < 1e-12)
                    return false;
                if (pivot != i)
                {
                    for (int c = i; c <= 4; c++)
                    {
                        double temp = aug[i, c];
                        aug[i, c] = aug[pivot, c];
                        aug[pivot, c] = temp;
                    }
                }
                double diag = aug[i, i];
                for (int c = i; c <= 4; c++)
                    aug[i, c] /= diag;
                for (int r = 0; r < 4; r++)
                {
                    if (r == i) continue;
                    double factor = aug[r, i];
                    if (Math.Abs(factor) < 1e-12) continue;
                    for (int c = i; c <= 4; c++)
                        aug[r, c] -= factor * aug[i, c];
                }
            }

            for (int i = 0; i < 4; i++)
                solution[i] = aug[i, 4];
            return true;
        }

        static void FitCubic3TermsDouble(double[] s, double[] v, out double a1, out double a2, out double a3)
        {
            FitQuartic4TermsDouble(s, v, out a1, out a2, out a3, out _);
        }

        // Returns (Lat, Lon) in Radians
        static (float, float) InverseProject(float x, float y, ProjectionParams p)
        {
            // Stereographic Inverse
            float xp = x - p.FalseEasting;
            float yp = y - p.FalseNorthing;
            float rho = XMath.Sqrt(xp * xp + yp * yp);

            if (rho < 1e-5f) return (p.Lat0, p.Lon0);

            float c = 2.0f * XMath.Atan2(rho, 2.0f * p.K0 * p.R);
            float sinc = XMath.Sin(c);
            float cosc = XMath.Cos(c);
            float sinPhi0 = XMath.Sin(p.Lat0);
            float cosPhi0 = XMath.Cos(p.Lat0);

            float lat = XMath.Asin(cosc * sinPhi0 + (yp * sinc * cosPhi0) / rho);

            // Atan2(y, x) order
            float term1 = xp * sinc;
            float term2 = rho * cosPhi0 * cosc - yp * sinPhi0 * sinc;
            float lon = p.Lon0 + XMath.Atan2(term1, term2);

            return (lat, lon);
        }

        // Returns (MapX, MapY) in Meters
        static (float, float) ProjectToMap(float lat, float lon, ProjectionParams p)
        {
            // Stereographic Forward
            float sinPhi = XMath.Sin(lat);
            float cosPhi = XMath.Cos(lat);
            float sinPhi0 = XMath.Sin(p.Lat0);
            float cosPhi0 = XMath.Cos(p.Lat0);
            float dLam = lon - p.Lon0;
            float cosDLam = XMath.Cos(dLam);
            float sinDLam = XMath.Sin(dLam);

            float denom = 1.0f + sinPhi0 * sinPhi + cosPhi0 * cosPhi * cosDLam;
            if (XMath.Abs(denom) < 1e-10f) denom = 1e-10f;

            float k = 2.0f * p.K0 * p.R / denom;

            float x = k * cosPhi * sinDLam + p.FalseEasting;
            float y = k * (cosPhi0 * sinPhi - sinPhi0 * cosPhi * cosDLam) + p.FalseNorthing;

            return (x, y);
        }

        // Great Circle Destination
        static (float, float) GetLatLon(float lat1, float lon1, float az, float dist, float R)
        {
            float angDist = dist / R;

            // Handle North Pole Singularity
            if (lat1 > 1.5700f) // > 89.95 degrees
            {
                // At North Pole, Azimuth 0 should point along Lon 180 (Grid +Y, "Up").
                // Azimuth 90 should point along Lon 90 (Grid +X, "Right").
                // dLam = PI - az satisfies this.
                return (1.5707963f - angDist, lon1 + 3.1415927f - az);
            }

            // Handle South Pole Singularity
            if (lat1 < -1.5700f) // < -89.95 degrees
            {
                // At South Pole, Azimuth 0 should point along Lon 0 (Grid +Y, "Up").
                // Azimuth 90 should point along Lon 90 (Grid +X, "Right").
                // dLam = az satisfies this.
                return (-1.5707963f + angDist, lon1 + az);
            }

            float s1 = XMath.Sin(lat1);
            float c1 = XMath.Cos(lat1);
            float sa = XMath.Sin(angDist);
            float ca = XMath.Cos(angDist);
            float sAz = XMath.Sin(az);
            float cAz = XMath.Cos(az);

            float lat2 = XMath.Asin(s1 * ca + c1 * sa * cAz);
            float lon2 = lon1 + XMath.Atan2(sAz * sa * c1, ca - s1 * XMath.Sin(lat2));

            return (lat2, lon2);
        }

        static float SampleBilinear(ArrayView<float> data, ArrayView<LevelInfo> infos, int level, float col, float row)
        {
            var info = infos[level];
            int w = info.Width;
            int h = info.Height;
            int offset = info.Offset;

            float c = XMath.Clamp(col, 0, w - 1.0001f);
            float r = XMath.Clamp(row, 0, h - 1.0001f);

            int x0 = (int)XMath.Floor(c);
            int y0 = (int)XMath.Floor(r);
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            if (x1 >= w) x1 = w - 1;
            if (y1 >= h) y1 = h - 1;

            float tx = c - x0;
            float ty = r - y0;

            float h00 = data[offset + y0 * w + x0];
            float h10 = data[offset + y0 * w + x1];
            float h01 = data[offset + y1 * w + x0];
            float h11 = data[offset + y1 * w + x1];

            // If any neighbor is invalid, assume invalid.
            if (!IsValid(h00) || !IsValid(h10) || !IsValid(h01) || !IsValid(h11)) return -32000.0f;

            float top = h00 + tx * (h10 - h00);
            float bottom = h01 + tx * (h11 - h01);
            return top + ty * (bottom - top);
        }

        static bool IsInside(MapParams map, LevelInfo level0Info, float x, float y, out float px, out float py, out int w, out int h)
        {
            var res = map.CRSToPixel(x, y);
            px = res.px;
            py = res.py;
            w = level0Info.Width;
            h = level0Info.Height;
            return px >= 0 && py >= 0 && px < w && py < h;
        }

        // --- Cubic helpers ---
        // Fit cubic coefficients for x(s) or y(s) given 4 samples (s0..s3). Constrain x(0) to x0 by subtracting anchor externally.
        // Solves for a1,a2,a3 in x(s) = a1*s + a2*s^2 + a3*s^3 using normal equations on a small system.
        static void FitCubic3Terms(float[] s, float[] v, out float a1, out float a2, out float a3)
        {
            double[] sd = Array.ConvertAll(s, x => (double)x);
            double[] vd = Array.ConvertAll(v, x => (double)x);
            FitQuartic4TermsDouble(sd, vd, out double da1, out double da2, out double da3, out double _);
            a1 = (float)da1;
            a2 = (float)da2;
            a3 = (float)da3;
        }

        static float EvalCubic(float x0, float a1, float a2, float a3, float a4, float s)
        {
            float s2 = s * s;
            float s3 = s2 * s;
            float s4 = s2 * s2;
            return x0 + a1 * s + a2 * s2 + a3 * s3 + a4 * s4;
        }

        static float EvalPlanarChord(RaySegment seg, float planarMeters)
        {
            float p2 = planarMeters * planarMeters;
            float p3 = p2 * planarMeters;
            return seg.PlanarToChordC1 * planarMeters + seg.PlanarToChordC2 * p2 + seg.PlanarToChordC3 * p3;
        }

        static (float dxds, float dyds) EvalCubicTangent(float a1, float a2, float a3, float a4, float b1, float b2, float b3, float b4, float s)
        {
            float s2 = s * s;
            float s3 = s2 * s;
            float dxds = a1 + 2f * a2 * s + 3f * a3 * s2 + 4f * a4 * s3;
            float dyds = b1 + 2f * b2 * s + 3f * b3 * s2 + 4f * b4 * s3;
            return (dxds, dyds);
        }

        // Compute AABB of current block in map space with guard band
        static void ComputeBlockAABB(MapParams map, LevelInfo levelInfo, int scale, int lx, int ly, float guardPixels, out float minX, out float minY, out float maxX, out float maxY)
        {
            // Pixel to CRS basis vectors
            float t1 = map.T1, t2 = map.T2, t4 = map.T4, t5 = map.T5;
            float originX = map.T0; // geo[0]
            float originY = map.T3; // geo[3]

            // Compute map pixel size (approx, average of axes)
            float pixCol = XMath.Sqrt(t1 * t1 + t4 * t4);
            float pixRow = XMath.Sqrt(t2 * t2 + t5 * t5);
            float mapRes = (pixCol + pixRow) * 0.5f;

            // Block size in pixels
            float sidePx = scale;
            float guard = guardPixels * mapRes;

            // Top-left pixel of block in level0 pixel space
            float px0 = lx * scale;
            float py0 = ly * scale;

            // Convert corners to CRS using affine
            // Corner 0: (px0, py0)
            float x0 = originX + t1 * px0 + t2 * py0;
            float y0 = originY + t4 * px0 + t5 * py0;
            // Corner 1: (px0 + sidePx, py0)
            float x1 = originX + t1 * (px0 + sidePx) + t2 * py0;
            float y1 = originY + t4 * (px0 + sidePx) + t5 * py0;
            // Corner 2: (px0, py0 + sidePx)
            float x2 = originX + t1 * px0 + t2 * (py0 + sidePx);
            float y2 = originY + t4 * px0 + t5 * (py0 + sidePx);
            // Corner 3: (px0 + sidePx, py0 + sidePx)
            float x3 = originX + t1 * (px0 + sidePx) + t2 * (py0 + sidePx);
            float y3 = originY + t4 * (px0 + sidePx) + t5 * (py0 + sidePx);

            minX = XMath.Min(XMath.Min(x0, x1), XMath.Min(x2, x3)) - guard;
            maxX = XMath.Max(XMath.Max(x0, x1), XMath.Max(x2, x3)) + guard;
            minY = XMath.Min(XMath.Min(y0, y1), XMath.Min(y2, y3)) - guard;
            maxY = XMath.Max(XMath.Max(y0, y1), XMath.Max(y2, y3)) + guard;
        }

        // Ray-AABB slab intersection: returns t_enter (>=0) along ray dir, or -1 if no hit
        // Also returns tExit via out parameter
        static float RayAabbIntersect(float rX, float rY, float dX, float dY, float minX, float minY, float maxX, float maxY, out float tExit)
        {
            // Avoid division by zero by using large numbers
            float invDx = (XMath.Abs(dX) > 1e-8f) ? (1.0f / dX) : 1e30f;
            float invDy = (XMath.Abs(dY) > 1e-8f) ? (1.0f / dY) : 1e30f;

            float t1 = (minX - rX) * invDx;
            float t2 = (maxX - rX) * invDx;
            float t3 = (minY - rY) * invDy;
            float t4 = (maxY - rY) * invDy;

            float tminX = XMath.Min(t1, t2);
            float tmaxX = XMath.Max(t1, t2);
            float tminY = XMath.Min(t3, t4);
            float tmaxY = XMath.Max(t3, t4);

            float tEnter = XMath.Max(tminX, tminY);
            tExit = XMath.Min(tmaxX, tmaxY);

            if (tExit < 0.0f) return -1.0f; // AABB behind ray
            if (tEnter > tExit) return -1.0f; // No overlap
            if (tEnter < 0.0f) return 0.0f; // Origin inside AABB
            return tEnter;
        }

        // Start-level heuristic based on footprint ~ dist * beamWidth
        static int ComputeStartLevel(float trueDist, float mapRes, int levels)
        {
            float footprint = trueDist * BEAM_WIDTH_RAD;
            int level = levels - 1;
            while (level > 0)
            {
                float side = (1 << (level * 2)) * mapRes;
                if (side <= footprint) break;
                level--;
            }
            return level;
        }

        static float SampleLevel0(ArrayView<float> data, ArrayView<LevelInfo> infos, int levels, int col, int row)
        {
            if (levels == 0) return 0;
            var info = infos[0];
            if (col < 0 || row < 0 || col >= info.Width || row >= info.Height) return 0;
            return data[info.Offset + row * info.Width + col];
        }

        public (double lat_deg, double lon_deg) GetObserverLatLon(int pixelX, int pixelY)
        {
            var primaryMap = BuildMapParams(ReferenceHorizonGenerator.LoadDEMs()[0]);
            var primaryProj = BuildProjectionParams(ReferenceHorizonGenerator.LoadDEMs()[0]);

            float centerPx = pixelX;
            float centerPy = pixelY;

            var (obsX_prim, obsY_prim) = primaryMap.PixelToCRS(centerPx, centerPy);
            var (obsLat_rad, obsLon_rad) = InverseProject(obsX_prim, obsY_prim, primaryProj);

            return (obsLat_rad * 180.0 / Math.PI, obsLon_rad * 180.0 / Math.PI);
        }

        static int PowInt(int baseVal, int exp)
        {
            int result = 1;
            for (int i = 0; i < exp; i++) result *= baseVal;
            return result;
        }

        static MapParams BuildMapParams(ElevationMap dem)
        {
            var srs = dem.SrsDescriptor;
            var geo = dem.GeoTransform;

            var colStepX = (float)geo[1];
            var rowStepX = (float)geo[2];
            var colStepY = (float)geo[4];
            var rowStepY = (float)geo[5];
            var det = colStepX * rowStepY - rowStepX * colStepY;
            var invDet = 1f / det;

            return new MapParams(
                (float)srs.R,
                (float)srs.k0,
                (float)srs.FalseEasting,
                (float)srs.FalseNorthing,
                invDet,
                (float)geo[0],
                (float)geo[1],
                (float)geo[2],
                (float)geo[3],
                (float)geo[4],
                (float)geo[5]);
        }

        internal static ProjectionParamsDouble BuildProjectionParamsDouble(ElevationMap dem)
        {
            var srs = dem.SrsDescriptor;
            return new ProjectionParamsDouble
            {
                R = srs.R,
                Lat0 = srs.lat0,
                Lon0 = srs.lon0,
                K0 = srs.k0,
                FalseEasting = srs.FalseEasting,
                FalseNorthing = srs.FalseNorthing
            };
        }

        static float[] FlattenElevation(float[,]? elevation)
        {
            if (elevation == null)
                throw new ArgumentNullException(nameof(elevation));
            int rows = elevation.GetLength(0);
            int cols = elevation.GetLength(1);
            long totalElements = (long)rows * cols; // Use long for total elements

            // Allocate single-dimensional array
            // totalElements will be < int.MaxValue (~1.75B elements < 2.147B elements)
            var flat = new float[(int)totalElements];

            // Copy data using Span<byte> in chunks to avoid int overflow in byte count
            int floatSize = sizeof(float); // 4 bytes
            long totalBytes = totalElements * floatSize; // This can exceed Int32.MaxValue (~7GB)

            unsafe
            {
                fixed (float* pElevation = elevation)
                fixed (float* pFlat = flat)
                {
                    byte* bElevation = (byte*)pElevation;
                    byte* bFlat = (byte*)pFlat;

                    long currentByteOffset = 0;
                    // Chunk size for Span<byte> and efficient copying (e.g., 1GB)
                    // Max size of Span<byte> is int.MaxValue, so this needs to be <= int.MaxValue.
                    int maxChunkSize = 1024 * 1024 * 1024; // 1GB

                    while (currentByteOffset < totalBytes)
                    {
                        long remainingBytes = totalBytes - currentByteOffset;
                        int copyCount = (int)Math.Min(remainingBytes, maxChunkSize);

                        Span<byte> sourceSpan = new Span<byte>(bElevation + currentByteOffset, copyCount);
                        Span<byte> destinationSpan = new Span<byte>(bFlat + currentByteOffset, copyCount);

                        sourceSpan.CopyTo(destinationSpan);
                        currentByteOffset += copyCount;
                    }
                }
            }
            return flat;
        }

        public static void WriteStopwatchTime(string msg, TimeSpan ts)
        {
            if (!Log.IsEnabled(Serilog.Events.LogEventLevel.Information))
                return;

            if (ts.TotalMinutes > 1.0)
            {
                Log.Information($"{msg}: {ts.TotalMinutes:F2} min");
            }
            else
            {
                Log.Information($"{msg}: {ts.TotalSeconds:F2} sec");
            }
        }
    }
}
