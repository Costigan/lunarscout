using GeoJSON.Net.Feature;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Algorithms.ScanReduceOperations;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using Newtonsoft.Json;
using OSGeo.GDAL; // For GDAL functionalities
using OSGeo.OSR;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

// =====================================================================================
// Use a flood algorithm to generate candidate rover traverses.
// This is based on the LOLA 20m DEM, which is 30400 by 30400.
// A subset of that DEM, called the Grid, will hold the data for this program.
// Both dimensions of Grid will be a multiple of the 128.
// already correctly implement auto-grouping.
// 1. They are loaded using _accelerator.LoadAutoGroupedStreamKernel.
// 2. Their kernel signatures in KernelImplementations.cs start with Index1D.
// 3. They are launched with a 1D extent, e.g., (Index1D)_numTotalCells.
// This setup allows ILGPU to manage thread grouping, which is ideal for CPU execution
// and general-purpose kernels not requiring explicit shared memory synchronization.
// =====================================================================================
namespace GridRunner
{
    public class MoonRoverPathfinder : IDisposable
    {
        // Set to true to enable debugging checks, messages and images
        public const bool DEBUGGING_NOW = false;
        public const bool WRITE_STEP_IMAGES = false; // Write progress images to disk

        // The slope data in a GeoTIFF format 
        public const string LOLA_TIF = @"D:\viper\maps\lola\ldem_80s_20m.tif";
        public const string SLOPE_TIF = @"D:\datasets\tiger_20m_repositioning_36_37_12hr\other\slope.gdal.compressed.tif";
        //public const string SLOPE_TIF = @"D:\viper\maps\lola\LDEM_80S_20M-2017-06-15.slope.tif";

        public const int DEM_WIDTH = 30400;
        public const int DEM_HEIGHT = 30400;

        public static Rectangle LOLA_20M_BOUNDS = new Rectangle(0, 0, 30400, 30400);

        public const int HALO_SIZE = 1;

        // the grid region must be a multiple of this, and it's origin in the LOLA product
        // must be a multiple of this too.  This simplifies loading horizons if that's needed.
        public const int PATCH_CONSTRAINT = 128;

        // Kernel0 is auto-grouped. Its delegate signature matches the kernel method minus the Index1D.
        Action<Index2D,                                 // index of the grid cell
            ArrayView2D<StateTuple, Stride2D.DenseX>,   // Current state
            Index2D                                     // Start location
            > _kernel0_initialize;

        // Kernel1 is explicitly grouped. Its delegate signature starts with KernelConfig.
        Action<KernelConfig,                                // Launch configuration (GridDim, GroupDim)
                SpecializedValue<Index2D>,
                ArrayView1D<Point2, Stride1D.Dense>,         // gpu_isActiveBlock
                int,                                         // active_group_count
                ArrayView1D<byte, Stride1D.Dense>,           // gpu_isBlockChanged
                ArrayView2D<StateTuple, Stride2D.DenseX>,    // gpu_state,
                ArrayView2D<byte, Stride2D.DenseX>,         // gpu_sun,
                ArrayView2D<float, Stride2D.DenseX>,         // gpu_slope,
                ArrayView2D<int, Stride2D.DenseX>,           // gpu_debug,
                float> _kernel1_driving;

        // Kernel2 is auto-grouped. Its delegate signature matches the kernel method minus the Index1D.
        Action<KernelConfig,
                SpecializedValue<Index2D>,
                ArrayView1D<Point2, Stride1D.Dense>,        // gpu_isActiveBlock
                int,                                        // active_group_count
                ArrayView2D<StateTuple, Stride2D.DenseX>,   // gpu_state,
                ArrayView2D<byte, Stride2D.DenseX>,         // gpu_sun,
                float,                                      // epochTimeAtWindowEnd
                byte> _kernel2_charging;                    // fuelCellNegativeIsIllegal

        Rectangle _bounds;          // region within the LOLA 20m DEM
        Rectangle _bounds_w_halo;

        Point _start;               // traverse start location

        //int _numTotalCells;         // _bounds.Width * _bounds.Height
        //int _blockSize, _numBlocksX, _numBlocksY, _numTotalBlocks;

        public delegate void SlopeDataGetter(byte[,] buffer, Rectangle bounds_in_buffer, Rectangle grid_bounds_in_20m);

        public delegate void SunDataGetter(byte[,] buffer, Rectangle bounds_in_buffer, Rectangle grid_bounds_in_20m, int level);
        int _sunStep = 3;       // The normal time step is 2 hours.  A sun step of 3 means 3 steps or 6 hours.

        #region Main Entry

        public StateTuple[,] RunSimulation(
            Rectangle bounds_in_lola,                           // The bounds of the grid within the LOLA 20m DEM
            Point start_within_grid,                         // The rover's start position in the 20m DEM (must be inside bounds but defined relative to the lola product)
            Point goal_within_grid,
            bool preferCPU = false,                     // Prefer a CPU accelerator if true
            SlopeDataGetter slopeDataGetter = null,   // Fetcher for slope data 
            SunDataGetter sunDataGetter = null,       // Fetcher for sun data
            float timeStepHours = 2.0f,                 // The time step in hours
            int maxSteps = 400,                         // The maximum number of steps to run.  ~6 months
            int sunStep = 6,
            bool negativeFuelCellIsIllegal = false)
        {
            //Debug.Assert(bounds.Left > 0 & bounds.Top > 0 & bounds.Bottom < LOLA_20M_BOUNDS.Bottom & bounds.Right < LOLA_20M_BOUNDS.Right,
            //    "Grid must be within LOLA DEM bounds (0,0) to (30400,30400) to allow a 1 cell halo");
            Debug.Assert(bounds_in_lola.Width % PATCH_CONSTRAINT == 0,
                "Grid X dimension must be a multiple of BLOCK_CONSTRAINT");
            Debug.Assert(bounds_in_lola.Height % PATCH_CONSTRAINT == 0,
                "Grid Y dimension must be a multiple of BLOCK_CONSTRAINT");

            var grid_size = new Index2D(bounds_in_lola.Width, bounds_in_lola.Height);
            var numTotalCells = bounds_in_lola.Width * bounds_in_lola.Height;
            var bounds_in_grid = new Rectangle(0, 0, bounds_in_lola.Width, bounds_in_lola.Height);

            Debug.Assert(bounds_in_grid.Contains(start_within_grid),
                $"[start {start_within_grid.X},{start_within_grid.Y} must be within the bounds of the simulation: {bounds_in_grid}");

            Index2D group_size = new Index2D(0, 0); // The group size will be calculated later
            Index2D group_array_size = new Index2D(0, 0); // The group array size will be calculated later

            byte[,] byte_buffer;

            // debugging
            var unoccupied_state = StateTuple.Unoccupied();

            using (var context = Context.Create(builder => builder.Default().EnableAlgorithms()))
            using (var accelerator = GetAccelerator(context, preferCPU))
            {
                if (accelerator == null)
                {
                    Console.WriteLine("Error: ILGPU couldn't create a CUDA accelerator. Exiting.");
                    return null;
                }

                LoadKernels(accelerator);

                group_size = CalculateOptimalGroupSize(accelerator, Interop.SizeOf<StateTuple>());

                Debug.Assert(bounds_in_lola.X % group_size.X == 0 && bounds_in_lola.Y % group_size.Y == 0,
                    $"Grid dimensions must be multiples of the group size: {group_size}.");

                group_array_size = new Index2D(bounds_in_lola.Width / group_size.X, bounds_in_lola.Height / group_size.Y);
                var group_count = group_array_size.X * group_array_size.Y;
                var group_bounds = new Rectangle(0, 0, group_array_size.X, group_array_size.Y);

                byte_buffer = new byte[group_size.X, group_size.Y];

                Console.WriteLine($"Running on:                           {accelerator.Name}");
                Console.WriteLine($"Accelerator Type:                     {accelerator.AcceleratorType}");
                Console.WriteLine($"Selected group size:                  {group_size}");
                Console.WriteLine($"Max Threads Per Group:                {accelerator.MaxNumThreadsPerGroup}");
                Console.WriteLine($"Max Shared Memory Per Group (Static): {accelerator.MaxSharedMemoryPerGroup} bytes");
                Console.WriteLine($"Warp Size:                            {accelerator.WarpSize}");
                Console.WriteLine($"Max Grid Size:                        {accelerator.MaxGridSize}");
                Console.WriteLine($"State Array Size:                     {Interop.SizeOf<StateTuple>() * bounds_in_lola.Width * bounds_in_lola.Height} bytes");
                Console.WriteLine();
                Console.WriteLine($"Grid:                                 {bounds_in_lola.Width}x{bounds_in_lola.Height} (Cells: {bounds_in_lola.Width * bounds_in_lola.Height})");
                Console.WriteLine($"Group Size:                           {group_size.X}x{group_size.Y}");
                Console.WriteLine($"Num Groups:                           {group_array_size.X}x{group_array_size.X} (Total: {group_count})");

                //var (cpu_slope_float, slope_projection, slope_transform) = ReadRectangleFromFloatGeotiff2(bounds, SLOPE_TIF);
                var (cpu_slope_float, slope_projection, slope_transform) = ReadRectangleFromFloatGeotiff2(new Rectangle(0, 0, bounds_in_lola.Width, bounds_in_lola.Height), SLOPE_TIF);
                //var cpu_slope = ToByteArray(cpu_slope_float, s => (byte)Math.Min(255f, Math.Max(0f, s * 10f)));  // Slope in 10ths of a degree
                var cpu_slope = cpu_slope_float;

                //WriteGeoTiff2(cpu_slope_float, "slope_float_test.tif", slope_projection, slope_transform);
                //WriteGeoTiff2(cpu_slope, "slope_byte_test.tif", slope_projection, slope_transform);

                //FillSlopeArrayWithTestData(cpu_slope);

                var cpu_state = new StateTuple[grid_size.X, grid_size.Y];
                var cpu_sun = new byte[grid_size.X, grid_size.Y];
                var cpu_debug = new int[grid_size.X, grid_size.Y];

                // Allocate DenseY, which means we want to iterate over the X dimension most quickly in the kernels.
                using (var gpu_state = accelerator.Allocate2DDenseX<StateTuple>(cpu_state))
                using (var gpu_grid_captured_initial_states = accelerator.Allocate2DDenseX<StateTuple>(cpu_state))
                using (var gpu_slope = accelerator.Allocate2DDenseX<float>(cpu_slope))
                using (var gpu_sun = accelerator.Allocate2DDenseX<byte>(cpu_sun))
                using (var gpu_debug = accelerator.Allocate2DDenseX<int>(cpu_debug))
                using (var gpu_debug_result = accelerator.Allocate1D<int>(1))
                using (var gpu_active_groups = accelerator.Allocate1D<Point2>(group_count))
                using (var gpu_group_modified = accelerator.Allocate1D<byte>(group_count))
                    try
                    {
                        var stream = (AcceleratorStream)accelerator.DefaultStream;

                        var sunlightCache = new Dictionary<(int blockIndex, int layer), byte[]>();
                        // Debugging
                        gpu_sun.MemSet(stream, 255, 0, grid_size.X * grid_size.Y);

                        gpu_slope.CopyFromCPU(stream, cpu_slope);

                        // Initialize the starting location
                        //var (grid_startX, grid_startY) = ((global_start.X - bounds.Left), (global_start.Y - bounds.Top));
                        var (grid_startX, grid_startY) = ((start_within_grid.X), (start_within_grid.Y));
                        var (startGroupX, startGroupY) = (grid_startX / group_size.X, grid_startY / group_size.Y);
                        var grid_start = new Index2D(grid_startX, grid_startY);

                        InitializeGPUState(stream, grid_start, gpu_state, cpu_state);

                        if (true)
                            CheckInitialGPUState(stream, grid_start, gpu_state, cpu_state);

                        if (DEBUGGING_NOW)
                            DebugCountMatchingStateTuples(accelerator, gpu_state, s => s.Equals(StateTuple.StartState()), 1);

                        Array.Clear(cpu_debug, 0, cpu_debug.Length);
                        gpu_debug.MemSet(stream, 0, 0, grid_size.X * grid_size.Y);

                        // Set up the active groups
                        var active_groups = new ActiveGroupManager(group_bounds, gpu_active_groups, gpu_group_modified).WithDebugging();
                        active_groups.Add(new Point2((short)startGroupX, (short)startGroupY));
                        Debug.Assert(active_groups.Count == 1);

                        // Time steps
                        for (int zStep = 0; zStep < maxSteps; ++zStep)
                        {
                            if (active_groups.Count < 1)
                                break;

                            if (!StateAtGoal(accelerator, gpu_state, goal_within_grid).Equals(unoccupied_state))
                            {
                                Console.WriteLine($"Goal reached at step {zStep} with {active_groups.Count} active groups.");
                                break;
                            }

                            var currentTime = RoverConstants.InitialTime + zStep * RoverConstants.TimeStepDurationHours;
                            var stepEndTime = RoverConstants.InitialTime + (zStep + 1) * RoverConstants.TimeStepDurationHours;
                            active_groups.ClearStepGroups();

                            Console.WriteLine($"Starting Z-step {zStep}/{maxSteps}. Window: [{currentTime}h to {stepEndTime}h]");
                            Console.Write($"  active_blocks={active_groups.Count}");

                            var zStepTimer = Stopwatch.StartNew();

                            if (false)
                                // Update sun blocks if needed
                                if (zStep % _sunStep == 0)
                                {
                                    foreach (var group_pt in active_groups.Enumerate())
                                    {
                                        var group_bounds_in_grid = GroupToGridBounds(ToPoint(group_pt));
                                        // Using byte_buffer temporarily
                                        sunDataGetter(byte_buffer, new Rectangle(0, 0, group_size.X, group_size.Y), bounds_in_lola, 0);
                                        gpu_sun.View
                                            .SubView(new Index2D(group_bounds_in_grid.X, group_bounds_in_grid.Y), group_size)
                                            .CopyFromCPU(stream, byte_buffer);
                                    }
                                    accelerator.Synchronize();
                                }

                            int iteration_counter_within_step = 0;

                            do
                            {
                                if (active_groups.Count < 1)
                                    break;

                                active_groups.SendGroupsToGPU(stream);

                                KernelConfig k1LaunchConfig = new KernelConfig(new Index2D(active_groups.Count, 1), group_size);
                                _kernel1_driving(k1LaunchConfig,
                                    new SpecializedValue<Index2D>(group_size),
                                    gpu_active_groups.View,
                                    active_groups.Count,
                                    gpu_group_modified.View,
                                    gpu_state.View,
                                    gpu_sun.View,
                                    gpu_slope.View,
                                    gpu_debug.View,
                                    stepEndTime);

                                active_groups.UpdateActiveGroupsFromGPU(stream, true);

                                if (false)       // DEBUGGING_NOW
                                {
                                    var finalStates = CopyStateTuplesToCPU(accelerator, gpu_state);
                                    var earliest_arrival = finalStates.Where(s => !s.Equals(unoccupied_state)).Max(s => s.ArrivalTime);
                                    var occupied_count = finalStates.Count(s => !s.IsUnoccupied());
                                    Console.WriteLine($"    occupied={occupied_count} latest_active={earliest_arrival} stepEnd={stepEndTime}");
                                }

                                //var active_indices = IndexesOfMatchingCells(CopyStateTuplesToCPU(accelerator, gpu_state), s => !s.Equals(unoccupied_state)).ToList();

                                // Use explicitly grouped kernel for GPU
                                //Console.WriteLine($"Finishing Iteration {iteration_counter_within_step + 1} zStep {zStep} ");  //active_count={active_indices.Count}

                                iteration_counter_within_step++;
                                currentTime = (zStep + 1) * RoverConstants.TimeStepDurationHours;

                            } while (active_groups.Count > 0 && iteration_counter_within_step < RoverConstants.MaxK2Iterations);

                            if (iteration_counter_within_step >= RoverConstants.MaxK2Iterations)
                                Console.WriteLine($"  Warning: Kernel1_Driving reached max iterations ({RoverConstants.MaxK2Iterations})");
                            Console.WriteLine($"  Kernel1_Driving completed in {iteration_counter_within_step} iterations.");

                            if (WRITE_STEP_IMAGES)
                            {
                                gpu_state.CopyToCPU(stream, cpu_state);
                                var rect = new Rectangle(0, 0, cpu_state.GetLength(0), cpu_state.GetLength(1));
                                var arrival_min = EnumerateTuples(cpu_state).Min(s => s.ArrivalTime);
                                var arrival_max = EnumerateTuples(cpu_state).Where(s => s.ArrivalTime != float.MaxValue).Max(s => s.ArrivalTime);
                                var delta = arrival_max - arrival_min;

                                Console.WriteLine($"  Writing GeoTIFF for Z-step {zStep} with arrival time range: " +
                                    $"{arrival_min} to {arrival_max} (delta={delta})");

                                Func<StateTuple, Color> color_mapper = state =>
                                {
                                    if (state.IsUnoccupied()) return Color.Transparent;
                                    if (state.ArrivalTime < arrival_min || state.ArrivalTime > arrival_max)
                                        return Color.FromArgb(255, 128, 128, 128); // Gray for out of bounds
                                    if (delta == 0) return Color.FromArgb(255, 32, 32, 32);

                                    int red = (int)(255 * (state.ArrivalTime - arrival_min) / (arrival_max - arrival_min));
                                    int blue = 255 - red;
                                    return Color.FromArgb(255, red, 100, blue); // Blue gradient based on arrival time
                                };

                                WriteStateTuplesToGeoTiff(rect, cpu_state, slope_projection, slope_transform, color_mapper, $"step_{zStep:D3}.tif");
                            }

                            if (false)  // debugging
                            {
                                // 2. Allocate a 1D result buffer of length 1

                                var flatView = gpu_debug.View.AsContiguous();
                                accelerator.Reduce<int, AddInt32>(stream, flatView, gpu_debug_result.View);

                                accelerator.Synchronize();

                                int debug_sum = gpu_debug_result.GetAsArray1D()[0];
                                if (debug_sum > 0)
                                {

                                }
                            }

                            active_groups.MoveStepGroupsToActiveGroups();
                            active_groups.SendGroupsToGPU(stream);

                            KernelConfig k2LaunchConfig = new KernelConfig(new Index2D(active_groups.Count, 1), group_size);
                            _kernel2_charging(k2LaunchConfig,
                                    new SpecializedValue<Index2D>(group_size),
                                    gpu_active_groups.View,
                                    active_groups.Count,
                                    gpu_state.View,
                                    gpu_sun.View,
                                    stepEndTime,
                                    0);

                            accelerator.Synchronize();

                            if (false)
                            {
                                gpu_debug.CopyToCPU(stream, cpu_debug);
                                WriteGeoTiff2(cpu_debug, $"debug_{zStep:D3}.tif", slope_projection, slope_transform);
                            }

                            zStepTimer.Stop();
                            Console.WriteLine();
                            Console.WriteLine($"  Z-step {zStep} completed in {zStepTimer.ElapsedMilliseconds} ms.");
                        }

                        gpu_state.CopyToCPU(stream, cpu_state);
                        return cpu_state;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Error during simulation: {ex.Message}");
                        Console.WriteLine(ex.StackTrace);
                        return null;
                    }
            }

            Point GlobalToGrid(Point global_pt) => new Point(global_pt.X - bounds_in_lola.Left, global_pt.Y - bounds_in_lola.Top);
            Point GridToGlobal(Point grid_pt) => new Point(grid_pt.X + bounds_in_lola.Left, grid_pt.Y + bounds_in_lola.Top);
            Point GridToGroup(Point grid_pt) => new Point(grid_pt.X / group_size.X, grid_pt.Y / group_size.Y);
            int GroupToGroupIndex(Point p) => p.X + p.Y * group_array_size.X;
            int GridToGroupIndex(Point grid_pt) => GroupToGroupIndex(GridToGroup(grid_pt));

            Point2 ToPoint2(Point p) => new Point2((short)p.X, (short)p.Y);
            Point ToPoint(Point2 p) => new Point(p.X, p.Y);


            Rectangle GroupToGridBounds(Point grid_pt) => new Rectangle(grid_pt.X * group_size.X, grid_pt.Y * group_size.Y, group_size.X, group_size.Y);

            void InitializeGPUState(AcceleratorStream stream, Index2D grid_start, MemoryBuffer2D<StateTuple, Stride2D.DenseX> gpu_state, StateTuple[,] cpu_state)
            {
                if (!preferCPU)
                {
                    _kernel0_initialize(grid_size, gpu_state.View, grid_start);
                }
                else
                {
                    var batch_size = 32;
                    Debug.Assert(grid_size.Y % batch_size == 0, "Grid height must be a multiple of 128 for CPU initialization.");
                    var unoccupied = StateTuple.Unoccupied();
                    var options = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };
                    Parallel.For(0, grid_size.Y / batch_size,
                        (i) =>
                            {
                                var start_row = i * batch_size;
                                var stop_row = start_row + batch_size;
                                for (var row = start_row; row < stop_row; row++)
                                    for (var col = 0; col < grid_size.X; col++)
                                        cpu_state[col, row] = unoccupied;

                            });

                    // Initialize the start cell
                    cpu_state[grid_start.X, grid_start.Y] = StateTuple.StartState();

                    gpu_state.View.CopyFromCPU(stream, cpu_state);
                }
            }

            void CheckInitialGPUState(AcceleratorStream stream, Index2D grid_start, MemoryBuffer2D<StateTuple, Stride2D.DenseX> gpu_state, StateTuple[,] cpu_state)
            {
                var batch_size = 32;
                Debug.Assert(grid_size.Y % batch_size == 0, "Grid height must be a multiple of 128 for CPU initialization.");
                var unoccupied = StateTuple.Unoccupied();
                var start_state = StateTuple.StartState();

                gpu_state.CopyToCPU(stream, cpu_state);

                var options = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };
                Parallel.For(0, grid_size.Y / batch_size,
                    (i) =>
                    {
                        var start_row = i * batch_size;
                        var stop_row = start_row + batch_size;
                        for (var row = start_row; row < stop_row; row++)
                            for (var col = 0; col < grid_size.X; col++)
                            {
                                var v = cpu_state[col, row];
                                if (!v.IsUnoccupied())
                                    if (!(row == grid_start.X && col == grid_start.Y && v.Equals(start_state)))
                                    {
                                        var c1 = v.ArrivalTime == start_state.ArrivalTime;
                                        var c2 = v.TotalWh == start_state.TotalWh;
                                        var c3 = v.BatteryWh == start_state.BatteryWh;
                                        var c4 = v.MinFuelCellWhScaledBy20 == start_state.MinFuelCellWhScaledBy20;
                                        var c5 = v.PreviousCell == start_state.PreviousCell;
                                        var c6 = v.Padding == start_state.Padding;
                                        if (c1 && c2 && c3 && c4 && c5 && c6)
                                            continue; // This is the start state
                                        throw new InvalidOperationException($"Initial state at ({col},{row}) is not unoccupied or start state: {v}");
                                    }

                            }
                    });
            }
        }

        void LoadKernels(Accelerator accelerator)
        {
            _kernel0_initialize = accelerator.LoadAutoGroupedStreamKernel<
                Index2D,                                    // index of the grid cell
                ArrayView2D<StateTuple, Stride2D.DenseX>,   // Current state
                Index2D                                     // Start location
                >(KernelImplementations.Kernel0_initialize);

            _kernel1_driving = accelerator.LoadStreamKernel<
                SpecializedValue<Index2D>,
                ArrayView1D<Point2, Stride1D.Dense>,        // gpu_isActiveBlock
                int,                                        // active_group_count
                ArrayView1D<byte, Stride1D.Dense>,          // gpu_isBlockChanged
                ArrayView2D<StateTuple, Stride2D.DenseX>,   // gpu_state
                ArrayView2D<byte, Stride2D.DenseX>,         // gpu_sun,
                ArrayView2D<float, Stride2D.DenseX>,        // gpu_slope
                ArrayView2D<int, Stride2D.DenseX>,         // gpu_debug
                float
                >(KernelImplementations.Kernel1_Driving);

            _kernel2_charging = accelerator.LoadStreamKernel<
                SpecializedValue<Index2D>,
                ArrayView1D<Point2, Stride1D.Dense>,        // gpu_isActiveBlock
                int,                                        // active_group_count
                ArrayView2D<StateTuple, Stride2D.DenseX>,   // gpu_currentState,
                ArrayView2D<byte, Stride2D.DenseX>,         // gpu_sun,
                float,
                byte
                >(KernelImplementations.Kernel2_CHARGING);
        }

        public class ActiveGroupManager
        {
            bool _debugging = true; // Debugging flag

            Rectangle bounds;
            MemoryBuffer1D<Point2, Stride1D.Dense> gpu_groups;
            MemoryBuffer1D<byte, Stride1D.Dense> gpu_modified;

            int cpu_modified_fill_ptr = 0; // Pointer for filling cpu_modified array

            HashSet<Point2> active_groups = new HashSet<Point2>();      // Active groups for the current iteration
            HashSet<Point2> step_groups = new HashSet<Point2>();        // Groups that were active during the current step (union of all active groups during step)

            Point2[] cpu_groups;
            byte[] cpu_modified;

            public ActiveGroupManager(Rectangle bounds, MemoryBuffer1D<Point2, Stride1D.Dense> gpu_groups, MemoryBuffer1D<byte, Stride1D.Dense> gpu_modified)
            {
                this.bounds = bounds;
                this.gpu_groups = gpu_groups;
                this.gpu_modified = gpu_modified;

                cpu_groups = new Point2[gpu_groups.Length];
                cpu_modified = new byte[gpu_modified.Length];
            }

            public ActiveGroupManager WithDebugging(bool debugging = true)
            {
                _debugging = debugging;
                return this;
            }

            public void Clear()
            {
                active_groups.Clear();
                step_groups.Clear();
                Array.Clear(cpu_groups, 0, cpu_groups.Length);
                Array.Clear(cpu_modified, 0, cpu_modified.Length);

                if (_debugging)
                    Console.WriteLine("ACTIVE_GROUPS: clear");
            }

            public void Add(Point2 group)
            {
                if (active_groups.Count >= gpu_groups.Length)
                    throw new InvalidOperationException("Active group pointer exceeded the length of the array.");
                active_groups.Add(group);

                if (_debugging)
                    Console.WriteLine($"ACTIVE_GROUPS: add {group} (count={active_groups.Count})");
            }

            public void RemoveActiveGroup(Point2 group)
            {
                active_groups.Remove(group);
                if (_debugging)
                    Console.WriteLine($"ACTIVE_GROUPS: remove {group} (count={active_groups.Count})");
            }

            public int Count => active_groups.Count;

            public IEnumerable<Point2> Enumerate() => active_groups;

            public void ClearStepGroups()
            {
                step_groups.Clear();
                if (_debugging)
                    Console.WriteLine($"ACTIVE_GROUPS: start step (count={active_groups.Count})");
            }

            public void SendGroupsToGPU(AcceleratorStream stream)
            {
                Debug.Assert(Count > 0);
                cpu_modified_fill_ptr = 0;
                foreach (var group in active_groups)
                    cpu_groups[cpu_modified_fill_ptr++] = group;
                Debug.Assert(cpu_modified_fill_ptr == Count);

                gpu_groups.View.CopyFromCPU(stream, new ReadOnlySpan<Point2>(cpu_groups, 0, Count));
                gpu_modified.MemSetToZero(stream);

                step_groups.UnionWith(active_groups);

                var show_all_groups = false;
                if (_debugging & show_all_groups)
                {
                    Console.WriteLine($"ACTIVE_GROUPS: send to GPU {Count} groups");
                    Console.WriteLine($"  Groups: {string.Join(", ", active_groups.Select(p => $"({p.X},{p.Y})"))}");
                }
                else if (_debugging)
                {
                    //Console.WriteLine($"ACTIVE_GROUPS: send to GPU {Count} groups. First 10: {string.Join(", ", active_groups.Take(10).Select(p => $"({p.X},{p.Y})"))}");
                    //Console.WriteLine($"ACTIVE_GROUPS: {Count}");
                    Console.Write($" {Count}");
                }
            }

            public void UpdateActiveGroupsFromGPU(AcceleratorStream stream, bool add_neighbors = true)
            {
                gpu_modified.View.CopyToCPU(stream, new Span<byte>(cpu_modified, 0, cpu_modified_fill_ptr));
                stream.Synchronize();

                var raw_count = 0;
                for (var i = 0; i < cpu_modified_fill_ptr; i++)
                    if (cpu_modified[i] > 0)
                        raw_count++;

                // Assume cpu_groups is up-to-date and cpu_modified_fill_ptr is the count of active groups
                Debug.Assert(cpu_modified_fill_ptr == active_groups.Count);
                active_groups.Clear();
                for (var i = 0; i < cpu_modified_fill_ptr; i++)
                    if (cpu_modified[i] > 0)
                    {
                        var group = cpu_groups[i];
                        active_groups.Add(group);
                        if (add_neighbors)
                            foreach (var neighbor in EnumerateSurroundingBlockPoints(group, bounds))
                                active_groups.Add(neighbor);
                    }

                //if (_debugging)
                //    Console.WriteLine($"ACTIVE_GROUPS: update from GPU.  raw count={raw_count} after neighbors added {active_groups.Count}");
            }

            public void MoveStepGroupsToActiveGroups()
            {
                (active_groups, step_groups) = (step_groups, active_groups); // Swap references
                step_groups.Clear();
                if (_debugging)
                    Console.WriteLine($"ACTIVE_GROUPS: move step groups to active groups.  Total active groups now {active_groups.Count}");
            }
        }

        #endregion Main Entry

        #region Kernels



        #endregion Kernels

        #region Problem Utilities

        public static List<Point> GetRouteBackwardsAsGridPoints(StateTuple[,] states, Point start, Point goal) => EnumeratePath(states, start, goal).ToList();

        public static List<Point> GetRouteForwardsAsGridPoints(StateTuple[,] states, Point start, Point goal)
        {
            var list = GetRouteBackwardsAsGridPoints(states, start, goal);
            list.Reverse();
            return list;
        }

        public static List<Point> GetRouteBackwardsCarefully(StateTuple[,] states, Point start, Point goal)
        {
            var result = new List<Point>();
            var hash = new HashSet<Point>();
            foreach (var pt in EnumeratePath(states, start, goal))
            {
                if (hash.Contains(pt))
                {
                    Console.WriteLine($"Duplicate point encountered: {pt}. Stopping path enumeration.");
                    break;
                }
                result.Add(pt);
                hash.Add(pt);
            }
            return result;
        }

        public static IEnumerable<Point> EnumeratePath(StateTuple[,] states, Point start, Point goal)
        {
            Debug.Assert(states != null);
            var (width, height) = (states.GetLength(0), states.GetLength(1));

            var x = goal.X;
            var y = goal.Y;

            if (x < 0 || x >= width || y < 0 || y >= height)
            {
                Console.WriteLine($"Target global coordinates ({x},{y}) are outside the grid ({width}x{height}).");
                yield break;
            }

            var unoccupied = StateTuple.Unoccupied();
            var delta_array = new Point[]
            {
                new Point(int.MaxValue, int.MaxValue),
                new Point(1,1),     // 1
                new Point(0,1),     // 2
                new Point(-1,1),    // 3
                new Point(1,0),     // 4
                new Point(int.MaxValue, int.MaxValue),
                new Point(-1,0),    // 6
                new Point(1,-1),    // 7
                new Point(0,-1),    // 8
                new Point(-1,-1)    // 9
            };

            while (true)
            {
                if (x < 0 || x >= width || y < 0 || y >= height)
                {
                    Console.WriteLine($"Reached out of bounds at ({x},{y}). Stopping path enumeration.");
                    yield break;
                }

                var state = states[x, y];

                if (state.Equals(unoccupied))
                    yield break;

                yield return new Point(x, y);

                var prevCellCode = state.PreviousCell;
                if (prevCellCode == -1)
                    yield break; // Reached the start cell

                var delta = delta_array[prevCellCode];

                x -= delta.X;
                y -= delta.Y;
            }
        }

        public static IEnumerable<(double lon, double lat)> EnumerateLonLat(IEnumerable<Point> route) => route.Select(p => RowCol2LonLatDeg(p));

        public static void WriteRouteGeoJson(string filePath, IEnumerable<(double Lon, double Lat)> route)
        {
            // 1) Build the LineString from your lat/lon pairs
            var line = new GeoJSON.Net.Geometry.LineString(
                route
                 .Select(p => new GeoJSON.Net.Geometry.Position(p.Lat, p.Lon))
                 .ToList()
            );

            // 2) Wrap it in a Feature
            var feat = new Feature(line);

            // 3) Create a FeatureCollection
            var fc = new FeatureCollection(new List<Feature> { feat });

            // 4) Serialize as pretty JSON
            string json = JsonConvert.SerializeObject(fc, Newtonsoft.Json.Formatting.Indented);

            // 5) Write to disk
            File.WriteAllText(filePath, json);
        }

        byte[] GenerateSunlightForBlock(float globalEpochTime, int blockFlatIdx, int versionId, bool isInnerActiveBlock)
        {
            int blockWidth = RoverConstants.BlockDimX;
            int blockHeight = RoverConstants.BlockDimY;
            byte[] blockSunlight = new byte[blockWidth * blockHeight];

            byte sunValue;
            if (isInnerActiveBlock)
            {
                sunValue = (byte)((versionId % 5) * 50 + 25);
            }
            else
            {
                sunValue = 0;
            }

            for (int i = 0; i < blockSunlight.Length; ++i)
            {
                blockSunlight[i] = sunValue;
            }
            return blockSunlight;
        }

        #endregion Problem Utilities

        #region CUDA Utilities

        /// <summary>
        /// Get an accelerator from the context.  ILGPU's GetPreferredDevices picks the device with the most
        /// memory, which can be the integrated graphics if you have a lot of memory. I'm changing this to
        /// use MaxNumThreads.
        /// </summary>
        /// <param name="context"></param>
        /// <param name="preferCPU"></param>
        /// <returns></returns>
        Accelerator GetAccelerator(Context context, bool preferCPU = false)
        {
            if (context == null)
                return null;
            try
            {
                return preferCPU
                    ? context.GetPreferredDevice(preferCPU).CreateAccelerator(context)
                    : context.Devices.OrderBy(d => -d.MaxNumThreads).First().CreateAccelerator(context);
                // Forcing a specific accelerator:
                // accelerator = new CudaAccelerator(context);
                // accelerator = new CPUAccelerator(context, 16); // CPU with 16 threads
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Could not create preferred GPU accelerator: {ex.Message}. Falling back to CPU.");
                return context.GetCPUDevice(0).CreateAccelerator(context);
            }
        }

        /// <summary>
        /// Calculates an "optimal" group dimension (TILE_COLS, TILE_ROWS)
        /// based on device constraints, prioritizing maximizing threads within shared memory limits.
        /// </summary>
        public static Index2D CalculateOptimalGroupSize(
            Accelerator accelerator,
            int elementSizeInBytes,
            int haloSize = 1)
        {
            // I want square groups of a multiple of 2
            var sides = new[] { 64, 32, 16, 8, 4 };

            var a = accelerator;
            var side = sides.FirstOrDefault(s => s <= a.MaxGroupSize.X & s <= a.MaxGroupSize.Y & s * s <= a.MaxNumThreadsPerGroup);
            return new Index2D(side, side);
        }

        // Static Utility Methods for Coordinate and Index Conversions (Revised)

        /// <summary>
        /// Calculates the flat index of the block that contains the given global grid coordinates.
        /// </summary>
        /// <param name="globalX">Global X coordinate of the cell.</param>
        /// <param name="globalY">Global Y coordinate of the cell.</param>
        /// <param name="blockDimX">Dimension of a block in X.</param>
        /// <param name="blockDimY">Dimension of a block in Y.</param>
        /// <param name="numBlocksXInGrid">Total number of blocks in the X dimension of the grid.</param>
        /// <returns>The flat index of the block.</returns>
        public static int GlobalCellToBlockIndex(int globalX, int globalY, int blockDimX, int blockDimY, int numBlocksXInGrid)
        {
            Debug.Assert(blockDimX > 0, "Block X dimension must be positive.");
            Debug.Assert(blockDimY > 0, "Block Y dimension must be positive.");
            Debug.Assert(numBlocksXInGrid > 0, "Number of blocks in X must be positive.");
            Debug.Assert(globalX >= 0, "Global X coordinate must be non-negative.");
            Debug.Assert(globalY >= 0, "Global Y coordinate must be non-negative.");

            int blockCoordX = globalX / blockDimX;
            int blockCoordY = globalY / blockDimY;
            return blockCoordY * numBlocksXInGrid + blockCoordX;
        }

        /// <summary>
        /// Calculates the block flat index and the linear offset within that block for given global grid coordinates.
        /// </summary>
        /// <param name="globalX">Global X coordinate of the cell.</param>
        /// <param name="globalY">Global Y coordinate of the cell.</param>
        /// <param name="blockDimX">Dimension of a block in X.</param>
        /// <param name="blockDimY">Dimension of a block in Y.</param>
        /// <param name="numBlocksXInGrid">Total number of blocks in the X dimension of the grid.</param>
        /// <returns>An unnamed tuple: (int blockFlatIndex, int offsetInBlock).</returns>
        public static (int blockIndex, int offsetInBlock) GetBlockIndexAndOffset(int globalX, int globalY, int blockDimX, int blockDimY, int numBlocksXInGrid)
        {
            Debug.Assert(blockDimX > 0, "Block X dimension must be positive.");
            Debug.Assert(blockDimY > 0, "Block Y dimension must be positive.");
            Debug.Assert(numBlocksXInGrid > 0, "Number of blocks in X must be positive.");
            Debug.Assert(globalX >= 0, "Global X coordinate must be non-negative.");
            Debug.Assert(globalY >= 0, "Global Y coordinate must be non-negative.");

            int blockCoordX = globalX / blockDimX;
            int blockCoordY = globalY / blockDimY;
            int blockFlatIndex = blockCoordY * numBlocksXInGrid + blockCoordX;

            int threadXInBlock = globalX % blockDimX;
            int threadYInBlock = globalY % blockDimY;
            int offsetInBlock = threadYInBlock * blockDimX + threadXInBlock;

            return (blockFlatIndex, offsetInBlock);
        }

        public struct GlobalCellPosition
        {
            public int X;
            public int Y;
            public GlobalCellPosition(int x, int y) { X = x; Y = y; }
            public (int x, int y) Destructure() => (X, Y);
        }

        public struct BlockIndex
        {
            public int Index;
            public BlockIndex(int index) => Index = index;
            public int Destructure() => Index;
        }

        public struct BlockIndexAndOffset
        {
            public BlockIndex BlockIndex;
            public int OffsetInBlock;
            public BlockIndexAndOffset(BlockIndex blockIndex, int offsetInBlock) => (BlockIndex, OffsetInBlock) = (blockIndex, offsetInBlock);
            public (int blockIndex, int offsetInBlock) Destructure() => (BlockIndex.Destructure(), OffsetInBlock);
        }

        /// <summary>
        /// Calculates the global grid coordinates (X, Y) from a block's flat index and a linear offset within that block.
        /// </summary>
        /// <param name="blockFlatIndex">The flat index of the block.</param>
        /// <param name="offsetInBlock">Linear offset within the block (0 to blockDimX*blockDimY - 1).</param>
        /// <param name="blockDimX">Dimension of a block in X.</param>
        /// <param name="blockDimY">Dimension of a block in Y.</param>
        /// <param name="numBlocksXInGrid">Total number of blocks in the X dimension of the grid.</param>
        /// <returns>An unnamed tuple: (int globalX, int globalY).</returns>
        public static (int globalX, int globalY) BlockOffsetToGlobalCell(int blockFlatIndex, int offsetInBlock, int blockDimX, int blockDimY, int numBlocksXInGrid)
        {
            Debug.Assert(blockDimX > 0, "Block X dimension must be positive.");
            Debug.Assert(blockDimY > 0, "Block Y dimension must be positive.");
            Debug.Assert(numBlocksXInGrid > 0, "Number of blocks in X must be positive.");
            Debug.Assert(blockFlatIndex >= 0, "Block flat index must be non-negative.");
            Debug.Assert(offsetInBlock >= 0 && offsetInBlock < (blockDimX * blockDimY), "Offset in block is out of valid range.");

            int blockCoordY = blockFlatIndex / numBlocksXInGrid;
            int blockCoordX = blockFlatIndex % numBlocksXInGrid;

            int blockOriginGlobalX = blockCoordX * blockDimX;
            int blockOriginGlobalY = blockCoordY * blockDimY;

            int threadYInBlock = offsetInBlock / blockDimX;
            int threadXInBlock = offsetInBlock % blockDimX;

            int globalX = blockOriginGlobalX + threadXInBlock;
            int globalY = blockOriginGlobalY + threadYInBlock;

            return (globalX, globalY);
        }

        /// <summary>
        /// Calculates the global grid coordinates (X, Y) of the origin (top-left cell) of a specified block.
        /// </summary>
        /// <param name="blockFlatIndex">The flat index of the block.</param>
        /// <param name="blockDimX">Dimension of a block in X.</param>
        /// <param name="blockDimY">Dimension of a block in Y.</param>
        /// <param name="numBlocksXInGrid">Total number of blocks in the X dimension of the grid.</param>
        /// <returns>An unnamed tuple: (int globalX, int globalY).</returns>
        public static (int globalX, int globalY) BlockOriginToGlobalCell(int blockFlatIndex, int blockDimX, int blockDimY, int numBlocksXInGrid)
        {
            Debug.Assert(blockDimX > 0, "Block X dimension must be positive.");
            Debug.Assert(blockDimY > 0, "Block Y dimension must be positive.");
            Debug.Assert(numBlocksXInGrid > 0, "Number of blocks in X must be positive.");
            Debug.Assert(blockFlatIndex >= 0, "Block flat index must be non-negative.");

            int blockCoordY = blockFlatIndex / numBlocksXInGrid;
            int blockCoordX = blockFlatIndex % numBlocksXInGrid;

            int globalX = blockCoordX * blockDimX;
            int globalY = blockCoordY * blockDimY;

            return (globalX, globalY);
        }

        public static bool IsHaloBlock(int blockIndex, int numBlocksX, int numBlocksY)
        {
            int blockX = blockIndex % numBlocksX;
            int blockY = blockIndex / numBlocksX;
            return (blockX == 0 | blockX == numBlocksX - 1 | blockY == 0 | blockY == numBlocksY - 1);
        }

        public static IEnumerable<int> EnumerateSurroundingBlockIds(int blockId, int numBlocksX, int numBlocksY)
        {
            var (blockX, blockY) = (blockId % numBlocksX, blockId / numBlocksX);
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                {
                    var (nbx, nby) = (blockX + dx, blockY + dy);
                    if (nbx >= 0 & nbx < numBlocksX & nby >= 0 & nby < numBlocksY)
                        yield return nby * numBlocksX + nbx;
                }
        }

        public static IEnumerable<Point2> EnumerateSurroundingBlockPoints(Point2 center, Rectangle bounds)
        {
            var (cx, cy) = (center.X, center.Y);
            var (left, top, right, bottom) = (bounds.Left, bounds.Top, bounds.Right, bounds.Bottom);
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                {
                    var (x2, y1) = (cx + dx, cy + dy);
                    if (x2 >= left & x2 < right & y1 >= top & y1 < bottom)
                        yield return new Point2((short)x2, (short)y1);
                }
        }

        public static int CountOccupied(StateTuple[] states) => states.Count(s => !s.IsUnoccupied());
        public static int CountUnOccupied(StateTuple[] states) => states.Count(s => s.IsUnoccupied());
        public static int FirstOccupiedAslinearGlobalCellIdx(StateTuple[] states) => Array.FindIndex(states, s => !s.IsUnoccupied());

        public IEnumerable<int> IndexesOfMatchingCells(StateTuple[] states, Func<StateTuple, bool> pred)
        {
            if (states == null || states.Length == 0)
                yield break;
            for (int i = 0; i < states.Length; i++)
            {
                if (pred(states[i]))
                    yield return i;
            }
        }

        public IEnumerable<StateTuple> EnumerateTuples(StateTuple[,] states)
        {
            if (states == null || states.Length == 0)
                yield break;
            for (int y = 0; y < states.GetLength(1); y++)
                for (int x = 0; x < states.GetLength(0); x++)
                    yield return states[x, y];
        }

        void DebugCountMatchingStateTuples(Accelerator accelerator, MemoryBuffer2D<StateTuple, Stride2D.DenseX> gpu_state, Func<StateTuple, bool> pred, int expected_count = -1)
        {
            var cpu_states = CopyStateTuplesToCPU(accelerator, gpu_state);
            var matching = IndexesOfMatchingCells(cpu_states, pred).ToList();
            Console.WriteLine($"Matching count: {matching.Count}");
            if (expected_count > -1)
                Debug.Assert(matching.Count == expected_count);
        }

        public StateTuple[] CopyStateTuplesToCPU(Accelerator accelerator, MemoryBuffer2D<StateTuple, Stride2D.DenseX> gpu_state, StateTuple[] buffer = null)
        {
            if (gpu_state == null)
            {
                Console.WriteLine("Error: Simulation not initialized or not run. Cannot get final states.");
                return Array.Empty<StateTuple>();
            }

            if (buffer == null)
            {
                buffer = new StateTuple[gpu_state.Length];
            }
            else if (buffer.Length != gpu_state.Length)
            {
                Console.WriteLine($"Error: Provided buffer size {buffer.Length} does not match expected size {gpu_state.Length}.");
                return Array.Empty<StateTuple>();
            }

            gpu_state.View.AsContiguous().CopyToCPU(accelerator.DefaultStream, buffer);
            accelerator.Synchronize();
            return buffer;
        }

        StateTuple StateAtGoal(Accelerator accelerator, MemoryBuffer2D<StateTuple, Stride2D.DenseX> gpu_state, Point goal_within_grid)
        {
            var single = new StateTuple[1, 1];
            gpu_state.View
                .SubView(new Index2D(goal_within_grid.X, goal_within_grid.Y), new Index2D(1, 1))
                .CopyToCPU((AcceleratorStream)accelerator.DefaultStream, single);
            return single[0, 0];
        }

        #endregion CUDA Utilities

        #region GDAL Utilities

        static unsafe float[,] ReadRectangleFromFloatGeotiff(Rectangle rect, string path)
        {
            if (path == null || !File.Exists(path))
                throw new FileNotFoundException($"File not found: {path}");
            GdalConfiguration.ConfigureGdal();
            //Gdal.AllRegister();
            using (Dataset dataset = Gdal.Open(path, Access.GA_ReadOnly))
            {
                if (dataset == null)
                    throw new FileNotFoundException($"Could not open file: {path}");
                Band band = dataset.GetRasterBand(1);
                if (band == null)
                    throw new Exception($"Could not get band from dataset: {path}");
                var buf = new float[rect.Width, rect.Height];
                fixed (float* pBuffer = buf)
                {
                    band.ReadRaster(rect.Left, rect.Top, rect.Width, rect.Height, (IntPtr)pBuffer, rect.Width, rect.Height, DataType.GDT_Float32, 0, 0);
                }

                var proj = dataset.GetProjection();
                double[] transform = new double[6];
                dataset.GetGeoTransform(transform);
                //WriteGeoTiff(buf, "test.tif", proj, transform);
                return buf;
            }
        }

        static unsafe (float[,] ary, string proj, double[] transform) ReadRectangleFromFloatGeotiff2(Rectangle rect, string path)
        {
            if (path == null || !File.Exists(path))
                throw new FileNotFoundException($"File not found: {path}");
            GdalConfiguration.ConfigureGdal();
            //Gdal.AllRegister();
            using (Dataset dataset = Gdal.Open(path, Access.GA_ReadOnly))
            {
                if (dataset == null)
                    throw new FileNotFoundException($"Could not open file: {path}");
                Band band = dataset.GetRasterBand(1);
                if (band == null)
                    throw new Exception($"Could not get band from dataset: {path}");
                var buf = new float[rect.Width, rect.Height];
                float[] pixelValue = new float[1];

                var (y_start, y_stop) = (rect.Top, rect.Bottom);
                var (x_start, x_stop) = (rect.Left, rect.Right);
                for (int y = y_start; y < y_stop; y++)
                    for (int x = x_start; x < x_stop; x++)
                    {
                        // Read a 1x1 window (a single pixel)
                        // Parameters for ReadRaster:
                        // xOff, yOff, xSize, ySize (of the source region)
                        // buffer, buf_xSize, buf_ySize (of the destination buffer)
                        // buf_type (DataType of the buffer)
                        // pixelSpace, lineSpace (advanced usage, can often be 0)
                        band.ReadRaster(x, y, 1, 1, pixelValue, 1, 1, 0, 0);
                        buf[x - x_start, y - y_start] = pixelValue[0]; // Store the pixel value in the buffer
                    }

                var proj = dataset.GetProjection();
                double[] original_transform = new double[6];
                dataset.GetGeoTransform(original_transform);

                var new_transform = new double[6];
                new_transform[0] = original_transform[0] + rect.Left * original_transform[1] + rect.Top * original_transform[2];
                new_transform[1] = original_transform[1];
                new_transform[2] = original_transform[2];
                new_transform[3] = original_transform[3] + rect.Left * original_transform[4] + rect.Top * original_transform[5];
                new_transform[4] = original_transform[4];
                new_transform[5] = original_transform[5];

                return (buf, proj, new_transform);
            }
        }

        public unsafe static void WriteGeoTiff(
            float[,] data,
            string filename,
            string projectionWkt = null,
            double[] geoTransform = null)
        {
            GdalConfiguration.ConfigureGdal();
            //Gdal.AllRegister();

            int width = data.GetLength(0);
            int height = data.GetLength(1);

            // Create the GeoTIFF file
            Driver driver = Gdal.GetDriverByName("GTiff");
            using (var ds = driver.Create(filename, width, height, 1, DataType.GDT_Float32, null))
            using (var band = ds.GetRasterBand(1))
            {
                // Set projection and geotransform if provided
                if (!string.IsNullOrEmpty(projectionWkt))
                    ds.SetProjection(projectionWkt);
                if (geoTransform != null && geoTransform.Length == 6)
                    ds.SetGeoTransform(geoTransform);

                // Write data to band 1
                fixed (float* pBuffer = data)
                {
                    band.WriteRaster(0, 0, width, height, (IntPtr)pBuffer, width, height, DataType.GDT_Float32, 0, 0);
                }
                band.FlushCache();
            }
        }

        public unsafe static void WriteGeoTiff2(
            float[,] data,
            string filename,
            string projectionWkt = null,
            double[] geoTransform = null)
        {
            GdalConfiguration.ConfigureGdal();
            //Gdal.AllRegister();

            int width = data.GetLength(0);
            int height = data.GetLength(1);

            // Create the GeoTIFF file
            Driver driver = Gdal.GetDriverByName("GTiff");
            using (var ds = driver.Create(filename, width, height, 1, DataType.GDT_Float32, null))
            using (var band = ds.GetRasterBand(1))
            {
                // Set projection and geotransform if provided
                if (!string.IsNullOrEmpty(projectionWkt))
                    ds.SetProjection(projectionWkt);
                if (geoTransform != null && geoTransform.Length == 6)
                    ds.SetGeoTransform(geoTransform);

                float[] pixelValue = new float[1];

                var (y_start, y_stop) = (0, height);
                var (x_start, x_stop) = (0, width);
                for (int y = y_start; y < y_stop; y++)
                    for (int x = x_start; x < x_stop; x++)
                    {
                        pixelValue[0] = data[x - x_start, y - y_start]; // Store the pixel value in the buffer
                        band.WriteRaster(x, y, 1, 1, pixelValue, 1, 1, 0, 0);
                    }

                band.FlushCache();
            }
        }

        public unsafe static void WriteGeoTiff2(
            int[,] data,
            string filename,
            string projectionWkt = null,
            double[] geoTransform = null)
        {
            GdalConfiguration.ConfigureGdal();
            //Gdal.AllRegister();

            int width = data.GetLength(0);
            int height = data.GetLength(1);

            // Create the GeoTIFF file
            Driver driver = Gdal.GetDriverByName("GTiff");
            using (var ds = driver.Create(filename, width, height, 1, DataType.GDT_Int32, null))
            using (var band = ds.GetRasterBand(1))
            {
                // Set projection and geotransform if provided
                if (!string.IsNullOrEmpty(projectionWkt))
                    ds.SetProjection(projectionWkt);
                if (geoTransform != null && geoTransform.Length == 6)
                    ds.SetGeoTransform(geoTransform);

                int[] pixelValue = new int[1];

                var (y_start, y_stop) = (0, height);
                var (x_start, x_stop) = (0, width);
                for (int y = y_start; y < y_stop; y++)
                    for (int x = x_start; x < x_stop; x++)
                    {
                        pixelValue[0] = data[x - x_start, y - y_start]; // Store the pixel value in the buffer
                        band.WriteRaster(x, y, 1, 1, pixelValue, 1, 1, 0, 0);
                    }

                band.FlushCache();
            }
        }

        static unsafe byte[,] ToByteArray(float[,] input, Func<float, byte> func)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));
            if (func == null)
                throw new ArgumentNullException(nameof(func));
            var (height, width) = (input.GetLength(0), input.GetLength(1));
            var output = new byte[height, width];
            for (int row = 0; row < height; row++)
                for (int col = 0; col < width; col++)
                    output[row, col] = func(input[row, col]);
            return output;
        }

        static void WriteStateTuplesToGeoTiff(
            StateTuple[,] stateData,
            string projectionWkt,
            double[] geoTransform,
            Func<StateTuple, Color> colorMappingFunction,
            string filename)
        {
            WriteStateTuplesToGeoTiff(new Rectangle(0, 0, stateData.GetLength(0), stateData.GetLength(1)), stateData, projectionWkt, geoTransform, colorMappingFunction, filename);
        }

        static void WriteStateTuplesToGeoTiff(
            Rectangle rect,
            StateTuple[,] stateData,
            string projectionWkt,
            double[] geoTransform,
            Func<StateTuple, Color> colorMappingFunction,
            string filename)
        {
            var bounds = new Rectangle(0, 0, stateData.GetLength(0), stateData.GetLength(1));
            if (!bounds.Contains(rect))
                throw new ArgumentOutOfRangeException("Rectangle dimensions exceed grid dimensions.");
            if (stateData == null)
                throw new ArgumentNullException(nameof(stateData));
            if (colorMappingFunction == null)
                throw new ArgumentNullException(nameof(colorMappingFunction));
            if (string.IsNullOrWhiteSpace(filename))
                throw new ArgumentNullException(nameof(filename));

            GdalConfiguration.ConfigureGdal();
            //Gdal.AllRegister();

            Driver driver = Gdal.GetDriverByName("GTiff");
            if (driver == null)
            {
                Console.WriteLine("Error: GeoTIFF driver not found. Ensure GDAL is correctly installed and configured.");
                throw new NotSupportedException("GeoTIFF driver not available in GDAL.");
            }

            try
            {
                using (var dataset = driver.Create(filename, rect.Width, rect.Height, 4, DataType.GDT_Byte, null))
                {
                    if (dataset == null)
                    {
                        Console.WriteLine($"Error: Could not create dataset for file '{filename}'. Check permissions and path.");
                        throw new InvalidOperationException($"Failed to create GDAL dataset for '{filename}'.");
                    }

                    // Set projection and geotransform if provided
                    if (!string.IsNullOrEmpty(projectionWkt))
                        dataset.SetProjection(projectionWkt);
                    if (geoTransform != null && geoTransform.Length == 6)
                        dataset.SetGeoTransform(geoTransform);

                    Band redBand = dataset.GetRasterBand(1);
                    Band greenBand = dataset.GetRasterBand(2);
                    Band blueBand = dataset.GetRasterBand(3);
                    Band alphaBand = dataset.GetRasterBand(4);

                    redBand.SetColorInterpretation(ColorInterp.GCI_RedBand);
                    greenBand.SetColorInterpretation(ColorInterp.GCI_GreenBand);
                    blueBand.SetColorInterpretation(ColorInterp.GCI_BlueBand);
                    alphaBand.SetColorInterpretation(ColorInterp.GCI_AlphaBand);

                    var gridWidth = rect.Width;
                    byte[] redRowBuffer = new byte[gridWidth];
                    byte[] greenRowBuffer = new byte[gridWidth];
                    byte[] blueRowBuffer = new byte[gridWidth];
                    byte[] alphaRowBuffer = new byte[gridWidth];

                    var (rleft, rtop, rwidth, rheight) = (rect.Left, rect.Top, rect.Width, rect.Height);

                    for (int y = 0; y < rheight; y++)
                    {
                        for (int x = 0; x < rwidth; x++)
                        {
                            var (stateX, stateY) = (rleft + x, rtop + y);
                            var currentTuple = stateData[stateX, stateY];
                            var pixelColor = colorMappingFunction(currentTuple);

                            redRowBuffer[x] = pixelColor.R;
                            greenRowBuffer[x] = pixelColor.G;
                            blueRowBuffer[x] = pixelColor.B;
                            alphaRowBuffer[x] = pixelColor.A;
                        }

                        redBand.WriteRaster(0, y, rwidth, 1, redRowBuffer, rwidth, 1, 0, 0);
                        greenBand.WriteRaster(0, y, rwidth, 1, greenRowBuffer, rwidth, 1, 0, 0);
                        blueBand.WriteRaster(0, y, rwidth, 1, blueRowBuffer, rwidth, 1, 0, 0);
                        alphaBand.WriteRaster(0, y, rwidth, 1, alphaRowBuffer, rwidth, 1, 0, 0);
                    }

                    dataset.FlushCache();
                    Console.WriteLine($"Successfully wrote RGBA TIFF to '{filename}'");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"An error occurred while writing the GeoTIFF file: {ex.Message}");
                Console.WriteLine($"Stack Trace: {ex.StackTrace}");
            }
        }

        static Color DefaultColorMapper(StateTuple state)
        {
            if (state.IsUnoccupied()) return Color.Black;
            if (state.TotalWh > 200000) return Color.FromArgb(255, (int)(state.BatteryWh / RoverConstants.BatteryCapacityWh * 200), 255, 100); // Example: Greenish, alpha based on battery
                                                                                                                                               // Add more logic to map states to colors
            return Color.FromArgb(128, Color.Gray); // Default for other occupied states with some transparency
        }

        void FillSlopeArrayWithTestData(byte[,] ary)
        {
            var (width, height) = (ary.GetLength(0), ary.GetLength(1));
            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    ary[x, y] = (byte)(x % 150);
                }
            }
        }

        #endregion GDAL Utilities

        #region projection utilities

        public const string LatLonProjection = "GEOGCS[\"Moon 2000\", DATUM[\"D_Moon_2000\", SPHEROID[\"Moon_2000_IAU_IAG\",1737400.0,0.0]], PRIMEM[\"Greenwich\",0], UNIT[\"Decimal_Degree\",0.0174532925199433]]";
        public const string PROJECTION_FILE = SLOPE_TIF;

        public static double[] SITE_AFFINE_TRANSFORM;
        public static string SITE_PROJECTION;
        public static SpatialReference SITE_LATLON_SPATIAL_REFERENCE;
        public static SpatialReference SITE_STEREOGRAPHIC_SPATIAL_REFERENCE;
        public static CoordinateTransformation map2LonLat_transform;
        public static CoordinateTransformation lonLat2Map_transform;

        public static void LoadProjectionInfo(string filename = PROJECTION_FILE)
        {
            if (SITE_AFFINE_TRANSFORM != null)
                return;
            using (var dataset = Gdal.Open(filename, Access.GA_ReadOnly))
            {
                SITE_AFFINE_TRANSFORM = new double[6];
                dataset.GetGeoTransform(SITE_AFFINE_TRANSFORM);

                SITE_PROJECTION = dataset.GetProjectionRef();

                var src = new SpatialReference(SITE_PROJECTION);
                var dst = new SpatialReference(LatLonProjection);

                map2LonLat_transform = new CoordinateTransformation(src, dst);
                lonLat2Map_transform = new CoordinateTransformation(dst, src);
            }
        }

        public static Point LatLonDeg2RowCol(double lat_deg, double lon_deg)
        {
            LoadProjectionInfo();
            // This assumes a north up image: a[2]=a[4]=0, a[1]=a[5]=pixel_size, a[0],a[3] position of top left
            var o = new double[3];
            lonLat2Map_transform.TransformPoint(o, lon_deg, lat_deg, 0d);
            var (x_geo, y_geo) = (o[0], o[1]);
            var col = (x_geo - SITE_AFFINE_TRANSFORM[0]) / SITE_AFFINE_TRANSFORM[1];
            var row = (y_geo - SITE_AFFINE_TRANSFORM[3]) / SITE_AFFINE_TRANSFORM[5];
            return new Point((int)col, (int)row);
        }

        public static (double lon_deg, double lat_deg) RowCol2LonLatDeg(Point p) => RowCol2LonLatDeg(p.Y, p.X);
        public static (double lon_deg, double lat_deg) RowCol2LonLatDeg(double row, double col)
        {
            LoadProjectionInfo();
            if (map2LonLat_transform == null || SITE_AFFINE_TRANSFORM == null)
            {
                Console.WriteLine(@"No coordinate transform has been provided to go from pixels to lat/lon");
                throw new Exception(@"No coordinate transform has been provided to go from pixels to lat/lon");
            }

            var x = col * SITE_AFFINE_TRANSFORM[1] + SITE_AFFINE_TRANSFORM[0];
            var y = row * SITE_AFFINE_TRANSFORM[5] + SITE_AFFINE_TRANSFORM[3];

            var o = new double[3];
            map2LonLat_transform.TransformPoint(o, x, y, 0d);
            return (o[0], o[1]);        // Note canonical order here
        }

        public static IEnumerable<(double crs_x, double crs_y)> Points2CRS(string geotiff_filename, IEnumerable<Point> points)
        {
            using (var dataset = Gdal.Open(geotiff_filename, Access.GA_ReadOnly))
            {
                var SITE_AFFINE_TRANSFORM = new double[6];
                dataset.GetGeoTransform(SITE_AFFINE_TRANSFORM);

                var SITE_PROJECTION = dataset.GetProjectionRef();

                var src = new SpatialReference(SITE_PROJECTION);
                var dst = new SpatialReference(LatLonProjection);
                var map2LonLat_transform = new CoordinateTransformation(src, dst);

                var o = new double[3];

                foreach (var pt in points)
                {
                    var (col, row) = (pt.X, pt.Y);
                    var x = col * SITE_AFFINE_TRANSFORM[1] + SITE_AFFINE_TRANSFORM[0];
                    var y = row * SITE_AFFINE_TRANSFORM[5] + SITE_AFFINE_TRANSFORM[3];

                    yield return (x, y);

                    //map2LonLat_transform.TransformPoint(o, x, y, 0d);
                    //yield return (o[0], o[1]);
                }
            }
        }

        #endregion projection utilities

        public void Dispose()
        {
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Point2 : IEquatable<Point2>
    {
        public short X;
        public short Y;

        public Point2(short x, short y) => (X, Y) = (x, y);

        public static Point2 Invalid => new Point2(-1, -1);
        public bool IsValid => X != -1;

        public bool Equals(Point2 other) => X == other.X && Y == other.Y;

        public override bool Equals(object obj) => obj is Point2 other && Equals(other);

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = 17;
                hash = hash * 23 + X.GetHashCode();
                hash = hash * 23 + Y.GetHashCode();
                return hash;
            }
        }
        public static bool operator ==(Point2 a, Point2 b) => a.Equals(b);
        public static bool operator !=(Point2 a, Point2 b) => !(a.Equals(b));
    }
}