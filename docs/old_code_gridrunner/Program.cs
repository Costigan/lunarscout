using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace GridRunner
{
    internal static class Program
    {
        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        [STAThread]
        static void OldMain()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Form1());
        }

        [STAThread]
        static void Main(string[] args)
        {
            Console.WriteLine("Moon Rover Pathfinder Simulation");

            GdalConfiguration.ConfigureGdal();
            //Gdal.AllRegister();

            // Parameters for the Compilation & Basic Execution Test
            var testSizeInBlocks = 3;
            int activeGridDimX = testSizeInBlocks * RoverConstants.BlockDimX;
            int activeGridDimY = testSizeInBlocks * RoverConstants.BlockDimY;
            int startX_active = activeGridDimX / 2; // Start in the middle of the active area
            int startY_active = activeGridDimY / 2;
            float initialEpochTime = 0.0f; // Rover starts at T=0
            int numZSteps = 1;             // Run for a single Z-step
            bool fuelCellNegativeIsIllegalMode = true;

            var start_latlon = (-88.5956, 360 + -67.9506);
            var goal_latlon = (-86.42666, 360 + -22.637756); // Move slightly north
            var start_pt = MoonRoverPathfinder.LatLonDeg2RowCol(start_latlon.Item1, start_latlon.Item2);
            var goal_pt = MoonRoverPathfinder.LatLonDeg2RowCol(goal_latlon.Item1, goal_latlon.Item2);

            var bounds_in_lola = new Rectangle(11008, 9856, 3840, 4736);
            var bounds_in_slope = new Rectangle(0, 0, 3840, 4736);
            var start = start_pt; // new Point(start_pt.X-bounds.Left, start_pt.Y-bounds.Top);
            var goal = goal_pt; // new Point(goal_pt.X - bounds.Left, goal_pt.Y - bounds.Top); // Adjusted for bounds

            var start_within_grid = new Point(start.X - bounds_in_slope.Left, start.Y - bounds_in_slope.Top);
            var goal_within_grid = new Point(goal.X - bounds_in_slope.Left, goal.Y - bounds_in_slope.Top);

            //var start = new Point(12971, 14319); // Start point in the middle of the grid
            //var goal = new Point(13507, 10156); // Goal point for the rover

            // Debugging
            //var bounds = new Rectangle(11008, 9856, 128, 256);
            //var start = new Point(11008, 9856); // Start point in the middle of the grid
            //var goal = new Point(11008, 9856 + 10 ); // Goal point for the rover

            // Run the example simulation
            RunSimulationExample(bounds_in_lola, start_within_grid, goal_within_grid);
            Console.WriteLine("Simulation example finished.");
        }

        /// <summary>
        /// Runs a Moon Rover Pathfinder simulation with specified parameters.
        /// </summary>
        static void RunSimulationExample(
            Rectangle bounds_in_lola,
            Point start_within_grid,
            Point goal_within_grid)
        {
            Console.WriteLine("\n--- Starting Simulation Example ---");
            //Console.WriteLine($"Grid Size: {bounds.Width}x{bounds.Height}");

            Stopwatch totalSimTimer = Stopwatch.StartNew();

            try
            {
                using (var pathfinder = new MoonRoverPathfinder())
                {
                    // --- Prepare Slope and Initial Data ---
                    // For this basic test, we'll generate simple slope data.
                    // In a real scenario, this would come from a file or a more complex generation.

                    // Run the simulation
                    var cpu_state = pathfinder.RunSimulation(
                        bounds_in_lola,
                        start_within_grid,
                        goal_within_grid,
                        preferCPU: false,
                        FillSlopeBlockWithTestData1,
                        FillSunBlockWithTestData1);

                    var backpath = MoonRoverPathfinder.GetRouteBackwardsCarefully(cpu_state, start_within_grid, goal_within_grid);
                    Console.WriteLine("backpath.Count = " + backpath.Count);

                    var crs_points = MoonRoverPathfinder.Points2CRS(MoonRoverPathfinder.LOLA_TIF, backpath.Select(p => new Point(p.X + bounds_in_lola.Left, p.Y + bounds_in_lola.Top))).ToList();
                    MoonRoverPathfinder.WriteRouteGeoJson("route.geojson", crs_points);
                }
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"\n--- SIMULATION EXAMPLE FAILED ---");
                Console.WriteLine($"Error: {ex.Message}");
                Console.WriteLine($"Stack Trace: {ex.StackTrace}");
                Console.ResetColor();
            }
            finally
            {
                totalSimTimer.Stop();
                Console.WriteLine($"\nTotal example execution time: {totalSimTimer.ElapsedMilliseconds} ms");
                Console.WriteLine("--- End of Simulation Example ---");
            }
        }

        static void FillSlopeBlockWithTestData1(byte[,] buffer, Rectangle bounds_in_buffer, Rectangle grid_bounds_in_20m)
        {
            var bounds = bounds_in_buffer;
            Debug.Assert(bounds.Left >= 0 & bounds.Top >= 0 & bounds.Right <= buffer.GetLength(0) & bounds.Bottom <= buffer.GetLength(1),
                "Block bounds must be within buffer dimensions");
            for (var y = bounds.Top; y < bounds.Bottom; ++y)
            {
                for (var x = bounds.Left; x < bounds.Right; ++x)
                {
                    buffer[x, y] = 1; // test data
                }
            }
        }

        static void FillSunBlockWithTestData1(byte[,] buffer, Rectangle bounds_in_buffer, Rectangle grid_bounds_in_20m, int level)
        {
            var bounds = bounds_in_buffer;
            Debug.Assert(bounds.Left >= 0 & bounds.Top >= 0 & bounds.Right <= buffer.GetLength(0) & bounds.Bottom <= buffer.GetLength(1),
                "Block bounds must be within buffer dimensions");
            for (var y = bounds.Top; y < bounds.Bottom; ++y)
            {
                for (var x = bounds.Left; x < bounds.Right; ++x)
                {
                    buffer[x, y] = 255; // test data
                }
            }
        }

        /// <summary>
        /// Generates initial slope data for the full grid (including halo)
        /// in block-contiguous layout.
        /// Halo regions are marked as impassable. Active regions are traversable.
        /// </summary>
        private static byte[] GenerateInitialSlopeData(
            int fullGridW, int fullGridH,
            int activeGridW, int activeGridH,
            int blockDimX, int blockDimY)
        {
            int numTotalCells = fullGridW * fullGridH;
            byte[] slopeData = new byte[numTotalCells];

            int numBlocksX = fullGridW / blockDimX;
            // int numBlocksY = fullGridH / blockDimY; // Not needed for flat indexing

            int haloCellsLeft = (fullGridW - activeGridW) / 2;
            int haloCellsRight = fullGridW - haloCellsLeft - activeGridW;
            int haloCellsTop = (fullGridH - activeGridH) / 2;
            int haloCellsBottom = fullGridH - haloCellsTop - activeGridH;

            Console.WriteLine($"Generating slope data for {fullGridW}x{fullGridH} grid.");
            Console.WriteLine($"Halo L/R/T/B: {haloCellsLeft}/{haloCellsRight}/{haloCellsTop}/{haloCellsBottom}");

            for (int globalY = 0; globalY < fullGridH; ++globalY)
            {
                for (int globalX = 0; globalX < fullGridW; ++globalX)
                {
                    byte slopeValue;
                    // Determine if the current global (x,y) is in a halo region
                    bool isHalo = (globalX < haloCellsLeft ||
                                   globalX >= fullGridW - haloCellsRight ||
                                   globalY < haloCellsTop ||
                                   globalY >= fullGridH - haloCellsBottom);

                    if (isHalo)
                    {
                        slopeValue = 255; // Impassable slope for halo regions
                    }
                    else
                    {
                        slopeValue = 5; // Traversable slope for active regions
                                        // You can add more complex patterns here if needed for testing
                    }

                    // Calculate index in block-contiguous layout
                    int blockCoordX = globalX / blockDimX;
                    int blockCoordY = globalY / blockDimY;
                    int blockFlatIndex = blockCoordY * numBlocksX + blockCoordX;

                    int threadXInBlock = globalX % blockDimX;
                    int threadYInBlock = globalY % blockDimY;

                    int blockBaseOffset = blockFlatIndex * (blockDimX * blockDimY);
                    int offsetInBlockChunk = threadYInBlock * blockDimX + threadXInBlock;
                    int finalIndex = blockBaseOffset + offsetInBlockChunk;

                    if (finalIndex < numTotalCells)
                    {
                        slopeData[finalIndex] = slopeValue;
                    }
                    else
                    {
                        Console.WriteLine($"Error generating slope data: Index {finalIndex} out of bounds for size {numTotalCells} at ({globalX},{globalY})");
                    }
                }
            }
            Console.WriteLine("Slope data generated.");
            return slopeData;
        }

        // Note: Sunlight data generation (`GenerateSunlightForBlock`) is handled internally
        // by MoonRoverPathfinder during the simulation run based on its logic.
        // For this basic test, we don't need to provide it upfront to InitializeSimulation.
        // The `initialSlopeDataBlockContiguous` is the primary external map data needed at init.
    }
}
