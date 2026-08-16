using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using System.Drawing.Drawing2D;
using System.Threading;
using System.Windows.Forms;

// =====================================================================================
// Note on Kernel0 and Kernel2 Auto-Grouping:
// The signatures for Kernel0_InitializeStates and Kernel2_CHARGING below
// (starting with Index1D linearGlobalCellIdx) are correct for auto-grouping.
// When loaded with LoadAutoGroupedStreamKernel and launched with a 1D extent,
// ILGPU will manage the thread grouping, making them suitable for CPU debugging
// without manual group size adjustments for these specific kernels.
// =====================================================================================

namespace GridRunner
{
    public static class KernelImplementations
    {


        // =====================================================================================
        // Device Helper Functions
        // =====================================================================================
        public static bool IsBetterState(in StateTuple newState, in StateTuple oldState)
        {
            if (newState.IsUnoccupied()) return false;
            if (oldState.IsUnoccupied()) return true;
            if (newState.TotalWh > oldState.TotalWh) return true;
            if (newState.TotalWh < oldState.TotalWh) return false;
            return newState.ArrivalTime < oldState.ArrivalTime;
        }

        public static void GetMoveDetails(int dx, int dy, out float energyCostWh, out float durationHours)
        {
            bool isDiagonal = (dx != 0 & dy != 0);
            if (isDiagonal)
            {
                energyCostWh = RoverConstants.DriveEnergyDiagonalWh;
                durationHours = RoverConstants.DriveTimeToDiagonalCellHr;
            }
            else
            {
                energyCostWh = RoverConstants.DriveEnergyAdjacentWh;
                durationHours = RoverConstants.DriveTimeToAdjacentCellHr;
            }
        }

        public static byte GetPreviousCellEncoding(int dx, int dy)
        {
            if (dx == 1 && dy == -1) return 7;
            if (dx == 0 && dy == -1) return 8;
            if (dx == -1 && dy == -1) return 9;
            if (dx == 1 && dy == 0) return 4;
            if (dx == -1 && dy == 0) return 6;
            if (dx == 1 && dy == 1) return 1;
            if (dx == 0 && dy == 1) return 2;
            if (dx == -1 && dy == 1) return 3;
            return RoverConstants.PreviousCellUnoccupied;
        }

        public static byte GetOppositeEncoding(int enc)
        {
            if (enc == 1) return 9;
            if (enc == 2) return 8;
            if (enc == 3) return 7;
            if (enc == 4) return 6;
            if (enc ==5) return 5;
            if (enc == 6) return 4;
            if (enc == 7) return 3;
            if (enc == 8) return 2;
            if (enc == 9) return 1;
            return RoverConstants.PreviousCellUnoccupied;
        }

        public static StateTuple ApplyStationaryLogic(
            StateTuple current_state, float durationHours, byte sunFractionByte, byte fuelCellNegativeIsIllegal)
        {
            if (current_state.IsUnoccupied() || durationHours <= 0.0f) return current_state;

            StateTuple new_state = current_state;
            float sunIntensity = sunFractionByte / 255.0f;
            float solarGenerationWatts = sunIntensity * RoverConstants.SolarGenerationFullSunWatts;
            float netPowerWatts = solarGenerationWatts - RoverConstants.StationaryPowerWatts;
            float energyChangeWh = netPowerWatts * durationHours;

            if (energyChangeWh > 0)
            {
                float batterySpaceToFullUsable = RoverConstants.BatteryCapacityWh - new_state.BatteryWh;
                float chargeToUsableBattery = XMath.Min(energyChangeWh, batterySpaceToFullUsable);
                new_state.BatteryWh += chargeToUsableBattery;
                new_state.TotalWh += chargeToUsableBattery;
                energyChangeWh -= chargeToUsableBattery;

                if (energyChangeWh > 0)
                {
                    float currentFuelCellWh = new_state.TotalWh - (new_state.BatteryWh + RoverConstants.BatteryMinOperatingWh);
                    float fuelCellSpaceToFull = RoverConstants.FuelCellCapacityWh - currentFuelCellWh;

                    if (fuelCellSpaceToFull > 0)
                    {
                        float energyToFuelCellEffective = energyChangeWh * RoverConstants.FuelCellChargeEfficiency;
                        float actualChargeToFuelCell = XMath.Min(energyToFuelCellEffective, fuelCellSpaceToFull);
                        new_state.TotalWh += actualChargeToFuelCell;
                    }
                }
            }
            else
            {
                float energyNeededWh = -energyChangeWh;

                float dischargeFromUsableBattery = XMath.Min(energyNeededWh, new_state.BatteryWh);
                new_state.BatteryWh -= dischargeFromUsableBattery;
                new_state.TotalWh -= dischargeFromUsableBattery;
                energyNeededWh -= dischargeFromUsableBattery;

                if (energyNeededWh > 0)
                {
                    new_state.TotalWh -= energyNeededWh;
                }
            }

            float finalFuelCellWh = new_state.TotalWh - (new_state.BatteryWh + RoverConstants.BatteryMinOperatingWh);
            if (fuelCellNegativeIsIllegal != 0 && finalFuelCellWh < -1e-3f)
            {
                return StateTuple.Unoccupied();
            }

            short scaledFinalFuelCellWh = (short)XMath.Floor(finalFuelCellWh / 20.0f);
            if (scaledFinalFuelCellWh < new_state.MinFuelCellWhScaledBy20)
            {
                new_state.MinFuelCellWhScaledBy20 = scaledFinalFuelCellWh;
            }
            if (new_state.BatteryWh < 0.0f) new_state.BatteryWh = 0.0f;

            float minPossibleTotalWh = RoverConstants.BatteryMinOperatingWh;
            if (finalFuelCellWh < 0) minPossibleTotalWh += finalFuelCellWh;

            if (new_state.TotalWh < RoverConstants.BatteryMinOperatingWh && !current_state.IsUnoccupied())
            {
                if (fuelCellNegativeIsIllegal != 0 || finalFuelCellWh < -1e-3f) return StateTuple.Unoccupied();
            }

            return new_state;
        }

        /// <summary>
        /// Convert from a global xy in the grid to an index into the block-oriented data in the slope or sun arrays
        /// </summary>
        /// <param name="globalX"></param>
        /// <param name="globalY"></param>
        /// <param name="blockDimX"></param>
        /// <param name="blockDimY"></param>
        /// <param name="numBlocksXInGrid"></param>
        /// <returns></returns>
        static int GlobalXY_to_BlockLinearIndex(
            int globalX, int globalY,
            int blockDimX, int blockDimY,
            int numBlocksXInGrid)
        {
            int blockCoordX = globalX / blockDimX;
            int blockCoordY = globalY / blockDimY;
            int blockFlatIndex = blockCoordY * numBlocksXInGrid + blockCoordX;
            int threadXInBlock = globalX % blockDimX;
            int threadYInBlock = globalY % blockDimY;
            int blockBaseOffset = blockFlatIndex * (blockDimX * blockDimY);
            int offsetInBlockChunk = threadYInBlock * blockDimX + threadXInBlock;
            return blockBaseOffset + offsetInBlockChunk;
        }

        public static void Kernel0_initialize(
            Index2D index,
            ArrayView2D<StateTuple, Stride2D.DenseX> gpu_currentState,
            Index2D start_position)
        {
            if (index == start_position)
                gpu_currentState[index] = StateTuple.StartState();
            else
                gpu_currentState[index] = StateTuple.Unoccupied();
        }

        // This kernel updates all of the grid cells in one block/group.  It iterates until no cell within this
        // group is written.  It writes the number of iterations back to the block activity array.
        // 
        public static void Kernel1_Driving(
            SpecializedValue<Index2D> group_size,
            ArrayView1D<Point2, Stride1D.Dense> gpu_active_groups,
            int active_group_count,
            ArrayView1D<byte, Stride1D.Dense> gpu_group_modified,
            ArrayView2D<StateTuple, Stride2D.DenseX> gpu_state,
            ArrayView2D<byte, Stride2D.DenseX> gpu_sun,
            ArrayView2D<float, Stride2D.DenseX> gpu_slope,
            ArrayView2D<int, Stride2D.DenseX> gpu_debug,
            float epochTimeAtWindowEnd)
        {
            var (x, y) = (Group.IdxX, Group.IdxY);                  // Local thread index within this group
            var group_id = gpu_active_groups[Grid.IdxX];
            var global_x = group_id.X * group_size.Value.X + x;     // Position of this thread in the global memory state array
            var global_y = group_id.Y * group_size.Value.Y + y;

            var global_extent_x = gpu_state.Extent.X;
            var global_extent_y = gpu_state.Extent.Y; 

            // After each do iteration, this is checked.  If it's 0, the group is idle and the loop ends.
            ref int group_idle_flag = ref SharedMemory.Allocate<int>();

            // Counts the do iterations
            ref int iteration_count = ref SharedMemory.Allocate<int>();

            Group.Barrier();

            if (Group.IsFirstThread)
            {
                group_idle_flag = 0;
                iteration_count = 0;
            }

            do
            {
                // This is needed because shared_iteration_change_flag is read at the end of the loop, and
                // All those reads must be finished before it's written to.
                Group.Barrier();

                if (Group.IsFirstThread)
                    group_idle_flag = 0;

                var slope = gpu_slope[global_x, global_y];
                var isLegalSlope = slope <= RoverConstants.SlopeThreshold;

                if (!isLegalSlope)
                {

                }

                var best_state = gpu_state[global_x, global_y];
                bool stateChangedInCell = false;

                // debugging
                int dx1 = 0;
                int dy1 = 0;

                if (isLegalSlope)
                {
                    // Iterate 8 neighbors (dx, dy offsets from current cell).    Iterate over X in the inner loop.
                    for (int neighbor_dy = -1; neighbor_dy <= 1; ++neighbor_dy)
                    {
                        for (int neighbor_dx = -1; neighbor_dx <= 1; ++neighbor_dx)
                        {
                            if (neighbor_dx == 0 && neighbor_dy == 0)
                                continue; // Skip self

                            int neighbor_x = global_x + neighbor_dx;
                            int neighbor_y = global_y + neighbor_dy;

                            if (neighbor_x < 0 || neighbor_x >= global_extent_x ||
                                neighbor_y < 0 || neighbor_y >= global_extent_y)
                                continue; // Neighbor is out of bounds

                            var neighborState = gpu_state[neighbor_x, neighbor_y];

                            // If neighbor is unoccupied, skip
                            if (neighborState.IsUnoccupied())
                                continue;

                            // The move is FROM neighbor TO current cell.
                            // dx_move, dy_move are from neighbor's perspective to current cell
                            int move_dx = -neighbor_dx;
                            int move_dy = -neighbor_dy;

                            GetMoveDetails(move_dx, move_dy, out float energyCost, out float durationHours);
                            
                            // debugging
                            if (energyCost < 0f)
                            {
                            }

                            // Update the neighbor state to reflect the move to the current cell
                            neighborState.ArrivalTime += durationHours; // Arrives at current cell
                            neighborState.BatteryWh -= energyCost;
                            neighborState.TotalWh -= energyCost;

                            if (neighborState.ArrivalTime > epochTimeAtWindowEnd | neighborState.BatteryWh < 0.0f)
                                continue;

                            if (IsBetterState(neighborState, best_state))
                            {
                                best_state = neighborState;
                                best_state.PreviousCell = GetPreviousCellEncoding(move_dx, move_dy);

                                stateChangedInCell = true;

                                // debugging
                                dx1 = move_dx;
                                dy1 = move_dy;
                            }
                        }
                    }

                    if (stateChangedInCell)
                    {
                        // Debugging
                        if (true)                                           // group origin 2240,4512
                        {
                            if (global_x == 2241 & global_y == 4528)        // writing upper left
                            {
                                var cul = gpu_state[2241, 4528];
                                var c00 = gpu_state[2242, 4529];
                                var clr = gpu_state[2243, 4530];
                                var bes = best_state;
                            }
                            if (global_x == 2242 & global_y == 4529)        // writing center
                            {
                                var cul = gpu_state[2241, 4528];
                                var c00 = gpu_state[2242, 4529];
                                var clr = gpu_state[2243, 4530];
                                var bes= best_state;
                            }
                            if (global_x == 2243 & global_y == 4530)        // writing lower right
                            {
                                var cul = gpu_state[2241, 4528];
                                var c00 = gpu_state[2242, 4529];
                                var clr = gpu_state[2243, 4530];
                                var bes = best_state;
                            }
                            int nx = global_x - dx1;
                            int ny = global_y - dy1;

                            if (nx >= 0 & nx < global_extent_x & ny >= 0 & ny < global_extent_y)
                            {
                                var neighbor = gpu_state[nx, ny];
                                if (neighbor.PreviousCell == GetOppositeEncoding(best_state.PreviousCell))
                                {
                                    var cul = gpu_state[2241, 4528];        // writing center with the problem
                                    var c00 = gpu_state[2242, 4529];
                                    var clr = gpu_state[2243, 4530];

                                    if (gpu_debug[global_x, global_y] < 255)
                                        gpu_debug[global_x, global_y] += 1; // Increment debug counter for this cell
                                }
                            }
                        }

                        gpu_state[global_x, global_y] = best_state;
                        Atomic.Or(ref group_idle_flag, 1);
                    }
                }

                if (Group.IsFirstThread)
                    Atomic.Add(ref iteration_count, 1);

                Group.Barrier();
                
            } while (group_idle_flag > 0);

            Group.Barrier();

            // If there was more than the initial iteration, mark this group as active for the next iteration
            if (Group.IsFirstThread)
                if (iteration_count > 1)
                    gpu_group_modified[Grid.IdxX] = 1; // Mark this block as active
        }

        // Kernel2: Signature `(Index1D ...)` is correct for auto-grouping.
        public static void Kernel2_CHARGING(
            SpecializedValue<Index2D> group_size,
            ArrayView1D<Point2, Stride1D.Dense> gpu_active_groups,
            int active_group_count,
            ArrayView2D<StateTuple, Stride2D.DenseX> gpu_state,
            ArrayView2D<byte, Stride2D.DenseX> gpu_sun,
            float epochTimeAtWindowEnd,
            byte fuelCellNegativeIsIllegal)
        {
            var (x, y) = (Group.IdxX, Group.IdxY);                  // Local thread index within this group
            var group_id = gpu_active_groups[Grid.IdxX];
            var global_x = group_id.X * group_size.Value.X + x;     // Position of this thread in the global memory state array
            var global_y = group_id.Y * group_size.Value.Y + y;

            var state = gpu_state[global_x, global_y];

            // No need to process unoccupied or out-of-time cells.
            // Assume the slope constraint in the driving kernel forces the cell to be unoccupied if it is not legal.
            // so don't check slope here.
            if (state.IsUnoccupied())
                return;

            if (state.ArrivalTime > epochTimeAtWindowEnd)
                gpu_active_groups[-1] = new Point2(-1, -1); // This should be an error.

            var time_delta = epochTimeAtWindowEnd - state.ArrivalTime;
            float power_during_charging, power_during_deployment;

            if (time_delta <= RoverConstants.SolarArrayDeploymentHrs)
            {
                var next_battery_wh = XMath.Min(
                    RoverConstants.BatteryCapacityWh, 
                    state.BatteryWh - RoverConstants.StationaryPowerWatts * time_delta);

                if (next_battery_wh <= 0f)
                    state = StateTuple.Unoccupied(); // Below 20% reserve, mark as unoccupied
                else
                {
                    // Update BatteryWh and TotalWh
                    var battery_delta_wh = next_battery_wh - state.BatteryWh;
                    state.BatteryWh = next_battery_wh;          // Update battery state
                    state.TotalWh = XMath.Min(                  // Update total charge
                        RoverConstants.InitialTotalWh, 
                        state.TotalWh + battery_delta_wh);
                    state.ArrivalTime = epochTimeAtWindowEnd;
                }
            }
            else
            {
                // We have enough time to charge.  This assumes you deploy the solar array
                // even if we're in darkness.  There's no power cost for the deployment, so
                // this is ok.

                var charge_time = time_delta - RoverConstants.SolarArrayDeploymentHrs;
                var sun_fraction = gpu_sun[global_x, global_y] / 255.0f; // Convert to fraction [0, 1]

                power_during_charging = sun_fraction * RoverConstants.SolarGenerationFullSunWatts - RoverConstants.StationaryPowerWatts;
                power_during_deployment = -RoverConstants.StationaryPowerWatts;

                var next_battery_wh = XMath.Min(
                    RoverConstants.BatteryCapacityWh, 
                    state.BatteryWh + power_during_charging * charge_time + power_during_deployment * RoverConstants.SolarArrayDeploymentHrs);

                if (next_battery_wh < 0f)
                    state = StateTuple.Unoccupied();
                else
                {
                    // Update BatteryWh and TotalWh
                    var battery_delta_wh = next_battery_wh - state.BatteryWh;
                    state.BatteryWh = next_battery_wh;          // Update battery state
                    state.TotalWh = XMath.Min(                  // Update total charge
                        RoverConstants.InitialTotalWh,
                        state.TotalWh + battery_delta_wh);
                    state.ArrivalTime = epochTimeAtWindowEnd;
                }
            }

            // Save the state.  I'm not writing it inside the if's because I want to support coalescing writes
            gpu_state[global_x, global_y] = state; // Write back the updated state
        }
    }
}