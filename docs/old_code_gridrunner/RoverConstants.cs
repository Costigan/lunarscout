namespace GridRunner
{
    public static class RoverConstants
    {
        // Grid and Time Parameters
        public const float TimeStepDurationHours = 2.0f;
        public const float CellSizeMeters = 20.0f;

        // Maximum iterations for Kernel1_Driving safety break
        public const int MaxK2Iterations = 500;

        // Times are in units of hours.
        public const float InitialTime = 0f;

        // Initial states (the whole StateTuple should be 0)
        public const float UnoccupiedTotalWh = 0f;
        public const float UnoccupiedArrivalTime = float.MaxValue;

        // Rover Movement Parameters
        public const float RoverSpeedKph = 2.0f; // km/h

        // The transfer times between cells.  Note these times are POSITIVE.  They will be inverted
        // when stored in the state tuple.  We expect to use these constants in the driving kernel
        public const float DriveTimeToAdjacentCellHr = (CellSizeMeters / 1000f) / RoverSpeedKph;
        public const float DriveTimeToDiagonalCellHr = 1.41421356237f * DriveTimeToAdjacentCellHr;

        // Rover Power System Parameters
        // The battery SOC will never be allowed below 20%.  Therefore, we won't represent that portion.
        public const float BatteryRawCapacityWh = 31400.0f;
        public const float BatteryCapacityWh = BatteryRawCapacityWh * 0.8f;
        public const float Battery20PercentReserveWh = BatteryRawCapacityWh * 0.2f;
        public const float FuelCellCapacityWh = 551000.0f;                  
        public const float BatteryMinOperatingWh = 0f;  // This doesn't include the 20% reserve

        public const float InitialTotalWh = BatteryCapacityWh + BatteryMinOperatingWh + FuelCellCapacityWh;

        public const float DrivingPowerWatts = 3925.0f;
        public const float StationaryPowerWatts = 2870.0f;
        public const float SolarGenerationFullSunWatts = 10600.0f;
        public const float FuelCellChargeEfficiency = 0.35f;

        public const float SolarArrayDeploymentHrs = 0.33f; // in hours, 20 minutes

        // Derived Rover Energy Costs
        public static readonly float DriveEnergyAdjacentWh = DrivingPowerWatts * DriveTimeToAdjacentCellHr;
        public static readonly float DriveEnergyDiagonalWh = DrivingPowerWatts * DriveTimeToDiagonalCellHr;

        // StateTuple Specific Constants
        public const byte PreviousCellUnoccupied = 0;
        public const byte PreviousCellChainTerminator = 255;

        // Operational Constraints
        public const float MaxSlopeDegrees = 15.0f;
        public const float SlopeThreshold = 15f;

        // Kernel Configuration Parameters (for ILGPU launch)
        public const int BlockDimX = 32;
        public const int BlockDimY = 32;
        public const int SharedMemHaloSize = 1; // Critical for K1 shared memory logic

        public static readonly int SharedStateDimX = BlockDimX + 2 * SharedMemHaloSize;
        public static readonly int SharedStateDimY = BlockDimY + 2 * SharedMemHaloSize;
    }
}