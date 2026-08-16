using System.Runtime.InteropServices;
using ILGPU.Algorithms; // For XMath

namespace GridRunner
{
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct StateTuple
    {
        /// <summary>
        /// Total charge contained by the battery and the fuel cell together.
        /// Unoccupied cells are initialized with 0.
        /// A full charge is BatteryCapacityWh  + FuelCellCapacityWh.
        /// </summary>
        public float TotalWh;

        /// <summary>
        /// Time in hours from an epoch (e.g., start of the simulation).  Its initial value
        /// for unoccupied cells is float.MaxValue.  The start cell is initialized with 0f.
        /// </summary>
        public float ArrivalTime;

        public float BatteryWh;
        public short MinFuelCellWhScaledBy20;
        public byte PreviousCell;
        public byte Padding; // Ensure struct size is multiple of 4 for some alignment benefits if any

        public override string ToString() => $"<state t={TotalWh:F2} a={ArrivalTime} b={BatteryWh:F2} p={PreviousCell}>";

        public bool IsUnoccupied() => TotalWh == RoverConstants.UnoccupiedTotalWh;

        public static StateTuple Unoccupied() => new StateTuple
        {
            TotalWh = RoverConstants.UnoccupiedTotalWh,
            ArrivalTime = RoverConstants.UnoccupiedArrivalTime, // Also NegativeInfinity
            BatteryWh = 0.0f,
            MinFuelCellWhScaledBy20 = 0, // Or short.MinValue if more appropriate for unoccupied
            PreviousCell = RoverConstants.PreviousCellUnoccupied,
            Padding = 0
        };

        public static StateTuple StartState() => new StateTuple
        {
            TotalWh = RoverConstants.InitialTotalWh,
            ArrivalTime = RoverConstants.InitialTime,
            BatteryWh = RoverConstants.BatteryCapacityWh, // Start with usable part of battery full
            MinFuelCellWhScaledBy20 = (short)XMath.Floor(RoverConstants.FuelCellCapacityWh / 20.0f),
            PreviousCell = RoverConstants.PreviousCellChainTerminator,
            Padding = 0
        };
    }
}
