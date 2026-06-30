namespace moonlib.horizon
{
    public struct RayTraceSample
    {
        public double DistanceMeters { get; set; }
        public double ElevationMeters { get; set; }
        public double Slope { get; set; }
        public double PixelX { get; set; }
        public double PixelY { get; set; }
    }

    public class EmulatorResult
    {
        public double[] Slopes { get; set; } = Array.Empty<double>();
        public List<RayTraceSample> Trace { get; set; } = new List<RayTraceSample>();
        
        /// <summary>
        /// Sample pixel locations used to generate the polynomial (QuadTree emulator only).
        /// Null for reference emulator results.
        /// </summary>
        public List<RayTraceSample>? PolynomialSamples { get; set; } = null;
        
        // Diagnostic: Observer lat/lon in radians
        public double ObserverLatRad { get; set; }
        public double ObserverLonRad { get; set; }
        
        // Diagnostic: Direction vector in ME frame (unit vector)
        public double DirectionX { get; set; }
        public double DirectionY { get; set; }
        public double DirectionZ { get; set; }

        public float ElevationDeg => (Slopes == null || Slopes.Length == 0) ? float.NaN : (float)(Math.Atan(Slopes.Max()) * 180d / Math.PI);

        /// <summary>
        /// Combines multiple EmulatorResult objects into a single result.
        /// Validates that distances are non-decreasing across all results.
        /// </summary>
        /// <param name="results">List of EmulatorResult objects to combine, ordered sequentially.</param>
        /// <returns>A single EmulatorResult containing all slopes and traces combined.</returns>
        /// <exception cref="ArgumentException">Thrown if distances decrease.</exception>
        public static EmulatorResult Combine(List<EmulatorResult> results)
        {
            if (results == null || results.Count == 0)
                return new EmulatorResult();

            if (results.Count == 1)
                return results[0];

            var combined = new EmulatorResult();
            var allSlopes = new List<double>();
            var allTraces = new List<RayTraceSample>();
            var allPolynomialSamples = new List<RayTraceSample>();
            double lastDistance = double.MinValue;

            for (int i = 0; i < results.Count; i++)
            {
                var result = results[i];
                
                // Add slopes
                allSlopes.AddRange(result.Slopes);

                // Add traces with validation
                foreach (var sample in result.Trace)
                {
                    if ((sample.DistanceMeters - lastDistance) < -1d)
                    {
                        break;
                        throw new ArgumentException(
                            $"Distance values must not decrease. " +
                            $"Found {sample.DistanceMeters} < {lastDistance} at result index {i}");
                    }
                    lastDistance = sample.DistanceMeters;
                    allTraces.Add(sample);
                }
                
                // Add polynomial samples if present
                if (result.PolynomialSamples != null)
                {
                    allPolynomialSamples.AddRange(result.PolynomialSamples);
                }
            }

            combined.Slopes = allSlopes.ToArray();
            combined.Trace = allTraces;
            combined.PolynomialSamples = allPolynomialSamples.Count > 0 ? allPolynomialSamples : null;
            return combined;
        }
    }
}
