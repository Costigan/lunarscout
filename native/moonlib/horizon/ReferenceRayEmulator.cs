using ILGPU.IR.Values;
using moonlib.math;

namespace moonlib.horizon
{
    public class ReferenceRayEmulator
    {
        public const double MoonRadius = 1737.4d;

        /// <summary>
        /// Runs the reference ray emulator to sample elevation slopes along a ray from the given origin and azimuth.
        /// Optionally writes a CSV trace and optionally enables unifiedStepMode for debugging.
        /// </summary>
        /// <param name="dem">The DEM to sample.</param>
        /// <param name="origin">Pixel origin (X,Y in pixels, Z in meters).</param>
        /// <param name="azimuthDeg">Azimuth in degrees clockwise from true North.</param>
        /// <param name="outputPath">CSV output path (ignored if suppressCsv is true).</param>
        /// <param name="suppressCsv">When true, suppresses writing the CSV trace.</param>
        /// <param name="unifiedStepMode">When true, forces fixed 1.2 meter steps up to 1 km or DEM edge to align samples with the QuadTree emulator for debugging.</param>
        /// <returns>An EmulatorResult containing slopes and trace data.</returns>
        public static EmulatorResult Run(ElevationMap dem, PixelOrigin origin, double azimuthDeg, string outputPath, bool suppressCsv = false, bool unifiedStepMode = false, double maxDistanceMeters = 1000000.0, double startDistanceMeters = 1.0)
        {
            var result = new EmulatorResult();
            var traceList = new List<RayTraceSample>();
            
            var (pixel_x, pixel_y, observer_height_m) = ((double)origin.X, (double)origin.Y, origin.Z);
            
            // Calculate Lat/Lon from Pixel
            var (observer_lat_deg, observer_lon_deg) = dem.Point2LatLonDeg(pixel_x, pixel_y);
            var (obs_lat_rad, obs_lon_rad) = (observer_lat_deg.ToRadians(), observer_lon_deg.ToRadians());

            // Get Observer Vector
            var observer_me_km = GetObserverVector3d(dem, pixel_y, pixel_x, obs_lat_rad, obs_lon_rad, observer_height_m);
            var me_to_obs_mat = GetRotationMatrixd(obs_lat_rad, obs_lon_rad);
            var obs_to_me_mat = me_to_obs_mat.Inverted();

            var max_range_m = maxDistanceMeters;
            double[] steps_km;
            if (unifiedStepMode)
            {
                // Generate 1.2 meter steps up to 5 km in kilometers
                int stepCount = (int)Math.Floor(Math.Min(5000d, maxDistanceMeters) / 1.2d);
                steps_km = Enumerable.Range(1, stepCount).Select(i => (i * 1.2d) / 1000d).ToArray();
            }
            else
            {
                var geo = dem.GeoTransform;
                var resX = Math.Sqrt(geo[1] * geo[1] + geo[4] * geo[4]);
                var resY = Math.Sqrt(geo[2] * geo[2] + geo[5] * geo[5]);
                var min_step_m = Math.Min(resX, resY) * 0.1; // Use 0.1 pixel step to match GPU high-res sampling
                
                steps_km = EnumerateErrorBoundSteps(max_range_m, min_step_m, startDistanceMeters).ToArray();
            }

            // Oversample to match ReferenceHorizonGenerator behavior (3 samples per 0.25 deg bin)
            // We assume azimuthDeg is the bin start. We test 3 offsets.
            // But ReferenceRayEmulator might be called with arbitrary azimuth.
            // We will test azimuthDeg, azimuthDeg + 0.08333, azimuthDeg + 0.16667
            
            double globalMaxSlope = double.MinValue;
            List<double> bestSlopes = null;
            List<RayTraceSample> bestTrace = null;

            double[] offsets = new double[] { 0.0, 0.0833333333333333, 0.1666666666666667 };
            
            // Capture direction for base azimuth (offset=0) for diagnostics
            Vector3d baseDir = default;

            foreach (var offset in offsets)
            {
                var currentAz = azimuthDeg + offset;
                var currentSlopes = new List<double>();
                var currentTrace = new List<RayTraceSample>();
                double currentMaxSlope = double.MinValue;

                // Calculate direction for specific azimuth
                var theta_true = currentAz.ToRadians();
                var angle = (Math.PI / 2d) - theta_true;
                var dir_obs_frame = new Vector3d(Math.Cos(angle), Math.Sin(angle), 0d);
                var dir_me_frame = Vector3d.Transform(dir_obs_frame, obs_to_me_mat);
                
                // Capture base direction for diagnostics
                if (offset == 0.0)
                    baseDir = dir_me_frame;

                // Use full raster extents; GetElevation already clamps to the edge when sampling
                var (dem_width, dem_height) = (dem.Width, dem.Height);
                
                for (var j = 0; j < steps_km.Length; j++)
                {
                    var d = steps_km[j];
                    var walker_me_km = observer_me_km + (dir_me_frame * d);
                    var (caster_lat_rad, caster_lon_rad) = VecME2LatLon(walker_me_km);
                    var (caster_lat_deg, caster_lon_deg) = (caster_lat_rad.ToDegrees(), caster_lon_rad.ToDegrees());
                    var (row, col) = dem.LonLatDeg2RowCol(caster_lon_deg, caster_lat_deg);

                    // Bounds check: stop marching once we leave the DEM
                    if (!(row >= 0 && row < dem_height && col >= 0 && col < dem_width))
                    {
                        // Only log OOB for the base ray (offset 0) to avoid noise, or if it's the best ray?
                        // For now suppress OOB logs for oversampling to keep output clean, or log only if best?
                        // System.Diagnostics.Debug.WriteLine($"OOB at step {j}...");
                        break;
                    }

                    // Calculate position of shadow caster in MOON_ME
                    var caster_elevation_m = dem.GetElevation(col, row);
                    var caster_radius_km = MoonRadius + (caster_elevation_m / 1000d);
                    var caster_z_me_km = caster_radius_km * Math.Sin(caster_lat_rad);
                    var caster_temp_km = caster_radius_km * Math.Cos(caster_lat_rad);
                    var caster_x_me_km = caster_temp_km * Math.Cos(caster_lon_rad);
                    var caster_y_me_km = caster_temp_km * Math.Sin(caster_lon_rad);
                    var caster_me_km = new Vector3d(caster_x_me_km, caster_y_me_km, caster_z_me_km);

                    // Convert to observer frame
                    var obs_to_caster_me_km = caster_me_km - observer_me_km;
                    var caster_obs_km = Vector3d.Transform(obs_to_caster_me_km, me_to_obs_mat);

                    var x = caster_obs_km.X;
                    var y = caster_obs_km.Y;
                    var z = caster_obs_km.Z;

                    var alen = Math.Sqrt((x * x) + (y * y));
                    var new_slope = z / alen;

                    if (new_slope > currentMaxSlope)
                    {
                        currentMaxSlope = new_slope;
                    }
                    currentSlopes.Add(new_slope);
                    
                    // Add to trace
                    currentTrace.Add(new RayTraceSample
                    {
                        DistanceMeters = d * 1000d,
                        ElevationMeters = caster_elevation_m,
                        Slope = new_slope,
                        PixelX = col,
                        PixelY = row
                    });
                }

                if (currentMaxSlope > globalMaxSlope || bestTrace == null)
                {
                    globalMaxSlope = currentMaxSlope;
                    bestSlopes = currentSlopes;
                    bestTrace = currentTrace;
                }
            }

            // Write CSV for the best trace
            if (!suppressCsv && bestTrace != null)
            {
                Console.WriteLine($"Writing to {Path.GetFullPath(outputPath)}");
                using (var writer = new StreamWriter(outputPath))
                {
                    writer!.WriteLine("step_index,PixelX,PixelY,DistanceMeters,ElevationMeters,Slope,Angle_deg");
                    // We need to re-calculate some values for CSV or store them in RayTraceSample extended?
                    // RayTraceSample is limited.
                    // For simplicity, just dump what we have or re-run the best ray?
                    // Re-running is expensive. Let's just output basic info or skipping detailed debug columns for now.
                    // Or we can store the debug strings in a list?
                    // The original code wrote inside the loop.
                    // Let's just skip CSV writing for oversampled emulator to save time/complexity, 
                    // or assume the caller doesn't need the detailed debug columns if they are suppressCsv=true usually.
                    // The test calls with suppressCsv=true.
                    for (var i = 0; i < bestTrace.Count; i++)
                    {
                        var t = bestTrace[i];
                        var angle_deg = (float)Math.Atan(t.Slope) * 180d / 3.14159265358979f;
                        writer.WriteLine($"{i},{t.PixelX},{t.PixelY},{t.DistanceMeters},{t.ElevationMeters},{t.Slope},{angle_deg}");
                    }
                }
            }

            System.Diagnostics.Debug.Assert(bestSlopes != null && bestTrace != null);
            result.Slopes = bestSlopes.ToArray();
            result.Trace = bestTrace;
            
            // Populate diagnostic values
            result.ObserverLatRad = obs_lat_rad;
            result.ObserverLonRad = obs_lon_rad;
            result.DirectionX = baseDir.X;
            result.DirectionY = baseDir.Y;
            result.DirectionZ = baseDir.Z;
            
            return result;

        /* Original single-ray implementation replaced by oversampling above
        static IEnumerable<double> EnumerateErrorBoundSteps... (kept)
        */
        }

        static IEnumerable<double> EnumerateErrorBoundSteps(double max_distance, double min_step_m, double start_dist_m)
        {
             // Replicating ReferenceHorizonGenerator.EnumerateErrorBoundSteps logic
             // horizon_angular_error_budget_deg = 0.01d
             double error_deg = 0.01d;
             var error_rad = error_deg.ToRadians();
             var d = start_dist_m;

             while (d <= max_distance)
             {
                 yield return d / 1000d;
                 var step = Math.Max(min_step_m, d * error_rad);
                 d += step;
             }
        }

        static Vector3d GetObserverVector3d(ElevationMap terrain, double row, double col, double lat_rad, double lon_rad, double observer_m)
        {
            var elevation_m = terrain.GetElevation(col, row) + observer_m;
            var radius_km = MoonRadius + (elevation_m / 1000d);
            var z = radius_km * Math.Sin(lat_rad);
            var c = radius_km * Math.Cos(lat_rad);
            var x = c * Math.Cos(lon_rad);
            var y = c * Math.Sin(lon_rad);
            return new Vector3d(x, y, z);
        }

        static Matrix4d GetRotationMatrixd(double lat_rad, double lon_rad)
        {
            var zaxis = new Vector3d(0d, 0d, 1d);
            var yaxis = new Vector3d(0d, 1d, 0d);
            var mat1 = Matrix4d.CreateFromAxisAngle(zaxis, -lon_rad);
            var mat2 = Matrix4d.CreateFromAxisAngle(yaxis, -((Math.PI / 2d) - lat_rad));
            var fixEnu = Matrix4d.CreateFromAxisAngle(zaxis, -Math.PI / 2d);
            var mat = mat1 * mat2 * fixEnu;
            return mat;
        }

        static (double lat_rad, double lon_rad) VecME2LatLon(Vector3d vec)
        {
            var lon_rad = Math.Atan2(vec.Y, vec.X);
            if (lon_rad < 0d) lon_rad += Math.PI * 2d;
            var alen = Math.Sqrt((vec.X * vec.X) + (vec.Y * vec.Y));
            var lat_rad = Math.Atan2(vec.Z, alen);
            return (lat_rad, lon_rad);
        }

        /// <summary>
        /// Runs the reference ray emulator across multiple nested DEMs, starting each DEM's ray cast 
        /// at the distance where the previous DEM went out of bounds.
        /// This emulates the behavior of ReferenceHorizonGenerator with nested DEMs.
        /// </summary>
        /// <param name="dems">List of nested DEMs, ordered from finest to coarsest resolution.</param>
        /// <param name="origin">Pixel origin in the first DEM's coordinate space (X,Y in pixels, Z in meters).</param>
        /// <param name="azimuthDeg">Azimuth in degrees clockwise from true North.</param>
        /// <param name="maxDistanceMeters">Maximum distance to cast the ray in meters.</param>
        /// <param name="suppressCsv">When true, suppresses writing CSV traces for individual DEMs.</param>
        /// <param name="unifiedStepMode">When true, forces fixed 1.2 meter steps for debugging.</param>
        /// <returns>A list of EmulatorResult objects, one per DEM that was sampled.</returns>
        public static List<EmulatorResult> RunMultiDem(List<ElevationMap> dems, PixelOrigin origin, double azimuthDeg, double maxDistanceMeters = 1000000.0, bool suppressCsv = true, bool unifiedStepMode = false)
        {
            if (dems == null || dems.Count == 0)
                throw new ArgumentException("DEMs list cannot be null or empty", nameof(dems));

            var results = new List<EmulatorResult>();
            double currentStartDistance = 1.0; // Start at 1 meter to match GPU kernel

            // First DEM always uses the provided origin
            var firstDem = dems[0];
            var outputPath = suppressCsv ? "" : $"reference_trace_dem0.csv";
            var result = Run(firstDem, origin, azimuthDeg, outputPath, suppressCsv, unifiedStepMode, maxDistanceMeters, currentStartDistance);
            results.Add(result);

            // If we have samples, determine where this DEM stopped
            if (result.Trace.Count > 0)
            {
                var lastSample = result.Trace[result.Trace.Count - 1];
                currentStartDistance = lastSample.DistanceMeters;
            }

            // Process remaining DEMs
            for (int demIndex = 1; demIndex < dems.Count; demIndex++)
            {
                var dem = dems[demIndex];
                
                // Check if we've exceeded max distance
                if (currentStartDistance >= maxDistanceMeters)
                    break;

                // For subsequent DEMs, we need to transform the origin's lat/lon to the new DEM's pixel coordinates
                var firstDemOrigin = new PixelOrigin { X = origin.X, Y = origin.Y, Z = origin.Z };
                var (obsLat, obsLon) = firstDem.Point2LatLonDeg(firstDemOrigin.X, firstDemOrigin.Y);
                var (newRow, newCol) = dem.LonLatDeg2RowCol(obsLon, obsLat);
                
                // Create origin in this DEM's coordinate space
                var demOrigin = new PixelOrigin 
                { 
                    X = (float)newCol, 
                    Y = (float)newRow, 
                    Z = origin.Z 
                };

                outputPath = suppressCsv ? "" : $"reference_trace_dem{demIndex}.csv";
                result = Run(dem, demOrigin, azimuthDeg, outputPath, suppressCsv, unifiedStepMode, maxDistanceMeters, currentStartDistance);
                results.Add(result);

                // Update starting distance for next DEM
                if (result.Trace.Count > 0)
                {
                    var lastSample = result.Trace[result.Trace.Count - 1];
                    currentStartDistance = lastSample.DistanceMeters;
                }
            }

            return results;
        }
    }
}
