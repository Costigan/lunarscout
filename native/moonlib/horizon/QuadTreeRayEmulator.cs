using moonlib.math;
using System.Diagnostics;

namespace moonlib.horizon
{
    public class QuadTreeRayEmulator
    {
        // Use constants from QuadTreeHorizonGenerator
        private const float METERS_TO_KILOMETERS = QuadTreeHorizonGenerator.METERS_TO_KILOMETERS;
        private const float KILOMETERS_TO_METERS = QuadTreeHorizonGenerator.KILOMETERS_TO_METERS;
        private const double METERS_TO_KILOMETERS_D = QuadTreeHorizonGenerator.METERS_TO_KILOMETERS_D;
        private const double KILOMETERS_TO_METERS_D = QuadTreeHorizonGenerator.KILOMETERS_TO_METERS_D;
        private const float BEAM_WIDTH_RAD = QuadTreeHorizonGenerator.BEAM_WIDTH_RAD;
        private const float INV_TAN_MAX_SLOPE = QuadTreeHorizonGenerator.INV_TAN_MAX_SLOPE;
        private const float ANGULAR_STEP_FACTOR = QuadTreeHorizonGenerator.ANGULAR_STEP_FACTOR;

        /// <summary>
        /// Runs the quad tree ray emulator to sample elevation slopes along a ray from the given origin and azimuth.
        /// Optionally writes a CSV trace and optionally enables unifiedStepMode for debugging.
        /// unifiedStepMode converts a 1.2 meter physical step into pixel-space using map resolution, ensuring both
        /// the reference and quad tree emulators sample at equivalent physical intervals to compare pixels.
        /// Sampling stops at 1 km or when the DEM bounds are exceeded.
        /// </summary>
        /// <param name="dem">The DEM to sample.</param>
        /// <param name="origin">Pixel origin (X,Y in pixels, Z in meters).</param>
        /// <param name="azimuthDeg">Azimuth in degrees clockwise from true North.</param>
        /// <param name="outputPath">CSV output path (ignored if suppressCsv is true).</param>
        /// <param name="suppressCsv">When true, suppresses writing the CSV trace.</param>
        /// <param name="unifiedStepMode">When true, forces 1.2 meter steps (converted to pixel increments) up to 1 km or DEM edge for debugging alignment with the reference emulator.</param>
        /// <param name="startDistanceMeters">Distance along the ray to start sampling (for multi-DEM support).</param>
        /// <returns>An EmulatorResult containing slopes and trace data.</returns>
        public static EmulatorResult Run(
            ElevationMap dem,
            PixelOrigin origin,
            double azimuthDeg,
            string outputPath,
            bool suppressCsv = false,
            bool unifiedStepMode = false,
            bool logCoefficients = false,
            double startDistanceMeters = 1.0)
        {
            Debug.Assert(startDistanceMeters >= 1.0);
            var result = new EmulatorResult();
            var traceList = new List<RayTraceSample>();

            if (string.IsNullOrEmpty(outputPath))
                suppressCsv = true;

            var geo = dem.GeoTransform;
            var projD = QuadTreeHorizonGenerator.BuildProjectionParamsDouble(dem);
            
            // Observer Setup
            double obsPx = (double)origin.X;
            double obsPy = (double)origin.Y;
            double observerOffset = (double)origin.Z;

            // Get Obs Lat/Lon (Double)
            var crsPt = dem.PixelToCRS(new PixelPoint(obsPx, obsPy));
            var (obsLat, obsLon) = QuadTreeHorizonGenerator.InverseProjectDouble(crsPt.X, crsPt.Y, projD);

            // Calculate Ray Segment (CPU Side - Double)
            double az = azimuthDeg.ToRadians();
            double maxDist = 1000000.0; // 1000km
            
            double R = projD.R;
            PixelBounds bounds = new PixelBounds { Width = dem.Width, Height = dem.Height };
            
            // Compute map resolution
            double pixCol = Math.Sqrt(geo[1] * geo[1] + geo[4] * geo[4]);
            double pixRow = Math.Sqrt(geo[2] * geo[2] + geo[5] * geo[5]);
            double mapRes = (pixCol + pixRow) * 0.5;
            double demWidthM = bounds.Width * mapRes;
            double demHeightM = bounds.Height * mapRes;
            double demSizeM = Math.Min(demWidthM, demHeightM);

            double rayLimit = Math.Min(maxDist, demSizeM * 1.2);
            var mapParams = BuildMapParams(dem);

            double observerHeightMeters = SampleBilinear(dem, (float)obsPx, (float)obsPy) + observerOffset;
            var observerVec = QuadTreeHorizonGenerator.LatLonToVectorMeters(obsLat, obsLon, projD.R + observerHeightMeters);
            var meToObs = QuadTreeHorizonGenerator.GetRotationMatrixd(obsLat, obsLon);
            var obsToMe = meToObs.Inverted();  // FIX: Invert matrix to get obs->ME transform (matches Reference)
            var dirMe = QuadTreeHorizonGenerator.ComputeDirectionVector(obsToMe, az);

            Span<QuadTreeHorizonGenerator.RaySample> sampleBuffer = stackalloc QuadTreeHorizonGenerator.RaySample[QuadTreeHorizonGenerator.MAX_RAY_SAMPLE_CAPACITY];
            // Use GPU's BuildRaySamples which generates evenly-spaced samples for polynomial fitting
            int sampleCount = QuadTreeHorizonGenerator.BuildRaySamples(
                observerVec,
                dirMe,
                startDistanceMeters, rayLimit,
                dem, mapRes, sampleBuffer);
            var samples = sampleBuffer[..sampleCount];
            if (sampleCount < 3)
            {
                throw new InvalidOperationException("Unable to build ray samples for emulator.");
            }

            if (!suppressCsv)
            {
                var samplePath = Path.ChangeExtension(outputPath, ".samples.txt");
                var lines = new string[sampleCount];
                for (int i = 0; i < sampleCount; i++)
                    lines[i] = $"{samples[i].DistanceMeters * QuadTreeHorizonGenerator.METERS_TO_KILOMETERS_D:F6}:{samples[i].PixelX:F6}:{samples[i].PixelY:F6}";
                File.WriteAllLines(samplePath, lines);
                Console.WriteLine($"Samples written to {Path.GetFullPath(samplePath)}");
            }

            // Use normalized polynomial fitting to match GPU (normalize s to [0,1] range)
            double x0p = samples[0].PixelX;
            double y0p = samples[0].PixelY;
            double sAnchor = samples[0].DistanceMeters;
            int n = sampleCount;
            double span = Math.Max(0.001, samples[n - 1].DistanceMeters - sAnchor);
            float sStart = (float)(samples[0].DistanceMeters * QuadTreeHorizonGenerator.METERS_TO_KILOMETERS_D);
            float sEnd = (float)(samples[n - 1].DistanceMeters * QuadTreeHorizonGenerator.METERS_TO_KILOMETERS_D);
            
            // Build arrays relative to anchor s, normalized to [0,1] range for stability
            Span<double> sArr = stackalloc double[QuadTreeHorizonGenerator.MAX_RAY_SAMPLE_CAPACITY];
            Span<double> vx = stackalloc double[QuadTreeHorizonGenerator.MAX_RAY_SAMPLE_CAPACITY];
            Span<double> vy = stackalloc double[QuadTreeHorizonGenerator.MAX_RAY_SAMPLE_CAPACITY];
            for (int k = 0; k < n; k++)
            {
                double ds = (samples[k].DistanceMeters - sAnchor) / span;
                sArr[k] = ds;
                vx[k] = samples[k].PixelX - x0p;
                vy[k] = samples[k].PixelY - y0p;
            }

            QuadTreeHorizonGenerator.FitQuartic4TermsDouble(sArr[..n], vx[..n], out double a1d, out double a2d, out double a3d, out double a4d);
            QuadTreeHorizonGenerator.FitQuartic4TermsDouble(sArr[..n], vy[..n], out double b1d, out double b2d, out double b3d, out double b4d);
            QuadTreeHorizonGenerator.FitPlanarToChordCubicWithTerrain(samples, mapRes, observerVec, R, out double chordC1, out double chordC2, out double chordC3, logCoefficients);
            
            // Rescale coefficients back to original s-domain (matching GPU)
            double inv = 1.0 / span;
            double inv2 = inv * inv;
            double inv3 = inv2 * inv;
            double inv4 = inv2 * inv2;
            a1d *= inv;
            a2d *= inv2;
            a3d *= inv3;
            a4d *= inv4;
            b1d *= inv;
            b2d *= inv2;
            b3d *= inv3;
            b4d *= inv4;
            
            // Cast to float for Kernel Emulation
            var seg = new RaySegment
            {
                X0 = (float)x0p, Y0 = (float)y0p,
                A1 = (float)a1d, A2 = (float)a2d, A3 = (float)a3d, A4 = (float)a4d,
                B1 = (float)b1d, B2 = (float)b2d, B3 = (float)b3d, B4 = (float)b4d,
                SStart = sStart,
                SEnd = sEnd,
                SStartChord = sStart,
                PlanarToChordC1 = (float)chordC1,
                PlanarToChordC2 = (float)chordC2,
                PlanarToChordC3 = (float)chordC3
            };

            if (logCoefficients)
            {
                Console.WriteLine("QuadTreeRayEmulator coefficients:");
                Console.WriteLine($"  X0={seg.X0:F9}  Y0={seg.Y0:F9}");
                Console.WriteLine($"  A: {seg.A1:E6}, {seg.A2:E6}, {seg.A3:E6}, {seg.A4:E6}");
                Console.WriteLine($"  B: {seg.B1:E6}, {seg.B2:E6}, {seg.B3:E6}, {seg.B4:E6}");
                Console.WriteLine($"  S range: [{seg.SStart:F3}, {seg.SEnd:F3}]");
                Console.WriteLine($"  SStartChord: {seg.SStartChord:F6}km");
                Console.WriteLine($"  PlanarToChord: C1={seg.PlanarToChordC1:E9}, C2={seg.PlanarToChordC2:E9}, C3={seg.PlanarToChordC3:E9}");
            }

            // Kernel Emulation (GPU Side Logic - Float)
            // Re-build float map/proj params for kernel debug output usage
            var map = mapParams;
            var proj = BuildProjectionParams(dem);
            float mapResFloat = (float)mapRes;

            var slopes = new List<double>();
            using (var writer = suppressCsv ? null : new StreamWriter(outputPath))
            {
                if (writer != null)
                    Console.WriteLine($"Writing slopes to {Path.GetFullPath(outputPath)}");
                if (!suppressCsv)
                {
                    writer!.WriteLine("step_index,dist_m,pixel_x,pixel_y,lat_deg,lon_deg,elevation_m,slope,x_local_km,z_local_m,x_local_m");
                }

                float s = seg.SStart;  // s is in kilometers
                float currentHorizonSlope = -1e30f;
                
                // Get Observer Z (bilinear at origin)
                float obsZ = (float)observerHeightMeters;

                int stepIndex = 0;

                // For unified mode, we want the first sample to be at 1.2m, not 0.
                if (unifiedStepMode)
                {
                    // Start sampling exactly 1.2 m from origin to align with reference steps.
                    s = 0.0012f;  // 1.2 meters in kilometers
                }

                while (s <= seg.SEnd && (!unifiedStepMode || s <= 5.0f))  // 5 km limit in unified mode
                {
                    float px = EvalCubic(seg.X0, seg.A1, seg.A2, seg.A3, seg.A4, s - seg.SStart);
                    float py = EvalCubic(seg.Y0, seg.B1, seg.B2, seg.B3, seg.B4, s - seg.SStart);
                    float planarDx = (px - seg.X0) * mapResFloat;
                    float planarDy = (py - seg.Y0) * mapResFloat;
                    float planarMeters = (float)Math.Sqrt(planarDx * planarDx + planarDy * planarDy);
                    
                    // For close distances, use s directly; for far distances, use polynomial-corrected chordDist
                    // The polynomial is fit to capture curvature at larger scales and may be inaccurate at short range
                    const float CLOSE_DISTANCE_THRESHOLD_KM = 0.5f; // 500 meters
                    float trueDist;
                    if (s < CLOSE_DISTANCE_THRESHOLD_KM)
                    {
                        // Use parameterized distance directly - at short range, s ≈ true distance
                        trueDist = s * 1000.0f;
                    }
                    else
                    {
                        // Use polynomial-corrected chord distance for larger distances
                        trueDist = (seg.SStartChord * 1000.0f) + EvalPlanarChord(seg, planarMeters);
                    }

                    // Bounds check
                    if (px < 0 || py < 0 || px >= dem.Width || py >= dem.Height)
                    {
                        // Stop marching once we leave the DEM to mirror production behavior
                        break;
                    }

                    // Sample Height
                    float bilinearH = SampleBilinear(dem, px, py);
                    
                    // Match GPU calculation - use flat-earth for close distances, spherical for far
                    float R_f = (float)R;
                    float obsZ_f = (float)observerHeightMeters;
                    float bilinearH_f = bilinearH;
                    
                    float accurateSlope;
                    if (s < CLOSE_DISTANCE_THRESHOLD_KM)
                    {
                        // Flat-earth approximation for short range: slope = dH / distance
                        float dH = bilinearH_f - obsZ_f;
                        accurateSlope = (trueDist > 1e-6f) ? (dH / trueDist) : -1e30f;
                    }
                    else
                    {
                        // Exact spherical calculation using Law of Cosines for far field
                        float r_o_f = R_f + obsZ_f;
                        float s_sq_f = (float)((double)trueDist * (double)trueDist); 
                        
                        // Fix precision: Calculate (r_p - r_o) as (bilinearH - obsZ) to avoid large number cancellation
                        float z_local_bi_f_val = ((bilinearH_f - obsZ_f) * (2.0f * R_f + bilinearH_f + obsZ_f) - s_sq_f) / (2.0f * r_o_f);
                        float x_sq_f = s_sq_f - z_local_bi_f_val * z_local_bi_f_val;
                        float x_local_bi_f = (x_sq_f > 0f) ? (float)Math.Sqrt(x_sq_f) : 1e-6f;
                        accurateSlope = z_local_bi_f_val / x_local_bi_f;
                    }
                    
                    if (float.IsNaN(accurateSlope) || float.IsInfinity(accurateSlope))
                    {
                        accurateSlope = -1e30f;
                    }
                    currentHorizonSlope = Math.Max(currentHorizonSlope, accurateSlope);

                    // Calc Lat/Lon for debug (can use float version for display)
                    var (casterX, casterY) = map.PixelToCRS(px, py);
                    var (lat, lon) = InverseProject(casterX, casterY, proj);

                    // For CSV trace, compute x_local and z_local for logging
                    float z_local_m, x_local_m;
                    if (s < CLOSE_DISTANCE_THRESHOLD_KM)
                    {
                        z_local_m = bilinearH_f - obsZ_f;
                        x_local_m = trueDist;
                    }
                    else
                    {
                        float r_o_f = R_f + obsZ_f;
                        float s_sq_f = trueDist * trueDist;
                        z_local_m = ((bilinearH_f - obsZ_f) * (2.0f * R_f + bilinearH_f + obsZ_f) - s_sq_f) / (2.0f * r_o_f);
                        float x_sq_f = s_sq_f - z_local_m * z_local_m;
                        x_local_m = (x_sq_f > 0f) ? (float)Math.Sqrt(x_sq_f) : 1e-6f;
                    }

                    slopes.Add(accurateSlope);
                    
                    traceList.Add(new RayTraceSample
                    {
                        DistanceMeters = trueDist,
                        ElevationMeters = bilinearH,
                        Slope = accurateSlope,
                        PixelX = px,
                        PixelY = py
                    });
                    
                    if (!suppressCsv)
                    {
                        float xLocalKm = x_local_m / 1000.0f;
                        writer!.WriteLine($"{stepIndex},{trueDist},{px},{py},{lat * 180.0 / Math.PI},{lon * 180.0 / Math.PI},{bilinearH},{accurateSlope},{xLocalKm},{z_local_m},{x_local_m}");
                    }

                    // Step - use adaptive margin-based stepping to match GPU
                    var (dxds, dyds) = EvalCubicTangent(seg.A1, seg.A2, seg.A3, seg.A4, seg.B1, seg.B2, seg.B3, seg.B4, s - seg.SStart);
                    float mag = (float)Math.Sqrt(dxds * dxds + dyds * dyds);
                    float ds;
                    if (unifiedStepMode)
                    {
                        ds = 0.0012f;  // 1.2 meters in kilometers
                    }
                    else
                    {
                        // Match GPU adaptive stepping with margin-based acceleration
                        float dsPixel = (mag > 1e-6f) ? (1.0f / mag) : 0.001f;
                        
                        // Compute margin-based step: larger steps when well below horizon
                        float margin = currentHorizonSlope - accurateSlope;
                        float dsMargin = (margin > 0f) ? (margin * trueDist * INV_TAN_MAX_SLOPE * METERS_TO_KILOMETERS) : 0f;
                        
                        // Angular error budget cap: step proportional to distance
                        float dsAngular = trueDist * ANGULAR_STEP_FACTOR * METERS_TO_KILOMETERS;
                        
                        // Use max of pixel step and margin step, capped by angular budget
                        ds = Math.Max(dsPixel, Math.Min(dsMargin, dsAngular));
                        
                        // Increase sampling frequency 4x for close distances (under 500m)
                        if (s < CLOSE_DISTANCE_THRESHOLD_KM)
                        {
                            ds *= 0.25f;
                        }
                    }
                    
                    s += ds;
                    stepIndex++;
                }
            result.Slopes = slopes.ToArray();
            result.Trace = traceList;
            
            // Convert polynomial samples to RayTraceSample for visualization
            var polynomialSamples = new List<RayTraceSample>();
            foreach (var sample in samples)
            {
                // Sample coordinates are already in pixel space (x, y)
                // We need to compute distance and elevation for completeness
                double distMeters = sample.DistanceMeters;
                float elevation = SampleBilinear(dem, (float)sample.PixelX, (float)sample.PixelY);
                
                polynomialSamples.Add(new RayTraceSample
                {
                    DistanceMeters = distMeters,
                    ElevationMeters = elevation,
                    Slope = 0.0, // Not computed for polynomial samples
                    PixelX = sample.PixelX,
                    PixelY = sample.PixelY
                });
            }
            result.PolynomialSamples = polynomialSamples;
            
            // Populate diagnostic values
            result.ObserverLatRad = obsLat;
            result.ObserverLonRad = obsLon;
            result.DirectionX = dirMe.X;
            result.DirectionY = dirMe.Y;
            result.DirectionZ = dirMe.Z;
            
            return result;
            }
        }

        // --- Helpers ---

        static float SampleBilinear(ElevationMap dem, float col, float row)
        {
            Debug.Assert(dem != null && dem.Elevation != null);
            int w = dem.Width;
            int h = dem.Height;
            float c = (float)Math.Max(0, Math.Min(col, w - 1.0001f));
            float r = (float)Math.Max(0, Math.Min(row, h - 1.0001f));

            int x0 = (int)Math.Floor(c);
            int y0 = (int)Math.Floor(r);
            int x1 = Math.Min(x0 + 1, w - 1);
            int y1 = Math.Min(y0 + 1, h - 1);

            float tx = c - x0;
            float ty = r - y0;

            float h00 = dem.Elevation[y0, x0];
            float h10 = dem.Elevation[y0, x1];
            float h01 = dem.Elevation[y1, x0];
            float h11 = dem.Elevation[y1, x1];

            float top = h00 + tx * (h10 - h00);
            float bottom = h01 + tx * (h11 - h01);
            return top + ty * (bottom - top);
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

        static (float, float) GetLatLon(float lat1, float lon1, float az, float dist, float R)
        {
            float angDist = dist / R;
            float s1 = (float)Math.Sin(lat1);
            float c1 = (float)Math.Cos(lat1);
            float sa = (float)Math.Sin(angDist);
            float ca = (float)Math.Cos(angDist);
            float sAz = (float)Math.Sin(az);
            float cAz = (float)Math.Cos(az);

            float lat2 = (float)Math.Asin(s1 * ca + c1 * sa * cAz);
            float lon2 = lon1 + (float)Math.Atan2(sAz * sa * c1, ca - s1 * (float)Math.Sin(lat2));
            return (lat2, lon2);
        }

        static (float, float) ProjectToMap(float lat, float lon, ProjectionParams p)
        {
            float sinPhi = (float)Math.Sin(lat);
            float cosPhi = (float)Math.Cos(lat);
            float sinPhi0 = (float)Math.Sin(p.Lat0);
            float cosPhi0 = (float)Math.Cos(p.Lat0);
            float dLam = lon - p.Lon0;
            float cosDLam = (float)Math.Cos(dLam);
            float sinDLam = (float)Math.Sin(dLam);

            float denom = 1.0f + sinPhi0 * sinPhi + cosPhi0 * cosPhi * cosDLam;
            if (Math.Abs(denom) < 1e-10f) denom = 1e-10f;
            
            float k = 2.0f * p.K0 * p.R / denom;
            
            float x = k * cosPhi * sinDLam + p.FalseEasting;
            float y = k * (cosPhi0 * sinPhi - sinPhi0 * cosPhi * cosDLam) + p.FalseNorthing;
            return (x, y);
        }

        public static (float, float) InverseProject(float x, float y, ProjectionParams p)
        {
            float xp = x - p.FalseEasting;
            float yp = y - p.FalseNorthing;
            float rho = (float)Math.Sqrt(xp * xp + yp * yp);
            
            if (rho < 1e-5f) return (p.Lat0, p.Lon0);

            float c = 2.0f * (float)Math.Atan2(rho, 2.0f * p.K0 * p.R);
            float sinc = (float)Math.Sin(c);
            float cosc = (float)Math.Cos(c);
            float sinPhi0 = (float)Math.Sin(p.Lat0);
            float cosPhi0 = (float)Math.Cos(p.Lat0);

            float lat = (float)Math.Asin(cosc * sinPhi0 + (yp * sinc * cosPhi0) / rho);
            
            float term1 = xp * sinc;
            float term2 = rho * cosPhi0 * cosc - yp * sinPhi0 * sinc;
            float lon = p.Lon0 + (float)Math.Atan2(term1, term2);
            return (lat, lon);
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
                (float)geo[5]
            );
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

        struct PixelBounds { public int Width; public int Height; }

        /// <summary>
        /// Runs the quad tree ray emulator across multiple nested DEMs, starting each DEM's ray cast 
        /// at the distance where the previous DEM went out of bounds.
        /// This emulates the behavior of QuadTreeHorizonGenerator with nested DEMs.
        /// </summary>
        /// <param name="dems">List of nested DEMs, ordered from finest to coarsest resolution.</param>
        /// <param name="origin">Pixel origin in the first DEM's coordinate space (X,Y in pixels, Z in meters).</param>
        /// <param name="azimuthDeg">Azimuth in degrees clockwise from true North.</param>
        /// <param name="maxDistanceMeters">Maximum distance to cast the ray in meters.</param>
        /// <param name="suppressCsv">When true, suppresses writing CSV traces for individual DEMs.</param>
        /// <param name="unifiedStepMode">When true, forces fixed 1.2 meter steps for debugging.</param>
        /// <returns>A list of EmulatorResult objects, one per DEM that was sampled.</returns>
        public static List<EmulatorResult> RunMultiDEM(List<ElevationMap> dems, PixelOrigin origin, double azimuthDeg, double maxDistanceMeters = 1000000.0, bool suppressCsv = true, bool unifiedStepMode = false)
        {
            if (dems == null || dems.Count == 0)
                throw new ArgumentException("DEMs list cannot be null or empty", nameof(dems));

            var results = new List<EmulatorResult>();
            double currentStartDistance = 1.0; // Start at 1 meter to match GPU kernel

            // First DEM always uses the provided origin
            var firstDem = dems[0];
            var outputPath = suppressCsv ? "" : $"quadtree_trace_dem0.csv";
            var result = Run(firstDem, origin, azimuthDeg, outputPath, suppressCsv, unifiedStepMode, logCoefficients: false, currentStartDistance);
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

                outputPath = suppressCsv ? "" : $"quadtree_trace_dem{demIndex}.csv";
                result = Run(dem, demOrigin, azimuthDeg, outputPath, suppressCsv, unifiedStepMode, logCoefficients: false, currentStartDistance);
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
