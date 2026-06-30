using moonlib.math;
using System.Diagnostics;
using System.Drawing;

namespace moonlib.horizon
{
    /// <summary>
    /// Generate and draw a viewshed
    /// </summary>
    public class ReferenceHorizonGenerator
    {
        public const double MoonRadius = 1737.4d;
        public const int HorizonSamples = 360 * 4;
        public const double HorizonSamplesD = HorizonSamples;
        public const int NearHorizonOversample = 3;
        public const int RayCastDistanceInPixels = 230;  // Ceiling[1/Tan[.25 deg]]
        public const float RayCastDistanceInPixelsF = RayCastDistanceInPixels;
        public const float NearFieldRayStep = 0.70710698f;
        public const float PI = 3.14159265358979f;

        public const double horizon_angular_error_budget_deg = 0.01d;

        public static ReferenceHorizonGenerator Singleton => _singleton ?? (_singleton = new ReferenceHorizonGenerator());
        public static ReferenceHorizonGenerator? _singleton;

        public static string[] DEM_names = new string[]
        {
                "/d/datasets/viper_v71_2024_medium/other/dem.tif",
                "/d/viper/maps/gsfc/site_20v2/Site20v2_final_adj_5mpp_surf.tif",
                //"/d/viper/maps/lola/LDEM_80S_20M-2017-06-15-processed.tif"

            //"/d/viper/maps/viper_v71_small/viper_v71_small_dem.tif",
            //"/d/viper/maps/viper_v71/viper_sfs_dem.tif",
            //"/d/viper/maps/gsfc/site_20v2/Site20v2_final_adj_5mpp_surf.tif",
            //"/d/viper/maps/lola/LDEM_80S_20M-2017-06-15-processed.tif",
        };

        public static List<ElevationMap>? DEMs = null;

        public ViewshedEntry[]? Entries;

        // {X = 2553 Y = 952}
        public ViewshedHorizon Generate() => Generate(new PixelOrigin { X = 2553, Y = 952, Z = 2f });

        public ViewshedHorizon Generate(PixelOrigin origin)
        {
            lock (this)
            {
                var dems = LoadDEMs();
                // Use precise pixel coordinates for better accuracy and alignment with QuadTreeGenerator
                var origin2 = new PixelOrigin { X = origin.X, Y = origin.Y, Z = origin.Z };
                return GenerateFromPixel(origin2, dems);
            }
        }

        public static List<ElevationMap> LoadDEMs()
        {
            lock (Singleton)
                return DEMs = DEM_names.Select(fn => GetTerrain(fn)).ToList();
        }

        // Renamed from Generate to allow overload with double pixels
        public ViewshedHorizon GenerateFromLatLon(LatLonOrigin origin, List<ElevationMap>? dems = null)
        {
            var (observer_lat_deg, observer_lon_deg, observer_height_m) = (origin.Latitude, origin.Longitude, origin.Z);
            if (dems == null)
                dems = DEM_names.Select(fn => GetTerrain(fn)).ToList();
            Debug.Assert(dems.Count > 0);
            var target_dem = dems[0];
            var (line, sample) = target_dem.LonLatDeg2RowCol(observer_lon_deg, observer_lat_deg);
            // line=row=y, sample=col=x
            // Pass directly to pixel generator
            var pixel_origin = new PixelOrigin { X = (float)sample, Y = (float)line, Z = observer_height_m };
            return GenerateFromPixel(pixel_origin, dems);
        }

        // New primary entry point taking precise pixel coordinates (x=col, y=row)
        public ViewshedHorizon GenerateFromPixel(PixelOrigin originPoint, List<ElevationMap>? dems = null)
        {
            var demList = dems ?? LoadDEMs();
            Debug.Assert(demList.Count > 0);

            var entries = new ViewshedEntry[HorizonSamples];
            var radians = ComputeHorizonSamples(originPoint, demList, 1000d * 1000d, entries);

            var elevationsDeg = new float[HorizonSamples];
            for (int i = 0; i < HorizonSamples; i++)
                elevationsDeg[i] = radians[i].ToDegrees();

            return new ViewshedHorizon
            {
                Elevations = elevationsDeg,
                Entries = entries
            };
        }

        /// <summary>
        /// Computes a horizon using the reference algorithm but clamps the maximum range.
        /// The returned angles are in radians to simplify blending with GPU results.
        /// </summary>
        public float[] ComputeLimitedHorizon(PixelOrigin originPoint, List<ElevationMap>? dems = null, double maxRangeMeters = 50.0, bool useParallel = true)
        {
            if (maxRangeMeters <= 0)
                throw new ArgumentOutOfRangeException(nameof(maxRangeMeters), "Max range must be positive.");
            var demList = dems ?? LoadDEMs();
            Debug.Assert(demList.Count > 0);
            return ComputeHorizonSamples(originPoint, demList, maxRangeMeters, null, useParallel);
        }

        private float[] ComputeHorizonSamples(PixelOrigin originPoint, List<ElevationMap> dems, double maxRangeMeters, ViewshedEntry[]? entriesOut, bool useParallel = true)
        {
            var (pixel_x, pixel_y, observer_height_m) = ((double)originPoint.X, (double)originPoint.Y, originPoint.Z);
            var target_dem = dems[0];

            // Calculate Lat/Lon from Pixel (needed for rotation matrix)
            var (observer_lat_deg, observer_lon_deg) = target_dem.Point2LatLonDeg(pixel_x, pixel_y);
            var (obs_lat_rad, obs_lon_rad) = (observer_lat_deg.ToRadians(), observer_lon_deg.ToRadians());

            // Use Pixel coordinates directly for 3D vector
            var observer_me_km = GetObserverVector3d(target_dem, pixel_y, pixel_x, obs_lat_rad, obs_lon_rad, observer_height_m);
            var me_to_obs_mat = GetRotationMatrixd(obs_lat_rad, obs_lon_rad);
            var obs_to_me_mat = me_to_obs_mat.Inverted();

            // Calculate min_step from Inner DEM (target_dem) resolution
            var geo = target_dem.GeoTransform;
            var resX = Math.Sqrt(geo[1] * geo[1] + geo[4] * geo[4]);
            var resY = Math.Sqrt(geo[2] * geo[2] + geo[5] * geo[5]);
            var min_step_m = Math.Min(resX, resY);

            var steps_km = EnumerateErrorBoundSteps(maxRangeMeters, horizon_angular_error_budget_deg, min_step_m).ToArray();
            int oversample_for_error = CalculateOversampleForError(maxRangeMeters, horizon_angular_error_budget_deg);
            var oversample = Math.Max(NearHorizonOversample, oversample_for_error);

            var radians = new float[HorizonSamples];
            bool trackEntries = entriesOut != null;

            Action<int> calc = i =>
            {
                int entry_dem_id = -1;
                double entry_range_km = -1d, entry_row = -1d, entry_col = -1d, entry_lat_deg = -1d, entry_lon_deg = -1d;
                double max_slope = double.MinValue;

                for (var oversample_index = 0; oversample_index < oversample; oversample_index++)
                {
                    var theta_true = 2d * Math.PI * ((i / HorizonSamplesD) + (oversample_index / (HorizonSamplesD * oversample)));
                    var angle = (Math.PI / 2d) - theta_true;
                    var dir_obs_frame = new Vector3d(Math.Cos(angle), Math.Sin(angle), 0d);
                    var dir_me_frame = Vector3d.Transform(dir_obs_frame, obs_to_me_mat);

                    var dem_index = 0;
                    var dem = dems[dem_index];
                    var (dem_width, dem_height) = (dem.Width, dem.Height);

                    for (var j = 0; j < steps_km.Length; j++)
                    {
                        var d = steps_km[j];
                        var walker_me_km = observer_me_km + (dir_me_frame * d);
                        var (caster_lat_rad, caster_lon_rad) = VecME2LatLon(walker_me_km);
                        var (caster_lat_deg, caster_lon_deg) = (caster_lat_rad.ToDegrees(), caster_lon_rad.ToDegrees());
                        var (row, col) = dem.LonLatDeg2RowCol(caster_lon_deg, caster_lat_deg);

                        while (!(row >= 0 && row < dem_height && col >= 0 && col < dem_width))
                        {
                            dem = ++dem_index < dems.Count ? dems[dem_index] : null;
                            if (dem == null)
                                goto finished_walking;
                            (dem_width, dem_height) = (dem.Width, dem.Height);
                            (row, col) = dem.LonLatDeg2RowCol(caster_lon_deg, caster_lat_deg);
                        }

                        if (!(row >= 0 && row < dem_height && col >= 0 && col < dem_width))
                            continue;

                        var caster_elevation_m = dem.GetElevation(col, row);
                        var caster_radius_km = MoonRadius + (caster_elevation_m / 1000d);
                        var caster_z_me_km = caster_radius_km * Math.Sin(caster_lat_rad);
                        var caster_temp_km = caster_radius_km * Math.Cos(caster_lat_rad);
                        var caster_x_me_km = caster_temp_km * Math.Cos(caster_lon_rad);
                        var caster_y_me_km = caster_temp_km * Math.Sin(caster_lon_rad);
                        var caster_me_km = new Vector3d(caster_x_me_km, caster_y_me_km, caster_z_me_km);

                        var obs_to_caster_me_km = caster_me_km - observer_me_km;
                        var caster_obs_km = Vector3d.Transform(obs_to_caster_me_km, me_to_obs_mat);

                        var x = caster_obs_km.X;
                        var y = caster_obs_km.Y;
                        var z = caster_obs_km.Z;

                        var alen = Math.Sqrt((x * x) + (y * y));
                        var new_slope = z / alen;

                        if (new_slope > max_slope)
                        {
                            max_slope = new_slope;
                            if (trackEntries)
                            {
                                entry_dem_id = dem_index;
                                entry_range_km = caster_obs_km.Length;
                                entry_row = row;
                                entry_col = col;
                                entry_lat_deg = caster_lat_deg;
                                entry_lon_deg = caster_lon_deg;
                            }
                        }
                    }

finished_walking:
                    { }
                }

                if (trackEntries)
                {
                    entriesOut![i] = new ViewshedEntry
                    {
                        DEM_id = entry_dem_id,
                        Range_m = (float)(entry_range_km * 1000d),
                        Point = new Point(entry_col.ToFixed(), entry_row.ToFixed()),
                        Latitude_deg = entry_lat_deg,
                        Longitude_deg = entry_lon_deg
                    };
                }

                radians[i] = (float)Math.Atan(max_slope);
            };

            if (useParallel)
            {
                Parallel.For(0, HorizonSamples, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount * 2 }, calc);
            }
            else
            {
                for (int i = 0; i < HorizonSamples; i++)
                    calc(i);
            }

            return radians;
        }

        static IEnumerable<double> EnumerateSteps(double max_distance, double factor = 2d)
        {
            var (start, step, stop) = (2d, 1d, 32d);

            while (start <= max_distance)
            {
                var stop1 = Math.Min(stop, max_distance);
                for (var d = start; d < stop1; d += step)
                    yield return d / 1000d;

                start = stop;
                step *= factor;
                stop *= factor;
            }
        }

        static IEnumerable<double> EnumerateErrorBoundSteps(
            double max_distance,
            double error_deg,
            double min_step_m)
        {
            var error_rad = error_deg.ToRadians();
            var d = min_step_m; // Start at min_step_m (usually ~1m) to align with pixel-based traversal

            while (d <= max_distance)
            {
                yield return d / 1000d;
                // Step is geometric growth relative to distance, clamped to minimum resolution
                var step = Math.Max(min_step_m, d * error_rad);
                d += step;
            }
        }

        static int CalculateOversampleForError(double range_m, double error_budget_deg) =>
            (int)Math.Ceiling((360d / error_budget_deg) / HorizonSamplesD);

        public static int ConvertHorizonIndexToQuadTreeIndex(int i)
        {
            return i;
        }

        (double lat_rad, double lon_rad) VecME2LatLon(Vector3d vec)
        {
            var lon_rad = Math.Atan2(vec.Y, vec.X);  // [0,2PI]
            if (lon_rad < 0d) lon_rad += Math.PI * 2d;
            var alen = Math.Sqrt((vec.X * vec.X) + (vec.Y * vec.Y));
            var lat_rad = Math.Atan2(vec.Z, alen);
            return (lat_rad, lon_rad);
        }

        (double lat_deg, double lon_deg) VecME2LatLonDeg(Vector3d vec)
        {
            var (lat_rad, lon_rad) = VecME2LatLon(vec);
            return (lat_rad.ToDegrees(), lon_rad.ToDegrees());
        }

        Matrix4d GetRotationMatrixd(double lat_rad, double lon_rad)
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

        Matrix4 GetRotationMatrix4(double lat_rad, double lon_rad)
        {
            // Make the rotation
            var zaxis = new Vector3(0f, 0f, 1f);
            var yaxis = new Vector3(0f, 1f, 0f);
            var mat1 = Matrix4.CreateFromAxisAngle(zaxis, -(float)lon_rad);
            var mat2 = Matrix4.CreateFromAxisAngle(yaxis, -((PI / 2f) - (float)lat_rad));
            var fixEnu = Matrix4.CreateFromAxisAngle(zaxis, -PI / 2f);
            var mat = mat1 * mat2 * fixEnu;
            return mat;
        }

        // The purpose of repeating this is to avoid converting from row,col to lat,lon again
        Vector3d GetVector3d(ElevationMap terrain, int row, int col, double lat_rad, double lon_rad)
        {
            var elevation_m = terrain.GetElevation(col, row);
            var radius_km = MoonRadius + (elevation_m / 1000d);
            var z = radius_km * Math.Sin(lat_rad);
            var c = radius_km * Math.Cos(lat_rad);
            var x = c * Math.Cos(lon_rad);
            var y = c * Math.Sin(lon_rad);
            return new Vector3d(x, y, z);
        }

        Vector3d GetObserverVector3d(ElevationMap terrain, int row, int col, double lat_rad, double lon_rad, double observer_m)
        {
            var elevation_m = terrain.GetElevation(col, row) + observer_m;
            var radius_km = MoonRadius + (elevation_m / 1000d);
            var z = radius_km * Math.Sin(lat_rad);
            var c = radius_km * Math.Cos(lat_rad);
            var x = c * Math.Cos(lon_rad);
            var y = c * Math.Sin(lon_rad);
            return new Vector3d(x, y, z);
        }

        Vector3d GetObserverVector3d(ElevationMap terrain, double row, double col, double lat_rad, double lon_rad, double observer_m)
        {
            var elevation_m = terrain.GetElevation(col, row) + observer_m;
            var radius_km = MoonRadius + (elevation_m / 1000d);
            var z = radius_km * Math.Sin(lat_rad);
            var c = radius_km * Math.Cos(lat_rad);
            var x = c * Math.Cos(lon_rad);
            var y = c * Math.Sin(lon_rad);
            return new Vector3d(x, y, z);
        }

        #region Utilities

        static ElevationMap GetTerrain(string path)
        {
            if (string.IsNullOrEmpty(path))
                throw new ArgumentException("Path cannot be null or empty", nameof(path));
            return new ElevationMap(path);
        }

        #endregion
    }

    public struct ViewshedEntry
    {
        public float Range_m;
        public int DEM_id;
        public Point Point;
        public double Latitude_deg;
        public double Longitude_deg;
    }

    public class ViewshedHorizon
    {
        public ViewshedEntry[] Entries;
        public float[] Elevations = new float[ReferenceHorizonGenerator.HorizonSamples];

        public ViewshedHorizon()
        {
            Entries = new ViewshedEntry[ReferenceHorizonGenerator.HorizonSamples];
            Elevations = new float[ReferenceHorizonGenerator.HorizonSamples];
            Clear();
        }

        public void Clear()
        {
            Array.Clear(Entries, 0, Entries.Length);
            for (int i = 0; i < Elevations.Length; i++)
                Elevations[i] = float.MinValue;
        }

        public float MaximumRange_m => (Entries == null || Entries.Length < 1) ? float.NaN : Entries.Max(e => e.Range_m);

        public int CountElevations(float threshold = float.MinValue) => Elevations.Count(e => e >= threshold);

        public void Merge(ViewshedHorizon other)
        {
            var (other_elevations, other_entries) = (other.Elevations, other.Entries);
            Debug.Assert(other_elevations != null && other_elevations.Length == Elevations.Length);
            Debug.Assert(other_entries != null && other_entries.Length == other_elevations.Length);

            for (var i = 0; i < Elevations.Length; i++)
                if (other_elevations[i] > Elevations[i])
                    Entries[i] = other_entries[i];
        }

        public void Write(string path)
        {
            using (var sw = new StreamWriter(path))
            {
                sw.WriteLine("azimuth,elevation,range,lat_deg,lon_deg,dem_id,x,y");
                for (var i = 0; i < Elevations.Length; i++)
                {
                    var e = Entries[i];
                    sw.WriteLine($"{i * 360f / Elevations.Length},{Elevations[i]},{e.Range_m},{e.Latitude_deg},{e.Longitude_deg},{e.DEM_id},{e.Point.X},{e.Point.Y}");
                }
            }
        }
    }
}
