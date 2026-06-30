using moonlib.math;
using moonlib.util;
using Serilog;
using System.Diagnostics;

#nullable disable

namespace moonlib.spice
{
    /// <summary>
    /// Wraps spice and doesn't load it if running in 32-bit mode.
    /// </summary>
    public class SpiceManager
    {
        public static bool Verbose = false;
        protected static SpiceManager _singleton;
        protected readonly SpiceMethods Methods;

        public SpiceManager()
        {
            Methods = Environment.Is64BitProcess && Environment.Is64BitOperatingSystem ? new FullSpiceMethods() : (SpiceMethods)new DummySpiceMethods();

            // Initialize the epoch

            double et = 0;
            CSpice.str2et_c("2023 December 1, 00:00:00 UTC", ref et);
            SpiceMethods.LocalEpoch = ViperDate.New(2023, 12, 1, 0, 0, 0, 0);
            SpiceMethods.LocalEpochEpochTime = et;
            _singleton = this;
        }

        public static long GetUnixTimestamp(DateTime date)
        {
            Debug.Assert(date.Kind == DateTimeKind.Utc);
            var str = date.ToString("YYYY MM D, hh:mm:ss UTC");     // to this format: "2023 December 1, 00:00:00 UTC"
            double et_date = 0d;
            CSpice.str2et_c("2023 December 1, 00:00:00 UTC", ref et_date);
            double et_unix_epoch = 0d;
            CSpice.str2et_c("1970 January 1, 00:00:00 UTC", ref et_unix_epoch);
            return (long)(et_date - et_unix_epoch);
        }

        public static bool IsLoaded => _singleton != null;

        public static SpiceManager Singleton => _singleton != null ? _singleton : _singleton = new SpiceManager();

        public void SunEarthAzEl(DateTime time, out double sunAzimuth, out double sunElevation, out double earthAzimuth, out double earthElevation)
        {
            Methods.SunEarthAzEl(time, out sunAzimuth, out sunElevation, out earthAzimuth, out earthElevation);
        }

        public void DSNAzEl(DateTime time, double lat, double lon, double height, out double dsnaz, out double dsnel)
        {
            Methods.DSNAzEl(time, lat, lon, height, out dsnaz, out dsnel);
        }

        public List<TimeInterval> GenerateStationPasses(string complex, bool north_pole, DateTime start, DateTime stop, TimeSpan? defaultStep = null) =>
            Methods.GenerateStationPasses(complex, north_pole, start, stop, defaultStep);

        public List<TimeInterval> GeneratePreciseStationPasses(string complex, bool north_pole, DateTime start, DateTime stop, TimeSpan? defaultStep = null) =>
            Methods.GeneratePreciseStationPasses(complex, north_pole, start, stop, defaultStep);

        public List<TimeInterval> CalculateCoverageGaps(DateTime start, DateTime stop, bool north_pole) =>
            Methods.CalculateCoverageGaps(start, stop, north_pole);

        public static void Furnish(string path) => FullSpiceMethods.Furnish(path);

        public void WriteTopoFrame(string frame_name, int frame_id, double lat, double lon, string path)
        {
            var normalized_lat = lat;
            var normalized_lon = NormalizeLon(lon);
            var colat = 90d - normalized_lat;
            var minus_colat = -colat;
            var minus_lon = -normalized_lon;

            var lat_str = lat.ToString("F8");
            var lon_str = lon.ToString("F8");
            var minus_lon_str = minus_lon.ToString("F8");
            var minus_colat_str = minus_colat.ToString("F8");

            var class_id = frame_id + 1000000;

            var text = "This is a site frame\r\n\r\n";
            text += $"longitude_deg    = {lon_str}\r\n";
            text += $"latitude_deg    = {lat_str}\r\n\r\n";

            text += $"Angles = (-normalized(longitude), -colatitude, 180)\r\n";
            text += $"  where normalization maps the angle to the interval [0,360)\r\n";
            text += $"  and colatitude = 90 - Normalized(latitude)\r\n\r\n";

            text += $"\\begindata\r\n\r\n";
            text += $"  FRAME_{frame_name} = {class_id}\r\n";
            text += $"  FRAME_{class_id}_NAME = '{frame_name}'\r\n";
            text += $"  FRAME_{class_id}_CLASS = 4\r\n";
            text += $"  FRAME_{class_id}_CLASS_ID = {class_id}\r\n";
            text += $"  FRAME_{class_id}_CENTER = {frame_id}\r\n\r\n";
            text += $"  OBJECT_{frame_id}_FRAME = '{frame_name}'\r\n\r\n";
            text += $"  TKFRAME_{class_id}_RELATIVE = 'MOON_ME'\r\n";
            text += $"  TKFRAME_{class_id}_SPEC = 'ANGLES'\r\n";
            text += $"  TKFRAME_{class_id}_UNITS = 'DEGREES'\r\n";
            text += $"  TKFRAME_{class_id}_AXES = (3, 2, 3 )\r\n";

            //text += $"  TKFRAME_{frame_id}_ANGLES = ( {minus_lon_str} {minus_colat_str} 180.0 )\r\n\r\n";
            text += $"  TKFRAME_{class_id}_ANGLES = ( {minus_lon_str} {minus_colat_str} 180.0 )\r\n\r\n";

            text += "\\begintext\r\n";

            File.WriteAllText(path, text);
        }

        double NormalizeLon(double l)
        {
            if (l < 0d) l += 360d;
            if (l > 360d) l -= 360d;
            return l;
        }

        public static DateTime Epoch = ViperDate.New(2000, 1, 1, 11, 58, 55, 816);
        public const int EarthId = CSpice.EarthId;
        public const int MoonId = CSpice.MoonId;
        public const int SunId = CSpice.SunId;

        // These two values are equivalent according to horizons (https://ssd.jpl.nasa.gov/horizons.cgi)
        public static DateTime ViperEpoch = ViperDate.New(2024, 1, 1, 0, 0, 0);
        public static double ViperET = 2460311.500000000d;

        public static Vector3d SunPosition(DateTime time)
        {
            var et = SpiceMethods.DateTimeToET(time);
            var state = new double[6];
            double lt = 0d;
            CSpice.spkgeo_c(SunId, et, "MOON_ME", MoonId, state, ref lt);
            return new Vector3d(state[0], state[1], state[2]);
        }

        public static Vector3d SunPosition_meters(DateTime time) => SunPosition(time) * 1000.0;

        public static Vector3d EarthPosition(DateTime time)
        {
            var et = SpiceMethods.DateTimeToET(time);
            var state = new double[6];
            double lt = 0d;
            CSpice.spkgeo_c(EarthId, et, "MOON_ME", MoonId, state, ref lt);
            return new Vector3d(state[0], state[1], state[2]);
        }

        public static Vector3d EarthPosition_meters(DateTime time) => EarthPosition(time) * 1000.0;

        public static List<Vector3d> SunPositions(DateTime start, DateTime stop, TimeSpan step)
        {
            var r = new List<Vector3d>();
            for (var time = start; time <= stop; time += step)
                r.Add(SunPosition(time));
            return r;
        }

        public static List<Vector3d> EarthPositions(DateTime start, DateTime stop, TimeSpan step)
        {
            var r = new List<Vector3d>();
            for (var time = start; time <= stop; time += step)
                r.Add(EarthPosition(time));
            return r;
        }

    }

    public abstract class SpiceMethods
    {
        public static DateTime Epoch = ViperDate.New(2000, 1, 1, 11, 58, 55, 816);

        public static DateTime LocalEpoch = ViperDate.New(2023, 12, 1, 0, 0, 0, 0);
        public static double LocalEpochEpochTime = 0d;
        public static DateTime ProtectSpiceEpoch = ViperDate.New(2020, 1, 1, 0, 0, 0, 0);

        public const int EarthId = 399;
        public const int MoonId = 301;
        public const int SunId = 10;
        public const double MoonRadius = 1737.4d;

        public abstract void SunEarthAzEl(DateTime time, out double sunAzimuth, out double sunElevation, out double earthAzimuth, out double earthElevation);
        public abstract void DSNAzEl(DateTime time, double lat, double lon, double height, out double dsnaz, out double dsnel);

        // Calculate spice seconds from Epoch based on a query done to spice when initializing the SpiceManager
        public static double DateTimeToET(DateTime time)
        {
            //Debug.Assert(time.Kind == DateTimeKind.Utc & time > ProtectSpiceEpoch);
            Debug.Assert(time.Kind == DateTimeKind.Utc);
            return (time - LocalEpoch).TotalSeconds + LocalEpochEpochTime;
        }

        // This doesn't completely invert DateTimeToET, but it's close.  The leap seconds (5) is hard coded
        public static DateTime ETToDateTime(double et) => Epoch.AddTicks((long)((et - 5d) * 10000000d));  // 5 leap seconds is appropriate for times around '23-24

        public abstract List<TimeInterval> GenerateStationPasses(string complex, bool north_pole, DateTime start, DateTime stop, TimeSpan? defaultStep = null);
        public abstract List<TimeInterval> GeneratePreciseStationPasses(string complex, bool north_pole, DateTime start, DateTime stop, TimeSpan? defaultStep = null);
        public abstract List<TimeInterval> CalculateCoverageGaps(DateTime start, DateTime stop, bool north_pole);
    }

    public class DummySpiceMethods : SpiceMethods
    {
        public DummySpiceMethods() { }

        public override List<TimeInterval> CalculateCoverageGaps(DateTime start, DateTime stop, bool north_pole)
        {
            throw new NotImplementedException();
        }

        public override void DSNAzEl(DateTime time, double lat, double lon, double height, out double dsnaz, out double dsnel)
        {
            dsnaz = dsnel = 0d;
        }

        public override List<TimeInterval> GeneratePreciseStationPasses(string complex, bool north_pole, DateTime start, DateTime stop, TimeSpan? defaultStep = default(TimeSpan?))
        {
            throw new NotImplementedException();
        }

        public override List<TimeInterval> GenerateStationPasses(string complex, bool north_pole, DateTime start, DateTime stop, TimeSpan? defaultStep = default(TimeSpan?))
        {
            return null;
        }

        public override void SunEarthAzEl(DateTime time, out double sunAzimuth, out double sunElevation, out double earthAzimuth, out double earthElevation)
        {
            sunAzimuth = sunElevation = earthAzimuth = earthElevation = 0d;
        }
    }

    public class FullSpiceMethods : SpiceMethods
    {
        public static bool KernelsLoaded = false;

        public FullSpiceMethods()
        {
            if (KernelsLoaded)
                return;

            var exe_path = System.Reflection.Assembly.GetExecutingAssembly().Location;
            var kernel_root = Path.Combine(Path.GetDirectoryName(exe_path), "StaticFiles");

            FurnishSpiceKernels(kernel_root, "kernels/metakernel.txt");
            KernelsLoaded = true;
        }

        public void FurnishSpiceKernels(string spiceKernelRoot, string rootFile)
        {
            string metakernelPath = Path.Combine(spiceKernelRoot, rootFile);
            foreach (string line in File.ReadAllLines(metakernelPath).Where(line => !string.IsNullOrEmpty(line) && line[0] != ' ' && line[0] != '#'))
                Furnish(Path.Combine(spiceKernelRoot, CanonicalizeDirectorySeparators(line)));
        }

        public static void Furnish(string path)
        {
            if (SpiceManager.Verbose)
                //Log.Debug($"Spice furnishing {path}");
                Console.WriteLine($"Spice furnishing {path}");
            CSpice.furnsh_c(path);
        }

        public static string CanonicalizeDirectorySeparators(string path)
        {
            path = path.Replace('/', Path.DirectorySeparatorChar);
            path = path.Replace('\\', Path.DirectorySeparatorChar);
            return path;
        }

        public override void SunEarthAzEl(DateTime time, out double sunAzimuth, out double sunElevation, out double earthAzimuth, out double earthElevation)
        {
            //sunAzimuth = sunElevation = earthAzimuth = earthElevation = 0d;
            //return;

            var et = DateTimeToET(time);
            var state = new double[6];
            double lt = 0d;
            double[] pos = { 0d, 0d, 0d };

            CSpice.spkgeo_c(SunId, et, "MOON_ME", MoonId, state, ref lt);

            CSpice.spkcpo_c("SUN", et, "SITE_TOPO", "CENTER", "NONE", pos, "MOON", "MOON_ME", state, ref lt);
            //var d = Math.Sqrt(state[0] * state[0] + state[1] * state[1] + state[2] * state[2]);
            //Console.WriteLine(@"HERMITE: Sun= dist={3} vec=[{0},{1},{2}]", state[0], state[1], state[2], d);

            sunAzimuth = -Math.Atan2(state[1], state[0]);
            var flatd = Math.Sqrt(state[0] * state[0] + state[1] * state[1]);
            sunElevation = Math.Atan2(state[2], flatd);

            CSpice.spkcpo_c("EARTH", et, "SITE_TOPO", "CENTER", "NONE", pos, "MOON", "MOON_ME", state, ref lt);
            //d = Math.Sqrt(state[0] * state[0] + state[1] * state[1] + state[2] * state[2]);
            //Console.WriteLine(@"HERMITE: Earth= dist={3} vec=[{0},{1},{2}]", state[0], state[1], state[2], d);

            earthAzimuth = -Math.Atan2(state[1], state[0]);
            flatd = Math.Sqrt(state[0] * state[0] + state[1] * state[1]);
            earthElevation = Math.Atan2(state[2], flatd);
        }

        public string[] DSNSiteNames = { "DSS-54", "DSS-24", "DSS-34" };
        public string[] DSNSiteFrames = { "DSS-54_TOPO", "DSS-24_TOPO", "DSS-34_TOPO" };
        public string[] DSNComplexNames = { "Madrid", "Goldstone", "Canberra" };
        public int[] DSNIds = { 399054, 399024, 399034 };

        public override void DSNAzEl(DateTime time, double rover_lat, double rover_lon, double height, out double dsnaz, out double dsnel)
        {
            const double stationHorizonMask = 10d * Math.PI / 190d;

            var roverLat = rover_lat * Math.PI / 180d;
            var roverLon = rover_lon * Math.PI / 180d;

            double[] traverseSite = { 0d, 0d, 0d };
            CSpice.pgrrec_c("MOON", roverLon, roverLat, 0d, MoonRadius, 0f, traverseSite);

            var pz = MoonRadius * Math.Sin(roverLat);
            var temp = MoonRadius * Math.Cos(roverLat);
            var px = temp * Math.Cos(roverLon);
            var py = temp * Math.Sin(roverLon);

            Debug.Assert(Math.Abs(px - traverseSite[0]) < 1d);
            Debug.Assert(Math.Abs(py - traverseSite[1]) < 1d);
            Debug.Assert(Math.Abs(pz - traverseSite[2]) < 1d);

            var et = DateTimeToET(time);
            var state = new double[6];
            var earthState = new double[6];
            var moonPositionFromStation = new double[3];
            double lt = 0d;

            CSpice.spkcpo_c("EARTH", et, "SITE_TOPO", "CENTER", "NONE", traverseSite, "MOON", "MOON_ME", earthState, ref lt);
            var earthAzimuth = -Math.Atan2(earthState[1], earthState[0]);
            var flatd1 = Math.Sqrt(earthState[0] * earthState[0] + earthState[1] * earthState[1]);
            var earthElevation = Math.Atan2(earthState[2], flatd1);

            var data = new List<DSNSiteData>();
            for (var i = 0; i < DSNSiteNames.Length; i++)
            {
                // Station to the traverse site
                CSpice.spkezp_c(MoonId, et, DSNSiteFrames[i], "NONE", DSNIds[i], moonPositionFromStation, ref lt);
                var moonElevation = Math.Atan2(moonPositionFromStation[2], Math.Sqrt(moonPositionFromStation[0] * moonPositionFromStation[0] + moonPositionFromStation[1] * moonPositionFromStation[1]));

                // Traverse site to the station
                CSpice.spkcpo_c(DSNSiteNames[i], et, "SITE_TOPO", "CENTER", "NONE", traverseSite, "MOON", "MOON_ME", state, ref lt);

                var flatd = Math.Sqrt(state[0] * state[0] + state[1] * state[1]);
                data.Add(new DSNSiteData
                {
                    SiteToStationAzimuth = -Math.Atan2(state[1], state[0]),
                    SiteToStationElevation = Math.Atan2(state[2], flatd),
                    StationToMoonElevation = moonElevation,
                    Site = DSNSiteNames[i]
                });
            }

            if (SpiceManager.Verbose)
                foreach (var d in data)
                    Log.Debug($"complex={d.Site} elev@complex={d.StationToMoonElevation}  complexElev@traverse={d.SiteToStationElevation} complexAz@traverse={d.SiteToStationAzimuth}");

            DSNSiteData bestData = null;
            const double highestElev = double.MinValue;
            foreach (var d in data)
            {
                if (d.StationToMoonElevation >= stationHorizonMask && d.SiteToStationElevation > highestElev)
                    bestData = d;
            }

            if (bestData == null)
            {
                if (SpiceManager.Verbose)
                    Log.Error($"Couldn't find a suitable ground complex.  Using {data[0].Site}");
                bestData = data[0];
            }
            dsnaz = bestData.SiteToStationAzimuth;
            dsnel = bestData.SiteToStationElevation;

            if (SpiceManager.Verbose)
            {
                Log.Debug($"moon center");
                Log.Debug($"  earth az={earthAzimuth}");
                Log.Debug($"  earth el={earthElevation}");
                Log.Debug($"To {bestData.Site}");
                Log.Debug($"  earth az={bestData.SiteToStationAzimuth}");
                Log.Debug($"  earth el={bestData.SiteToStationElevation}");
            }
        }
        /// <summary>
        /// Generate a csv file containing station passes
        /// </summary>
        /// <param name="complex">Madrid, Goldstone or Canberra</param>
        /// <param name="north_pole">The rover's position is approximated using one or the other pole.  True if np, false if sp</param>
        /// <param name="start">Start time for the file</param>
        /// <param name="stop">Stop time</param>
        public override List<TimeInterval> GenerateStationPasses(string complex, bool north_pole, DateTime start, DateTime stop, TimeSpan? defaultStep = null)
        {
            const double threshold = 10.5d * Math.PI / 180d;
            var step = defaultStep ?? new TimeSpan(1, 0, 0);
            var canonical_name = DSNComplexNames.FirstOrDefault(c => c.Equals(complex));
            if (canonical_name == null)
                return null;
            var canonical_index = Array.IndexOf(DSNComplexNames, canonical_name);
            var site_frame = DSNSiteFrames[canonical_index];
            var dsn_name = DSNSiteNames[canonical_index];
            var target_position = new double[] { 0d, 0d, north_pole ? MoonRadius : -MoonRadius };

            var rows = new List<TimeInterval>();

            var t1 = FirstStepBelowThreshold(dsn_name, site_frame, start, stop, step, target_position, threshold);
            if (t1 == null) return rows;
            var t2 = FirstStepAboveThreshold(dsn_name, site_frame, t1.Value, stop, step, target_position, threshold);
            if (t2 == null) return rows;

            var startOfPass = t2.Value;
            while (startOfPass.Ticks > 0L && startOfPass <= stop)
            {
                var endOfPass = FirstStepBelowThreshold(dsn_name, site_frame, startOfPass, stop, step, target_position, threshold);
                if (endOfPass == null) break;
                rows.Add(new TimeInterval { Start = startOfPass, Stop = endOfPass.Value - step });
                if (SpiceManager.Verbose)
                    Console.WriteLine($"{startOfPass} to {endOfPass} length={endOfPass - startOfPass}");
                var nextPass = FirstStepAboveThreshold(dsn_name, site_frame, endOfPass.Value, stop, step, target_position, threshold);
                if (nextPass == null) break;
                startOfPass = nextPass.Value;
            }

            return rows;
        }

        public override List<TimeInterval> GeneratePreciseStationPasses(string complex, bool north_pole, DateTime start, DateTime stop, TimeSpan? defaultStep = null)
        {
            const double threshold = 10.5d * Math.PI / 180d;
            var scanFuzz = new TimeSpan(0, 5, 0);  // When scanning for the end of the pass, add this to the start of the pass to be sure the scan starts inside
            var step = defaultStep ?? new TimeSpan(1, 0, 0);
            var canonical_name = DSNComplexNames.FirstOrDefault(c => c.Equals(complex));
            if (canonical_name == null)
                return null;
            var canonical_index = Array.IndexOf(DSNComplexNames, canonical_name);
            var site_frame = DSNSiteFrames[canonical_index];
            var dsn_name = DSNSiteNames[canonical_index];
            var target_position = new double[] { 0d, 0d, north_pole ? MoonRadius : -MoonRadius };
            var rows = new List<TimeInterval>();

            var t1 = FirstStepBelowThreshold(dsn_name, site_frame, start, stop, step, target_position, threshold);
            if (t1 == null) return rows;
            var t2 = FirstStepAboveThreshold(dsn_name, site_frame, t1.Value, stop, step, target_position, threshold);
            if (t2 == null) return rows;
            t2 = FindCrossing(dsn_name, site_frame, target_position, threshold, t2.Value - step, t2.Value);

            var startOfPass = t2.Value;
            while (startOfPass.Ticks > 0L && startOfPass <= stop)
            {
                var endOfPass = FirstStepBelowThreshold(dsn_name, site_frame, startOfPass.Add(scanFuzz), stop, step, target_position, threshold); // approx end of pass
                if (endOfPass == null) break;
                endOfPass = FindCrossing(dsn_name, site_frame, target_position, threshold, endOfPass.Value - step, endOfPass.Value);  // accurate end of pass
                if (endOfPass == null) break;
                rows.Add(new TimeInterval { Start = startOfPass, Stop = endOfPass.Value });
                var nextPass = FirstStepAboveThreshold(dsn_name, site_frame, endOfPass.Value.Add(scanFuzz), stop, step, target_position, threshold);
                if (nextPass == null) break;
                startOfPass = FindCrossing(dsn_name, site_frame, target_position, threshold, nextPass.Value - step, nextPass.Value);  // accurate startOfPass
            }

            return rows;
        }

        DateTime FindCrossing(string dsn_name, string site_frame, double[] target_position, double threshold, DateTime time1, DateTime time2)
        {
            if (time2 < time1)
            {
                var t = time2;
                time2 = time1;
                time1 = t;
            }
            var t_low = DateTimeToET(time1);
            var t_high = DateTimeToET(time2);
            var v_low = RoverElevationFromDSN(dsn_name, site_frame, t_low, target_position) - threshold;
            var v_high = RoverElevationFromDSN(dsn_name, site_frame, t_high, target_position) - threshold;

            const double fuzz = 1d;
            Debug.Assert(v_low * v_high < 0d);  // They must straddle the threshold

            while (Math.Abs(t_low - t_high) > fuzz)
            {
                //Console.WriteLine(Math.Abs(t_low - t_high));
                var t_mid = (t_low + t_high) / 2d;
                var v_mid = RoverElevationFromDSN(dsn_name, site_frame, ETToDateTime(t_mid), target_position) - threshold;

                if (v_low * v_mid < 0d)
                {
                    t_high = t_mid;
                }
                else
                {
                    t_low = t_mid;
                    v_low = v_mid;
                }
            }
            return ETToDateTime((t_low + t_high) / 2d);
        }

        DateTime? FirstStepAboveThreshold(string dsn_name, string site_frame, DateTime start, DateTime stop, TimeSpan step, double[] target_position, double threshold)
        {
            for (var time = start; time <= stop; time += step)
                if (RoverElevationFromDSN(dsn_name, site_frame, time, target_position) >= threshold)
                    return time;
            return null;
        }

        DateTime? FirstStepBelowThreshold(string dsn_name, string site_frame, DateTime start, DateTime stop, TimeSpan step, double[] target_position, double threshold)
        {
            for (var time = start; time <= stop; time += step)
                if (RoverElevationFromDSN(dsn_name, site_frame, time, target_position) < threshold)
                    return time;
            return null;
        }

        double RoverElevationFromDSN(string dsn_name, string site_frame, DateTime time, double[] target_position) =>
            RoverElevationFromDSN(dsn_name, site_frame, DateTimeToET(time), target_position);

        double RoverElevationFromDSN(string dsn_name, string site_frame, double et, double[] target_position)
        {
            var targetState = new double[6];
            double lt = 0d;
            CSpice.spkcpt_c(target_position, "MOON", "MOON_ME", et, site_frame, "OBSERVER", "NONE", dsn_name, targetState, ref lt);
            return Math.Atan2(targetState[2], Math.Sqrt(targetState[0] * targetState[0] + targetState[1] * targetState[1]));
        }

        public override List<TimeInterval> CalculateCoverageGaps(DateTime start, DateTime stop, bool north_pole)
        {
            var passes = DSNComplexNames.Select(n => GeneratePreciseStationPasses(n, north_pole, start, stop)).ToList();
            var combined = passes[0].Concat(passes[1]).Concat(passes[2]).ToList();
            var sorted = TimeInterval.Sorted(combined);
            var coalesced = TimeInterval.Coalesce(sorted);
            var coalesced_gaps = TimeInterval.GapList(coalesced);
            return coalesced_gaps;
        }

        internal class DSNSiteData
        {
            internal double SiteToStationAzimuth;
            internal double SiteToStationElevation;
            internal double StationToMoonElevation;
            internal string Site;
        }
    }

    // TODO: There is another implementation of this called Interval.  Combine them.
    public struct TimeInterval
    {
        public DateTime Start;
        public DateTime Stop;

        public override string ToString() => $"[{Start}->{Stop}]";

        public bool IsDisjoint(TimeInterval o) => Stop < o.Start || Start > o.Stop;
        public bool Intersects(TimeInterval o) => !IsDisjoint(o);
        public bool Before(TimeInterval o) => Stop < o.Start;
        public bool After(TimeInterval o) => Start > o.Stop;
        public TimeInterval Gap(TimeInterval o) => Before(o) ? new TimeInterval { Start = Stop, Stop = o.Start } : new TimeInterval { Start = o.Stop, Stop = Start };
        public TimeInterval Union(TimeInterval o) => IsDisjoint(o) ? new TimeInterval() : new TimeInterval { Start = ViperDate.New(Math.Min(Start.Ticks, o.Start.Ticks)), Stop = ViperDate.New(Math.Max(Stop.Ticks, o.Stop.Ticks)) };

        public static List<TimeInterval> GapList(List<TimeInterval> l)
        {
            var r = new List<TimeInterval>();
            for (var i = 1; i < l.Count; i++)
                if (l[i - 1].IsDisjoint(l[i]))
                    r.Add(l[i - 1].Gap(l[i]));
            return r;
        }

        public static List<TimeInterval> Sorted(List<TimeInterval> l) => l.OrderBy(i => i.Start).ToList();
        public static List<TimeInterval> Coalesce(List<TimeInterval> l)
        {
            var r = new List<TimeInterval>();
            TimeInterval? current = null;
            foreach (var i in l)
            {
                if (current == null)
                    current = i;
                else if (current.Value.Intersects(i))
                    current = current.Value.Union(i);
                else
                {
                    r.Add(current.Value);
                    current = i;
                }
            }
            if (current.HasValue)
                r.Add(current.Value);
            return r;
        }
    }
}

