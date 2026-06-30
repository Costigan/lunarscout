using System.Globalization;

namespace moonlib.horizon
{
    public class SrsDescriptor
    {
        public enum ProjType { LongLat, Stereographic, Unsupported }
        public ProjType Type;
        public double R;        // sphere radius
                                // for stereographic
        public double lat0;     // φ₀ in radians
        public double lon0;     // λ₀ in radians
        public double k0;       // scale factor (often 1)
        public double x0, y0;   // false easting/northing (offsets)

        public double FalseEasting => x0;
        public double FalseNorthing => y0;
    }

    public static class MoonSrsLambdaFactory
    {
        static double DegToRad(double deg) => deg * Math.PI / 180.0;
        static double RadToDeg(double rad) => rad * 180.0 / Math.PI;

        public static SrsDescriptor ParseSrs(string wktLike)
        {
            // crude parser: split by spaces, find key=value tokens.
            var sd = new SrsDescriptor();
            var parts = wktLike.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            var dict = new Dictionary<string, string>();
            foreach (var tok in parts)
            {
                if (tok.StartsWith("+"))
                {
                    var eq = tok.Substring(1).Split('=');
                    if (eq.Length == 2)
                        dict[eq[0]] = eq[1];
                    else
                        dict[eq[0]] = "";  // flag
                }
            }

            // radius (default to Moon's radius if missing)
            if (dict.TryGetValue("R", out var rstr))
            {
                if (!double.TryParse(rstr, NumberStyles.Float, CultureInfo.InvariantCulture, out sd.R))
                    throw new ArgumentException("Bad R value");
            }
            else
            {
                // Default to Moon's mean radius in meters
                sd.R = 1737400.0;
            }

            // determine proj
            if (dict.TryGetValue("proj", out var p))
            {
                if (p == "longlat")
                {
                    sd.Type = SrsDescriptor.ProjType.LongLat;
                }
                else if (p == "stere")
                {
                    sd.Type = SrsDescriptor.ProjType.Stereographic;
                    // parse stereographic parameters
                    if (dict.TryGetValue("lat_0", out var lat0s))
                        sd.lat0 = DegToRad(double.Parse(lat0s, CultureInfo.InvariantCulture));
                    else
                        sd.lat0 = 0;
                    if (dict.TryGetValue("lon_0", out var lon0s))
                        sd.lon0 = DegToRad(double.Parse(lon0s, CultureInfo.InvariantCulture));
                    else
                        sd.lon0 = 0;
                    if (dict.TryGetValue("k", out var ks))
                        sd.k0 = double.Parse(ks, CultureInfo.InvariantCulture);
                    else
                        sd.k0 = 1.0;
                    if (dict.TryGetValue("x_0", out var x0s))
                        sd.x0 = double.Parse(x0s, CultureInfo.InvariantCulture);
                    else
                        sd.x0 = 0.0;
                    if (dict.TryGetValue("y_0", out var y0s))
                        sd.y0 = double.Parse(y0s, CultureInfo.InvariantCulture);
                    else
                        sd.y0 = 0.0;
                }
                else
                {
                    sd.Type = SrsDescriptor.ProjType.Unsupported;
                }
            }
            else
            {
                sd.Type = SrsDescriptor.ProjType.Unsupported;
            }

            return sd;
        }



        /// <summary>
        /// Returns a lambda mapping (lon_rad, lat_rad) → (x, y) in the target SRS.
        /// Both source and target must be in your supported set.
        /// Input and outputs are in radians for angles, meters for x,y.
        /// Throws if unsupported combination.
        /// </summary>
        public static Func<CRSPoint, CRSPoint> MakeLambda(string srcSrsDef, string dstSrsDef)
        {
            var src = ParseSrs(srcSrsDef);
            var dst = ParseSrs(dstSrsDef);

            return MakeLambda(src, dst);
        }

        public static Func<CRSPoint, CRSPoint> MakeLambda(SrsDescriptor src, SrsDescriptor dst)
        {
            if (src.Type == SrsDescriptor.ProjType.Unsupported ||
                dst.Type == SrsDescriptor.ProjType.Unsupported)
            {
                throw new NotSupportedException("Unsupported SRS pair");
            }

            Func<CRSPoint, CRSPoint> toLongLat;
            switch (src.Type)
            {
                case SrsDescriptor.ProjType.LongLat:
                    toLongLat = pt => pt;
                    break;
                case SrsDescriptor.ProjType.Stereographic:
                    {
                        double R = src.R;
                        double φ0 = src.lat0;
                        double λ0 = src.lon0;
                        double k0 = src.k0;
                        double x0 = src.x0;
                        double y0 = src.y0;

                        double sinφ0 = Math.Sin(φ0);
                        double cosφ0 = Math.Cos(φ0);

                        toLongLat = pt =>
                        {
                            double xp = pt.X - x0;
                            double yp = pt.Y - y0;
                            double ρ = Math.Sqrt(xp * xp + yp * yp);

                            // Debug logging
                            //Console.WriteLine($"--- toLongLat Debug (Stereographic) ---");
                            //Console.WriteLine($"  Input pt: ({pt.X}, {pt.Y})");
                            //Console.WriteLine($"  SRS: R={R}, lat0={RadToDeg(φ0)}, lon0={RadToDeg(λ0)}, k0={k0}, x0={x0}, y0={y0}");
                            //Console.WriteLine($"  xp={xp}, yp={yp}, rho={ρ}");

                            if (ρ <= 1e-12)
                            {
                                //Console.WriteLine($"  rho near zero, returning ({RadToDeg(λ0)}, {RadToDeg(φ0)})");
                                return new CRSPoint(λ0, φ0);
                            }

                            double c = 2 * Math.Atan2(ρ, 2 * k0 * R);
                            double sin_c = Math.Sin(c);
                            double cos_c = Math.Cos(c);

                            double lat = Math.Asin(
                                cos_c * sinφ0 + (yp * sin_c * cosφ0) / ρ);

                            double lon = λ0 + Math.Atan2(
                                xp * sin_c,
                                ρ * cosφ0 * cos_c - yp * sinφ0 * sin_c);
                            //Console.WriteLine($"  c={c}, sin_c={sin_c}, cos_c={cos_c}");
                            //Console.WriteLine($"  Output (lon, lat) radians: ({lon}, {lat}) -> Degrees: ({RadToDeg(lon)}, {RadToDeg(lat)})");
                            return new CRSPoint(lon, lat);
                        };
                        break;
                    }
                default:
                    throw new NotSupportedException($"Unsupported source SRS: {src.Type}");
            }

            Func<CRSPoint, CRSPoint> fromLongLat;
            switch (dst.Type)
            {
                case SrsDescriptor.ProjType.LongLat:
                    fromLongLat = pt => pt;
                    break;
                case SrsDescriptor.ProjType.Stereographic:
                    {
                        double R = dst.R;
                        double φ0 = dst.lat0;
                        double λ0 = dst.lon0;
                        double k0 = dst.k0;
                        double x0 = dst.x0;
                        double y0 = dst.y0;

                        double sinφ0 = Math.Sin(φ0);
                        double cosφ0 = Math.Cos(φ0);

                        fromLongLat = pt =>
                        {
                            double sinφ = Math.Sin(pt.Y);
                            double cosφ = Math.Cos(pt.Y);
                            double dλ = pt.X - λ0;
                            double cos_dλ = Math.Cos(dλ);
                            double sin_dλ = Math.Sin(dλ);

                            double denom = 1 + sinφ0 * sinφ + cosφ0 * cosφ * cos_dλ;
                            double k = 2 * k0 * R / denom;

                            double x = k * cosφ * sin_dλ + x0;
                            double y = k * (cosφ0 * sinφ - sinφ0 * cosφ * cos_dλ) + y0;
                            return new CRSPoint(x, y);
                        };
                        break;
                    }
                default:
                    throw new NotSupportedException($"Unsupported destination SRS: {dst.Type}");
            }

            return pt =>
            {
                var result = toLongLat(pt);
                return fromLongLat(result);
            };
        }

        public static CRSPoint ToLambdaInputUnits(CRSPoint point, SrsDescriptor? srs)
        {
            if (srs is null)
                return point;
            if (srs.Type == SrsDescriptor.ProjType.LongLat)
                return new CRSPoint(DegToRad(point.X), DegToRad(point.Y));
            return point;
        }

        public static CRSPoint FromLambdaOutputUnits(CRSPoint point, SrsDescriptor? srs)
        {
            if (srs is null)
                return point;
            if (srs.Type == SrsDescriptor.ProjType.LongLat)
                return new CRSPoint(RadToDeg(point.X), RadToDeg(point.Y));
            return point;
        }

        public static (double k, double gamma) GetDistortion(CRSPoint point, SrsDescriptor srs)
        {
            if (srs.Type == SrsDescriptor.ProjType.LongLat)
            {
                // k = MapUnits / Meters
                // MapUnits = Degrees
                // 1 meter approx (180 / (PI * R)) degrees
                // We neglect the cos(phi) term for longitude because this is a simplification 
                // sufficient for scaling ray distances in basic cases.
                double kFactor = 180.0 / (Math.PI * srs.R);
                return (kFactor, 0.0);
            }

            if (srs.Type != SrsDescriptor.ProjType.Stereographic)
                return (1.0, 0.0);

            double R = srs.R;
            double φ0 = srs.lat0;
            double λ0 = srs.lon0;
            double k0 = srs.k0;
            double x0 = srs.x0;
            double y0 = srs.y0;

            // Convert input CRS (assumed radians if coming from ToLambdaInputUnits context, but standard usage 
            // of this factory usually implies input is in Dataset units (meters for Stereographic).
            // However, the point passed here should be (Lon, Lat) in radians to calculate distortion easily.
            // Let's assume the caller passes (Lon, Lat) in radians.

            double λ = point.X;
            double φ = point.Y;

            double sinφ = Math.Sin(φ);
            double cosφ = Math.Cos(φ);
            double sinφ0 = Math.Sin(φ0);
            double cosφ0 = Math.Cos(φ0);
            double dλ = λ - λ0;
            double cosdλ = Math.Cos(dλ);
            double sindλ = Math.Sin(dλ);

            // Scale factor k
            double denom = 1 + sinφ0 * sinφ + cosφ0 * cosφ * cosdλ;
            double k = (2 * k0) / denom; // Standard formula for sphere stereographic scale relative to R? 
            // Actually k = k0 * 2 / (1 + ...) matches the projection formula deriv.
            // But standard definition is point scale factor. 
            // For conformal projection, scale is isotropic.
            // k = k_center * 2 / (1 + sin(phi)*sin(phi0) + cos(phi)*cos(phi0)*cos(dlam))

            // Convergence gamma
            // For polar (phi0 = 90): gamma = lam - lam0
            // For polar (phi0 = -90): gamma = -(lam - lam0)
            // General: tan(gamma) = sin(dlam) / (cos(phi0) * tan(phi) - sin(phi0) * cos(dlam)) ?
            // Or use simple derivative of projected coordinates.
            
            // Let's compute projected x,y derivatives to be robust.
            // x = k_proj * R * cos(phi) * sin(dlam)
            // y = k_proj * R * (cos(phi0)*sin(phi) - sin(phi0)*cos(phi)*cos(dlam))
            // where k_proj = 2*k0 / (1 + ...)
            
            // Convergence is angle of projected meridian (Y axis) relative to True North.
            // Actually usually defined as True North relative to Grid North.
            // We want to rotate our True North azimuths to Grid Azimuths.
            // Grid Azimuth = True Azimuth - Gamma.
            
            double y_x = -sindλ * sinφ0; // derived from standard formula simplifications
            double x_y = cosdλ * sinφ0 - Math.Tan(φ) * cosφ0;
            
            // Simpler formula for convergence on Sphere:
            // tan(gamma) = sin(dlam) * sin(phi) / ... no that's not it.
            // For Stereographic: gamma = (lambda - lambda0) * sin(phi0) ?? No that's Conic.
            
            // Robust way: Calculate North vector in projected plane.
            // North is (0, 1) in Lat/Lon space.
            // Calculate projection of (lon, lat) and (lon, lat + epsilon).
            
            // However, we can implement the general formula:
            // tan(gamma) = (cos(phi0) * sin(dlam)) / (sin(phi0)*cos(phi) - cos(phi0)*sin(phi)*cos(dlam))
            // But wait, standard definition for Gamma:
            // Gamma is the angle measured clockwise from True North to Grid North.
            // So Grid = True - Gamma.
            
            // Robust Gamma (Grid Convergence) calculation:
            // Calculate the direction of True North in the projected plane.
            // We project (φ, λ) and a point slightly North (φ + ε, λ).
            double eps = 1e-5;
            double phi_north = φ + eps;
            if (phi_north > Math.PI / 2) phi_north = Math.PI / 2 - eps; // Clamp

            // Local projection helper to avoid code duplication
            // Returns (x, y)
            (double, double) Project(double p_phi, double p_dlam)
            {
                double s_phi = Math.Sin(p_phi);
                double c_phi = Math.Cos(p_phi);
                double c_dlam = Math.Cos(p_dlam);
                double s_dlam = Math.Sin(p_dlam);
                double den = 1 + sinφ0 * s_phi + cosφ0 * c_phi * c_dlam;
                if (Math.Abs(den) < 1e-12) den = 1e-12; // Safety
                double A = 2 * k0 * R / den;
                return (A * c_phi * s_dlam, A * (cosφ0 * s_phi - sinφ0 * c_phi * c_dlam));
            }

            var (x_curr, y_curr) = Project(φ, dλ);
            var (x_north, y_north) = Project(phi_north, dλ);

            double dx = x_north - x_curr;
            double dy = y_north - y_curr;

            double gamma = Math.Atan2(dx, dy);
            return (k, gamma);
        }
    }
}
