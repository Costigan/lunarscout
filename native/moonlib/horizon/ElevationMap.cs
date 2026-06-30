using moonlib.math;
using OSGeo.OSR;
using System.Diagnostics;
using System.Drawing;

namespace moonlib.horizon
{
    public struct PixelPoint
    {
        public double X;
        public double Y;
        public PixelPoint(double col, double row) => (X, Y) = (col, row);
        public PixelPoint(Point p) => (X, Y) = (p.X, p.Y);
        public PixelPoint(PointF p) => (X, Y) = (p.X, p.Y);

        public (double col, double row) Destructure() => (X, Y);
        public override string ToString() => $"<PP {X},{Y}>";
    }

    public struct CRSPoint
    {
        public double X;
        public double Y;
        public CRSPoint(double x, double y) => (X, Y) = (x, y);
        public CRSPoint(Point p) => (X, Y) = (p.X, p.Y);
        public CRSPoint(PointF p) => (X, Y) = (p.X, p.Y);
        public (double x, double y) Destructure() => (X, Y);
        public override string ToString() => $"<CRS {X},{Y}>";
    }

    public class ElevationMap
    {
        public const string LongLatProj = "+proj=longlat +R=1737400 +no_defs";

        /// <summary>
        /// Elevation raster in meters relative to the lunar datum.
        /// </summary>
        public float[,]? Elevation;

        public string Projection = string.Empty;
        public string Proj4 = string.Empty;
        public SrsDescriptor SrsDescriptor = new SrsDescriptor();
        public Func<CRSPoint, CRSPoint> SrsToLongLat = (pt) => throw new InvalidOperationException("SRS conversion not configured");
        public Func<CRSPoint, CRSPoint> LonLatToSRS = (pt) => throw new InvalidOperationException("SRS conversion not configured");
        public Func<CRSPoint, CRSPoint> SrsToLongLatReference = (pt) => throw new InvalidOperationException("SRS conversion not configured");
        public Func<CRSPoint, CRSPoint> LonLatToSRSReference = (pt) => throw new InvalidOperationException("SRS conversion not configured");

        public double[] GeoTransform = Array.Empty<double>();
        public string? Path; // Made nullable

        private int _width;
        private int _height;
        public int Width => Elevation != null ? Elevation.GetLength(1) : _width;
        public int Height => Elevation != null ? Elevation.GetLength(0) : _height;
        public Size Size => new Size(Width, Height);


        public ElevationMap(float[,] elevation, string? proj4 = null) 
        {
            Elevation = elevation;
            _height = elevation.GetLength(0);
            _width = elevation.GetLength(1);

            Proj4 = proj4 ?? "+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 +k=1 +R=1737400 +units=m +no_defs";
            SrsDescriptor = MoonSrsLambdaFactory.ParseSrs(Proj4);
            GeoTransform = new double[6] { 0, 1, 0, 0, 0, 1 }; // Default GeoTransform
        }

        public ElevationMap(string path, bool loadRaster = true)
        {
            Path = path; // Initialized here
            using var ds = OSGeo.GDAL.Gdal.Open(path, OSGeo.GDAL.Access.GA_ReadOnly);
            if (ds == null)
                throw new ArgumentException($"Could not open DEM file: {path}");

            _width = ds.RasterXSize;
            _height = ds.RasterYSize;

            if (loadRaster)
            {
                var band = ds.GetRasterBand(1);
                // Removed strict check for Float32. GDAL ReadRaster converts automatically.
                Elevation = new float[_height, _width];
                unsafe
                {
                    fixed (float* p = &Elevation[0, 0])
                    {
                        band.ReadRaster(0, 0, _width, _height, (IntPtr)p, _width, _height, OSGeo.GDAL.DataType.GDT_Float32, 0, 0);
                    }
                }
            }

            Projection = ds.GetProjectionRef() ?? string.Empty;
            var srs = new SpatialReference(Projection);
            srs.ExportToProj4(out string? proj4Str); // Changed to nullable string for out parameter
            Proj4 = proj4Str ?? string.Empty; // Handle potential null proj4Str

            SrsDescriptor = MoonSrsLambdaFactory.ParseSrs(Proj4)!;

            GeoTransform = new double[6];
            ds.GetGeoTransform(GeoTransform);

            // Use Factory for main
            SrsToLongLat = MoonSrsLambdaFactory.MakeLambda(Proj4, LongLatProj);
            LonLatToSRS = MoonSrsLambdaFactory.MakeLambda(LongLatProj, Proj4);
            
            // Use OSR for reference
            SrsToLongLatReference = CreateOsrTransform(Proj4, LongLatProj, false, true);
            LonLatToSRSReference = CreateOsrTransform(LongLatProj, Proj4, true, false);

            if (GeoTransform is null || GeoTransform.Length < 6)
                throw new InvalidOperationException("GeoTransform is not defined.");

            double det = GeoTransform[1] * GeoTransform[5] - GeoTransform[2] * GeoTransform[4];
            if (Math.Abs(det) < 1e-12)
                throw new InvalidOperationException("GeoTransform cannot be inverted.");
        }
        


        public Rectangle BoundingBox => new Rectangle(0, 0, Width, Height);

        public ElevationMap(float[,] elevation, string projection, double[] geoTransform)
        {
            Elevation = elevation;
            Projection = projection ?? string.Empty;
            GeoTransform = geoTransform ?? Array.Empty<double>();
            Path = null; // Explicitly null for this constructor

            // Provide default non-throwing lambdas for safety against CS8618 (already handled by field initializers)
            // SrsToLongLat = (pt) => throw new InvalidOperationException("SRS conversion not configured for this ElevationMap");
            // LonLatToSRS = (pt) => throw new InvalidOperationException("SRS conversion not configured for this ElevationMap");
            // SrsDescriptor = default; // SrsDescriptor is a class, so initialize it as new SrsDescriptor()
            SrsDescriptor = new SrsDescriptor(); // Explicitly new

            if (Projection == null || string.Empty.Equals(Projection))
            {
                // Projection is empty, SrsDescriptor and lambdas remain default-initialized
                Proj4 = string.Empty;
            }
            else if (IsEPSG(Projection))
            {
                var srs = new SpatialReference(null);
                var err = srs.ImportFromEPSG(int.Parse(Projection.Replace("EPSG:", "")));
                if (err != 0)
                    throw new Exception($"ImportFromEPSG({Projection.Replace("EPSG:", "")}) failed with error code {err}");
                srs.ExportToProj4(out string? proj4Str);
                Proj4 = proj4Str ?? string.Empty;
                SrsDescriptor = MoonSrsLambdaFactory.ParseSrs(Proj4);
            }
            else if (IsWkt(Projection))
            {
                var srs = new SpatialReference(null);
                // Make a mutable copy for ImportFromWkt if needed, or ensure Projection is not readonly if passed as ref
                string mutableProjection = Projection;
                var err = srs.ImportFromWkt(ref mutableProjection);
                if (err != 0)
                    throw new Exception($"ImportFromProj4({Projection}) failed with error code {err}");
                srs.ExportToProj4(out string? proj4Str);
                Proj4 = proj4Str ?? string.Empty;
                SrsDescriptor = MoonSrsLambdaFactory.ParseSrs(Proj4);
            }
            else
            {
                var proj4 = Projection;
                var srs = new SpatialReference(null);
                var err = srs.ImportFromProj4(proj4);
                if (err != 0)
                    throw new Exception($"ImportFromProj4({proj4}) failed with error code {err}");
                Proj4 = proj4;
                srs.ExportToWkt(out string? wktStr, new string[] { }); // Changed to nullable string for out parameter
                Projection = wktStr ?? string.Empty; // Handle potential null wktStr
                SrsDescriptor = MoonSrsLambdaFactory.ParseSrs(Proj4);
            }
            if (!string.IsNullOrEmpty(Proj4))
            {
                SrsToLongLat = MoonSrsLambdaFactory.MakeLambda(Proj4, LongLatProj);
                LonLatToSRS = MoonSrsLambdaFactory.MakeLambda(LongLatProj, Proj4);
                
                SrsToLongLatReference = CreateOsrTransform(Proj4, LongLatProj, false, true);
                LonLatToSRSReference = CreateOsrTransform(LongLatProj, Proj4, true, false);
            }
            
            if (Elevation != null)
            {
                _width = Elevation.GetLength(1);
                _height = Elevation.GetLength(0);
            }
        }

        public bool IsWkt(string wkt) => wkt.Contains('[');
        public bool IsEPSG(string epsg) => epsg.StartsWith("EPSG:");

        private Func<CRSPoint, CRSPoint> CreateOsrTransform(string srcDef, string dstDef, bool inputIsRadians, bool outputIsRadians)
        {
            SpatialReference src = new SpatialReference(null);
            if (IsEPSG(srcDef)) 
                src.ImportFromEPSG(int.Parse(srcDef.Substring(5)));
            else if (IsWkt(srcDef)) 
            { 
                string s = srcDef; 
                src.ImportFromWkt(ref s); 
            }
            else 
                src.ImportFromProj4(srcDef);
            
            src.SetAxisMappingStrategy(AxisMappingStrategy.OAMS_TRADITIONAL_GIS_ORDER);

            SpatialReference dst = new SpatialReference(null);
            if (IsEPSG(dstDef)) 
                dst.ImportFromEPSG(int.Parse(dstDef.Substring(5)));
            else if (IsWkt(dstDef)) 
            { 
                string s = dstDef; 
                dst.ImportFromWkt(ref s); 
            }
            else 
                dst.ImportFromProj4(dstDef);
            
            dst.SetAxisMappingStrategy(AxisMappingStrategy.OAMS_TRADITIONAL_GIS_ORDER);

            CoordinateTransformation ct = new CoordinateTransformation(src, dst);

            return (pt) =>
            {
                double[] val = new double[] { pt.X, pt.Y, 0 };
                if (inputIsRadians)
                {
                    val[0] = val[0] * 180.0 / Math.PI;
                    val[1] = val[1] * 180.0 / Math.PI;
                }

                ct.TransformPoint(val);

                if (outputIsRadians)
                {
                    val[0] = val[0] * Math.PI / 180.0;
                    val[1] = val[1] * Math.PI / 180.0;
                }
                return new CRSPoint(val[0], val[1]);
            };
        }

        public PixelPoint CRSToPixel(CRSPoint crs_pt)
        {
            double det = GeoTransform[1] * GeoTransform[5] - GeoTransform[2] * GeoTransform[4];
            double dx = crs_pt.X - GeoTransform[0];
            double dy = crs_pt.Y - GeoTransform[3];
            double col = (GeoTransform[5] * dx - GeoTransform[2] * dy) / det;
            double row = (-GeoTransform[4] * dx + GeoTransform[1] * dy) / det;
            return new PixelPoint(col, row);
        }

        public CRSPoint PixelToCRS(PixelPoint pixel_pt)
        {
            double x = GeoTransform[0] + GeoTransform[1] * pixel_pt.X + GeoTransform[2] * pixel_pt.Y;
            double y = GeoTransform[3] + GeoTransform[4] * pixel_pt.X + GeoTransform[5] * pixel_pt.Y;
            return new CRSPoint(x, y);
        }

        public double GetElevation(double col, double row)
        {
            Debug.Assert(Elevation != null);
            var x1 = (int)col;
            var y1 = (int)row;
            int x2 = Math.Min(x1 + 1, Width - 1);
            int y2 = Math.Min(y1 + 1, Height - 1);
            double q11 = Elevation[y1, x1];         // Don't check bounds and return some default (for now)
            double q12 = Elevation[y2, x1];
            double q21 = Elevation[y1, x2];
            double q22 = Elevation[y2, x2];
            // From https://en.wikipedia.org/wiki/Bilinear_interpolation, note denominator is 1 in this case, so is omitted
            var r = q11 * (x2 - col) * (y2 - row) + q21 * (col - x1) * (y2 - row) + q12 * (x2 - col) * (row - y1) + q22 * (col - x1) * (row - y1);
            return r;
        }

        public double GetElevationClipped(double col, double row)
        {
            Debug.Assert(Elevation != null);
            var x1 = (int)col;
            var y1 = (int)row;
            int x2 = x1 + 1;
            int y2 = y1 + 1;
            if (x1 < 0 || x2 >= Elevation.GetLength(1) || y1 < 0 || y2 >= Elevation.GetLength(0))
                return double.NaN; // Out of bounds

            double q11 = Elevation[y1, x1];         // Don't check bounds and return some default (for now)
            double q12 = Elevation[y2, x1];
            double q21 = Elevation[y1, x2];
            double q22 = Elevation[y2, x2];
            // From https://en.wikipedia.org/wiki/Bilinear_interpolation, note denominator is 1 in this case, so is omitted
            var r = q11 * (x2 - col) * (y2 - row) + q21 * (col - x1) * (y2 - row) + q12 * (x2 - col) * (row - y1) + q22 * (col - x1) * (row - y1);
            return r;
        }

        public Vector3d GetPointInMoonME(PixelPoint pt_pixel)
        {
            // Parse the SRS descriptor from the Proj4 string
            var srs = SrsDescriptor;

            // Get CRS coordinates from pixel
            var crs_point = PixelToCRS(pt_pixel);

            double lonRad, latRad;
            if (srs.Type == SrsDescriptor.ProjType.LongLat)
            {
                // CRS is longitude (deg), latitude (deg) or (rad)? Assume degrees, convert to radians
                lonRad = crs_point.X * Math.PI / 180.0;
                latRad = crs_point.Y * Math.PI / 180.0;
            }
            else if (srs.Type == SrsDescriptor.ProjType.Stereographic)
            {
                // Convert stereographic (x, y) to (lon, lat) in radians
                (lonRad, latRad) = SrsToLongLat(crs_point).Destructure();
            }
            else
            {
                // Unsupported SRS
                return Vector3d.Zero;
            }

            // Get elevation at this pixel
            double elev = GetElevation(pt_pixel.X, pt_pixel.Y); // meters above lunar datum

            // Use the sphere radius from the SRS descriptor
            double R = srs.R;
            double r = R + elev;

            // Convert (lon, lat, r) to Moon-centered Cartesian coordinates
            double cosLat = Math.Cos(latRad);
            double x = r * cosLat * Math.Cos(lonRad);
            double y = r * cosLat * Math.Sin(lonRad);
            double z = r * Math.Sin(latRad);

            return new Vector3d(x, y, z);
        }

        public (Vector3d moon_me, double lonRad, double latRad) GetMoonCoordinates(PixelPoint pt_pixel)
        {
            // Parse the SRS descriptor from the Proj4 string
            var srs = SrsDescriptor;

            // Get CRS coordinates from pixel
            var crs_point = PixelToCRS(pt_pixel);

            double lonRad, latRad;
            if (srs.Type == SrsDescriptor.ProjType.LongLat)
            {
                // CRS is longitude (deg), latitude (deg) or (rad)? Assume degrees, convert to radians
                lonRad = crs_point.X * Math.PI / 180.0;
                latRad = crs_point.Y * Math.PI / 180.0;
            }
            else if (srs.Type == SrsDescriptor.ProjType.Stereographic)
            {
                // Convert stereographic (x, y) to (lon, lat) in radians
                (lonRad, latRad) = SrsToLongLat(crs_point).Destructure();
            }
            else
            {
                // Unsupported SRS
                return (Vector3d.Zero, 0, 0);
            }

            // Get elevation at this pixel
            double elev = GetElevation(pt_pixel.X, pt_pixel.Y); // meters above lunar datum
            //double elev = GetElevationClipped(pt_pixel.X, pt_pixel.Y); // meters above lunar datum

            //if (double.IsNaN(elev))
            //{
            //    // Handle no-data by returning a point on the sphere at the given lat/lon
            //    elev = 0;
            //}

            // Use the sphere radius from the SRS descriptor
            double R = srs.R;
            double r = R + elev;

            // Convert (lon, lat, r) to Moon-centered Cartesian coordinates
            double cosLat = Math.Cos(latRad);
            double x = r * cosLat * Math.Cos(lonRad);
            double y = r * cosLat * Math.Sin(lonRad);
            double z = r * Math.Sin(latRad);

            return (new Vector3d(x, y, z), lonRad, latRad);
        }

        public Matrix4d GetMatrix(int line, int sample)
        {
            return GetMoonMEToENU(line, sample);
        }

        public Matrix4d GetMoonMEToENU(int line, int sample)
        {
            var (vec, lon_rad, lat_rad) = GetMoonCoordinates(new PixelPoint(sample, line));

            double cosLat = Math.Cos(lat_rad);
            double sinLat = Math.Sin(lat_rad);
            double cosLon = Math.Cos(lon_rad);
            double sinLon = Math.Sin(lon_rad);

            // Standard ENU basis vectors in MME frame
            Vector3d up = new Vector3d(cosLat * cosLon, cosLat * sinLon, sinLat);
            Vector3d east = new Vector3d(-sinLon, cosLon, 0);
            Vector3d north = new Vector3d(-sinLat * cosLon, -sinLat * sinLon, cosLat);

            // Row-major matrix for V_enu = (V_me - P_obs) * M
            // where M columns are basis vectors.
            // Row 0 = [ex, nx, ux, 0]
            // Row 1 = [ey, ny, uy, 0]
            // Row 2 = [ez, nz, uz, 0]
            Matrix4d rotation = new Matrix4d(
                east.X, north.X, up.X, 0,
                east.Y, north.Y, up.Y, 0,
                east.Z, north.Z, up.Z, 0,
                0, 0, 0, 1
            );

            Matrix4d translation = Matrix4d.CreateTranslation(-vec);
            return translation * rotation;
        }

        public Matrix4d MoonME_to_ZDown(int line, int sample)
        {
            const float moonRadius = 1737.4f;

            // 1. Convert latitude and longitude in radians
            var (vec, lon_rad, lat_rad) = GetMoonCoordinates(new PixelPoint(sample, line));

            double cosLat = Math.Cos(lat_rad);
            double sinLat = Math.Sin(lat_rad);
            double cosLon = Math.Cos(lon_rad);
            double sinLon = Math.Sin(lon_rad);

            // 2. Calculate the surface position in the MME frame (the origin of our local frame)
            double xs = moonRadius * cosLat * cosLon;
            double ys = moonRadius * cosLat * sinLon;
            double zs = moonRadius * sinLat;

            // 3. Define the basis vectors of the NED frame expressed in MME coordinates
            // Down (uD): Points toward the Moon's center (opposite of the normal)
            Vector3d uD = new Vector3d(-cosLat * cosLon, -cosLat * sinLon, -sinLat);

            // East (uE): Points along the parallel of latitude
            Vector3d uE = new Vector3d(-sinLon, cosLon, 0);

            // North (uN): Points toward the North Pole, tangent to the surface (uE cross uD)
            Vector3d uN = new Vector3d(-sinLat * cosLon, -sinLat * sinLon, cosLat);

            // 4. Construct the Rotation Matrix
            // Since your library uses Row3 for translation and basis vectors as columns
            // for the transformation v * M, we place our basis vectors in the columns.
            Matrix4d rotation = new Matrix4d(
                uN.X, uE.X, uD.X, 0,
                uN.Y, uE.Y, uD.Y, 0,
                uN.Z, uE.Z, uD.Z, 0,
                0, 0, 0, 1
            );

            // 5. Create the translation matrix to shift MME origin to the surface location
            Matrix4d translation = Matrix4d.CreateTranslation(-xs, -ys, -zs);

            // 6. Combine: Translate first, then rotate into the local orientation
            return translation * rotation;
        }

        /// <summary>
        /// Calculate the azimuth and elevation of a point in MOON_ME frame (usually the sun or earth)
        /// </summary>
        /// <param name="point_me">Point in MOON_ME frame</param>
        /// <param name="line">Y index of pixel in this terrain patch</param>
        /// <param name="sample">X index of pixel in this terrain patch</param>
        /// <returns>(output azimuth in radians, output elevation in radians</returns>
        public (float azimuth_rad, float elevation_rad) GetAzEl(Vector3d point_me, int line, int sample)
        {
            //var (vec, lat_rad, lon_rad) = GetPointInMeAndLatLon(line, sample);
            //var zaxis = new Vector3d(0d, 0d, 1d);
            //var yaxis = new Vector3d(0d, 1d, 0d);
            //var a = Matrix4d.CreateFromAxisAngle(zaxis, -lon_rad);
            //var b = Matrix4d.CreateFromAxisAngle(yaxis, -(Math.PI / 2 - lat_rad));
            //var c = Matrix4d.CreateTranslation(-vec);

            //var mat = c * a * b;
            //TODO: This is getting called lots of times
            var mat = GetMatrix(line, sample);
            return GetAzEl(point_me, mat);
        }

        public (float azimuth_rad, float elevation_rad) GetAzEl(Vector3d point_me, Matrix4d mat)
        {
            var temp = new Vector3d();
            ApplyMatrix(ref point_me, ref mat, ref temp);
            var (x, y, z) = (temp.X, temp.Y, temp.Z);
            // In ENU frame: X=East, Y=North. For clockwise-from-North azimuth: Atan2(East, North)
            var azimuth_rad = (float)Math.Atan2(x, y);  // [-PI,PI]
            if (azimuth_rad < 0) azimuth_rad += (float)(2 * Math.PI);  // [0,2PI]
            var alen = Math.Sqrt(x * x + y * y);
            var elevation_rad = (float)Math.Atan2(z, alen);
            return (azimuth_rad, elevation_rad);
        }

        /// <summary>
        /// Computes a DEM surface normal at a pixel in the local ENU frame.
        /// Uses central differences from neighboring surface points transformed by the supplied MoonME->ENU matrix.
        /// </summary>
        public Vector3d GetSurfaceNormalEnu(int line, int sample, Matrix4d moonMeToEnu)
        {
            int left = Math.Max(sample - 1, 0);
            int right = Math.Min(sample + 1, Width - 1);
            int up = Math.Max(line - 1, 0);
            int down = Math.Min(line + 1, Height - 1);

            if (left == right || up == down)
                return new Vector3d(0.0, 0.0, 1.0);

            var pLeft = GetPointInMoonME(new PixelPoint(left, line));
            var pRight = GetPointInMoonME(new PixelPoint(right, line));
            var pUp = GetPointInMoonME(new PixelPoint(sample, up));
            var pDown = GetPointInMoonME(new PixelPoint(sample, down));

            var enuLeft = new Vector3d();
            var enuRight = new Vector3d();
            var enuUp = new Vector3d();
            var enuDown = new Vector3d();
            ApplyMatrix(ref pLeft, ref moonMeToEnu, ref enuLeft);
            ApplyMatrix(ref pRight, ref moonMeToEnu, ref enuRight);
            ApplyMatrix(ref pUp, ref moonMeToEnu, ref enuUp);
            ApplyMatrix(ref pDown, ref moonMeToEnu, ref enuDown);

            var tangentSample = enuRight - enuLeft;
            var tangentLine = enuDown - enuUp;
            var normal = Vector3d.Cross(tangentSample, tangentLine);

            if (normal.LengthSquared < 1e-20)
                return new Vector3d(0.0, 0.0, 1.0);

            normal.Normalize();
            if (normal.Z < 0.0)
                normal = -normal;

            return normal;
        }

        public static void ApplyMatrix(ref Vector3d vec, ref Matrix4d mat, ref Vector3d output)
        {
            output.X = vec.X * mat.Row0.X + vec.Y * mat.Row1.X + vec.Z * mat.Row2.X + mat.Row3.X;
            output.Y = vec.X * mat.Row0.Y + vec.Y * mat.Row1.Y + vec.Z * mat.Row2.Y + mat.Row3.Y;
            output.Z = vec.X * mat.Row0.Z + vec.Y * mat.Row1.Z + vec.Z * mat.Row2.Z + mat.Row3.Z;
        }

        public (double lat_deg, double lon_deg) Point2LatLonDeg(Point p)
        {
            var crs_pt = PixelToCRS(new PixelPoint(p));
            if (SrsDescriptor != null && SrsDescriptor.Type == SrsDescriptor.ProjType.LongLat)
            {
                // CRS is already in degrees for LongLat; return directly.
                return (crs_pt.Y, crs_pt.X);
            }
            var (lonRad, latRad) = SrsToLongLat(crs_pt).Destructure();
            return (latRad.ToDegrees(), lonRad.ToDegrees());
        }

        public (double lat_deg, double lon_deg) Point2LatLonDeg(double x, double y)
        {
            var crs_pt = PixelToCRS(new PixelPoint(x, y));
            if (SrsDescriptor != null && SrsDescriptor.Type == SrsDescriptor.ProjType.LongLat)
            {
                // CRS is already in degrees for LongLat; return directly.
                return (crs_pt.Y, crs_pt.X);
            }
            var (lonRad, latRad) = SrsToLongLat(crs_pt).Destructure();
            return (latRad.ToDegrees(), lonRad.ToDegrees());
        }

        public (double row, double col) LonLatDeg2RowCol(double lon_deg, double lat_deg)
        {
            CRSPoint crs_pt;
            if (SrsDescriptor != null && SrsDescriptor.Type == SrsDescriptor.ProjType.LongLat)
            {
                // Data is stored in degrees; no conversion to radians.
                crs_pt = new CRSPoint(lon_deg, lat_deg);
            }
            else
            {
                // Use LongLat (radians) -> Dataset SRS
                double lonRad = lon_deg.ToRadians();
                double latRad = lat_deg.ToRadians();
                crs_pt = LonLatToSRS(new CRSPoint(lonRad, latRad));
            }

            var pixel_pt = CRSToPixel(crs_pt);
            return (pixel_pt.Y, pixel_pt.X);
        }

        public (double row, double col) LonLatRad2RowCol(double lon_rad, double lat_rad)
        {
            CRSPoint crs_pt;
            if (SrsDescriptor != null && SrsDescriptor.Type == SrsDescriptor.ProjType.LongLat)
            {
                // Input already radians; convert to degrees for data in degrees
                crs_pt = new CRSPoint(lon_rad.ToDegrees(), lat_rad.ToDegrees());
            }
            else
            {
                // LongLat (radians) -> Dataset SRS
                crs_pt = LonLatToSRS(new CRSPoint(lon_rad, lat_rad));
            }

            var pixel_pt = CRSToPixel(crs_pt);
            return (pixel_pt.Y, pixel_pt.X);
        }
    }
}
