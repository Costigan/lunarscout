using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;

namespace moonlib.spice
{
    public class CSpice
    {
        private const string LibraryName = "cspice";

        static CSpice()
        {
            NativeLibrary.SetDllImportResolver(typeof(CSpice).Assembly, ResolveCSpiceLibrary);
        }

        //public SpiceWrapper Wrapper;

        public const int LadeeId = -12;
        public const int LadeeIdForCK = -12000;
        public const int MoonId = 301;
        public const double MoonRadius = 1737.4d;
        public const double MoonFlattening = 0d;
        public const int SunId = 10;
        public const int EarthId = 399;
        public const int MarsId = 4;
        public const double DegToRad = 3.141592653589793238D / 180d;
        public const double RadToDeg = 180d / 3.141592653589793238D;
        public static double[] XAxis = { 1d, 0d, 0d };
        public static double[] YAxis = { 0d, 1d, 0d };
        public static double[] ZAxis = { 0d, 0d, 1d };

        //private static readonly string Pictur = "MON DD,YYYY  HR:MN:SC.####";

        /// <summary>
        /// Load an ephemeris file for use by the readers.  Return that file's handle, to be used by other SPK routines to refer to the file.
        /// </summary>
        /// <param name="filename">Name of the file to be loaded.</param>
        /// <param name="handle">Loaded file's handle. (output)</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void spklef_c([MarshalAs(UnmanagedType.LPStr)] string filename,
                                           ref int handle);

        /// <summary>
        /// Return the state (position and velocity) of a target body relative to an observing body, optionally corrected for light
        /// time (planetary aberration) and stellar aberration. 
        /// </summary>
        /// <param name="targ">Target body name.</param>
        /// <param name="et">Observer epoch.</param>
        /// <param name="refer">Reference frame of output state vector.</param>
        /// <param name="abcorr">Aberration correction flag.</param>
        /// <param name="obs">Observing body name.</param>
        /// <param name="starg">State of target.</param>
        /// <param name="lt">One way light time between observer and target.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void spkezr_c([MarshalAs(UnmanagedType.LPStr)] string targ,
                                           double et,
                                           [MarshalAs(UnmanagedType.LPStr)] string refer,
                                           [MarshalAs(UnmanagedType.LPStr)] string abcorr,
                                           [MarshalAs(UnmanagedType.LPStr)] string obs,
                                           double[] starg,
                                           ref double lt);

        /// <summary>
        /// Return the state (position and velocity) of a target body 
        /// relative to an observing body, optionally corrected for light
        /// time(planetary aberration) and stellar aberration.
        /// </summary>
        /// <param name="targ">Target body name.</param>
        /// <param name="et">Observer epoch.</param>
        /// <param name="refer">Reference frame of output state vector.</param>
        /// <param name="abcorr">Aberration correction flag.</param>
        /// <param name="obs">Observing body name.</param>
        /// <param name="starg">State of target.</param>
        /// <param name="lt">One way light time between observer and target.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void spkez_c(int targ,
                                   double et,
                                   [MarshalAs(UnmanagedType.LPStr)] string refer,
                                   [MarshalAs(UnmanagedType.LPStr)] string abcorr,
                                   int obs,
                                   double[] starg,
                                   ref double lt);

        /// <summary>
        /// Load one or more SPICE kernels into a program.
        /// </summary>
        /// <param name="filename">Name of SPICE kernel file (text or binary).</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void furnsh_c([MarshalAs(UnmanagedType.LPStr)] string filename);

        /// <summary>
        /// Unload a SPICE kernel.
        /// </summary>
        /// <param name="filename">Name of SPICE kernel file (text or binary).</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void unload_c([MarshalAs(UnmanagedType.LPStr)] string filename);

        /// <summary>
        /// Convert a string representing an epoch to a double precision value representing the number of TDB seconds
        /// past the J2000 epoch corresponding to the input epoch.
        /// </summary>
        /// <param name="pictur">A string representing an epoch.</param>
        /// <param name="et">The equivalent value in seconds past J2000, TDB. (output)</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void str2et_c([MarshalAs(UnmanagedType.LPStr)] string pictur, ref double et);

        /// <summary>
        /// Convert from rectangular coordinates to latitudinal coordinates.
        /// </summary>
        /// <param name="rectan">Rectangular coordinates of a point.</param>
        /// <param name="radius">Distance of the point from the origin.</param>
        /// <param name="longitude">Longitude of the point in radians.</param>
        /// <param name="latitude">Latitude of the point in radians.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void reclat_c(double[] rectan,
                                           ref double radius,
                                           ref double longitude,
                                           ref double latitude);


        /// <summary>
        /// Convert from latitudinal coordinates to rectangular coordinates.
        /// </summary>
        /// <param name="radius">Distance of a point from the origin.</param>
        /// <param name="longitude">Longitude of point in radians.</param>
        /// <param name="latitude">Latitude of point in radians.</param>
        /// <param name="rectan">Rectangular coordinates of a point.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void latrec_c(double radius,
                double longitude,
                double latitude,
                double[] rectan);

        /// <summary>
        /// Return the position of a target body relative to an observing body, optionally corrected for light time (planetary aberration) and stellar aberration.
        /// </summary>
        /// <param name="targ">Target body name.</param>
        /// <param name="et">Observer epoch.</param>
        /// <param name="refFrame">Reference frame of output position vector.</param>
        /// <param name="abcorr">Aberration correction flag.</param>
        /// <param name="obs">Observing body name.</param>
        /// <param name="ptarg">Position of target. (output)</param>
        /// <param name="lt">One way light time between observer and target. (output)</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void spkpos_c([MarshalAs(UnmanagedType.LPStr)] string targ,
                                           double et,
                                           [MarshalAs(UnmanagedType.LPStr)] string refFrame,
                                           [MarshalAs(UnmanagedType.LPStr)] string abcorr,
                                           [MarshalAs(UnmanagedType.LPStr)] string obs,
                                           double[] ptarg,
                                           ref double lt);

        /// <summary>
        /// Return the position of a target body relative to an observing body, optionally corrected for light time (planetary aberration) and stellar aberration.
        /// </summary>
        /// <param name="targ">Target body NAIF ID code.</param>
        /// <param name="et">Observer epoch.</param>
        /// <param name="refFrame">Reference frame of output position vector.</param>
        /// <param name="abcorr">Aberration correction flag.</param>
        /// <param name="obs">Observing body NAIF ID code.</param>
        /// <param name="ptarg">Position of target.</param>
        /// <param name="lt">One way light time between observer and target.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void spkezp_c(int targ,
                                           double et,
                                           [MarshalAs(UnmanagedType.LPStr)] string refFrame,
                                           [MarshalAs(UnmanagedType.LPStr)] string abcorr,
                                           int obs,
                                           double[] ptarg,
                                           ref double lt);

        /// <summary>
        /// Compute the geometric position of a target body relative to an observing body.
        /// </summary>
        /// <param name="targ">Target body.</param>
        /// <param name="et">Target epoch.</param>
        /// <param name="refFrame">Target reference frame.</param>
        /// <param name="obs">Observing body.</param>
        /// <param name="pos">Position of target. (output)</param>
        /// <param name="lt">Light time. (output)</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void spkgps_c(int targ,
                                           double et,
                                           [MarshalAs(UnmanagedType.LPStr)] string refFrame,
                                           int obs,
                                           double[] pos,
                                           ref double lt);


        /// <summary>
        /// Compute the geometric state (position and velocity) of a target body relative to an observing body.
        /// </summary>
        /// <param name="targ">Target body.</param>
        /// <param name="et">Target epoch.</param>
        /// <param name="refFrame">Target reference frame.</param>
        /// <param name="obs">Observing body.</param>
        /// <param name="state">State of target. (output)</param>
        /// <param name="lt">Light time. (output)</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void spkgeo_c(int targ,
                                           double et,
                                           [MarshalAs(UnmanagedType.LPStr)] string refFrame,
                                           int obs,
                                           double[] state,
                                           ref double lt);

        /// <summary>
        /// Return the state of a specified target relative to an "observer," 
        /// where the observer has constant position in a specified reference
        /// frame.The observer's position is provided by the calling program 
        /// rather than by loaded SPK files. 
        /// </summary>
        /// <param name="target">Name of target ephemeris object</param>
        /// <param name="et">Observation epoch</param>
        /// <param name="outref">Reference frame of output state</param>
        /// <param name="refloc">Output reference frame evaluation locus</param>
        /// <param name="abcorr">Aberration correction</param>
        /// <param name="obspos">Observer position relative to center of motion</param>
        /// <param name="obsctr">Center of motion of observer</param>
        /// <param name="obsref">Frame of observer position</param>
        /// <param name="state">Output: State of target with respect to observer</param>
        /// <param name="lt">Output: One way light time between target and observer</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void spkcpo_c([MarshalAs(UnmanagedType.LPStr)] string target,
                                   double et,
                                   [MarshalAs(UnmanagedType.LPStr)] string outref,
                                   [MarshalAs(UnmanagedType.LPStr)] string refloc,
                                   [MarshalAs(UnmanagedType.LPStr)] string abcorr,
                                   double[] obspos,
                                   [MarshalAs(UnmanagedType.LPStr)] string obsctr,
                                   [MarshalAs(UnmanagedType.LPStr)] string obsref,
                                   double[] state,
                                   ref double lt);

        /// <summary>
        /// Return the state, relative to a specified observer, of a target
        /// having constant position in a specified reference frame. The
        /// target's position is provided by the calling program rather than by
        /// loaded SPK files.
        /// </summary>
        /// <param name="trgpos">Target position relative to center of motion</param>
        /// <param name="trgctr">Center of motion of target</param>
        /// <param name="trgref">Frame of target position</param>
        /// <param name="et">Observation epoch</param>
        /// <param name="outref">Reference frame of output state</param>
        /// <param name="refloc">Output reference frame evaluation locus</param>
        /// <param name="abcorr">Aberration correction</param>
        /// <param name="obsrvr">Name of observing ephemeris object</param>
        /// <param name="state">State of target with respect to observer</param>
        /// <param name="lt">One way light time between target and observer</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void spkcpt_c(double[] trgpos,
                                  [MarshalAs(UnmanagedType.LPStr)] string trgctr,
                                  [MarshalAs(UnmanagedType.LPStr)] string trgref,
                                  double et,
                                  [MarshalAs(UnmanagedType.LPStr)] string outref,
                                  [MarshalAs(UnmanagedType.LPStr)] string refloc,
                                  [MarshalAs(UnmanagedType.LPStr)] string abcorr,
                                  [MarshalAs(UnmanagedType.LPStr)] string obsrvr,
                                  double[] state,
                                  ref double lt);

        /// <summary>
        /// Convert rectangular coordinates to planetographic coordinates.
        /// </summary>
        /// <param name="body">Body with which coordinate system is associated.</param>
        /// <param name="rectan">Rectangular coordinates of a point.</param>
        /// <param name="re">Equatorial radius of the reference spheroid.</param>
        /// <param name="f">Flattening coefficient.</param>
        /// <param name="lon">Planetographic longitude of the point (radians). (output)</param>
        /// <param name="lat">Planetographic latitude of the point (radians). (output)</param>
        /// <param name="alt">Altitude of the point above reference spheroid. (output)</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void recpgr_c([MarshalAs(UnmanagedType.LPStr)] string body,
                                           double[] rectan,
                                           double re,
                                           double f,
                                           ref double lon,
                                           ref double lat,
                                           ref double alt);

        /// <summary>
        /// Convert planetographic coordinates to rectangular coordinates.
        /// </summary>
        /// <param name="body">Body with which coordinate system is associated.</param>
        /// <param name="lon">Planetographic longitude of a point (radians).</param>
        /// <param name="lat">Planetographic latitude of a point (radians).</param>
        /// <param name="alt">Altitude of a point above reference spheroid.</param>
        /// <param name="re">Equatorial radius of the reference spheroid.</param>
        /// <param name="f">Flattening coefficient.</param>
        /// <param name="rectan">Rectangular coordinates of the point.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void pgrrec_c([MarshalAs(UnmanagedType.LPStr)] string body,
                                           double lon,
                                           double lat,
                                           double alt,
                                           double re,
                                           double f,
                                           double[] rectan);

        //[3]

        /// <summary>
        /// Convert an input time from ephemeris seconds past J2000 to Calendar, Day-of-Year, or Julian Date format, UTC.
        /// </summary>
        /// <param name="et">is the input epoch, ephemeris seconds past J2000.</param>
        /// <param name="format">is the format of the output time string.</param>
        /// <param name="prec">is the number of digits of precision to which fractional seconds (for Calendar and Day-of-Year formats) or days (for Julian Date format) are to be computed. If PREC is zero or smaller, no decimal point is appended to the output string. If PREC is greater than 14, it is treated as 14.</param>
        /// <param name="lenout">The allowed length of the output string.  This length must large enough to hold the output string plus the null terminator.  If the output string is expected to have x characters, lenout must be x + 1.</param>
        /// <param name="utcstr">is the output time string equivalent to the input epoch, in the specified format.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        static extern void et2utc_c(double et,
                                            [MarshalAs(UnmanagedType.LPStr)] string format,
                                            int prec,
                                            int lenout,
                                            [MarshalAs(UnmanagedType.LPStr)] StringBuilder utcstr);

        /// <summary>
        /// Find nearest point on a triaxial ellipsoid to a specified line,
        /// and the distance from the ellipsoid to the line. If the line
        /// intersects the ellipsoid, dist is zero.
        /// </summary>
        /// <param name="a">Length of ellipsoid's semi-axis in the x direction</param>
        /// <param name="b">Length of ellipsoid's semi-axis in the y direction</param>
        /// <param name="c">Length of ellipsoid's semi-axis in the z direction</param>
        /// <param name="linept">Point on line</param>
        /// <param name="linedr">Direction vector of line</param>
        /// <param name="pnear">Nearest point on ellipsoid to line (output)</param>
        /// <param name="dist">Distance of ellipsoid from line (output)</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void npedln_c(double a,
                                           double b,
                                           double c,
                                           double[] linept,
                                           double[] linedr,
                                           double[] pnear,
                                           ref double dist);

        /// <summary>
        /// Find the nearest point on a line to a specified point, and find 
        /// the distance between the two points. 
        /// </summary>
        /// <param name="linpt">Point on a line</param>
        /// <param name="lindir">the line's direction vector</param>
        /// <param name="point">A second point</param>
        /// <param name="pnear">Nearest point on the line to point</param>
        /// <param name="dist">Distance between point and pnear</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void nplnpt_c(double[] linpt,
                                           double[] lindir,
                                           double[] point,
                                           double[] pnear,
                                           ref double dist);

        /// <summary>
        /// Determine the intersection of a line-of-sight vector with the surface of an ellipsoid.
        /// </summary>
        /// <param name="positn">Position of the observer in body-fixed frame.</param>
        /// <param name="u">Vector from the observer in some direction.</param>
        /// <param name="a">Length of the ellipsoid semi-axis along the x-axis.</param>
        /// <param name="b">Length of the ellipsoid semi-axis along the y-axis.</param>
        /// <param name="c"></param>
        /// <param name="point"></param>
        /// <param name="found"></param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void surfpt_c(double[] positn,
                                           double[] u,
                                           double a,
                                           double b,
                                           double c,
                                           double[] point,
                                           ref int found);

        [StructLayout(LayoutKind.Sequential)]
        public struct SpiceEllipse
        {
            public double centerX;
            public double centerY;
            public double centerZ;
            public double semiMajorX;
            public double semiMajorY;
            public double semiMajorZ;
            public double semiMinorX;
            public double semiMinorY;
            public double semiMinorZ;
        }

        /// <summary>
        /// Find the limb of a triaxial ellipsoid, viewed from a specified point.
        /// </summary>
        /// <param name="a">Length of ellipsoid semi-axis lying on the x-axis.</param>
        /// <param name="b">Length of ellipsoid semi-axis lying on the y-axis.</param>
        /// <param name="c">Length of ellipsoid semi-axis lying on the z-axis.</param>
        /// <param name="viewpt">Location of viewing point.</param>
        /// <param name="limb">Limb of ellipsoid as seen from viewing point. (output)</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void edlimb_c(double a,
                                           double b,
                                           double c,
                                           double[] viewpt,
                                           ref SpiceEllipse limb);

        /// <summary>
        /// Convert a CSPICE ellipse to a center vector and two generating vectors.  The selected generating vectors are semi-axes of the ellipse.
        /// </summary>
        /// <param name="ellipse">A CSPICE ellipse.</param>
        /// <param name="center">Center of the ellipse (output)</param>
        /// <param name="smajor">semi-major axis of the ellipse (output)</param>
        /// <param name="sminor">semi-minor axis of the ellipse (output)</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void el2cgv_c(ref SpiceEllipse ellipse,
                                           double[] center,
                                           double[] smajor,
                                           double[] sminor);

        public static string et2utc_c_temp(double et, string format, int prec, int lenout)
        {
            var buf = new StringBuilder(' ', lenout);
            et2utc_c(et, format, prec, lenout, buf);
            return buf.ToString();
        }

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        // private static extern void timout_c(ref double et, [MarshalAs(UnmanagedType.LPStr)]string pictur, ref int lenout, char[] buf);
        public static extern void timout_c(double et,
                                           [MarshalAs(UnmanagedType.LPStr)] string pictur,
                                           int lenout,
                                           char[] buf);

        [StructLayout(LayoutKind.Sequential)]
        public struct SpicePlane
        {
            public double normal0;
            public double normal1;
            public double normal2;
            public double constant;
        }

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void psv2pl_c(double[] point,
                                           double[] span1,
                                           double[] span2,
                                           ref SpicePlane plane);

        /// <summary>
        /// Find the intersection of a ray and a plane.
        /// </summary>
        /// <param name="vertex">Vertex of a ray</param>
        /// <param name="dir">Direction vector of a ray</param>
        /// <param name="plane">A CSPICE plane.</param>
        /// <param name="nxpts">Number of intersection points of ray and plane (output)</param>
        /// <param name="xpt">Intersection point, if nxpts = 1 (output)</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void inrypl_c(double[] vertex,
                                           double[] dir,
                                           ref SpicePlane plane,
                                           ref int nxpts,
                                           double[] xpt);

        /// <summary>
        /// Find the nearest point on an ellipse to a specified point, both 
        /// in three-dimensional space, and find the distance between the 
        ///  ellipse and the point. 
        /// </summary>
        /// <param name="point">Point whose distance to an ellipse is to be found.</param>
        /// <param name="ellips">A CSPICE ellipse.</param>
        /// <param name="pnear">Nearest point on ellipse to input point.</param>
        /// <param name="dist">Distance of input point to ellipse.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void npelpt_c(double[] point,
                                           ref SpiceEllipse ellips,
                                           double[] pnear,
                                           ref double dist);

        /// <summary>
        /// Find the intersection of an ellipse and a plane.
        /// </summary>
        /// <param name="ellips">A CSPICE ellipse.</param>
        /// <param name="plane">A CSPICE plane.</param>
        /// <param name="nxpts">Number of intersection points of plane and ellipse. (output)</param>
        /// <param name="xpt1">Intersection point (output)</param>
        /// <param name="xpt2">Intersection point (output)</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void inelpl_c(ref SpiceEllipse ellips,
                                           ref SpicePlane plane,
                                           ref int nxpts,
                                           double[] xpt1,
                                           double[] xpt2);

        /// <summary>
        /// Negate a 3D vector
        /// </summary>
        /// <param name="v1">input vector</param>
        /// <param name="vout">output vector</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void vminus_c(double[] v1,
                                           double[] vout);

        /// <summary>
        /// Add two 3 dimensional vectors. 
        /// </summary>
        /// <param name="v1">First vector to be added. </param>
        /// <param name="v2">Second vector to be added.</param>
        /// <param name="vout">Sum vector, v1 + v2.  vout can overwrite either v1 or v2. </param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void vadd_c(double[] v1,
                                         double[] v2,
                                         double[] vout);

        /// <summary>
        /// Compute the difference between two 3-dimensional, double precision vectors.
        /// </summary>
        /// <param name="v1">First vector (minuend).</param>
        /// <param name="v2">Second vector (subtrahend).</param>
        /// <param name="vout">Difference vector, v1 - v2. vout can overwrite either v1 or v2.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void vsub_c(double[] v1,
                                         double[] v2,
                                         double[] vout);

        /// <summary>
        /// Compute the normalized cross product of two 3-vectors.
        /// </summary>
        /// <param name="v1">Left vector for cross product.</param>
        /// <param name="v2">Right vector for cross product.</param>
        /// <param name="vout">Normalized cross product (v1xv2) / |v1xv2|. (output)</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void ucrss_c(double[] v1,
                                          double[] v2,
                                          double[] vout);

        /// <summary>
        /// Return the matrix that transforms position vectors from one 
        /// specified frame to another at a specified epoch.
        /// </summary>
        /// <param name="from">Name of the frame to transform from.</param>
        /// <param name="to">Name of the frame to transform to.</param>
        /// <param name="et">Epoch of the rotation matrix.</param>
        /// <param name="rotate">A rotation matrix.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void pxform_c([MarshalAs(UnmanagedType.LPStr)] string from,
                                           [MarshalAs(UnmanagedType.LPStr)] string to,
                                           double et,
                                           double[,] rotate);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static double Distance3(double[] a, double[] b)
        {
            return Math.Sqrt(Math.Pow(a[0] - b[0], 2d) +
                             Math.Pow(a[1] - b[1], 2d) +
                             Math.Pow(a[2] - b[2], 2d));
        }

        public static double Length(double[] a)
        {
            return Math.Sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
        }

        /// <summary>
        /// Pack three scalar components into a vector. 
        /// </summary>
        /// <param name="x">x</param>
        /// <param name="y">y</param>
        /// <param name="z">z</param>
        /// <param name="v">Equivalent 3-vector.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void vpack_c(double x,
                                          double y,
                                          double z,
                                          double[] v);

        /// <summary>
        /// This routine returns the 3x3 identity matrix
        /// </summary>
        /// <param name="m">Output: the 3x3 identity matrix</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void ident_c(double[,] m);

        /// <summary>
        /// Multiply a 3x3 double precision matrix with a 3-dimensional double precision vector.
        /// </summary>
        /// <param name="m1">3x3 double precision matrix</param>
        /// <param name="vin">3-dimensional double precision vector</param>
        /// <param name="vout">3-dimensinoal double precision vector. vout is the product m1*vin</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void mxv_c(double[,] m1,
                                        double[] vin,
                                        double[] vout);

        /// <summary>
        /// mtxv_c multiplies the transpose of a 3x3 matrix on the left with a vector on the right.
        /// </summary>
        /// <param name="m1">3x3 double precision matrix.</param>
        /// <param name="vin">3-dimensional double precision vector.</param>
        /// <param name="vout">3-dimensional double precision vector. vout is the product m1**t * vin.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void mtxv_c(double[,] m1,
                                        double[] vin,
                                        double[] vout);

        /// <summary>
        /// Find a unit quaternion corresponding to a specified rotation matrix.
        /// </summary>
        /// <param name="r">A rotation matrix.</param>
        /// <param name="q">A unit quaternion representing `r'.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void m2q_c(double[,] r,
                                        double[] q);

        /// <summary>
        /// Multiply two 3x3 matrices.
        /// </summary>
        /// <param name="m1">3x3 double precision matrix.</param>
        /// <param name="m2">3x3 double precision matrix.</param>
        /// <param name="mout">3x3 double precision matrix. mout is the product m1*m2.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void mxm_c(double[,] m1,
                                        double[,] m2,
                                        double[,] mout);

        /// <summary>
        /// Construct a rotation matrix that rotates vectors by a specified 
        /// angle about a specified axis.
        /// </summary>
        /// <param name="axis">Rotation axis.</param>
        /// <param name="angle">Rotation angle, in radians.</param>
        /// <param name="r">Rotation matrix corresponding to axis and angle.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void axisar_c(double[] axis,
                                           double angle,
                                           double[,] r);

        /// <summary>
        /// Transform a vector to a new coordinate system rotated by angle 
        /// radians about axis iaxis.  This transformation rotates v1 by 
        /// -angle radians about the specified axis.
        /// </summary>
        /// <param name="v1">Vector whose coordinate system is to be rotated</param>
        /// <param name="angle">Angle of rotation in radians</param>
        /// <param name="iaxis">Axis of rotation (X=1, Y=2, Z=3)</param>
        /// <param name="vout">Resulting vector [angle] expressed in the new coordinate system</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void rotvec_c(double[] v1,
                                           double angle,
                                           int iaxis,
                                           double[] vout);


        /// <summary>
        /// rotmat_c applies a rotation of angle radians about axis iaxis to a 
        /// matrix.This rotation is thought of as rotating the coordinate
        /// system.
        /// </summary>
        /// <param name="m1">Matrix to be rotated.</param>
        /// <param name="angle">Angle of rotation (radians).</param>
        /// <param name="iaxis">Axis of rotation (X=1, Y=2, Z=3).</param>
        /// <param name="mout">Resulting rotated matrix.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void rotmat_c(double[,] m1,
                                           double angle,
                                           int iaxis,
                                           double[,] mout);

        /// <summary>
        /// Compute the magnitude of a double precision, 3-dimensional vector.
        /// </summary>
        /// <param name="v1">Vector whose magnitude is to be found</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double vnorm_c(double[] v1);

        /// <summary>
        /// Compute the dot product of two double precision, 3-dimensional vectors.
        /// </summary>
        /// <param name="v1">This may be any 3-dimensional, double precision vector.</param>
        /// <param name="v2">This may be any 3-dimensional, double precision vector.</param>
        /// <returns>The function returns the value of the dot product of v1 and v2.</returns>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double vdot_c(double[] v1, double[] v2);

        /// <summary>
        /// Compute the cross product of two 3-dimensional vectors.
        /// </summary>
        /// <param name="v1">Left hand vector for cross product.</param>
        /// <param name="v2">Right hand vector for cross product</param>
        /// <param name="vout">Cross product v1xv2.  vout can overwrite either v1 or v2.</param>
        /// <returns></returns>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double vcrss_c(double[] v1,
                                            double[] v2,
                                            double[] vout);

        /// <summary>
        /// Compute the rectangular coordinates of the sub-observer point on 
        /// a target body at a specified epoch, optionally corrected for 
        /// light time and stellar aberration.
        /// </summary>
        /// <param name="method">Computation method.</param>
        /// <param name="target">Name of target body.</param>
        /// <param name="et">Epoch in ephemeris seconds past J2000 TDB.</param>
        /// <param name="fixref">Body-fixed, body-centered target body frame.</param>
        /// <param name="abcorr">Aberration correction.</param>
        /// <param name="obsrvr">Name of observing body.</param>
        /// <param name="spoint">Sub-observer point on the target body.</param>
        /// <param name="trgepc">Sub-observer point epoch.</param>
        /// <param name="srfvec">Vector from observer to sub-observer point.</param>
        /// <returns></returns>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double subpnt_c([MarshalAs(UnmanagedType.LPStr)] string method,
                                             [MarshalAs(UnmanagedType.LPStr)] string target,
                                             double et,
                                             [MarshalAs(UnmanagedType.LPStr)] string fixref,
                                             [MarshalAs(UnmanagedType.LPStr)] string abcorr,
                                             [MarshalAs(UnmanagedType.LPStr)] string obsrvr,
                                             double[] spoint,
                                             ref double trgepc,
                                             double[] srfvec);

        /// <summary>
        /// Fetch from the kernel pool the double precision values of an item
        /// associated with a body, where the body is specified by an integer ID
        /// code.
        /// </summary>
        /// <param name="bodyid">Body ID code.</param>
        /// <param name="item">Item for which values are desired. ("RADII" ...)</param>
        /// <param name="maxn">Maximum number of values that may be returned.</param>
        /// <param name="dim">Number of values returned.</param>
        /// <param name="values">Values.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void bodvcd_c(Int32 bodyid,
                                           [MarshalAs(UnmanagedType.LPStr)] string item,
                                           Int32 maxn,
                                           ref Int32 dim,
                                           double[] values);

        /// <summary>
        /// Find the transformation to the right-handed frame having a 
        /// given vector as a specified axis and having a second given 
        /// vector lying in a specified coordinate plane. 
        /// </summary>
        /// <param name="axdef">Vector defining a principal axis.</param>
        /// <param name="indexa">Principal axis number of axdef (X=1, Y=2, Z=3).</param>
        /// <param name="plndef">Vector defining (with axdef) a principal plane.</param>
        /// <param name="indexp">Second axis number (with indexa) of principal plane.</param>
        /// <param name="mout">Output rotation matrix.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void twovec_c(double[] axdef,
                                           int indexa,
                                           double[] plndef,
                                           int indexp,
                                           double[,] mout);

        /// <summary>
        /// Generate the inverse of a 3x3 matrix.
        /// </summary>
        /// <param name="m1">Matrix to be inverted.</param>
        /// <param name="mout">Inverted matrix (m1)**-1.  If m1 is singular, then mout will be the zero matrix.   mout can overwrite m1. </param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void invert_c(double[,] m1,
                                           double[,] mout);

        /// <summary>
        /// Multiply the transpose of a 3x3 matrix and a 3x3 matrix.
        /// </summary>
        /// <param name="m1">3x3 double precision matrix.</param>
        /// <param name="m2">3x3 double precision matrix.</param>
        /// <param name="mout">The produce m1 transpose times m2.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void mtxm_c(double[,] m1,
                                         double[,] m2,
                                         double[,] mout);

        /// <summary>
        /// Find the rotation matrix corresponding to a specified unit quaternion. 
        /// </summary>
        /// <param name="q">A unit quaternion.</param>
        /// <param name="r">A rotation matrix corresponding to `q'.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void q2m_c(double[] q,
                                        double[,] r);

        /// <summary>
        /// Factor a rotation matrix as a product of three rotations about
        /// specified coordinate axes.
        /// </summary>
        /// <param name="m1">A rotation matrix to be factored.</param>
        /// <param name="axis3">Number of third rotation axis</param>
        /// <param name="axis2">Number of second rotation axis</param>
        /// <param name="axis1">Number of first rotation axis</param>
        /// <param name="angle3">Third Euler angle</param>
        /// <param name="angle2">Second Euler angle</param>
        /// <param name="angle1">First Euler angle</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void m2eul_c(double[,] m1,
                                 int axis3,
                                 int axis2,
                                 int axis1,
                                 ref double angle3,
                                 ref double angle2,
                                 ref double angle1);

        /// <summary>
        /// Construct a rotation matrix from a set of Euler angles
        /// </summary>
        /// <param name="angle3">Third Euler angle</param>
        /// <param name="angle2">Second Euler angle</param>
        /// <param name="angle1">First Euler angle</param>
        /// <param name="axis3">Number of third rotation axis</param>
        /// <param name="axis2">Number of second rotation axis</param>
        /// <param name="axis1">Number of first rotation axis</param>
        /// <param name="m1">Product of the 3 rotations</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void eul2m_c(
            double angle3,
            double angle2,
            double angle1,
            int axis3,
            int axis2,
            int axis1,
            double[,] m1);

        /// <summary>
        /// Look up the frame ID code associated with a string.
        /// </summary>
        /// <param name="frname">The name of some reference frame.</param>
        /// <param name="frcode">The SPICE ID code of the frame.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void namfrm_c([MarshalAs(UnmanagedType.LPStr)] string frname,
                                           ref Int32 frcode);

        /// <summary>
        /// Convert planetocentric latitude and longitude of a surface 
        /// point on a specified body to rectangular coordinates.
        /// </summary>
        /// <param name="body">NAIF integer code of an extended body</param>
        /// <param name="longitude">Longitude of point in radians</param>
        /// <param name="latitude">Latitude of point in radians</param>
        /// <param name="rectan">Rectangular coordinates of the point</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void srfrec_c(int body,
                                   double longitude,
                                   double latitude,
                                   double[] rectan);

        /// <summary>
        /// Open a new CK file, returning the handle of the opened file.
        /// </summary>
        /// <param name="fname">The name of the CK file to be opened.</param>
        /// <param name="ifname">The internal filename for the CK.</param>
        /// <param name="ncomch">The number of characters to reserve for comments.</param>
        /// <param name="handle">The handle of the opened CK file.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void ckopn_c([MarshalAs(UnmanagedType.LPStr)] string fname,
                                          [MarshalAs(UnmanagedType.LPStr)] string ifname,
                                          int ncomch,
                                          ref int handle);

        /// <summary>
        /// Close an open CK file.
        /// </summary>
        /// <param name="handle">Handle of the CK file to be closed.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void ckcls_c(int handle);

        /// <summary>
        /// Add a type 3 segment to a C-kernel.
        /// </summary>
        /// <param name="handle">Handle of an open CK file.</param>
        /// <param name="begtim">The beginning encoded SCLK of the segment.</param>
        /// <param name="endtim">The ending encoded SCLK of the segment.</param>
        /// <param name="inst">The NAIF instrument ID code.</param>
        /// <param name="refFrame">The reference frame of the segment.</param>
        /// <param name="avflag">True if the segment will contain angular velocity.</param>
        /// <param name="segid">Segment identifier.</param>
        /// <param name="nrec">Number of pointing records.</param>
        /// <param name="sclkdp">Encoded SCLK times.</param>
        /// <param name="quats">Quaternions representing instrument pointing.</param>
        /// <param name="avvs">Angular velocity vectors.</param>
        /// <param name="nints">Number of intervals.</param>
        /// <param name="starts">Encoded SCLK interval start times.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void ckw03_c(int handle,
                                          double begtim,
                                          double endtim,
                                          int inst,
                                          [MarshalAs(UnmanagedType.LPStr)] string refFrame,
                                          int avflag, // SpiceBoolean
                                          [MarshalAs(UnmanagedType.LPStr)] string segid,
                                          int nrec,
                                          double[] sclkdp,
                                          double[] quats, //[][4],
                                          double[] avvs,  //[][3],
                                          int nints,
                                          double[] starts);

        //           handle     is the handle of the CK file to which the segment will 
        //              be written. The file must have been opened with write 
        //              access. 
        // 
        //   begtim     is the beginning encoded SCLK time of the segment. This 
        //              value should be less than or equal to the first time in 
        //              the segment. 
        // 
        //   endtim     is the encoded SCLK time at which the segment ends. 
        //              This value should be greater than or equal to the last 
        //              time in the segment. 
        // 
        //   inst       is the NAIF integer ID code for the instrument. 
        // 
        //   ref        is a character string which specifies the  
        //              reference frame of the segment. This should be one of 
        //              the frames supported by the SPICELIB routine NAMFRM 
        //              which is an entry point of FRAMEX. 
        // 
        //              The rotation matrices represented by the quaternions
        //              that are to be written to the segment transform the
        //              components of vectors from the inertial reference frame
        //              specified by ref to components in the instrument fixed
        //              frame. Also, the components of the angular velocity
        //              vectors to be written to the segment should be given
        //              with respect to ref.
        //
        //              ref should be the name of one of the frames supported
        //              by the SPICELIB routine NAMFRM.
        //
        //
        //   avflag     is a boolean flag which indicates whether or not the 
        //              segment will contain angular velocity. 
        // 
        //   segid      is the segment identifier.  A CK segment identifier may 
        //              contain up to 40 characters, excluding the terminating
        //              null.
        // 
        //   nrec       is the number of pointing instances in the segment. 
        // 
        //   sclkdp     are the encoded spacecraft clock times associated with 
        //              each pointing instance. These times must be strictly 
        //              increasing. 
        // 
        //   quats      is an array of SPICE-style quaternions representing a
        //              sequence of C-matrices. See the discussion of "Quaternion
        //              Styles" in the Particulars section below.
        //
        //              The C-matrix represented by the ith quaternion in
        //              quats is a rotation matrix that transforms the
        //              components of a vector expressed in the inertial
        //              frame specified by ref to components expressed in
        //              the instrument fixed frame at the time sclkdp[i].
        //
        //              Thus, if a vector V has components x, y, z in the
        //              inertial frame, then V has components x', y', z' in
        //              the instrument fixed frame where:
        //
        //                   [ x' ]     [          ] [ x ]
        //                   | y' |  =  |   cmat   | | y |
        //                   [ z' ]     [          ] [ z ]
        //
        //   avvs       are the angular velocity vectors ( optional ).
        //
        //              The ith vector in avvs gives the angular velocity of
        //              the instrument fixed frame at time sclkdp[i]. The
        //              components of the angular velocity vectors should
        //              be given with respect to the inertial reference frame
        //              specified by ref.
        //
        //              The direction of an angular velocity vector gives
        //              the right-handed axis about which the instrument fixed
        //              reference frame is rotating. The magnitude of the
        //              vector is the magnitude of the instantaneous velocity
        //              of the rotation, in radians per second.
        //
        //              If avflag is FALSE then this array is ignored by the
        //              routine; however it still must be supplied as part of
        //              the calling sequence.
        //
        //   nints      is the number of intervals that the pointing instances
        //              are partitioned into.
        //
        //   starts     are the start times of each of the interpolation
        //              intervals. These times must be strictly increasing
        //              and must coincide with times for which the segment
        //              contains pointing.


        /// <summary>
        /// Get pointing (attitude) for a specified spacecraft clock time.
        /// </summary>
        /// <param name="inst">NAIF ID of instrument, spacecraft, or structure.</param>
        /// <param name="sclkdp">Encoded spacecraft clock time.</param>
        /// <param name="tol">Time tolerance.</param>
        /// <param name="refFrame">Reference frame.</param>
        /// <param name="cmat">C-matrix pointing data.</param>
        /// <param name="clkout">Output encoded spacecraft clock time.</param>
        /// <param name="found">True when requested pointing is available.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void ckgp_c(int inst,
                                         double sclkdp,
                                         double tol,
                                         [MarshalAs(UnmanagedType.LPStr)] string refFrame,
                                         double[,] cmat,
                                         ref double clkout,
                                         ref int found);


        /// <summary>
        /// Convert ephemeris seconds past J2000 (ET) to continuous encoded  
        /// spacecraft clock (`ticks').  Non-integral tick values may be 
        /// returned.
        /// </summary>
        /// <param name="sc">NAIF spacecraft ID code.</param>
        /// <param name="et">Ephemeris time, seconds past J2000.</param>
        /// <param name="sclkdp">SCLK, encoded as ticks since spacecraft clock start.  sclkdp need not be integral.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void sce2c_c(int sc,
                                         double et,
                                         ref double sclkdp);

        /// <summary>
        /// Convert encoded spacecraft clock (`ticks') to ephemeris seconds past J2000 (ET).
        /// </summary>
        /// <param name="sc">NAIF spacecraft ID code.</param>
        /// <param name="sclkdp">SCLK, encoded as ticks since spacecraft clock start</param>
        /// <param name="et">Ephemeris time, seconds past J2000.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void sct2e_c(int sc,
                                         double sclkdp,
                                         ref double et);

        /// <summary>
        /// Convert encoded spacecraft clock ticks to character clock format.
        /// </summary>
        /// <param name="sc">NAIF spacecraft identification code.</param>
        /// <param name="ticks">Encoded representation of a spacecraft clock count.</param>
        /// <param name="lenout">Maximum allowed length of output string.</param>
        /// <param name="clkstr">Character representation of a clock count.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void scfmt_c(int sc,
                                          double ticks,
                                          int lenout,
                                          StringBuilder clkstr);

        /// <summary>
        /// Retrieve or set the list of error message items
        /// to be output when an error is detected.
        /// </summary>
        /// <param name="op">The operation:  "GET" or "SET".</param>
        /// <param name="lenout">Length of list for output.</param>
        /// <param name="clkstr">is an input argument when op is "SET."  It takes the 
        ///    values,  "ABORT",  "IGNORE", "REPORT", "RETURN", and 
        ///    "DEFAULT".</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void errprt_c([MarshalAs(UnmanagedType.LPStr)] string op,
                                          int lenout,
                                          StringBuilder clkstr);

        /// <summary>
        /// Retrieve or set the default error action.
        /// </summary>
        /// <param name="op">indicates the operation -- "GET" or "SET".  "GET" means,
        ///    "Set action to the current value of the error response
        ///    action."  "SET" means, "update the error response action to
        ///    the value indicated by action."</param>
        /// <param name="lenout">Length of list for output.</param>
        /// <param name="clkstr"></param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void erract_c([MarshalAs(UnmanagedType.LPStr)] string op,
                                          int lenout,
                                          StringBuilder clkstr);

        /// <summary>
        /// True if an error condition has been signalled via sigerr_c.
        /// failed_c is the CSPICE status indicator.
        /// </summary>
        /// <returns></returns>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int failed_c();

        /// <summary>
        /// Reset the CSPICE error status to a value of "no error."
        /// As a result, the status routine, failed_c, will return a value
        /// of SPICEFALSE
        /// </summary>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void reset_c();

        /// <summary>
        /// Set the cardinality of a SPICE cell of any data type.
        /// </summary>
        /// <param name="card">Cardinality of (number of elements in) the cell.</param>
        /// <param name="cell">The cell.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void scard_c(int card,
                                          ref SpiceDoubleCell cell);

        /// <summary>
        /// Return the cardinality (number of intervals) of a double precision window.
        /// </summary>
        /// <param name="cell">Input window </param>
        /// <returns>The function returns the window cardinality of the window.</returns>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int wncard_c(ref SpiceDoubleCell cell);

        /// <summary>
        /// Find the coverage window for a specified object in a specified CK
        /// file.
        /// </summary>
        /// <param name="ck">is the name of a C-kernel.</param>
        /// <param name="idcode">is the integer ID code of an object, normally a
        /// spacecraft structure or instrument, for which
        /// pointing data are expected to exist in the specified
        /// CK file.</param>
        /// <param name="needav">is a logical variable indicating whether only
        /// segments having angular velocity are to be considered</param>
        /// when determining coverage.  When `needav' is
        /// SPICETRUE, segments without angular velocity don't
        /// contribute to the coverage window; when `needav' is
        /// SPICEFALSE, all segments for `idcode' may contribute
        /// to the coverage window.</param>
        /// <param name="level">is the level (granularity) at which the coverage 
        ///          is examined.  Allowed values and corresponding 
        ///          meanings are: 
        ///
        ///             "SEGMENT"    The output coverage window contains
        ///                          intervals defined by the start and
        ///                          stop times of segments for the object
        ///                          designated by `idcode'.
        ///
        ///             "INTERVAL"   The output coverage window contains
        ///                          interpolation intervals of segments
        ///                          for the object designated by
        ///                          `idcode'.  For type 1 segments, which
        ///                          don't have interpolation intervals,
        ///                          each epoch associated with a pointing
        ///                          instance is treated as a singleton
        ///                          interval; these intervals are added
        ///                          to the coverage window.
        ///
        ///                          All interpolation intervals are
        ///                          considered to lie within the segment
        ///                          bounds for the purpose of this
        ///                          summary:  if an interpolation
        ///                          interval extends beyond the segment
        ///                          coverage interval, only its
        ///                          intersection with the segment
        ///                          coverage interval is considered to
        ///                          contribute to the total coverage.</param>
        /// <param name="tol">is a tolerance value expressed in ticks of the
        ///          spacecraft clock associated with IDCODE.  Before each
        ///          interval is inserted into the coverage window, the
        ///          interval is intersected with the segment coverage
        ///          interval, then if the intersection is non-empty, it
        ///          is expanded by `tol': the left endpoint of the
        ///          intersection interval is reduced by `tol' and the
        ///          right endpoint is increased by `tol'. Adjusted
        ///          interval endpoints, when expressed as encoded SCLK,
        ///          never are less than zero ticks.  Any intervals that
        ///          overlap as a result of the expansion are merged.
        /// 
        ///          The coverage window returned when tol > 0 indicates
        ///          the coverage provided by the file to the CK readers
        ///          ckgpav_c and ckgp_c when that value of `tol' is
        ///          passed to them as an input.</param>
        /// <param name="timsys">is a string indicating the time system used in the
        ///          output coverage window.  `timsys' may have the
        ///          values:
        ///
        ///              "SCLK"    Elements of `cover' are expressed in 
        ///                        encoded SCLK ("ticks"), where the 
        ///                        clock is associated with the object 
        ///                        designated by `idcode'. 
        ///
        ///              "TDB"     Elements of `cover' are expressed as 
        ///                        seconds past J2000 TDB.</param>
        /// <param name="cell">is an initialized CSPICE window data structure.
        ///          `cover' optionally may contain coverage data on
        ///          input; on output, the data already present in `cover'
        ///          will be combined with coverage found for the object
        ///          designated by `idcode' in the file `ck'.
        ///
        ///          If `cover' contains no data on input, its size and
        ///          cardinality still must be initialized.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void ckcov_c([MarshalAs(UnmanagedType.LPStr)] string ck,
                                          int idcode,
                                          int needav,
                                          [MarshalAs(UnmanagedType.LPStr)] string level,
                                          double tol,
                                          [MarshalAs(UnmanagedType.LPStr)] string timsys,
                                          ref SpiceDoubleCell cell);

        [StructLayout(LayoutKind.Sequential)]
        public unsafe struct SpiceDoubleCell
        {
            public int typ;
            public int length;
            public int size;
            public int reserved;
            public int bool1;
            public int bool2;
            public int bool3;
            public double* ptr1;
            public double* ptr2;
        }

        /// <summary>
        /// Fetch a particular interval from a double precision window.
        /// emission) at a specified surface point of a target body.
        /// </summary>
        /// <param name="cell">Input window.</param>
        /// <param name="n">Index of interval to be fetched.</param>
        /// <param name="left">Left endpoint of the nth interval.</param>
        /// <param name="right">Right endpoint of the nth interval.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void wnfetd_c(ref SpiceDoubleCell cell,
                                           int n,
                                           ref double left,
                                           ref double right);

        /// <summary>
        /// Find the illumination angles (phase, solar incidence, and
        /// emission) at a specified surface point of a target body.</summary>
        /// <param name="method">Computation method.</param>
        /// <param name="target">Name of target body.</param>
        /// <param name="et">Epoch in ephemeris seconds past J2000 TDB.</param>
        /// <param name="fixref">Body-fixed, body-centered target body frame.</param>
        /// <param name="abcorr">Desired aberration correction.</param>
        /// <param name="obsrvr">Name of observing body.</param>
        /// <param name="spoint">Body-fixed coordinates of a target surface point.</param>
        /// <param name="trgepc">Target surface point epoch.</param>
        /// <param name="srfvec">Vector from observer to target surface point.</param>
        /// <param name="phase">Phase angle at the surface point.</param>
        /// <param name="solar">Solar incidence angle at the surface point.</param>
        /// <param name="emissn">Emission angle at the surface point.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void ilumin_c([MarshalAs(UnmanagedType.LPStr)] string method,
                                           [MarshalAs(UnmanagedType.LPStr)] string target,
                                           double et,
                                           [MarshalAs(UnmanagedType.LPStr)] string fixref,
                                           [MarshalAs(UnmanagedType.LPStr)] string abcorr,
                                           [MarshalAs(UnmanagedType.LPStr)] string obsrvr,
                                           double[] spoint,
                                           ref double trgepc,
                                           double[] srfvec,
                                           ref double phase,
                                           ref double solar,
                                           ref double emissn);

        /// <summary>
        /// Fetch from the kernel pool the double precision values of an item
        /// associated with a body, where the body is specified by an integer ID
        /// code.
        /// </summary>
        /// <param name="bodyid">Body ID code.</param>
        /// <param name="item">Item for which values are desired. ("RADII", "NUT_PREC_ANGLES", etc. )</param>
        /// <param name="maxn">Maximum number of values that may be returned.</param>
        /// <param name="dim">Number of values returned.</param>
        /// <param name="values">Values.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void bodvcd_c(int bodyid,
                                           [MarshalAs(UnmanagedType.LPStr)] string item,
                                           int maxn,
                                           ref int dim,
                                           ref double values);

        /// <summary>
        /// Determine the set of osculating conic orbital elements that
        /// corresponds to the state (position, velocity) of a body at
        /// some epoch.
        /// </summary>
        /// <param name="state">State of body at epoch of elements (array of 6)</param>
        /// <param name="et">Epoch of elements</param>
        /// <param name="mu">Gravitational parameter (GM) of primary body</param>
        /// <param name="elts">Equivalent conic elements (array of 8)
        ///    The equivalent conic elements describing the orbit
        ///      of the body around its primary. The elements are,
        ///      in order:
        ///
        ///            rp      Perifocal distance.
        ///            ecc     Eccentricity.
        ///            inc     Inclination.
        ///            lnode   Longitude of the ascending node.
        ///            argp    Argument of periapsis.
        ///            m0      Mean anomaly at epoch.
        ///            t0      Epoch.
        ///            mu      Gravitational parameter.
        ///
        ///      The epoch of the elements is the epoch of the input
        ///      state. Units are km, rad, rad/sec. The same elements
        ///      are used to describe all three types (elliptic,
        ///      hyperbolic, and parabolic) of conic orbit.
        /// </param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void oscelt_c(double[] state,
                                           double et,
                                           double mu,
                                           double[] elts);


        /// <summary>
        /// Determine the state (position, velocity) of an orbiting body
        /// from a set of elliptic, hyperbolic, or parabolic orbital
        /// elements.
        /// </summary>
        /// <param name="elts">Conic elements (see oscelt_c)</param>
        /// <param name="et">Input time</param>
        /// <param name="state">State of orbiting body at et</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void conics_c(double[] elts,
                                           double et,
                                           double[] state);

        /// <summary>
        /// This entry point provides toolkit programmers a method for 
        /// programmatically inserting double precision data into the 
        /// kernel pool
        /// </summary>
        /// <param name="name">The kernel pool name to associate with dvals.</param>
        /// <param name="n">The number of values to insert.</param>
        /// <param name="dvals">An array of values to insert into the kernel pool.</param>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void pdpool_c([MarshalAs(UnmanagedType.LPStr)] string name,
                                           int n,
                                           ref double dvals);

        //public static string SpiceKernelDirectory = @"E:\Tasks\UVS_Spice\SpiceKernels\"; // @"..\..\..\SpiceKernels\";
        //public static string SpiceKernelDirectory = @"c:\UVS\svn\src\TestData\kernels\";
        public static string SpiceKernelDirectory = @"kernels\";

        public static string StandardKernelFiles =
            @"de421.bsp, moon_pa_de421_1900-2050.bpc, moon_080317.tf, moon_assoc_me.tf, pck00010.tpc, naif0010.tls";

        //, naif0010.tls.pc

        public void LoadStandardKernels()
        {
            LoadStandardKernels(StandardKernelFiles);
        }

        public void LoadStandardKernels(string kernels)
        {
            string[] filenames = kernels.Split(',').Select(name => name.Trim()).ToArray();
            LoadStandardKernels(filenames);
        }

        public void LoadStandardKernels(string[] filenames)
        {
            foreach (var filename in filenames)
            {
                string fullname = Path.GetFullPath(Path.Combine(SpiceKernelDirectory, filename));
                Console.WriteLine("Furnishing {0}", fullname);
                furnsh_c(fullname);
            }
        }

        public void LoadEphemeris(string filename)
        {
            furnsh_c(filename);
        }

        public double[,] GetMoonFixedToMoonSolarTransform(double et)
        {
            // Convert epoch to ephemeris time
            double lt = 0d;

            // Ecliptic Normal
            // normal to ecliptic plane is positive Z axis, transform to Lunar ME frame
            double[] zAxis = { 0d, 0d, 1d };
            var xform = new double[3, 3];
            var eclipticNormal = new double[3];
            pxform_c("ECLIPJ2000", "MOON_ME", et, xform);
            mxv_c(xform, zAxis, eclipticNormal);

            // Earth-Moon vector in ME frame
            var earthMoon = new double[3];
            spkpos_c("MOON", et, "MOON_ME", "NONE", "EARTH", earthMoon, ref lt);

            // Earth-Sun vector in ME frame
            var earthSun = new double[3];
            spkpos_c("SUN", et, "MOON_ME", "NONE", "EARTH", earthSun, ref lt);

            var sunMoon = new double[3];
            spkpos_c("MOON", et, "MOON_ME", "NONE", "SUN", sunMoon, ref lt);
            double mag2 = vnorm_c(sunMoon);
            sunMoon[0] /= mag2;
            sunMoon[1] /= mag2;
            sunMoon[2] /= mag2;

            // Get selenographic to local time transformation according to Ken Galal's formula:
            //
            // x1 = (R_Moon - R_Sun) / |R_Moon - R_Sun|,
            // x2 = (x_1 \cross Ecliptic_Normal) / |x_1 \cross Ecliptic_Normal|
            // x3 = (x_1 \cross x_2) / | x_1 \cross x_2|
            //
            // X1
            var x1 = new double[3];
            x1[0] = earthMoon[0] - earthSun[0];
            x1[1] = earthMoon[1] - earthSun[1];
            x1[2] = earthMoon[2] - earthSun[2];
            double mag = vnorm_c(x1);
            x1[0] /= mag;
            x1[1] /= mag;
            x1[2] /= mag;

            // X2
            var x2 = new double[3];
            vcrss_c(x1, eclipticNormal, x2);
            mag = vnorm_c(x2);
            x2[0] /= mag;
            x2[1] /= mag;
            x2[2] /= mag;

            // X3
            var x3 = new double[3];
            vcrss_c(x1, x2, x3);
            mag = vnorm_c(x3);
            x3[0] /= mag;
            x3[1] /= mag;
            x3[2] /= mag;

            for (int i = 0; i < 3; ++i)
            {
                xform[0, i] = x1[i];
                xform[1, i] = x2[i];
                xform[2, i] = x3[i];
            }
            return xform;
        }

        public static double ToDegrees(double rad) => rad * 180d / 3.14159265358979323846626d;
        public static float ToDegrees(float rad) => rad * 180f / 3.14159265358979323846626f;

        public static void Normalize(double[] vec3)
        {
            double m = vnorm_c(vec3);
            vec3[0] /= m;
            vec3[1] /= m;
            vec3[2] /= m;
        }

        public static void SpiceToEngineeringQuaternion(double[] s, double[] e)
        {
            e[0] = -s[1];
            e[1] = -s[2];
            e[2] = -s[3];
            e[3] = s[0];
        }

        #region CK Extent

        public static unsafe bool CkGetCoverage(string filename, out long start, out long stop)
        {
            //var idcode = LadeeId;
            const int idcode = -12000;
            const int needav = 0;
            const string level = "SEGMENT";
            const int tol = 0;
            const string timsys = "SCLK";

            const int SPICE_DP = 1;
            const int SPICE_CELL_CTRLSZ = 6;
            const int MAXIV = 100000;
            const int WINSIZ = (2 * MAXIV);

            //size=WINSIZ

            var SPICE_CELL_name = new double[SPICE_CELL_CTRLSZ + WINSIZ];
            var name = new SpiceDoubleCell
            {
                typ = SPICE_DP,
                length = 0,
                size = WINSIZ,
                reserved = 0,
                bool1 = 1,
                bool2 = 0,
                bool3 = 0
            };

            fixed (double* p1 = SPICE_CELL_name)
            {
                name.ptr1 = p1;
                name.ptr2 = p1 + SPICE_CELL_CTRLSZ * sizeof(double);

                scard_c(0, ref name);
                ckcov_c(filename, idcode, needav, level, tol, timsys, ref name);
            }

            int card = wncard_c(ref name);

            long earliest = long.MaxValue;
            long latest = long.MinValue;
            for (int i = 0; i < card; i++)
            {
                double b = 0d, e = 0d;
                wnfetd_c(ref name, i, ref b, ref e);

                var begin = (long)b;
                var end = (long)e;
                earliest = Math.Min(earliest, begin);
                latest = Math.Max(latest, end);
            }
            start = earliest;
            stop = latest;
            return true;
        }

        private static IntPtr ResolveCSpiceLibrary(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
        {
            if (!string.Equals(libraryName, LibraryName, StringComparison.Ordinal))
                return IntPtr.Zero;

            foreach (var candidate in EnumerateCandidateLibraryPaths())
            {
                if (!File.Exists(candidate))
                    continue;
                if (NativeLibrary.TryLoad(candidate, out var handle))
                    return handle;
            }

            return IntPtr.Zero;
        }

        private static IEnumerable<string> EnumerateCandidateLibraryPaths()
        {
            var fileName = "libcspice.so";
            var assemblyDir = Path.GetDirectoryName(typeof(CSpice).Assembly.Location);
            if (!string.IsNullOrWhiteSpace(assemblyDir))
                yield return Path.Combine(assemblyDir, fileName);

            yield return Path.Combine(AppContext.BaseDirectory, fileName);

            foreach (var ancestor in EnumerateAncestorDirectories())
                yield return Path.Combine(ancestor, "native", "third_party", "cspice", GetRuntimeIdDirectory(), fileName);
        }

        private static IEnumerable<string> EnumerateAncestorDirectories()
        {
            var directory = new DirectoryInfo(AppContext.BaseDirectory);
            while (directory is not null)
            {
                yield return directory.FullName;
                directory = directory.Parent;
            }
        }

        private static string GetRuntimeIdDirectory()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                return "linux-x64";
            throw new PlatformNotSupportedException("CSPICE loading is only configured for Linux.");
        }

        #endregion

    }
}
