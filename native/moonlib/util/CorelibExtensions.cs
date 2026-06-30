using moonlib.math;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Globalization;
using System.Text.RegularExpressions;

namespace moonlib.util
{
    public static class CorelibExtensions
    {
        public static PointF ToPointF(this Point a) => new PointF(a.X.ToCoord(), a.Y.ToCoord());
        public static PointF ToPointF(this PointD a) => new PointF((float)a.X, (float)a.Y);

        public static Point ToPoint(this PointD p) => new Point(p.X.ToFixed(), p.Y.ToFixed());
        public static Point ToPoint(this PointF p) => new Point(p.X.ToFixed(), p.Y.ToFixed());

        public static PointD ToPointD(this Point p) => new PointD(p.X, p.Y);
        public static PointD ToPointD(this PointF p) => new PointD(p.X, p.Y);

        /// <summary>
        /// Creates a unit vector based on a single point.
        /// https://stackoverflow.com/questions/10011687/c-sharp-normalize-like-a-vector
        /// </summary>
        /// <param name="A"></param>
        /// <returns></returns>
        public static PointF Normalize(this PointF A)
        {
            float distance = Magnitude(A);

            if(distance != 0f)
                return new PointF(A.X / distance, A.Y / distance);
            return new PointF(0.0f, 0.0f);
        }

        /// <summary>
        /// Returns the point magnitude (like Vector for a pointF type)
        /// </summary>
        /// <param name="A"></param>
        /// <returns></returns>
        public static float Magnitude(this PointF A)
        {
            return (float)Math.Sqrt(A.X * A.X + A.Y * A.Y);
        }

        public static PointF Perpendicular(this PointF A, PointF B)
        {
            PointF diffVector = A.Sub(B);
            PointF result = new PointF(-diffVector.Y, diffVector.X);
            return result;
        }
       

        public static float ToCoord(this int i) => i;

        // TODO: When this is used to go from map to pixel coordinates, then it shouldn't have the 0.5f per agreement with Edward.
        // I'm not changing it here, because I don't know where else this is used right now.  (6/24/2021)
        public static int ToFixed(this float f) => (int)(f + 0.5f);
        public static int ToFixed(this double f) => (int)(f + 0.5d);

        public static int Round(this double v) => (int)Math.Round(v);
        public static int Round(this float v) => (int)Math.Round(v);
        public static float ToPercent(this float v) => v * 100f;
        public static float ToPercent(this double v) => (float)(v * 100d);

        public static Rectangle ToRectangle(this RectangleF r) => new Rectangle(Round(r.Left), Round(r.Top), Round(r.Width), Round(r.Height));
        public static RectangleF ToRectangleF(this Rectangle r) => new RectangleF(r.Left, r.Top, r.Width, r.Height);

        //public static PointF ToPointF(this Point p) => new PointF(p.X, p.Y);

        public static string Format(this TimeSpan span) => span.ToString(Constants.DefaultDurationTimeFormat, CultureInfo.InvariantCulture);

        // DateTime

        public static string Format(this DateTime dt) => dt.ToString(Constants.FutureDateFormat, CultureInfo.InvariantCulture);
        public static string ToFilename(this DateTime dt) => dt.ToString(Constants.DateToFilenameFormat);

        public static DateTime ClipHours(this DateTime d) => new DateTime(d.Year, d.Month, d.Day, 0, 0, 0, DateTimeKind.Utc);

        // From the microsoft documentation: https://docs.microsoft.com/en-us/dotnet/api/system.string.contains?view=netframework-4.8
        public static bool Contains(this String str, String substring, StringComparison comp)
        {
            if (substring == null)
                throw new ArgumentNullException("substring",
                                             "substring cannot be null.");
            else if (!Enum.IsDefined(typeof(StringComparison), comp))
                throw new ArgumentException("comp is not a member of StringComparison",
                                         "comp");

            return str.IndexOf(substring, comp) >= 0;
        }

        public static string MidString(this string str, string pattern1, string pattern2)
        {
            var index1 = str.IndexOf(pattern1);
            var index2 = str.IndexOf(pattern2);
            if (!(index1 >= 0 && index2 >= 0 && index2 > index1))
                return string.Empty;
            var result = str.Substring(index1 + pattern1.Length, index2 - (index1 + pattern1.Length));
            return result;
        }

        public static DateTime ParseDate(this string str)
        {
            if(str is null)
                return ViperDate.Default;

            var l = str.Length;
            if (l == 0) throw new Exception("string cannot be of zero length");
            if (str[l - 1] == 'Z')
            {
                if(str.Contains("T"))
                    return DateTime.Parse(str, null, DateTimeStyles.RoundtripKind);
                
                if (str[l-5] == '.')
                    return DateTime.ParseExact(str, Constants.DefaultDateFormat, CultureInfo.InvariantCulture, DateTimeStyles.AdjustToUniversal);

                return DateTime.ParseExact(
                    str,
                    Constants.FutureDateFormat,
                    CultureInfo.InvariantCulture,
                    DateTimeStyles.AdjustToUniversal);
            }

            var t = DateTime.ParseExact(str, "yyyy-MM-ddTHH:mm:ss", CultureInfo.InvariantCulture);
            return new DateTime(t.Year, t.Month, t.Day, t.Hour, t.Minute, t.Second, DateTimeKind.Utc);
        }

        public static DateTime ParseDateFilename(this string str)
        {
            var t = DateTime.ParseExact(str, Constants.DateFileFormat, CultureInfo.InvariantCulture);
            return new DateTime(t.Year, t.Month, t.Day, t.Hour, t.Minute, t.Second, DateTimeKind.Utc);
        }

        public static DateTime ForceUTC(this DateTime d) => new DateTime(d.Year, d.Month, d.Day, d.Hour, d.Minute, d.Second, d.Millisecond, DateTimeKind.Utc);

        public static DateTime StartOfMonth(this DateTime dt) => ViperDate.New(dt.Year, dt.Month, 1);

        public static DateTime StartOfWeek(this DateTime dt, DayOfWeek startOfWeek = DayOfWeek.Sunday)
        {
            int diff = (7 + (dt.DayOfWeek - startOfWeek)) % 7;
            return dt.AddDays(-1 * diff).Date;
        }

        public static DateTime StartOfDay(this DateTime dt) => dt.Date;

        public static DateTime StartOfHour(this DateTime dt) => ViperDate.New(dt.Year, dt.Month, dt.Day, dt.Hour, 0, 0);
        
        // These treat points as vectors

        public static PointF Add(this PointF a, PointF b) => new PointF(a.X + b.X, a.Y + b.Y);
        public static PointF Add(this PointF a, float x, float y) => new PointF(a.X + x, a.Y + y);

        public static PointF Sub(this PointF a, PointF b) => new PointF(a.X - b.X, a.Y - b.Y);

        public static PointF Mult(this PointF p, float f) => new PointF(p.X * f, p.Y * f);
        public static float Dot(this PointF a, PointF b) => a.X * b.X + a.Y * b.Y;
        public static float Length(this PointF a) => (float)Math.Sqrt(a.X * a.X + a.Y * a.Y);

        public static PointF Interpolate(this PointF p1, PointF p2, float fraction) => new PointF(p1.X + (p2.X - p1.X) * fraction, p1.Y + (p2.Y - p1.Y) * fraction);

        public static int Rem(this int a, int b)
        {
            Math.DivRem(a, b, out int result);
            return result;
        }

        public static byte Clip(this byte a, byte min, byte max) => a < min ? min : (a > max ? max : a);

        public static float Clip(this float a, float min, float max) => a < min ? min : (a > max ? max : a);        // Assumes min<=max

        public static float MapTo(this float a, float fromlow, float fromhigh, float tolow, float tohigh)
        {
            var v = a.Clip(fromlow, fromhigh);
            var frac = (v - fromlow) / (fromhigh - fromlow);
            return tolow + frac * (tohigh - tolow);
        }


        public static Regex LatitudeRegex = new Regex("^\\s*(\\d+)d\\s*(\\d+)'\\s*(\\d*\\.\\d*)\"([NS])$", RegexOptions.IgnoreCase | RegexOptions.Compiled | RegexOptions.Singleline);

        public static Regex LongitudeRegex = new Regex("^\\s*(\\d+)d\\s*(\\d+)'\\s*(\\d*\\.\\d*)\"([EW])$", RegexOptions.IgnoreCase | RegexOptions.Compiled | RegexOptions.Singleline);

        public static bool TryToLatitude(this string s, out double r)
        {
            if (double.TryParse(s, out r)) return true;
            var match = LatitudeRegex.Match(s);
            if (!match.Success) return false;
            Debug.Assert(match.Groups.Count == 5);
            var deg = match.Groups[1].Value;
            var min = match.Groups[2].Value;
            var sec = match.Groups[3].Value;
            var dir = match.Groups[4].Value;
            if (!int.TryParse(deg, out int degi) || !int.TryParse(min, out int mini) || !double.TryParse(sec, out double secd)) return false;
            r = degi + mini / 60d + secd / 3600d;
            if (dir.ToUpperInvariant().Equals("S"))
                r = -r;
            if (-90d <= r && r <= 90d) return true;
            r = 0d;
            return false;
        }

        public static bool TryToLongitude(this string s, out double r)
        {
            if (double.TryParse(s, out r)) return true;
            var match = LongitudeRegex.Match(s);
            if (!match.Success) return false;
            Debug.Assert(match.Groups.Count == 5);
            var deg = match.Groups[1].Value;
            var min = match.Groups[2].Value;
            var sec = match.Groups[3].Value;
            var dir = match.Groups[4].Value;
            if (!int.TryParse(deg, out int degi) || !int.TryParse(min, out int mini) || !double.TryParse(sec, out double secd)) return false;
            r = degi + mini / 60d + secd / 3600d;
            if (dir.ToUpperInvariant().Equals("W"))
                r = -r;
            if (-180d <= r && r <= 180d) return true;
            r = 0d;
            return false;
        }

        public static string ToDegHMS(this double val)
        {
            var sign = val < 0d ? -1d : 1d;
            var v = sign * val;
            var deg = (int)v;
            var rem = (v - deg) * 60d;
            var min = (int)rem;
            var sec = (rem - min) * 60d;
            return (sign < 0 ? "-" : "") + deg + "d" + min.ToString("00") + "'" + sec.ToString("00.00\"");
        }

        public static IEnumerable<T> Enumerate<T>(this T[,] a) where T : struct
        {
            var (h, w) = (a.GetLength(0), a.GetLength(1));
            for (var y = 0; y < h; y++)
                for (var x = 0; x < w; x++)
                    yield return a[y, x];
        }

        public static Point Center(this Rectangle r) => new Point(r.Left + r.Width / 2, r.Top + r.Height / 2);
        public static Rectangle SpanFrom(this Point c, Point span) => new Rectangle(c.X - span.X, c.Y - span.Y, span.X * 2 + 1, span.Y * 2 + 1);

        public static Rectangle Intersection(this Rectangle a, Rectangle b)
        {
            if (!a.IntersectsWith(b))
                return (new Rectangle(a.X, a.Y, 0, 0));
            var x = Math.Max(a.X, b.X);
            var y = Math.Max(a.Y, b.Y);
            var bottom = Math.Min(a.Bottom, b.Bottom);
            var right = Math.Min(a.Right, b.Right);
            return new Rectangle(x, y, right - x, bottom - y);
        }

        public static bool Intersects(this Rectangle r, Point p, int threshold = 3) =>
            ((Math.Abs(r.Top - p.Y) <= threshold || Math.Abs(r.Bottom - p.Y) <= threshold) && r.Left - threshold <= p.X && p.X <= r.Right + threshold) ||
            ((Math.Abs(r.Right - p.X) <= threshold || Math.Abs(r.Left - p.X) <= threshold) && r.Top - threshold <= p.Y && p.Y <= r.Bottom + threshold);

        public static bool Intersects(this RectangleF r, PointF p, float threshold = 3f) =>
            ((Math.Abs(r.Top - p.Y) <= threshold || Math.Abs(r.Bottom - p.Y) <= threshold) && r.Left - threshold <= p.X && p.X <= r.Right + threshold) ||
            ((Math.Abs(r.Right - p.X) <= threshold || Math.Abs(r.Left - p.X) <= threshold) && r.Top - threshold <= p.Y && p.Y <= r.Bottom + threshold);

        public static PointF BottomRight(this RectangleF r) => new PointF(r.Right, r.Bottom);

        public static string FilenameAppend(this string path, string postfix, string? extension = null) => Path.Combine(Path.GetDirectoryName(path) ?? ".", Path.GetFileNameWithoutExtension(path) + postfix + (extension ?? Path.GetExtension(path)));

        public static byte[,] ToByteArray(this float[,] a, Func<float, byte> conv)
        {
            Debug.Assert(a != null && conv != null);
            var height = a.GetLength(0);
            var width = a.GetLength(1);
            var r = new byte[height, width];
            for (var row = 0; row < height; row++)
                for (var col = 0; col < width; col++)
                    r[row, col] = conv(a[row, col]);
            return r;
        }

        public static unsafe byte[,]? ToByteArray2(this Bitmap bmp)
        {
            Debug.Assert(bmp != null);
            if (bmp.PixelFormat != PixelFormat.Format8bppIndexed)
                return null;
            var height = bmp.Height;
            var width = bmp.Width;
            var bytes = new byte[height, width];
            var bmpdata = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height), ImageLockMode.ReadOnly, bmp.PixelFormat);
            for (var row = 0; row < height; row++)
            {
                var rowptr = (byte*)(bmpdata.Scan0 + row * bmpdata.Stride);
                for (var col = 0; col < width; col++)
                    bytes[row, col] = rowptr[col];
            }
            bmp.UnlockBits(bmpdata);
            return bytes;
        }

        public static unsafe void CopyFrom(this Bitmap bmp, byte[][] ary)
        {
            var (w, h) = (bmp.Width, bmp.Height);
            Debug.Assert(ary.GetLength(0) == h && ary[0].GetLength(0) == w && bmp.PixelFormat == PixelFormat.Format8bppIndexed);
            var bmpData = bmp.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.WriteOnly, bmp.PixelFormat);
            for (var row = 0; row < h; row++)
            {
                var rowptr = (byte*)(bmpData.Scan0 + row * bmpData.Stride);
                var ary_row = ary[row];
                for (var col = 0; col < w; col++)
                    rowptr[col] = ary_row[col];
            }
            bmp.UnlockBits(bmpData);
        }

        public static unsafe void CopyFrom(this Bitmap bmp, byte[,] ary)
        {
            var (w, h) = (bmp.Width, bmp.Height);
            Debug.Assert(ary.GetLength(0) == h && ary.GetLength(1) == w && bmp.PixelFormat == PixelFormat.Format8bppIndexed);
            var bmpData = bmp.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.WriteOnly, bmp.PixelFormat);
            for (var row = 0; row < h; row++)
            {
                var rowptr = (byte*)(bmpData.Scan0 + row * bmpData.Stride);
                for (var col = 0; col < w; col++)
                    rowptr[col] = ary[row, col];
            }
            bmp.UnlockBits(bmpData);
        }

        public static byte[,] ToMaskArray(this byte[,] a, Func<float, bool> threshold)
        {
            Debug.Assert(a != null && threshold != null);
            var height = a.GetLength(0);
            var width = a.GetLength(1);
            var r = new byte[height, width];
            for (var row = 0; row < height; row++)
                for (var col = 0; col < width; col++)
                    r[row, col] = threshold(a[row, col]) ? (byte)1 : (byte)0;
            return r;
        }

        public static byte[,] Copy(this byte[,] a)
        {
            Debug.Assert(a != null);
            var b = new byte[a.GetLength(0), a.GetLength(1)];
            Array.Copy(a, b, a.GetLength(0) * a.GetLength(1));
            return b;
        }

        public static byte[,] Map(this byte[,] a, Func<byte, byte> func)
        {
            Debug.Assert(a != null && func != null);
            var height = a.GetLength(0);
            var width = a.GetLength(1);
            var b = new byte[height, width];
            for (var row = 0; row < height; row++)
                for (var col = 0; col < width; col++)
                    b[row, col] = func(a[row, col]);
            return b;
        }

        public static byte[,] Map(this float[,] a, Func<byte, byte> func)
        {
            Debug.Assert(a != null && func != null);
            var height = a.GetLength(0);
            var width = a.GetLength(1);
            var b = new byte[height, width];
            for (var row = 0; row < height; row++)
                for (var col = 0; col < width; col++)
                    b[row, col] = func((byte)a[row, col]);
            return b;
        }

        public static float[,] Mult(this byte[,] a, float f)
        {
            var height = a.GetLength(0);
            var width = a.GetLength(1);
            var r = new float[height, width];
            for (var row = 0; row < height; row++)
                for (var col = 0; col < width; col++)
                    r[row, col] = f * a[row, col];
            return r;
        }

        public static string CommaSeparated<T>(this IEnumerable<T> items)
        {
            string r = string.Empty;
            foreach (var item in items)
            {
                if (r.Length == 0)
                    r = r + item;
                else
                    r = r + ", " + item;
            }
            return r;
        }

        public static string AppendToFilename(this string path, string to_append)
        {
            var dir = Path.GetDirectoryName(path);
            var filename = Path.GetFileNameWithoutExtension(path);
            var ext = Path.GetExtension(path);
            var filename2 = filename + to_append + ext;
            return dir == null ? filename2 : Path.Combine(dir, filename2);
        }

        /// <summary>
        /// Push an item onto a stack.  This allows initializers to work normally, e.g., new Stack<MouseMode> { new MouseMode() };
        /// </summary>
        /// <param name="stack"></param>
        /// <param name="mode"></param>
        public static void Add<T>(this Stack<T> stack, T mode) => stack.Push(mode);

        /*
        public static System.Windows.Vector Rotate(this System.Windows.Vector v, double degrees)
        {
            return v.RotateRadians(degrees * Math.PI / 180d);
        }

        public static System.Windows.Vector RotateRadians(this System.Windows.Vector v, double radians)
        {
            var ca = Math.Cos(radians);
            var sa = Math.Sin(radians);
            return new System.Windows.Vector(ca * v.X - sa * v.Y, sa * v.X + ca * v.Y);
        }
        */

        public unsafe static void SetPixels8Bit(this Bitmap target, Func<int, int, byte> func)
        {
            Debug.Assert(target != null);
            if (target.PixelFormat != PixelFormat.Format8bppIndexed)
                throw new Exception("The bitmap is not Format8bppIndexed");
            int width = target.Width, height = target.Height;
            var bmpdata = target.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, target.PixelFormat);
            for (var row = 0; row < height; row++)
            {
                var rowptr = (byte*)(bmpdata.Scan0 + row * bmpdata.Stride);
                for (var col = 0; col < width; col++)
                    rowptr[col] = func(row, col);
            }
            target.UnlockBits(bmpdata);
        }

        public unsafe static void CombineInto(this Bitmap target, Bitmap source, Point offset, Func<int, int, byte, byte, byte> func)
        {
            if (target.PixelFormat != PixelFormat.Format8bppIndexed || source.PixelFormat != PixelFormat.Format8bppIndexed)
                throw new Exception("The bitmap is not Format8bppIndexed");
            int target_width = target.Width, target_height = target.Height, source_width = source.Width, source_height = source.Height;
            var target_data = target.LockBits(new Rectangle(0, 0, target_width, target_height), ImageLockMode.ReadWrite, target.PixelFormat);
            var source_data = source.LockBits(new Rectangle(0, 0, source_width, source_height), ImageLockMode.ReadOnly, source.PixelFormat);
            for (var source_row = 0; source_row < source_height; source_row++)
            {
                var target_row = source_row + offset.Y;
                if (target_row < 0 || target_row >= target_height) continue;
                var source_rowptr = (byte*)(source_data.Scan0 + source_row * source_data.Stride);
                var target_rowptr = (byte*)(target_data.Scan0 + target_row * target_data.Stride);
                for (var source_col = 0; source_col < source_width; source_col++)
                {
                    var target_col = source_col + offset.X;
                    if (target_col < 0 || target_col >= target_width) continue;
                    target_rowptr[target_col] = func(source_row, source_col, source_rowptr[source_col], target_rowptr[target_col]);
                }
            }
            source.UnlockBits(source_data);
            target.UnlockBits(target_data);
        }

        public static byte[] ToByteArray(this Bitmap bmp)
        {
            Debug.Assert(bmp != null);
            if (bmp.PixelFormat != PixelFormat.Format8bppIndexed)
                throw new ArgumentException("The bitmap is not Format8bppIndexed");
            var bitmapData = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height), ImageLockMode.ReadOnly, bmp.PixelFormat);
            var length = bitmapData.Stride * bitmapData.Height;
            var bytes = new byte[length];
            System.Runtime.InteropServices.Marshal.Copy(bitmapData.Scan0, bytes, 0, length);
            bmp.UnlockBits(bitmapData);
            return bytes;
        }

        public static unsafe int[,] ToIntArray2(this Bitmap bmp)
        {
            if (bmp.PixelFormat != PixelFormat.Format32bppArgb)
                throw new ArgumentException("The bitmap is not Format32bppArgb");
            var height = bmp.Height;
            var width = bmp.Width;
            var result = new int[height, width];
            var bmpdata = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height), ImageLockMode.ReadOnly, bmp.PixelFormat);
            for (var row = 0; row < height; row++)
            {
                var rowptr = (int*)(bmpdata.Scan0 + row * bmpdata.Stride);
                for (var col = 0; col < width; col++)
                    result[row, col] = rowptr[col];
            }
            bmp.UnlockBits(bmpdata);
            return result;
        }

        public static IEnumerable<byte> EnumerateBytes(this Bitmap bmp) => bmp.ToByteArray();

        public static long[] GetByteHistogram(this Bitmap bmp)
        {
            var buf = new long[256];
            foreach (var b in bmp.EnumerateBytes())
                buf[b]++;
            return buf;
        }

        public static TSource MinBy<TSource, TKey>(this IEnumerable<TSource> source, Func<TSource, TKey> selector) => source.MinBy(selector, null!);

        public static TSource MinBy<TSource, TKey>(this IEnumerable<TSource> source, Func<TSource, TKey> selector, IComparer<TKey> comparer)
        {
            if (source == null) throw new ArgumentNullException("source");
            if (selector == null) throw new ArgumentNullException("selector");
            comparer = comparer ?? Comparer<TKey>.Default;

            using (var sourceIterator = source.GetEnumerator())
            {
                if (!sourceIterator.MoveNext())
                    throw new InvalidOperationException("Sequence contains no elements");
                var min = sourceIterator.Current;
                var minKey = selector(min);
                while (sourceIterator.MoveNext())
                {
                    var candidate = sourceIterator.Current;
                    var candidateProjected = selector(candidate);
                    if (comparer.Compare(candidateProjected, minKey) < 0)
                    {
                        min = candidate;
                        minKey = candidateProjected;
                    }
                }
                return min;
            }
        }

        public static (int, int) GetMinMax(this Bitmap bmp)
        {
            var max = int.MinValue;
            var min = int.MaxValue;
            foreach (var v in bmp.EnumerateBytes())
            {
                max = Math.Max(max, v);
                min = Math.Min(min, v);
            }
            return (min, max);
        }

        //public static double ToDegrees(this double angle_rad) => angle_rad * 180d / 3.14159265358979323846626d;
        //public static double ToRadians(this double angle_deg) => angle_deg * Math.PI / 180d;
        //public static float ToDegrees(this float angle_rad) => angle_rad * 180f / 3.14159265358979323846626f;
        //public static float ToRadians(this float angle_deg) => angle_deg * 3.14159265358979323846626f / 180f;

        public static string DegreeToLongitudeString(this double d)
        {
            string eastWest = "E";
            if (d > 180d)
            {
                eastWest = "W";
                d = 360d - d;
            }
            var deg = (int)d;
            double rem = d - deg;
            rem *= 60d;
            var min = (int)rem;
            rem = rem - min;
            rem *= 60d;
            var sec = (int)rem;
            rem = rem - sec;
            rem *= 100d;
            var hundredths = (int)rem;
            return string.Format("{0,2}d{1:D2}'{2:D2}.{3:D2}\"{4}", deg, min, sec, hundredths, eastWest);
        }

        public static string DegreeToLatitudeString(this double d)
        {
            string northSouth = "N";
            if (d < 0d)
            {
                northSouth = "S";
                d = -d;
            }
            var deg = (int)d;
            double rem = d - deg;
            rem *= 60d;
            var min = (int)rem;
            rem = rem - min;
            rem *= 60d;
            var sec = (int)rem;
            rem = rem - sec;
            rem *= 100d;
            var hundredths = (int)rem;
            return string.Format("{0,2}d{1:D2}'{2:D2}.{3:D2}\"{4}", deg, min, sec, hundredths, northSouth);
        }

        public static string ToDegMinSecString(this double v)
        {
            var a = Math.Abs(v);
            var deg = Math.Floor(a);
            a = 60d * (a - deg);
            var min = Math.Floor(a);
            a = 60d * (a - min);
            var sec = a;
            var str = string.Format("{0:###}d{1:00}'{2:00.###}\"", deg, min, sec);
            if (v < 0d) str = "-" + str;
            return str;
        }

        public static void Do<T>(this IEnumerable<T> stream, Action<T> action)
        {
            foreach (var item in stream)
                action(item);
        }

        #region general object extensions
        
        public static void ClearNAddRange<T>(this ObservableCollection<T> baseCollection, IList<T> objects)
        {
            if (baseCollection is null)
                return;

            baseCollection.Clear();
            baseCollection.AddRange(objects);
        }
        

        public static void AddRange<T>(this ObservableCollection<T> baseCollection, IList<T> objects)
        {
            if (baseCollection is null)
                return;

            objects.ForEach(x => baseCollection.Add(x));
        }


        public static void Sort<T>(this ObservableCollection<T> collection)
        {
            //Createing a list and using icompare of the activtiy class to sort
            //shows to be faster than using linq by a little bit.
            List<T> temp;
            temp = new List<T>(collection);
            temp.Sort();
            collection.ClearNAddRange(temp);
        }



        #endregion

        #region URL Object Extensions

        /// <summary>
        /// Takes a URI and replaces the user and password with **** so that
        /// we can keep identities and credential secure to logs and debug sessions.
        /// </summary>
        /// <param name="incomingURI"></param>
        /// <returns></returns>
        public static string Obfuscate(this Uri incomingURI)
        {
            if (incomingURI is null)
                return string.Empty;
            
            string puburl = string.Empty;

            if (!string.IsNullOrEmpty(incomingURI.UserInfo) && incomingURI.UserInfo.Contains(":"))
            {
                var locOfSemi = incomingURI.UserInfo.IndexOf(":");
                string passString = incomingURI.UserInfo.Substring(locOfSemi + 1);
                string userString = incomingURI.UserInfo.Substring(0, locOfSemi);
                puburl = incomingURI.ToString().Replace(passString, new string('*', passString.Length));
            }
            else
                puburl = incomingURI.ToString();

            return puburl;
        }


        #endregion

        #region Activity and ITreeNode Extensions
        
        

        #endregion

        public static void Normalize(this double[] a)
        {
            if (a.Length == 3)
            {
                double d = Math.Sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
                a[0] /= d;
                a[1] /= d;
                a[2] /= d;
            }
            else
            {
                var sum = 0d;
                for (var i = 0; i < a.Length; i++)
                    sum += a[i] * a[i];
                var d = Math.Sqrt(sum);
                for (var i = 0; i < a.Length; i++)
                    a[i] = a[i] / d;
            }
        }

        public static Color Lighten(this Color c, int d = 10) => Color.FromArgb(Clamp(c.R + d), Clamp(c.G + d), Clamp(c.B + d));
        public static Color Darken(this Color c, int d = 10) => Color.FromArgb(Clamp(c.R - d), Clamp(c.G - d), Clamp(c.B - d));
        public static int Clamp(int c) => c < 0 ? 0 : c > 255 ? 255 : c;
    }
}
