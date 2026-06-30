using System.Drawing;

namespace moonlib.math
{
    public static class Extensions
    {
        public static float Distance(this PointF a, PointF b)
        {
            var dx = a.X - b.X;
            var dy = a.Y - b.Y;
            return (float)Math.Sqrt(dx * dx + dy * dy);
        }

        public static float Distance(this Point a, Point b)
        {
            var dx = a.X - b.X;
            var dy = a.Y - b.Y;
            return (float)Math.Sqrt(dx * dx + dy * dy);
        }

        /// <summary>
        /// Distance from p0 to a line that passes through p1 and p2
        /// </summary>
        /// <returns></returns>
        public static float Distance(this PointF p0, PointF p1, PointF p2)
        {
            var (x0, y0, x1, y1, x2, y2) = (p0.X, p0.Y, p1.X, p1.Y, p2.X, p2.Y);
            return Math.Abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / (float)Math.Sqrt((y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1));
        }

        public static int ToFixed(this float f) => (int)(f + 0.5f);
        public static int ToFixed(this double f) => (int)(f + 0.5d);

        public static double ToDegrees(this double angle_rad) => angle_rad * 180d / 3.14159265358979323846626d;
        public static double ToRadians(this double angle_deg) => angle_deg * Math.PI / 180d;
        public static float ToDegrees(this float angle_rad) => angle_rad * 180f / 3.14159265358979323846626f;
        public static float ToRadians(this float angle_deg) => angle_deg * 3.14159265358979323846626f / 180f;

        public static bool Equivalent(this PointF a, PointF b, float epsilon = 0.01f) => a.Distance(b) < epsilon;
        public static PointF Plus(this PointF a, PointF b) => new PointF(a.X + b.X, a.Y + b.Y);
        public static PointF Plus(this PointF a, SizeF b) => new PointF(a.X + b.Width, a.Y + b.Height);
        public static PointF Minus(this PointF a, PointF b) => new PointF(a.X - b.X, a.Y - b.Y);
        public static PointF Times(this PointF a, float b) => new PointF(a.X * b, a.Y * b);
        public static PointF Corner(RectangleF r) => new PointF(r.Right, r.Bottom);

        public static Point Plus(this Point a, Point b) => new Point(a.X + b.X, a.Y + b.Y);
        public static Point Minus(this Point a, Point b) => new Point(a.X - b.X, a.Y - b.Y);

        public static int Clamp(this int a, int low, int high) => a < low ? low : (a > high ? high : a);
        public static float Clamp(this float a, float low, float high) => a < low ? low : (a > high ? high : a);
        public static double Clamp(this double a, double low, double high) => a < low ? low : (a > high ? high : a);

        public static void ForEach<T>(this IEnumerable<T> enumeration, Action<T> action)
        {
            foreach (T item in enumeration)
                action(item);
        }

        public static void Populate<T>(this T[] arr, T value)
        {
            for (int i = 0; i < arr.Length; i++)
                arr[i] = value;
        }

        public static double[] ToArray(this Vector3d v) => new double[] { v.X, v.Y, v.Z };

        public static Vector3 ToVector3(this Vector3d v) => new Vector3((float)v.X, (float)v.Y, (float)v.Z);
        public static float Distance(this Vector3 a, Vector3 b) => (float)Math.Sqrt((a.X - b.X) * (a.X - b.X) + (a.Y - b.Y) * (a.Y - b.Y) + (a.Z - b.Z) * (a.Z - b.Z));
        public static double Distance(this Vector3d a, Vector3d b) => Math.Sqrt((a.X - b.X) * (a.X - b.X) + (a.Y - b.Y) * (a.Y - b.Y) + (a.Z - b.Z) * (a.Z - b.Z));

        public static double Elevation(this Vector3d v) => Math.Atan2(v.Z, Math.Sqrt(v.X * v.X + v.Y * v.Y)) * 180d / Math.PI;
        public static double Azimuth(this Vector3d v) => Math.Atan2(v.Y, v.X) * 180d / Math.PI;

        /// <summary>
        /// Checks if the given floating-point values are similar.
        /// </summary>
        /// <param name="a">First value.</param>
        /// <param name="b">Second value.</param>
        /// <returns>true if both values are similar but not necessarily equal, false if they are different</returns>
        public static bool IsSimilar(this double a, double b)
        {
            // I don't understand why this isn't simpler Math.Abs(a-b)<epsilon ?
            if (a == b)
            {
                return true;
            }

            if (a < 0.0)
            {
                if (b >= 0.0)
                {
                    return b - a < 5E-15;
                }

                a = 0.0 - a;
                b = 0.0 - b;
            }
            else if (a > 0.0)
            {
                if (b <= 0.0)
                {
                    return a - b < 5E-15;
                }
            }
            else if (a == 0.0)
            {
                if (b > -5E-15)
                {
                    return b < 5E-15;
                }

                return false;
            }

            double num = ((!(a > b)) ? (a / b) : (b / a));
            return num > 0.999999999999995;
        }

        public static void Deconstruct(this PointF p, out float x, out float y) => (x, y) = (p.X, p.Y);

        /// <summary>
        /// This encoding is a little weird.  Only the degrees are negative to represent a negative angle.
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public static (int deg, int min, double sec) ToDegMinSec(this double a)
        {
            bool negative = false;
            double v;
            if (a < 0d)
                (negative, v) = (true, -a);
            else
                v = a;
            var deg = (int)v;
            var deg1 = (int)Math.Round(v);
            if (deg != deg1 && Math.Abs(deg1 - v) < 1e-10)
                (deg, v) = (deg1, deg1);
            v = (v - deg) * 60d;
            var min = (int)v;
            var min1 = (int)Math.Round(v);
            if (min != min1 && Math.Abs(min1 - v) < 1e-10)
                (min, v) = (min1, min1);
            var sec = (v - min) * 60d;
            return (negative ? -deg : deg, min, sec);
        }
    }
}
