using moonlib.horizon;
using moonlib.math;
using System.Diagnostics;
using System.Drawing;

namespace moonlib
{
    public static class Utilities
    {
        // This is used in the point-caster approach to determine how many pixels can be skipped without missing horizon details.
        public static int GetPatchStep(float distance_between_centers_in_pixels)
        {
            // The observer could be on the side of the patch closest to the caster patch, and
            // the caster pixels could be on the side of the patch closest to the observer patch.
            // So subtract the patch size to get the minimum distance between any two pixels in the patches
            var distance_in_pixels = distance_between_centers_in_pixels - TerrainPatch.PatchSizePixels;

            const float horizon_resolution_in_radians = 0.25f * 3.141592653589f / 180f;
            var step = ((int)Math.Floor(distance_in_pixels * horizon_resolution_in_radians)).Clamp(1, TerrainPatch.PatchSizePixels / 2);
            return step;
        }

        public static int GetPatchStep(Point observer, Point caster)
        {
            var (d1, d2) = (observer.X - caster.X, observer.Y - caster.Y);
            var distance = Math.Sqrt(d1 * d1 + d2 * d2);
            var distance_in_pixels = distance - TerrainPatch.PatchSizePixels;

            const double horizon_resolution_in_radians = 0.25d * 3.141592653589d / 180d;
            var step = ((int)Math.Floor(distance_in_pixels * horizon_resolution_in_radians)).Clamp(1, TerrainPatch.PatchSizePixels / 2);
            return step;
        }

        public static (List<Point> in_observer_dem, List<Point> in_caster_dems) GetPatchIds(ElevationMap observer_dem, List<ElevationMap>? caster_dems = null)
        {
            if (observer_dem == null)
                throw new ArgumentNullException(nameof(observer_dem));
            var center_bbox = observer_dem.BoundingBox;

            var center_ids = new List<Point>();
            for (int y = center_bbox.Top; y < center_bbox.Bottom; y += TerrainPatch.PatchSizePixels)
                for (int x = center_bbox.Left; x < center_bbox.Right; x += TerrainPatch.PatchSizePixels)
                    center_ids.Add(new Point(x, y));

            if (caster_dems == null)
                return (center_ids, new List<Point>());

            var outer_bbox = GetBoundingBox(observer_dem, caster_dems);
            var outer_ids = new List<Point>();
            for (int y = outer_bbox.Top; y < outer_bbox.Bottom; y += TerrainPatch.PatchSizePixels)
                for (int x = outer_bbox.Left; x < outer_bbox.Right; x += TerrainPatch.PatchSizePixels)
                    if (!center_bbox.Contains(new Point(x, y)))
                        outer_ids.Add(new Point(x, y));

            return (center_ids, outer_ids);
        }

        public static PointF CenterOfPatch(Point location) => new PointF(location.X + TerrainPatch.PatchSizePixels / 2.0f, location.Y + TerrainPatch.PatchSizePixels / 2.0f);

        public static List<Point> GetPatchLocationsInSpiralOrder(Point center_patch, List<ElevationMap> maps)
        {
            var all_ids = new HashSet<Point>();
            var bbox = GetBoundingBox(maps[0], maps.Skip(1).ToList());
            for (int y = bbox.Top; y < bbox.Bottom; y += TerrainPatch.PatchSizePixels)
                for (int x = bbox.Left; x < bbox.Right; x += TerrainPatch.PatchSizePixels)
                    all_ids.Add(new Point(x, y));

            var locations = new List<Point>();
            foreach (var patchLoc in EnumeratePatchLocationsInSpiralPattern(center_patch))
            {
                if (all_ids.Contains(patchLoc))
                {
                    locations.Add(patchLoc);
                    if (locations.Count >= all_ids.Count)
                        break;
                }
            }
            return locations;
        }

        public static Rectangle RoundBoundingBox(Rectangle bbox, int patch_size = TerrainPatch.PatchSizePixels)
        {
            if (patch_size <= 0)
                throw new ArgumentOutOfRangeException(nameof(patch_size), "Patch size must be positive.");

            int left = FloorToMultiple(bbox.Left, patch_size);
            int top = FloorToMultiple(bbox.Top, patch_size);
            int right = CeilToMultiple(bbox.Right, patch_size);
            int bottom = CeilToMultiple(bbox.Bottom, patch_size);
            return Rectangle.FromLTRB(left, top, right, bottom);
        }

        /// <summary>
        /// Get the bounding box in pixel coordinates of an observer map of that map plus a set of other maps with potentially
        /// different projections and resolutions.
        /// </summary>
        /// <param name="observer_elevation_map"></param>
        /// <param name="shadow_caster_elevation_maps"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentNullException"></exception>
        /// <exception cref="ArgumentException"></exception>
        public static Rectangle GetBoundingBox(ElevationMap observer_elevation_map, List<ElevationMap> shadow_caster_elevation_maps)
        {
            if (observer_elevation_map is null)
                throw new ArgumentNullException(nameof(observer_elevation_map));
            if (shadow_caster_elevation_maps is null)
                throw new ArgumentNullException(nameof(shadow_caster_elevation_maps));
            var center_bbox = observer_elevation_map.BoundingBox;
            int left = center_bbox.Left;
            int top = center_bbox.Top;
            int right = center_bbox.Right;
            int bottom = center_bbox.Bottom;
            foreach (var other_map in shadow_caster_elevation_maps)
            {
                var bbox = GetBoundingBox(observer_elevation_map, other_map);
                if (bbox.Left < left) left = bbox.Left;
                if (bbox.Top < top) top = bbox.Top;
                if (bbox.Right > right) right = bbox.Right;
                if (bbox.Bottom > bottom) bottom = bbox.Bottom;
            }
            return Rectangle.FromLTRB(left, top, right, bottom);
        }

        /// <summary>
        /// Calculates the bounding box of a second elevation map in the pixel coordinate system of a center elevation map.
        /// </summary>
        /// <param name="observer_elevation_map">Center ElevationMap</param>
        /// <param name="shadow_caster_elevation_map">Other ElevationMap (note that this can be the center map also)</param>
        /// <returns></returns>
        public static Rectangle GetBoundingBox(ElevationMap observer_elevation_map, ElevationMap shadow_caster_elevation_map)
        {
            if (observer_elevation_map is null)
                throw new ArgumentNullException(nameof(observer_elevation_map));
            if (shadow_caster_elevation_map is null)
                throw new ArgumentNullException(nameof(shadow_caster_elevation_map));
            if (observer_elevation_map.GeoTransform is null || observer_elevation_map.GeoTransform.Length < 6)
                throw new ArgumentException("Center map must have a valid GeoTransform.", nameof(observer_elevation_map));
            if (shadow_caster_elevation_map.GeoTransform is null || shadow_caster_elevation_map.GeoTransform.Length < 6)
                throw new ArgumentException("Other map must have a valid GeoTransform.", nameof(shadow_caster_elevation_map));
            if (shadow_caster_elevation_map.Elevation is null)
                throw new ArgumentException("Other map must have elevation data.", nameof(shadow_caster_elevation_map));

            // Convert the other map's four edges into world space (using pixel edges),
            // then reproject those coordinates into the center map's pixel space.
            var otherWidth = shadow_caster_elevation_map.Elevation.GetLength(1);
            var otherHeight = shadow_caster_elevation_map.Elevation.GetLength(0);

            var shadowSrs = shadow_caster_elevation_map.SrsDescriptor ?? throw new InvalidOperationException("Shadow caster map missing SRS descriptor.");
            var observerSrs = observer_elevation_map.SrsDescriptor ?? throw new InvalidOperationException("Observer map missing SRS descriptor.");

            var func = MoonSrsLambdaFactory.MakeLambda(shadowSrs, observerSrs);

            double minCol = double.PositiveInfinity;
            double maxCol = double.NegativeInfinity;
            double minRow = double.PositiveInfinity;
            double maxRow = double.NegativeInfinity;

            foreach (var pt_src_pixel in EnumerateCorners(shadow_caster_elevation_map))
            {
                var srcCrs = shadow_caster_elevation_map.PixelToCRS(pt_src_pixel);
                srcCrs = MoonSrsLambdaFactory.ToLambdaInputUnits(srcCrs, shadowSrs);
                var dstCrs = func(srcCrs);
                dstCrs = MoonSrsLambdaFactory.FromLambdaOutputUnits(dstCrs, observerSrs);
                var (pt_obs_pixel_col, pt_obs_pixel_row) = observer_elevation_map.CRSToPixel(dstCrs).Destructure();

                if (pt_obs_pixel_col < minCol) minCol = pt_obs_pixel_col;
                if (pt_obs_pixel_col > maxCol) maxCol = pt_obs_pixel_col;
                if (pt_obs_pixel_row < minRow) minRow = pt_obs_pixel_row;
                if (pt_obs_pixel_row > maxRow) maxRow = pt_obs_pixel_row;
            }

            int left = (int)Math.Floor(minCol);
            int top = (int)Math.Floor(minRow);
            int right = (int)Math.Ceiling(maxCol);
            int bottom = (int)Math.Ceiling(maxRow);

            return Rectangle.FromLTRB(left, top, right, bottom);

            IEnumerable<PixelPoint> EnumerateCorners(ElevationMap map)
            {
                yield return new PixelPoint(0, 0);
                yield return new PixelPoint(map.Width, 0);
                yield return new PixelPoint(map.Width, map.Height);
                yield return new PixelPoint(0, map.Height);
            }
        }

        private static int FloorToMultiple(int value, int step)
        {
            int remainder = value % step;
            if (remainder == 0)
                return value;
            if (remainder > 0)
                return value - remainder;
            return value - (remainder + step);
        }

        private static int CeilToMultiple(int value, int step)
        {
            int remainder = value % step;
            if (remainder == 0)
                return value;
            if (remainder > 0)
                return value + (step - remainder);
            return value - remainder;
        }

        private static double DegToRad(double degrees) => degrees * Math.PI / 180.0;

        private static double RadToDeg(double radians) => radians * 180.0 / Math.PI;

        /// <summary>
        /// Enumerates terrain patches in a spiral pattern starting from the specified center patch.
        /// </summary>
        /// <remarks>The method iterates over terrain patches in a spiral pattern, starting from the
        /// specified center patch and expanding outward.  The enumeration does not include the center patch itself.
        /// The order of enumeration is clockwise, beginning from the patch directly to the right of the center patch,
        /// and the patches are tightly packed without gaps.
        /// </remarks>
        /// <param name="centerLoc">The central terrain patch from which the spiral enumeration begins.</param>
        /// <returns>An enumerable collection of <see cref="Point"/> objects representing the terrain patches in the order
        /// of the spiral pattern.</returns>
        public static IEnumerable<Point> EnumeratePatchLocationsInSpiralPattern(Point centerLoc)
        {
            // Spiral outwards from the center patch, skipping the center itself.
            // The spiral order is: right, down, left, up, increasing the step size every two turns.
            // Each step moves by 1 patch grid unit, and the pixel location is calculated using PatchSize.
            int gx = centerLoc.X / TerrainPatch.PatchSize.Width;
            int gy = centerLoc.Y / TerrainPatch.PatchSize.Height;
            int dx = 1, dy = 0; // Start moving right (in patch grid units)
            int segmentLength = 1;
            int stepsTaken = 0;
            int segmentPassed = 0;

            // We'll yield patches in an infinite spiral. The caller should break as needed.
            // To avoid infinite loop, let's yield up to a reasonable number (e.g., 10000 patches)
            int maxPatches = 10000;
            int yielded = 0;
            gx += 1; // Start to the right of center (in patch grid units)
            while (yielded < maxPatches)
            {
                int px = gx * TerrainPatch.PatchSize.Width;
                int py = gy * TerrainPatch.PatchSize.Height;
                yield return new Point(px, py);
                yielded++;
                stepsTaken++;
                if (stepsTaken == segmentLength)
                {
                    // Change direction: right->down->left->up->right...
                    stepsTaken = 0;
                    segmentPassed++;
                    int temp = dx;
                    dx = -dy;
                    dy = temp;
                    if (segmentPassed % 2 == 0)
                    {
                        segmentLength++;
                    }
                }
                gx += dx;
                gy += dy;
            }
            yield break;
        }

        public static bool LineSegmentIntersectsRectangle(Point p1, Point p2, Rectangle rect)
        {
            if (rect.Contains(p1) || rect.Contains(p2))
                return true;

            int left = rect.Left;
            int right = rect.Right;
            int top = rect.Top;
            int bottom = rect.Bottom;

            int p1x = p1.X;
            int p1y = p1.Y;
            int p2x = p2.X;
            int p2y = p2.Y;

            if (SegmentsIntersect(p1x, p1y, p2x, p2y, left, top, right, top))
                return true;

            if (SegmentsIntersect(p1x, p1y, p2x, p2y, right, top, right, bottom))
                return true;

            if (SegmentsIntersect(p1x, p1y, p2x, p2y, right, bottom, left, bottom))
                return true;

            if (SegmentsIntersect(p1x, p1y, p2x, p2y, left, bottom, left, top))
                return true;

            return false;
        }

        // Helper: returns true if segments (p1,p2) and (q1,q2) intersect
        private static bool SegmentsIntersect(int p1x, int p1y, int p2x, int p2y,
                                              int q1x, int q1y, int q2x, int q2y)
        {
            int o1 = Orientation(p1x, p1y, p2x, p2y, q1x, q1y);
            int o2 = Orientation(p1x, p1y, p2x, p2y, q2x, q2y);
            int o3 = Orientation(q1x, q1y, q2x, q2y, p1x, p1y);
            int o4 = Orientation(q1x, q1y, q2x, q2y, p2x, p2y);

            // General case
            if (o1 != o2 && o3 != o4)
                return true;

            // Special cases
            if (o1 == 0 && OnSegment(p1x, p1y, q1x, q1y, p2x, p2y)) return true;
            if (o2 == 0 && OnSegment(p1x, p1y, q2x, q2y, p2x, p2y)) return true;
            if (o3 == 0 && OnSegment(q1x, q1y, p1x, p1y, q2x, q2y)) return true;
            if (o4 == 0 && OnSegment(q1x, q1y, p2x, p2y, q2x, q2y)) return true;

            return false;
        }

        // Returns orientation: 0=colinear, 1=clockwise, 2=counterclockwise
        private static int Orientation(int ax, int ay, int bx, int by, int cx, int cy)
        {
            long val = (long)(bx - ax) * (cy - ay) - (long)(by - ay) * (cx - ax);
            if (val == 0) return 0;
            return (val > 0) ? 1 : 2;
        }

        // Returns true if q lies on segment pr
        private static bool OnSegment(int px, int py, int qx, int qy, int rx, int ry)
        {
            return qx <= Math.Max(px, rx) && qx >= Math.Min(px, rx)
                && qy <= Math.Max(py, ry) && qy >= Math.Min(py, ry);
        }

        public static T[] LoadBinaryArray<T>(string v)
        {
            var type = typeof(T);
            int typeSize = System.Runtime.InteropServices.Marshal.SizeOf<T>();
            if (!(type == typeof(byte) || type == typeof(short) || type == typeof(int) || type == typeof(long) ||
                  type == typeof(float) || type == typeof(double) || type == typeof(char)))
                throw new NotSupportedException($"Type {type} is not supported for binary array loading.");

            var fileInfo = new FileInfo(v);
            long fileLength = fileInfo.Length;
            if (fileLength % typeSize != 0)
                throw new InvalidOperationException($"File size {fileLength} is not a multiple of type size {typeSize}.");
            
            long elementCount = fileLength / typeSize;
            if (elementCount > Array.MaxLength)
                throw new InvalidOperationException($"File contains {elementCount} elements, which exceeds the maximum array size.");

            T[] arr = new T[(int)elementCount];

            using var fs = new FileStream(v, FileMode.Open, FileAccess.Read);
            unsafe
            {
#pragma warning disable CS8500
                fixed (T* arrPtr = arr)
                {
                    byte* basePtr = (byte*)arrPtr;
                    long totalBytes = fileLength;
                    long bytesReadTotal = 0;
                    int chunkSize = 1024 * 1024 * 1024; // 1GB chunks

                    while (bytesReadTotal < totalBytes)
                    {
                        long remaining = totalBytes - bytesReadTotal;
                        int currentChunk = (int)Math.Min(remaining, chunkSize);
                        
                        // Create a span for the current chunk relative to the current position
                        var span = new Span<byte>(basePtr + bytesReadTotal, currentChunk);
                        int read = fs.Read(span);
                        if (read == 0) throw new EndOfStreamException();
                        bytesReadTotal += read;
                    }
                }
#pragma warning restore CS8500
            }
            return arr;
        }

        public static T[] WriteBinaryArray<T>(string path, T[] data)
        {
            var type = typeof(T);
            int typeSize = System.Runtime.InteropServices.Marshal.SizeOf<T>();
            if (!(type == typeof(byte) || type == typeof(short) || type == typeof(int) || type == typeof(long) ||
                  type == typeof(float) || type == typeof(double) || type == typeof(char)))
                throw new NotSupportedException($"Type {type} is not supported for binary array writing.");
            
            using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
            unsafe
            {
#pragma warning disable CS8500
                fixed (T* arrPtr = data)
                {
                    byte* basePtr = (byte*)arrPtr;
                    long totalBytes = (long)data.Length * typeSize;
                    long bytesWrittenTotal = 0;
                    int chunkSize = 1024 * 1024 * 1024; // 1GB chunks

                    while (bytesWrittenTotal < totalBytes)
                    {
                        long remaining = totalBytes - bytesWrittenTotal;
                        int currentChunk = (int)Math.Min(remaining, chunkSize);
                        
                        var span = new ReadOnlySpan<byte>(basePtr + bytesWrittenTotal, currentChunk);
                        fs.Write(span);
                        bytesWrittenTotal += currentChunk;
                    }
                }
#pragma warning restore CS8500
            }
            return data;
        }

        public static void DrawCasterPoints(string path, float[] x, float[] y)
        {
            Debug.Assert(x != null && y != null);
            Debug.Assert(x.Length == y.Length);
            var (xmin,xmax,ymin,ymax) = (float.MaxValue, float.MinValue, float.MaxValue, float.MinValue);
            for (var i = 0; i < x.Length; i++)
            {
                if (x[i] < xmin) xmin = x[i];
                if (x[i] > xmax) xmax = x[i];
                if (y[i] < ymin) ymin = y[i];
                if (y[i] > ymax) ymax = y[i];
            }

            int imgWidth = 1000;
            int imgHeight = 1000;
            using var bmp = new Bitmap(imgWidth, imgHeight);
            using var g = Graphics.FromImage(bmp);

            var xFactor = (imgWidth - 1) / (xmax - xmin);
            var yFactor = (imgHeight - 1) / (ymax - ymin);

            Point CasterPositionToPoint(float x, float y)
            {
                int px = (int)((x - xmin) * xFactor);
                int py = (int)((y - ymin) * yFactor);
                return new Point(px, py);
            }

            g.Clear(Color.Black);
            for (var i = 0; i < x.Length; i++)
            {
                var pt = CasterPositionToPoint(x[i], y[i]);
                float intensity = 1f; // (z[i] + 1000f) / 2000f;
                intensity = Math.Clamp(intensity, 0f, 1f);
                var color = Color.FromArgb((int)(intensity * 255), (int)(intensity * 255), (int)(intensity * 255));
                bmp.SetPixel(pt.X, imgHeight - 1 - pt.Y, color);
            }

            var p1 = CasterPositionToPoint(-100, 0);
            var p2 = CasterPositionToPoint(100, 0);
            var p3 = CasterPositionToPoint(0, -100);
            var p4 = CasterPositionToPoint(0, 100);

            g.DrawLine(Pens.Red, p1, p2);
            g.DrawLine(Pens.Red, p3, p4);

            bmp.Save(path);
        }

        public static T[] PreloadArray<T>(int length, T defaultValue)
        {
            var arr = new T[length];
            for (int i = 0; i < length; i++)
                arr[i] = defaultValue;
            return arr;
        }
    }
}
