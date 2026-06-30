using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;

namespace moonlib.util
{
    /// <summary>
    /// Utilities for finding contiguous regions within a grid
    /// </summary>
    public class ContiguousRegion
    {
        /// <summary>
        /// Divide an array of values into contiguous regions.  Each cell in the array is one of 255 types.  0 means no type.
        /// THIS COPIES THE INPUT ARRAY.
        /// </summary>
        /// <param name="zone_array">A 2D byte array labeling each pixel/cell as a particular zone type.  0 indicates 'no type'.</param>
        /// <returns>
        /// 1. region_indexes: An array containing the region indexes.  Each contiguous region will be assigned one index.
        ///    Non-contiguous regions of the same type will have different zone indexes.  A value of 0 indicates a pixel isn't part of any region.
        /// 2. region_count: The number of distinct regions.  
        /// 3. region_sizes: A list giving the sizes in pixels of each distinct zone, indexed by the region indexes.
        ///    The length is region_count + 1 and region_sizes[0] is the count of pixels not in any region</returns>
        public static (int[,] region_indexes, int region_count, List<int> region_sizes) IdentifyRegions(byte[,] zone_array)
        {
            zone_array = Copy(zone_array);
            var (height, width) = (zone_array.GetLength(0), zone_array.GetLength(1));
            var regions = new int[height, width];
            var region_sizes = new List<int> { 0 };  // Zone 0 doesn't exist.  Give it a size of 0.
            var region_counter = 0;
            for (var row = 0; row < height; row++)
                for (var col = 0; col < width; col++)
                {
                    var v = zone_array[row, col];
                    if (v == 255)
                        continue;
                   
                    ++region_counter;           // This cell hasn't been seen yet.
                    var count = FloodZone(region_counter, row, col, v);
                    region_sizes.Add(count);
                }

            var count_of_pixels_in_all_regions = region_sizes.Sum();
            var total_pixels = height * width;
            var count_of_pixels_not_in_a_region = total_pixels - count_of_pixels_in_all_regions;
            region_sizes[0] = count_of_pixels_not_in_a_region;

            return (regions, region_sizes.Count - 1, region_sizes);

            int FloodZone(int id, int row, int col, byte zoneType)
            {
                regions[row, col] = id;
                zone_array[row, col] = 255;

                int lcol;
                for (lcol = col - 1; lcol >= 0 && zone_array[row, lcol] == zoneType; lcol--)
                {
                    regions[row, lcol] = id;
                    zone_array[row, lcol] = 255;
                }
                lcol++;
                int rcol;
                for (rcol = col + 1; rcol < width && zone_array[row, rcol] == zoneType; rcol++)
                {
                    regions[row, rcol] = id;
                    zone_array[row, rcol] = 255;
                }
                rcol--;

                Debug.Assert(lcol == 0 || zone_array[row, lcol - 1] != zoneType);
                Debug.Assert(rcol == width - 1 || zone_array[row, rcol + 1] != zoneType);

                var count = rcol - lcol + 1;

                if (row > 0)
                {
                    var rowminus1 = row - 1;
                    for (var i = lcol; i <= rcol; i++)
                    {
                        if (zone_array[rowminus1, i] == zoneType)
                            count += FloodZone(id, rowminus1, i, zoneType);
                    }
                }
                if (row < height - 1)
                {
                    var rowplus1 = row + 1;
                    for (var i = lcol; i <= rcol; i++)
                    {
                        if (zone_array[rowplus1, i] == zoneType)
                            count += FloodZone(id, rowplus1, i, zoneType);
                    }
                }

                return count;
            }
        }

        public static byte[,] Copy(byte[,] a)
        {
            var (height, width) = (a.GetLength(0), a.GetLength(1));
            var r = new byte[height, width];
            for (var row = 0; row < height; row++)
                for (var col = 0; col < width; col++)
                    r[row, col] = a[row, col];
            return r;
        }

        /// <summary>
        /// Return the list of pixel coordinates that are in same region as the seed point (including the seed)
        /// </summary>
        /// <param name="region_indexes"></param>
        /// <param name="seed"></param>
        /// <returns></returns>
        public static List<Point> GetRegionPixels(int[,] region_indexes, Point seed)
        {
            var (height, width) = (region_indexes.GetLength(0), region_indexes.GetLength(1));

            var pixels_found = new HashSet<Point>();
            var seed_index = region_indexes[seed.Y, seed.X];
            var pixels_found_count = Flood(seed_index, seed.Y, seed.X, region_indexes);

            int Flood(int index, int row, int col, int[,] indexes)
            {
                pixels_found.Add(new Point(col, row));

                int lcol;
                for (lcol = col - 1; lcol >= 0 && indexes[row, lcol] == index; lcol--)
                    pixels_found.Add(new Point(lcol, row));

                lcol++;
                int rcol;
                for (rcol = col + 1; rcol < width && indexes[row, rcol] == index; rcol++)
                    pixels_found.Add(new Point(rcol, row));
                rcol--;

                Debug.Assert(lcol == 0 || indexes[row, lcol - 1] != index);
                Debug.Assert(rcol == width - 1 || indexes[row, rcol + 1] != index);

                var count = rcol - lcol + 1;
                if (row > 0)
                {
                    var rowminus1 = row - 1;
                    for (var i = lcol; i <= rcol; i++)
                    {
                        if (indexes[rowminus1,i] == index && !FoundBefore(rowminus1, i))
                            count += Flood(index, rowminus1, i, indexes);
                    }
                }
                if (row < height - 1)
                {
                    var rowplus1 = row + 1;
                    for (var i = lcol; i <= rcol; i++)
                    {
                        if (indexes[rowplus1, i] == index && !FoundBefore(rowplus1, i))
                            count += Flood(index, rowplus1, i, indexes);
                    }
                }

                return count;
            }

            Debug.Assert(pixels_found.Count == pixels_found_count);

            return pixels_found.ToList();

            bool FoundBefore(int r, int c) => pixels_found.Contains(new Point(c, r));
        }
    }
}
