using ILGPU.Runtime.Cuda;
using System.Diagnostics;
using System.Globalization;
using System.IO.Compression;
using System.Runtime.CompilerServices;

namespace moonlib.util
{
    public static unsafe class OldUtilities
    {
        /// <summary>
        /// Instead of using "false" we can obfuscate this to remove compile warnings. 
        /// If we comment out the code then it may rot
        /// </summary>
        /// <returns></returns>
        public static bool debug_false => false;

        /// <summary>
        /// Instead of using "true" we can obfuscate this to remove compile warnings. 
        /// If we comment out the code then it may rot
        /// </summary>
        /// <returns></returns>
        public static bool debug_true => true;

        public static List<List<T>> Split<T>(int n, IEnumerable<T> source)
        {
            var r = Enumerable.Range(0, n).Select(i => new List<T>()).ToList();
            var count = 0;
            foreach (var s in source)
                r[count++ % n].Add(s);
            return r;
        }

        public static void PrintActionDuration(Action a, string? comment = null)
        {
            var stopwatch = Stopwatch.StartNew();
            a();
            stopwatch.Stop();
            if (comment == null) comment = string.Empty;
            Console.WriteLine($"{comment} action took {stopwatch.Elapsed}");
        }

        public static T CreateJaggedArray<T>(params int[] lengths)
        {
            Debug.Assert(lengths != null);
            return (T)Initialize(typeof(T).GetElementType()!, 0, lengths);

            object Initialize(Type type, int index, int[] lens)
            {
                Array array = Array.CreateInstance(type, lens[index]);
                Type elementType = type.GetElementType();
                if (elementType != null)
                {
                    for (int i = 0; i < lens[index]; i++)
                    {
                        array.SetValue(
                            Initialize(elementType, index + 1, lens), i);
                    }
                }
                return array;
            }
        }

        public static (float, float) GetMinMax(this float[,] array)
        {
            var max = float.MinValue;
            var min = float.MaxValue;
            var height = array.GetLength(0);
            var width = array.GetLength(1);
            for (var row = 0; row < height; row++)
                for (var col = 0; col < width; col++)
                {
                    var v = array[row, col];
                    max = Math.Max(max, v);
                    min = Math.Min(min, v);
                }
            return (min, max);
        }

        public static void WriteByteArray(byte[] buf, string path, bool compress = false)
        {
            using (var bw = new BinaryWriter(compress
                ? (Stream)new GZipStream(File.OpenWrite(path), CompressionMode.Compress)
                : (Stream)File.Create(path)))
                bw.Write(buf, 0, buf.Length);
        }

        public static byte[] ReadByteArray(string path, int initialSize = 0, bool compress = false)
        {
            if (!".gz".Equals(System.IO.Path.GetExtension(path).ToLower()) && compress == false)
                return File.ReadAllBytes(path);

            var ms = initialSize == 0 ? new MemoryStream() : new MemoryStream(initialSize);
            using (var fs = File.Open(path, FileMode.Open))
            using (var gz = new GZipStream(fs, CompressionMode.Decompress))
                gz.CopyTo(ms);
            var buf = ms.GetBuffer();
            if (ms.Length == buf.Length)
                return buf;
            var buf1 = new byte[ms.Length];
            Array.Copy(buf, buf1, ms.Length);
            return buf1;
        }

        /// <summary>
        /// Shuffle (permute) an array using the Durnstenfeld version of the Fischer-Yates shuffle
        /// https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#The_modern_algorithm
        /// </summary>
        /// <typeparam name="T">Array element type.</typeparam>
        /// <param name="array">Array to shuffle.</param>
        public static void Shuffle<T>(T[] array, Random random)
        {
            int n = array.Length;
            for (int i = 0; i < n; i++)
            {
                // Use Next on random instance with an argument.
                // ... The argument is an exclusive bound.
                //     So we will not go past the end of the array.
                int r = i + random.Next(n - i);
                T t = array[r];
                array[r] = array[i];
                array[i] = t;
            }
        }

        public static void Shuffle<T>(List<T> list, Random random)
        {
            int n = list.Count;
            for (int i = 0; i < n; i++)
            {
                // Use Next on random instance with an argument.
                // ... The argument is an exclusive bound.
                //     So we will not go past the end of the array.
                int r = i + random.Next(n - i);
                T t = list[r];
                list[r] = list[i];
                list[i] = t;
            }
        }

        public static T[][] MakeJaggedArray<T>(int height, int width) => Enumerable.Range(0, height).Select(row => new T[width]).ToArray();

        public static T2[][] ConvertJaggedArray<T1, T2>(T1[][] a, Func<T1, T2> f)
        {
            var b = MakeJaggedArray<T2>(a.Length, a[0].Length);
            for (var r = 0; r < a.Length; r++)
            {
                var (arow, brow) = (a[r], b[r]);
                for (var c = 0; c < arow.Length; c++)
                    brow[c] = f(arow[c]);
            }
            return b;
        }

        public static T2[,] ConvertArray<T1, T2>(T1[,] a, Func<T1, T2> f)
        {
            var (height, width) = (a.GetLength(0), a.GetLength(1));
            var b = new T2[height, width];
            for (var r = 0; r < height; r++)
                for (var c = 0; c < width; c++)
                    b[r, c] = f(a[r, c]);
            return b;
        }

        public static T[][] ConvertToJaggedArray<T>(T[,] a)
        {
            var (height, width) = (a.GetLength(0), a.GetLength(1));
            var b = MakeJaggedArray<T>(height, width);
            for (var r = 0; r < height; r++)
            {
                var brow = b[r];
                for (var c = 0; c < width; c++)
                    brow[c] = a[r, c];
            }
            return b;
        }

        #region Bitmap Data Access

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte GetDataBit(byte* data, int index) => (byte)((*(data + (index >> 3)) >> (index & 0x7)) & 1);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte GetDataBit(byte[] data, int bitindex) => (byte)((data[bitindex >> 3] >> (7 - (bitindex & 0x7))) & 1);
        public static byte GetDataBit(byte[] data, long bitindex) => (byte)((data[(int)(bitindex >> 3)] >> (7 - (int)(bitindex & 0x7))) & 1);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte GetDataBit(byte[] data, int imageno, int height, int width, int row, int col)
        {
            long bitindex = imageno * (height * (long)width) + row * width + col;
            return GetDataBit(data, bitindex);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SetDataBit(byte* data, int index, byte value)
        {
            byte* wordPtr = data + (index >> 3);
            int v = *wordPtr & (byte)~(0x80 >> (index & 7)); 			// clear bit, note first pixel in the byte is most significant (1000 0000)
            *wordPtr = (byte)(v | ((value & 1) << (7 - (index & 7))));  // set bit, if value is 1
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SetDataBit(byte[] data, int bitindex, byte value)
        {
            int wordPtr = bitindex >> 3;
            int v = data[wordPtr] & (byte)~(1 << (7 - (bitindex & 7))); 			// clear bit, note first pixel in the byte is most significant (1000 0000)
            data[wordPtr] = (byte)(v | ((value & 1) << (7 - (bitindex & 7))));      // set bit, if value is 1  (TODO WORKING HERE!!!!)
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SetDataBit(byte[] data, int imageno, int height, int width, int row, int col, byte value)
        {
            int bitindex = imageno * (height * width) + row * width + col;
            SetDataBit(data, bitindex, value);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte GetDataQBit(byte* data, int index) => (byte)((*(data + (index >> 1)) >> (4 * (index & 1))) & 0xF);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SetDataQBit(byte* data, int index, byte value)
        {
            byte* wordPtr = data + (index >> 1);
            *wordPtr &= (byte)~(0xF0 >> (4 * (index & 1))); // clears qbit located at index, note like bit the qbit corresponding to the first pixel is the most significant (0xF0)
            *wordPtr |= (byte)((value & 0x0F) << (4 - (4 * (index & 1)))); // applys qbit to n
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte GetDataByte(byte* data, int index) => *(data + index);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SetDataByte(byte* data, int index, byte value)
        {
            *(data + index) = value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort GetDataUInt16(ushort* data, int index) => *(data + index);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SetDataUInt16(ushort* data, int index, ushort value)
        {
            *(data + index) = value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint GetDataUInt32(uint* data, int index) => *(data + index);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SetDataUInt32(uint* data, int index, uint value)
        {
            *(data + index) = value;
        }

        #endregion

        public static void WriteFloatArray(float[,] ary, string path, Func<float, float>? func = null)
        {
            var (height, width) = (ary.GetLength(0), ary.GetLength(1));
            using (var bw = new BinaryWriter(File.OpenWrite(path)))
            {
                if (func == null)
                {
                    for (int row = 0; row < height; row++)
                        for (int col = 0; col < width; col++)
                            bw.Write(ary[row, col]);
                }
                else
                {
                    for (int row = 0; row < height; row++)
                        for (int col = 0; col < width; col++)
                            bw.Write(func(ary[row, col]));
                }
            }
        }

        public static float[,] ReadFloatArray(string path, int width, int height)
        {
            var result = new float[height, width];
            using (var br = new BinaryReader(File.OpenRead(path)))
                for (int row = 0; row < height; row++)
                    for (int col = 0; col < width; col++)
                        result[row, col] = br.ReadSingle();
            return result;
        }

        public static byte[,] ReadByteArray(string path, int width, int height)
        {
            var result = new byte[height, width];
            using (var br = new BinaryReader(File.OpenRead(path)))
                for (int row = 0; row < height; row++)
                    for (int col = 0; col < width; col++)
                        result[row, col] = br.ReadByte();
            return result;
        }

        static string? CachedExecutableDirectory;

        public static string ExecutableDirectory
            => CachedExecutableDirectory
            ?? (CachedExecutableDirectory = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location)!);

        private static string[] TryParseFormats = {
            "yyyy-MM-ddTHH:mm:ss",
            "yyyy-MM-ddTHH-mm-ss",
            "yyyy-MM-ddTHH:mm:ssZ",
            "yyyy-MM-ddTHH-mm-ssZ",
            "O", // ISO 8601 round-trip
            "s"  // Sortable date/time pattern
        };

        public static bool TryParseDateTime(string s, out DateTime dt)
        {
            // Use moonlib.util.OldUtilities.TryParseDateTime if available
            if (DateTime.TryParseExact(s, TryParseFormats, CultureInfo.InvariantCulture, DateTimeStyles.AdjustToUniversal | DateTimeStyles.AssumeUniversal, out dt))
            {
                if (dt.Kind == DateTimeKind.Utc)
                    return true;
                // Treat both Local and Unspecified as UTC (no conversion)
                dt = DateTime.SpecifyKind(dt, DateTimeKind.Utc);
                return true;
            }
            // Fallback: DateTime.TryParse
            if (DateTime.TryParse(s, null, System.Globalization.DateTimeStyles.RoundtripKind, out dt) ||
                DateTime.TryParse(s, out dt))
            {
                if (dt.Kind == DateTimeKind.Utc)
                    return true;
                // Treat both Local and Unspecified as UTC (no conversion)
                dt = DateTime.SpecifyKind(dt, DateTimeKind.Utc);
                return true;
            }
            return false;
        }
    }
}
