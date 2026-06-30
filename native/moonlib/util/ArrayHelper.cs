using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Runtime.CompilerServices;

namespace moonlib.util
{
    public static unsafe class ArrayHelper
    {
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

        public static void WriteBuffer(byte[] buf, string path)
        {
            using (var fs = File.Create(path))
            using (var gz = new GZipStream(fs, CompressionMode.Compress))
                gz.Write(buf, 0, buf.Length);
        }

        public static byte[] ReadBuffer(string path, int initialSize = 0)
        {
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

        #region Bitmap Data Access

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte GetDataBit(byte* data, int index) => (byte)((*(data + (index >> 3)) >> (index & 0x7)) & 1);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte GetDataBit(byte[] data, int bitindex) => (byte)((data[bitindex >> 3] >> (7 - (bitindex & 0x7))) & 1);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte GetDataBit(byte[] data, int imageno, int height, int width, int row, int col)
        {
            int bitindex = imageno * (height * width) + row * width + col;
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

        #region Conversions

        public static T[,] Copy<T>(T[,] a)
        {
            var (height, width) = (a.GetLength(0), a.GetLength(1));
            var r = new T[height, width];
            for (var row = 0; row < height; row++)
                for (var col = 0; col < width; col++)
                    r[row, col] = a[row, col];
            return r;
        }

        public static byte[,] ThresholdToByteArray<T>(T[,] array, T threshold, byte is_haven, byte is_not_haven) where T : IComparable
        {
            var (height, width) = (array.GetLength(0), array.GetLength(1));
            var thresholded = new byte[height, width];
            for (var row = 0; row < height; row++)
                for (var col = 0; col < width; col++)
                {
                    if (array[row, col].CompareTo(threshold) != 1)
                        thresholded[row, col] = is_haven;
                    else
                        thresholded[row, col] = is_not_haven;
                }
            return thresholded;
        }

        public static byte[,] ApplyBoolean<T>(T[,] array, Func<T, bool> pred)
        {
            var (height, width) = (array.GetLength(0), array.GetLength(1));
            var result = new byte[height, width];
            for (var row = 0; row < height; row++)
                for (var col = 0; col < width; col++)
                    result[row, col] = pred(array[row, col]) ? (byte)1 : (byte)0;
            return result;
        }

        public static int CountBoolean<T>(T[,] array, Func<T, bool> pred)
        {
            var (height, width) = (array.GetLength(0), array.GetLength(1));
            var result = 0;
            for (var row = 0; row < height; row++)
                for (var col = 0; col < width; col++)
                    if (pred(array[row, col]))
                        result++;
            return result;
        }

        public static byte[,] ApplyBoolean<T>(T[,] a, T[,] b, Func<T, T, bool> pred)
        {
            var (height, width) = (a.GetLength(0), a.GetLength(1));
            var result = new byte[height, width];
            for (var row = 0; row < height; row++)
                for (var col = 0; col < width; col++)
                    result[row, col] = pred(a[row, col], b[row, col]) ? (byte)1 : (byte)0;
            return result;
        }

        #endregion

        #region search

        public static Point FirstTrue(byte[,] a)
        {
            var (height, width) = (a.GetLength(0), a.GetLength(1));
            for (var row = 0; row < height; row++)
                for (var col = 0; col < width; col++)
                    if (a[row, col] != 0)
                        return new Point(col, row);
            return new Point(-1, -1);
        }

        public static IEnumerable<Point> EnumerateTrue(byte[,] a)
        {
            var (height, width) = (a.GetLength(0), a.GetLength(1));
            for (var row = 0; row < height; row++)
                for (var col = 0; col < width; col++)
                    if (a[row, col] != 0)
                        yield return new Point(col, row);
        }

        public static IEnumerable<Point> EnumeratePredicate<T1>(T1[,] a, Func<T1, bool> pred)
        {
            var (height, width) = (a.GetLength(0), a.GetLength(1));
            for (var row = 0; row < height; row++)
                for (var col = 0; col < width; col++)
                    if (pred(a[row, col]))
                        yield return new Point(col, row);
        }

        public static IEnumerable<Point> EnumeratePredicate<T1, T2>(T1[,] a, T2[,] b, Func<T1, T2, bool> pred)
        {
            var (height, width) = (a.GetLength(0), a.GetLength(1));
            for (var row = 0; row < height; row++)
                for (var col = 0; col < width; col++)
                    if (pred(a[row, col], b[row, col]))
                        yield return new Point(col, row);
        }

        public static IEnumerable<(Point point, T value)> EnumerateValues<T>(T[,] a)
        {
            var (height, width) = (a.GetLength(0), a.GetLength(1));
            for (var row = 0; row < height; row++)
                for (var col = 0; col < width; col++)
                    yield return (new Point(col, row), a[row, col]);
        }

        #endregion
    }
}
