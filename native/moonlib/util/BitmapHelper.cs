using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.CompilerServices;
//using System.Windows.Media.Imaging;

#nullable disable

namespace moonlib.util
{
    /// <summary>
    /// Description of BitmapHelper.
    /// </summary>
    public static unsafe class BitmapHelper
    {
        /// <summary>
        /// gets the number of Bits Per Pixel (BPP)
        /// </summary>
        /// <param name="bitmap"></param>
        /// <returns></returns>
        public static int GetBPP(Bitmap bitmap)
        {
            switch (bitmap.PixelFormat)
            {
                case PixelFormat.Format1bppIndexed: return 1;
                case PixelFormat.Format4bppIndexed: return 4;
                case PixelFormat.Format8bppIndexed: return 8;
                case PixelFormat.Format16bppArgb1555:
                case PixelFormat.Format16bppGrayScale:
                case PixelFormat.Format16bppRgb555:
                case PixelFormat.Format16bppRgb565: return 16;
                case PixelFormat.Format24bppRgb: return 24;
                case PixelFormat.Format32bppArgb:
                case PixelFormat.Format32bppPArgb:
                case PixelFormat.Format32bppRgb: return 32;
                case PixelFormat.Format48bppRgb: return 48;
                case PixelFormat.Format64bppArgb:
                case PixelFormat.Format64bppPArgb: return 64;
                default: throw new ArgumentException(String.Format("The bitmap's pixel format of {0} was not recognized.", bitmap.PixelFormat), "bitmap");
            }
        }

        /// <summary>
        /// Bytes per row for a bitmap, assuming 1 bit per pixel
        /// </summary>
        /// <param name="width"></param>
        /// <returns></returns>
        public static int BytesPerRow(int width) => 4 * ((((width + 7) / 8) + 3) / 4);

        /*
        public static int BytesPerRow(int width)
        {
            int rem;
            int stride = Math.DivRem(width, 8, out rem);  // divide 8bpp by 8 to get  1bpp
            if (rem > 0) stride += 1;
            Math.DivRem(stride, 4, out rem); // round newStride up to multiple of 4 bytes
            Math.DivRem(4 - rem, 4, out rem);
            stride += rem;
            return stride;
        }
        */

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int BitsPerRow(int width) => 8 * BytesPerRow(width);

        #region PixelFormat conversion

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint ConvertRgb555ToRGBA(uint val)
        {
            uint red = ((val & 0x7C00) >> 10);
            uint green = ((val & 0x3E0) >> 5);
            uint blue = (val & 0x1F);

            return ((red << 3 | red >> 2) << 24) |
                ((green << 3 | green >> 2) << 16) |
                ((blue << 3 | blue >> 2) << 8) |
                0xFF;
        }

       	[MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint ConvertRgb565ToRGBA(uint val)
        {
            uint red = ((val & 0xF800) >> 11);
            uint green = ((val & 0x7E0) >> 5);
            uint blue = (val & 0x1F);

            return ((red << 3 | red >> 2) << 24) |
                ((green << 2 | green >> 4) << 16) |
                ((blue << 3 | blue >> 2) << 8) |
                0xFF;
        }

       	[MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint ConvertArgb1555ToRGBA(uint val)
        {
            uint alpha = ((val & 0x8000) >> 15);
            uint red = ((val & 0x7C00) >> 10);
            uint green = ((val & 0x3E0) >> 5);
            uint blue = (val & 0x1F);

            return ((red << 3 | red >> 2) << 24) |
                ((green << 3 | green >> 2) << 16) |
                ((blue << 3 | blue >> 2) << 8) |
                ((alpha << 8) - alpha); // effectively alpha * 255, only works as alpha will be either 0 or 1
        }

       	[MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint EncodeAsRGBA(byte red, byte green, byte blue, byte alpha)
        {
            return (uint)((red << 24) |
                (green << 16) |
                (blue << 8) |
                alpha);
        }

        public static void HSVToRGB(double H, double S, double V, out double R, out double G, out double B)
        {
            if (H == 1.0)
                H = 0.0;

            double step = 1.0 / 6.0;
            double vh = H / step;

            var i = (int)Math.Floor(vh);

            double f = vh - i;
            double p = V * (1.0 - S);
            double q = V * (1.0 - (S * f));
            double t = V * (1.0 - (S * (1.0 - f)));

            switch (i)
            {
                case 0:
                    {
                        R = V;
                        G = t;
                        B = p;
                        break;
                    }
                case 1:
                    {
                        R = q;
                        G = V;
                        B = p;
                        break;
                    }
                case 2:
                    {
                        R = p;
                        G = V;
                        B = t;
                        break;
                    }
                case 3:
                    {
                        R = p;
                        G = q;
                        B = V;
                        break;
                    }
                case 4:
                    {
                        R = t;
                        G = p;
                        B = V;
                        break;
                    }
                case 5:
                    {
                        R = V;
                        G = p;
                        B = q;
                        break;
                    }
                default:
                    {
                        // not possible - if we get here it is an internal error
                        throw new ArgumentException();
                    }
            }
        }

        #endregion

        #region working with drawing bitmaps

        public static Bitmap ToFormat32bppArgb(this Bitmap source)
        {
            if (source.PixelFormat == PixelFormat.Format32bppArgb)
                return source;
            var bmp = new Bitmap(source.Width, source.Height, PixelFormat.Format32bppArgb);
            using (var g = Graphics.FromImage(bmp))
                g.DrawImageUnscaled(source, 0, 0);
            return bmp;
        }

        public static unsafe Bitmap ToFormat32bppArgb(int[,] ary)
        {
            Debug.Assert(ary != null);
            var height = ary.GetLength(0);
            var width = ary.GetLength(1);
            var bmp = new Bitmap(width, height, PixelFormat.Format32bppArgb);
            var bmpdata = bmp.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, bmp.PixelFormat);
            for (var row = 0; row < height; row++)
            {
                var rowptr = (int*)(bmpdata.Scan0 + row * bmpdata.Stride);
                for (var col = 0; col < width; col++)
                    rowptr[col] = ary[row, col];
            }
            bmp.UnlockBits(bmpdata);
            return bmp;
        }

        public static unsafe Bitmap ToFormat32bppArgb(int[,] ary, Func<int, Int32> conv)
        {
            Debug.Assert(ary != null);
            var height = ary.GetLength(0);
            var width = ary.GetLength(1);
            var bmp = new Bitmap(width, height, PixelFormat.Format32bppArgb);
            var bmpdata = bmp.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, bmp.PixelFormat);
            for (var row = 0; row < height; row++)
            {
                var rowptr = (int*)(bmpdata.Scan0 + row * bmpdata.Stride);
                for (var col = 0; col < width; col++)
                    rowptr[col] = conv(ary[row, col]);
            }
            bmp.UnlockBits(bmpdata);
            return bmp;
        }

        public static unsafe Bitmap ToFormat32bppArgb(byte[,] ary, Func<byte, Int32> conv = null)
        {
            Debug.Assert(ary != null);
            if (conv == null) conv = b => Color.FromArgb(b, b, b).ToArgb();
            var height = ary.GetLength(0);
            var width = ary.GetLength(1);
            var bmp = new Bitmap(width, height, PixelFormat.Format32bppArgb);
            var bmpdata = bmp.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, bmp.PixelFormat);
            for (var row = 0; row < height; row++)
            {
                var rowptr = (int*)(bmpdata.Scan0 + row * bmpdata.Stride);
                for (var col = 0; col < width; col++)
                    rowptr[col] = conv(ary[row, col]);
            }
            bmp.UnlockBits(bmpdata);
            return bmp;
        }

        public static unsafe Bitmap ToFormat32bppArgb(float[,] ary, Func<float, Int32> conv = null)
        {
            Debug.Assert(ary != null);
            if (conv == null) conv = b => Color.FromArgb((int)(b*255f), (int)(b * 255f), (int)(b * 255f)).ToArgb();
            var height = ary.GetLength(0);
            var width = ary.GetLength(1);
            var bmp = new Bitmap(width, height, PixelFormat.Format32bppArgb);
            var bmpdata = bmp.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, bmp.PixelFormat);
            for (var row = 0; row < height; row++)
            {
                var rowptr = (int*)(bmpdata.Scan0 + row * bmpdata.Stride);
                for (var col = 0; col < width; col++)
                    rowptr[col] = conv(ary[row, col]);
            }
            bmp.UnlockBits(bmpdata);
            return bmp;
        }

        public static unsafe Bitmap ToFormat32bppArgb(byte[][] ary, Func<byte, Int32> conv = null)
        {
            Debug.Assert(ary != null);
            if (conv == null) conv = b => Color.FromArgb(b, b, b).ToArgb();
            var height = ary.Length;
            var width = ary[0].Length;
            var bmp = new Bitmap(width, height, PixelFormat.Format32bppArgb);
            var bmpdata = bmp.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, bmp.PixelFormat);
            for (var row = 0; row < height; row++)
            {
                var rowptr = (int*)(bmpdata.Scan0 + row * bmpdata.Stride);
                var rowary = ary[row];
                for (var col = 0; col < width; col++)
                    rowptr[col] = conv(rowary[col]);
            }
            bmp.UnlockBits(bmpdata);
            return bmp;
        }

        public static unsafe Bitmap ToFormat8bppIndexed(byte[,] ary, Bitmap bmp = null)
        {
            Debug.Assert(ary != null);
            var height = ary.GetLength(0);
            var width = ary.GetLength(1);
            if (bmp == null)
            {
                bmp = new Bitmap(width, height, PixelFormat.Format8bppIndexed);
                var p = bmp.Palette;   // Initialize the palette just in case
                for (var i = 0; i < 256; i++) p.Entries[i] = Color.FromArgb(i,i,i);
                bmp.Palette = p;
            }
            Debug.Assert(bmp.Width == width && bmp.Height == height);
            var bmpdata = bmp.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, bmp.PixelFormat);
            for (var row = 0; row < height; row++)
            {
                var rowptr = (byte*)(bmpdata.Scan0 + row * bmpdata.Stride);
                for (var col = 0; col < width; col++)
                    rowptr[col] = ary[row, col];
            }
            bmp.UnlockBits(bmpdata);
            return bmp;
        }

        public static unsafe Bitmap ToFormat8bppIndexed(float[,] ary, Func<float, byte> conversion = null)
        {
            Debug.Assert(ary != null);
            var f = conversion ?? (v => (byte)v);
            var height = ary.GetLength(0);
            var width = ary.GetLength(1);
            var bmp = new Bitmap(width, height, PixelFormat.Format8bppIndexed);
            var bmpdata = bmp.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, bmp.PixelFormat);
            for (var row = 0; row < height; row++)
            {
                var rowptr = (byte*)(bmpdata.Scan0 + row * bmpdata.Stride);
                for (var col = 0; col < width; col++)
                    rowptr[col] = f(ary[row, col]);
            }
            bmp.UnlockBits(bmpdata);

            // Initialize the palette just in case
            var p = bmp.Palette;
            for (var i = 0; i < 256; i++) p.Entries[i] = Color.LightGray;
            bmp.Palette = p;

            return bmp;
        }

        public static unsafe Bitmap ToFormat8bppIndexed(Bitmap src, Func<byte, byte> conversion = null)
        {
            Debug.Assert(src != null);
            var f = conversion ?? (v => (byte)v);
            var height = src.Height;
            var width = src.Width;
            Debug.Assert(src.PixelFormat == PixelFormat.Format8bppIndexed);
            var src_data = src.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadOnly, src.PixelFormat);
            var dst = new Bitmap(width, height, PixelFormat.Format8bppIndexed);
            var dst_data = dst.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, dst.PixelFormat);
            for (var row = 0; row < height; row++)
            {
                var dst_ptr = (byte*)(dst_data.Scan0 + row * dst_data.Stride);
                var src_ptr = (byte*)(src_data.Scan0 + row * src_data.Stride);
                for (var col = 0; col < width; col++)
                    dst_ptr[col] = f(src_ptr[col]);
            }
            dst.UnlockBits(dst_data);
            src.UnlockBits(src_data);
            dst.Palette = src.Palette;
            return dst;
        }

        public static unsafe Bitmap ToFormat8bppIndexed(float[,] ary1, float[,] ary2, Func<float, float, byte> conversion)
        {
            Debug.Assert(ary1 != null & ary2 != null);
            var height = ary1.GetLength(0);
            var width = ary1.GetLength(1);
            Debug.Assert(height == ary2.GetLength(0) & width == ary2.GetLength(1));

            var bmp = new Bitmap(width, height, PixelFormat.Format8bppIndexed);
            var bmpdata = bmp.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, bmp.PixelFormat);
            for (var row = 0; row < height; row++)
            {
                var rowptr = (byte*)(bmpdata.Scan0 + row * bmpdata.Stride);
                for (var col = 0; col < width; col++)
                    rowptr[col] = conversion(ary1[row, col], ary2[row, col]);
            }
            bmp.UnlockBits(bmpdata);

            // Initialize the palette just in case
            var p = bmp.Palette;
            for (var i = 0; i < 256; i++) p.Entries[i] = Color.LightGray;
            bmp.Palette = p;

            return bmp;
        }

        /// <summary>
        /// Returns a 2D byte array containing the pixel data or null if the pixel format isn't 8 bit indexed
        /// </summary>
        /// <param name="path"></param>
        /// <returns></returns>
        public static byte[,] Image8bppToByteArray(string path)
        {
            using (var bmp = BitmapLoader.LoadBitmap(path) as Bitmap)
                return bmp.ToByteArray2();
        }

        public static int[,] Image32bppToIntArray(string path)
        {
            using (var bmp = Image.FromFile(path) as Bitmap)
                return bmp.ToIntArray2();
        }

        /*
        public static Bitmap LoadPNG(string file)
        {
            //using (Bitmap PNG = new Bitmap(1, 1, PixelFormat.Format8bppIndexed))
            // PNG.Save("Test.png", ImageFormat.Png);
            PngBitmapDecoder pngDec = new PngBitmapDecoder(new Uri(file),
               BitmapCreateOptions.PreservePixelFormat,
               BitmapCacheOption.OnDemand);
            BmpBitmapEncoder bmpEnc = new BmpBitmapEncoder();
            bmpEnc.Frames.Add(pngDec.Frames[0]);
            Bitmap bmp = null;
            using (MemoryStream stream = new MemoryStream())
            {
                bmpEnc.Save(stream);
                bmp = new Bitmap(stream);
            }
            return bmp;
        }
        */

        public static void LoadPalette(Bitmap bmp, Color[] colors)
        {
            Debug.Assert(bmp.PixelFormat == PixelFormat.Format8bppIndexed);
            var p = bmp.Palette;
            for (var i = 0; i < colors.Length; i++)
                p.Entries[i] = colors[i];
            bmp.Palette = p;
        }

        public static void LoadPalette(Bitmap bmp, List<(Color color, int from, int to)> spec)
        {
            Debug.Assert(bmp.PixelFormat == PixelFormat.Format8bppIndexed);
            var p = bmp.Palette;
            foreach (var (color, from, to) in spec)
                for (var i = from; i < to; i++)
                    p.Entries[i] = color;
            bmp.Palette = p;
        }

        public static void LoadPalette(Bitmap bmp, int from, int to, Func<int, float, Color> getter)
        {
            Debug.Assert(bmp.PixelFormat == PixelFormat.Format8bppIndexed);
            var p = bmp.Palette;
            if (from + 1 == to)
                p.Entries[from] = getter(from, 0f);
            else
            {
                for (var i = from; i < to; i++)
                    p.Entries[i] = getter(i, (i - from) / ((to - 1) - (float)from));
            }
            bmp.Palette = p;
        }

        public static int[,] ToFormatPixelArray(float[,] ary, Func<float, Int32> conv = null)
        {
            Debug.Assert(ary != null);
            if (conv == null) conv = b => Color.FromArgb((int)(b * 255f), (int)(b * 255f), (int)(b * 255f)).ToArgb();
            var (height, width) = (ary.GetLength(0), ary.GetLength(1));
            var result = new int[height, width];

            for (var row = 0; row < height; row++)
                for (var col = 0; col < width; col++)
                    result[row,col] = conv(ary[row, col]);

            return result;
        }

        #endregion

        public static ImageFormat FormatFromFile(string filename)
        {
            switch (Path.GetExtension(filename).ToLower())
            {
                case ".png":
                    return ImageFormat.Png;
                case ".tif":
                case ".tiff":
                    return ImageFormat.Tiff;
                default:
                    return ImageFormat.Png;
            }
        }

        //TODO: Remove these.  They duplicate code in moonlib.util.ArrayHelper

        #region Bitmap Data Access

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte GetDataBit(byte* data, int index) => (byte)((*(data + (index >> 3)) >> (index & 0x7)) & 1);

        // NOTE: Changed the address of the bit to a long to handle the larger bit array for the SfS test site.
        // I don't know what this does to performance on other tasks.

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
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
        public static void SetDataBit(byte[] data, long bitindex, byte value)
        {
            int wordPtr = (int)(bitindex >> 3);
            int v = data[wordPtr] & (byte)~(1 << (7 - (int)(bitindex & 7))); 			// clear bit, note first pixel in the byte is most significant (1000 0000)
            data[wordPtr] = (byte)(v | ((value & 1) << (7 - (int)(bitindex & 7))));     // set bit, if value is 1  (TODO WORKING HERE!!!!)
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SetDataBit(byte[] data, int imageno, int height, int width, int row, int col, byte value)
        {
            long bitindex = imageno * (height * (long)width) + row * width + col;
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

        public static int StrideForWidth(int width)
        {
            int stride = Math.DivRem(width, 8, out int rem);  // divide 8bpp by 8 to get  1bpp
            if (rem > 0) stride += 1;
            Math.DivRem(stride, 4, out rem); // round newStride up to multiple of 4 bytes
            Math.DivRem(4 - rem, 4, out rem);
            stride += rem;
            return stride;
        }

        public static unsafe Bitmap DifferenceImage(byte[,] a, byte[,] b, Color same, Color different)
        {
            var (height, width) = (a.GetLength(0), a.GetLength(1));
            var (same_argb, different_argb) = (same.ToArgb(), different.ToArgb());
            var bmp = new Bitmap(width, height, PixelFormat.Format32bppArgb);
            var bmpdata = bmp.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, bmp.PixelFormat);
            for (var row = 0; row < height; row++)
            {
                var rowptr = (int*)(bmpdata.Scan0 + row * bmpdata.Stride);
                for (var col = 0; col < width; col++)
                    rowptr[col] = a[row, col] == b[row, col] ? same_argb : different_argb;
            }
            bmp.UnlockBits(bmpdata);
            return bmp;
        }

        public static unsafe Bitmap BooleanImage(byte[,] a, Color true_color, Color false_color)
        {
            var (height, width) = (a.GetLength(0), a.GetLength(1));
            var (true_argb, false_argb) = (true_color.ToArgb(), false_color.ToArgb());
            var bmp = new Bitmap(width, height, PixelFormat.Format32bppArgb);
            var bmpdata = bmp.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, bmp.PixelFormat);
            for (var row = 0; row < height; row++)
            {
                var rowptr = (int*)(bmpdata.Scan0 + row * bmpdata.Stride);
                for (var col = 0; col < width; col++)
                    rowptr[col] = a[row, col] == 1 ? true_argb : false_argb;
            }
            bmp.UnlockBits(bmpdata);
            return bmp;
        }
    }

}
