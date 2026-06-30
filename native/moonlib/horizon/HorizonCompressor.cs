using System;
using System.Diagnostics;

namespace moonlib.horizon
{
    public static class HorizonCompressor
    {
        private const int expectedHorizonLength = 1440;
        private const float minHorizonElevationDeg = -50f;
        private const float maxHorizonElevationDeg = 50f;
        private const float elevationToShortScale = short.MaxValue / maxHorizonElevationDeg;
        private const float shortToElevationScale = maxHorizonElevationDeg / short.MaxValue;
        private const short maxSigned15bit = 16383;
        private const short minSigned15bit = -16384;
        private const short maxSigned7bit = 63;
        private const short minSigned7bit = -64;

        #region Byte Arrays
        /// <summary>
        /// 
        /// </summary>
        /// <param name="buf"></param>
        /// <returns></returns>
        /// 
        /// If the top bit of a byte is 0, then the value is a 7 bit signed number
        /// If the top bit is 1, then the value is a 15 bit signed value
        /// 
        public static byte[] Encode(short[] buf)
        {
            if (buf == null) throw new ArgumentNullException(nameof(buf));
            if (buf.Length == 0) return Array.Empty<byte>();

            int size = GetEncodedLength(buf);
            var result = new byte[size];
            int written = Encode(buf, result);
            Debug.Assert(written == size);
            return result;
        }

        public static byte[] Encode(float[] buf)
        {
            if (buf == null) throw new ArgumentNullException(nameof(buf));
            ValidateHorizonLength(buf.Length);

            Span<short> quantized = stackalloc short[expectedHorizonLength];
            QuantizeHorizon(buf, quantized);

            int size = GetEncodedLength(quantized);
            var result = new byte[size];
            int written = Encode(quantized, result);
            Debug.Assert(written == size);
            return result;
        }

        public static short[] Decode(byte[] buf)
        {
            if (buf == null || buf.Length == 0)
                return Array.Empty<short>();

            int size = GetDecodedLength(buf);
            if (size == 0) return Array.Empty<short>();

            var result = new short[size];
            int written = Decode(buf, result);
            if (written == size) return result;

            var trimmed = new short[written];
            result.AsSpan(0, written).CopyTo(trimmed);
            return trimmed;
        }

        public static float[] DecodeToFloat(byte[] buf)
        {
            if (buf == null || buf.Length == 0)
                return Array.Empty<float>();

            int size = GetDecodedLength(buf);
            if (size != expectedHorizonLength)
            {
                throw new ArgumentException(
                    $"Expected encoded horizon with {expectedHorizonLength} samples.",
                    nameof(buf));
            }

            var result = new float[expectedHorizonLength];
            int written = Decode(buf, result);
            Debug.Assert(written == expectedHorizonLength);
            return result;
        }

        #endregion

        #region Spans

        public static int Encode(ReadOnlySpan<short> input, Span<byte> output)
        {
            if (input.Length == 0) return 0;

            int written = 0;
            short prev = 0;

            for (int i = 0; i < input.Length; i++)
            {
                short val;
                if (i == 0)
                {
                    val = input[0];
                    prev = val;
                    
                    var (high, low) = ShortToBytes(val);
                    output[written++] = high;
                    output[written++] = low;
                    continue;
                }

                // Calculate delta and clamp to 15-bit range
                int delta = input[i] - prev;
                if (delta < minSigned15bit) delta = minSigned15bit;
                else if (delta > maxSigned15bit) delta = maxSigned15bit;
                
                val = (short)delta;
                prev = (short)(prev + val);

                if (val >= 0)
                {
                    if (val < maxSigned7bit)
                    {
                        output[written++] = (byte)val;
                    }
                    else
                    {
                        var (high, low) = ShortToBytes(val);
                        high |= 0b10000000;
                        output[written++] = high;
                        output[written++] = low;
                    }
                }
                else
                {
                    if (val >= minSigned7bit)
                    {
                        output[written++] = (byte)(val & 0b01111111);
                    }
                    else
                    {
                        var (high, low) = ShortToBytes(val);
                        Debug.Assert((high >> 7) == 1);
                        output[written++] = high;
                        output[written++] = low;
                    }
                }
            }

            return written;
        }

        public static int Encode(ReadOnlySpan<float> input, Span<byte> output)
        {
            ValidateHorizonLength(input.Length);

            Span<short> quantized = stackalloc short[expectedHorizonLength];
            QuantizeHorizon(input, quantized);
            return Encode(quantized, output);
        }

        public static int Decode(ReadOnlySpan<byte> input, Span<short> output)
        {
            if (input.Length < 2) return 0;

            int read = 2, written = 1;
            short acc = ToShort(input[0], input[1]);
            output[0] = acc;

            while (read < input.Length && written < output.Length)
            {
                byte b1 = input[read++];
                if ((b1 & 0b10000000) == 0)
                {
                    var s = Bit7ToShort(b1);
                    acc += s;
                    output[written++] = acc;
                }
                else
                {
                    if (read >= input.Length) break;
                    byte low = input[read++];
                    byte high = (byte)(((b1 << 1) & 0b10000000) | (b1 & 0b01111111));
                    var s = ToShort(high, low);
                    acc += s;
                    output[written++] = acc;
                }
            }

            return written;
        }

        public static int Decode(ReadOnlySpan<byte> input, Span<float> output)
        {
            if (input.Length < 2) return 0;

            int size = GetDecodedLength(input);
            if (size != expectedHorizonLength)
            {
                throw new ArgumentException(
                    $"Expected encoded horizon with {expectedHorizonLength} samples.",
                    nameof(input));
            }

            if (output.Length < expectedHorizonLength)
                throw new ArgumentException("Output span is too small.", nameof(output));

            Span<short> temp = stackalloc short[expectedHorizonLength];
            int written = Decode(input, temp);
            if (written != expectedHorizonLength)
            {
                throw new InvalidOperationException(
                    $"Decoded {written} samples; expected {expectedHorizonLength}.");
            }

            for (int i = 0; i < expectedHorizonLength; i++)
            {
                output[i] = ShortToElevation(temp[i]);
            }

            return written;
        }

        #endregion

        #region Utilities

        static short ToShort(short high, short low) => (short)((high << 8) + low);

        static (byte high, byte low) ShortToBytes(short number)
        {
            var high = (byte)(number >> 8);
            var low = (byte)(number & 255);
            return (high, low);
        }

        static short Bit7ToShort(byte b)
        {
            var b1 = 0b01111111 & b;            // ignore the high bit
            var b2 = (b1 << 1) & 0b10000000;
            var b3 = b1 | b2;
            var s1 = (sbyte)b3;

            return (short)s1;
        }

        static short Bit15ToShort(int b)
        {
            var bit14 = b & 0b100000000000000;      // Get the sign bit of a 15 bit quantity
            var bit15 = bit14 << 1;                 // shift it to the 15th bit
            var b2 = b | bit15;                     // or it in, extending the sign bit
            var s1 = (short)b2;
            return s1;
        }

        static int GetEncodedLength(ReadOnlySpan<short> input)
        {
            if (input.Length == 0) return 0;

            int size = 2; // First sample is always stored as raw short.
            short prev = input[0];
            for (int i = 1; i < input.Length; i++)
            {
                // Calculate delta and clamp to 15-bit range
                int delta = input[i] - prev;
                if (delta < minSigned15bit) delta = minSigned15bit;
                else if (delta > maxSigned15bit) delta = maxSigned15bit;
                
                short val = (short)delta;
                prev = (short)(prev + val);

                if (val >= 0)
                {
                    size += val < maxSigned7bit ? 1 : 2;
                }
                else
                {
                    size += val >= minSigned7bit ? 1 : 2;
                }
            }

            return size;
        }

        static int GetDecodedLength(ReadOnlySpan<byte> input)
        {
            if (input.Length < 2) return 0;

            int read = 2, written = 1;
            while (read < input.Length)
            {
                byte b1 = input[read++];
                if ((b1 & 0b10000000) == 0)
                {
                    written++;
                }
                else
                {
                    if (read >= input.Length) break;
                    read++;
                    written++;
                }
            }

            return written;
        }

        static void ValidateHorizonLength(int length)
        {
            if (length != expectedHorizonLength)
            {
                throw new ArgumentException(
                    $"Expected {expectedHorizonLength} horizon samples.",
                    nameof(length));
            }
        }

        static void QuantizeHorizon(ReadOnlySpan<float> input, Span<short> output)
        {
            ValidateHorizonLength(input.Length);
            if (output.Length < expectedHorizonLength)
                throw new ArgumentException("Output span is too small.", nameof(output));

            for (int i = 0; i < expectedHorizonLength; i++)
            {
                output[i] = ElevationToShort(input[i]);
            }
        }

        static short ElevationToShort(float elevationDeg)
        {
            float clamped = Math.Clamp(elevationDeg, minHorizonElevationDeg, maxHorizonElevationDeg);
            int quantized = (int)MathF.Round(clamped * elevationToShortScale, MidpointRounding.AwayFromZero);
            quantized = Math.Clamp(quantized, -short.MaxValue, short.MaxValue);
            return (short)quantized;
        }

        static float ShortToElevation(short value)
        {
            return value * shortToElevationScale;
        }

        #endregion
    }
}
