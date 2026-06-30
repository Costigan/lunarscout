using FluentAssertions;
using moonlib.horizon;

namespace moonlib.tests
{
    [TestClass]
    [TestCategory("Fast")]
    /// <summary>
    /// Tests the <see cref="HorizonCompressor"/> class, validating that the variable-length encoding/decoding
    /// (7-bit for small deltas, 15-bit for large deltas) works correctly for various input scenarios.
    /// </summary>
    public class HorizonCompressorTests
    {
        /// <summary>
        /// Ensures encoding returns an empty array when input is empty.
        /// </summary>
        [TestMethod]
        public void Encode_ShouldHandleEmptyInput()
        {
            // Arrange
            var input = new short[] { };

            // Act
            var result = HorizonCompressor.Encode(input);

            // Assert
            result.Should().BeEmpty();
        }

        /// <summary>
        /// Ensures encoding works for a single element array.
        /// </summary>
        [TestMethod]
        public void Encode_ShouldHandleSingleElement()
        {
            // Arrange
            var input = new short[] { 123 };

            // Act
            var result = HorizonCompressor.Encode(input);
            var decoded = HorizonCompressor.Decode(result);

            // Assert
            decoded.Should().Equal(input);
        }

        /// <summary>
        /// Ensures encoding works for positive values that fit within the 7-bit range (0-127 approximately, though specifics depend on implementation).
        /// </summary>
        [TestMethod]
        public void Encode_ShouldHandleAllPositive7BitValues()
        {
            // Arrange
            var input = new short[] { 0, 10, 63 };

            // Act
            var result = HorizonCompressor.Encode(input);
            var decoded = HorizonCompressor.Decode(result);

            // Assert
            decoded.Should().Equal(input);
        }

        /// <summary>
        /// Ensures encoding works for negative values that fit within the 7-bit range.
        /// </summary>
        [TestMethod]
        public void Encode_ShouldHandleAllNegative7BitValues()
        {
            // Arrange
            var input = new short[] { -1, -10, -64 };

            // Act
            var result = HorizonCompressor.Encode(input);
            var decoded = HorizonCompressor.Decode(result);

            // Assert
            decoded.Should().Equal(input);
        }

        /// <summary>
        /// Ensures encoding handles a mix of values requiring 7-bit and 15-bit representation.
        /// </summary>
        [TestMethod]
        public void Encode_ShouldHandleMixed7BitAnd15BitValues()
        {
            // Arrange
            var input = new short[] { 0, 100, -100, 1000, 16383, 2000, -2000, -16384 };

            // Act
            var result = HorizonCompressor.Encode(input);
            var decoded = HorizonCompressor.Decode(result);

            // Assert
            decoded.Should().Equal(input);
        }

        /// <summary>
        /// Ensures encoding handles boundary values (max/min for 7-bit and 15-bit limits).
        /// </summary>
        [TestMethod]
        public void Encode_ShouldHandleBoundaryValues()
        {
            // Arrange
            var input = new short[] { 63, -64, 1000, 4000, 16383, 3000, -3000, -16384 };

            // Act
            var result = HorizonCompressor.Encode(input);
            var decoded = HorizonCompressor.Decode(result);

            // Assert
            decoded.Should().Equal(input);
        }

        /// <summary>
        /// Ensures encoding can process a larger array without error.
        /// </summary>
        [TestMethod]
        public void Encode_ShouldHandleLargeArray()
        {
            // Arrange
            var input = new short[1000];
            for (int i = 0; i < input.Length; i++) input[i] = (short)(i % 100);

            // Act
            var result = HorizonCompressor.Encode(input);
            var decoded = HorizonCompressor.Decode(result);

            // Assert
            decoded.Should().Equal(input);
        }

        /// <summary>
        /// Ensures decoding returns an empty array when input is empty.
        /// </summary>
        [TestMethod]
        public void Decode_ShouldHandleEmptyInput()
        {
            // Arrange
            var input = new byte[] { };

            // Act
            var result = HorizonCompressor.Decode(input);

            // Assert
            result.Should().BeEmpty();
        }

        /// <summary>
        /// Ensures decoding works for a single element.
        /// </summary>
        [TestMethod]
        public void Decode_ShouldHandleSingleElement()
        {
            // Arrange
            var input = HorizonCompressor.Encode(new short[] { 123 });

            // Act
            var result = HorizonCompressor.Decode(input);

            // Assert
            result.Should().Equal(new short[] { 123 });
        }

        /// <summary>
        /// Ensures decoding works for positive 7-bit encoded values.
        /// </summary>
        [TestMethod]
        public void Decode_ShouldHandleAllPositive7BitValues()
        {
            // Arrange
            var input = HorizonCompressor.Encode(new short[] { 0, 10, 63 });

            // Act
            var result = HorizonCompressor.Decode(input);

            // Assert
            result.Should().Equal(new short[] { 0, 10, 63 });
        }

        /// <summary>
        /// Ensures decoding works for negative 7-bit encoded values.
        /// </summary>
        [TestMethod]
        public void Decode_ShouldHandleAllNegative7BitValues()
        {
            // Arrange
            var input = HorizonCompressor.Encode(new short[] { -1, -10, -64 });

            // Act
            var result = HorizonCompressor.Decode(input);

            // Assert
            result.Should().Equal(new short[] { -1, -10, -64 });
        }

        /// <summary>
        /// Ensures decoding works for a mix of 7-bit and 15-bit encoded values.
        /// </summary>
        [TestMethod]
        public void Decode_ShouldHandleMixed7BitAnd15BitValues()
        {
            var ary = new short[] { 0, 100, -100, 1000, 5000, 16383, 4000, 1000, -1000, -3000, -16384 };
            // Arrange
            var input = HorizonCompressor.Encode(ary);

            // Act
            var result = HorizonCompressor.Decode(input);

            // Assert
            result.Should().Equal(ary);
        }

        /// <summary>
        /// Ensures decoding works for boundary values.
        /// </summary>
        [TestMethod]
        public void Decode_ShouldHandleBoundaryValues()
        {
            var ary = new short[] { 15, 63, 15, -64, 1000, 16382, 2000, -1000, -16383 };
            // Arrange
            var input = HorizonCompressor.Encode(ary);

            // Act
            var result = HorizonCompressor.Decode(input);

            // Assert
            result.Should().Equal(ary);
        }

        /// <summary>
        /// Ensures decoding can process a large array correctly.
        /// </summary>
        [TestMethod]
        public void Decode_ShouldHandleLargeArray()
        {
            // Arrange
            var input = new short[1000];
            for (int i = 0; i < input.Length; i++) input[i] = (short)(i % 100);
            var encoded = HorizonCompressor.Encode(input);

            // Act
            var result = HorizonCompressor.Decode(encoded);

            // Assert
            result.Should().Equal(input);
        }

        /// <summary>
        /// Verifies consistency between encoding and decoding for a specific set of values.
        /// </summary>
        [TestMethod]
        public void EncodeAndDecode1_ShouldBeConsistent()
        {
            // Arrange
            var input = new short[] { 0, 100, -100, 3000, 16383, 2000, -5000, -16384 };

            // Act
            var encoded = HorizonCompressor.Encode(input);
            var decoded = HorizonCompressor.Decode(encoded);

            // Assert
            decoded.Should().Equal(input);
        }

        /// <summary>
        /// Verifies round-trip consistency using Span-based APIs.
        /// </summary>
        [TestMethod]
        public void EncodeDecode_RoundTrip_ShouldMatchOriginal()
        {
            short[] data = new short[] { 1234, 1240, 1300, 1200, 1150, 16383, 0, -16384 };
            Span<byte> encoded = new byte[64];
            Span<short> decoded = new short[data.Length];

            int bytesWritten = HorizonCompressor.Encode(data, encoded);
            int itemsDecoded = HorizonCompressor.Decode(encoded.Slice(0, bytesWritten), decoded);

            decoded.Slice(0, itemsDecoded).ToArray().Should().Equal(data);
        }

        /// <summary>
        /// Verifies that encoding empty input to a span writes 0 bytes.
        /// </summary>
        [TestMethod]
        public void Encode_EmptyInput_ShouldWriteNothing()
        {
            var output = new byte[16];
            HorizonCompressor.Encode(Span<short>.Empty, output).Should().Be(0);
        }

        /// <summary>
        /// Verifies that decoding empty input from a span reads 0 items.
        /// </summary>
        [TestMethod]
        public void Decode_EmptyInput_ShouldWriteNothing()
        {
            var output = new short[16];
            HorizonCompressor.Decode(Span<byte>.Empty, output).Should().Be(0);
        }

        /// <summary>
        /// Verifies round-trip consistency for a single element using Span APIs.
        /// </summary>
        [TestMethod]
        public void EncodeDecode_SingleElement_ShouldMatchOriginal()
        {
            short[] data = new short[] { -1234 };
            Span<byte> encoded = new byte[4];
            Span<short> decoded = new short[1];

            int written = HorizonCompressor.Encode(data, encoded);
            int read = HorizonCompressor.Decode(encoded.Slice(0, written), decoded);

            read.Should().Be(1);
            decoded[0].Should().Be(data[0]);
        }

        /// <summary>
        /// Data-driven test to verify round-trip consistency for multiple interesting input scenarios.
        /// </summary>
        [DataTestMethod]
        [DataRow(new short[] { 0, 0, 0, 0 })]
        [DataRow(new short[] { 0, 10, -10, 20, -20 })]
        [DataRow(new short[] { 1, 2, 3, 4, 5, 6, 7 })]
        [DataRow(new short[] { -1, -2, -3, -4, -5 })]
        [DataRow(new short[] { 0, 16383 })]
        [DataRow(new short[] { 0, -16384 })]
        [DataRow(new short[] { short.MaxValue, (short)(short.MaxValue - 1) })]
        [DataRow(new short[] { 0, 63, -64 })]
        [DataRow(new short[] { -64, 0, 63, 1000, -1000, 2000, 16383, 3200, -1000, -16384 })]
        [DataRow(new short[] { 500, 501, 502 })]
        [DataRow(new short[] { -500, -501, -502 })]
        [DataRow(new short[] { 42, 42, 42, 42 })]
        [DataRow(new short[] { short.MaxValue })]
        [DataRow(new short[] { short.MinValue })]
        [DataRow(new short[] { 0, 100, 50, 110, 40, 100 })]
        [DataRow(new short[] { -2000, -1800, -2200, -1700 })]
        public void EncodeDecode_RoundTrip_DataRows(short[] input)
        {
            Span<byte> encoded = new byte[64];
            Span<short> decoded = new short[input.Length];
            int written = HorizonCompressor.Encode(input, encoded);
            int read = HorizonCompressor.Decode(encoded.Slice(0, written), decoded);
            decoded.Slice(0, read).ToArray().Should().Equal(input);
        }

        /// <summary>
        /// Verifies that decoding insufficient bytes does not throw and leaves output mostly unchanged.
        /// </summary>
        [TestMethod]
        public void Decode_InsufficientBytes_ShouldNotThrow_AndOutputUnchanged()
        {
            var input = new byte[] { 0x00 };
            var output = new short[5] { 1, 2, 3, 4, 5 };
            Action act = () => HorizonCompressor.Decode(input, output);
            act.Should().NotThrow();
            // Output should remain unchanged except possibly the first element
            output[1].Should().Be(2);
            output[2].Should().Be(3);
            output[3].Should().Be(4);
            output[4].Should().Be(5);
        }

        /// <summary>
        /// Verifies that encoding null input throws ArgumentNullException.
        /// </summary>
        [TestMethod]
        public void Encode_NullInput_ShouldThrowArgumentNullException()
        {
            Action act = () => HorizonCompressor.Encode((short[])null!);
            act.Should().Throw<ArgumentNullException>();
        }

        [TestMethod]
        public void Encode_ShortArray_DeltaOutOfRange_ShouldClampAndSucceed()
        {
            var input = new short[] { -20000, 20000 };
            var encoded = HorizonCompressor.Encode(input);
            var decoded = HorizonCompressor.Decode(encoded);

            // -20000 to 20000 is 40000 delta.
            // Clamped to 16383.
            // Decoded[0] = -20000
            // Decoded[1] = -20000 + 16383 = -3617
            decoded[0].Should().Be(-20000);
            decoded[1].Should().Be(-3617);
        }

        [TestMethod]
        public void Encode_ShortSpan_DeltaOutOfRange_ShouldClampAndSucceed()
        {
            short[] input = new short[] { -20000, 20000 };
            var output = new byte[8];
            int written = HorizonCompressor.Encode(input, output);
            
            var decoded = new short[2];
            HorizonCompressor.Decode(output.AsSpan(0, written), decoded);
            
            decoded[0].Should().Be(-20000);
            decoded[1].Should().Be(-3617);
        }

        [TestMethod]
        public void EncodeFloat_Array_ShouldRoundTripToQuantizedShorts()
        {
            var input = new float[1440];
            input[0] = -100f;   // clamps to -50
            input[1] = -50f;
            input[2] = -30f;
            input[3] = -10f;
            input[4] = 0f;
            input[5] = 10f;
            input[6] = 30f;
            input[7] = 50f;
            input[8] = 50f;
            for (int i = 9; i < input.Length; i++)
            {
                // Smoothly vary to avoid large deltas
                input[i] = 50f - ((i - 8) % 200) * 0.1f;
            }

            var encoded = HorizonCompressor.Encode(input);
            var decoded = HorizonCompressor.Decode(encoded);

            var expected = new short[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                expected[i] = Quantize(input[i]);
            }

            decoded.Should().Equal(expected);
        }

        [TestMethod]
        public void EncodeFloat_Span_ShouldMatchArrayOverload()
        {
            var input = new float[1440];
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = (i % 97) - 48.5f;
            }

            var byArray = HorizonCompressor.Encode(input);
            Span<byte> bySpan = new byte[input.Length * 2];
            int written = HorizonCompressor.Encode(input, bySpan);

            bySpan.Slice(0, written).ToArray().Should().Equal(byArray);
        }

        [DataTestMethod]
        [DataRow(1439)]
        [DataRow(1441)]
        public void EncodeFloat_Array_InvalidLength_ShouldThrow(int length)
        {
            var input = new float[length];
            Action act = () => HorizonCompressor.Encode(input);
            act.Should().Throw<ArgumentException>();
        }

        [DataTestMethod]
        [DataRow(1439)]
        [DataRow(1441)]
        public void EncodeFloat_Span_InvalidLength_ShouldThrow(int length)
        {
            var input = new float[length];
            var output = new byte[2880];
            Action act = () => HorizonCompressor.Encode(input, output);
            act.Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void DecodeToFloat_Array_ShouldMatchQuantizedRoundTrip()
        {
            var input = new float[1440];
            for (int i = 0; i < input.Length; i++)
            {
                // Smooth ramp from -50 to +50
                input[i] = -50f + (i * 100f / 1439f);
            }

            var encoded = HorizonCompressor.Encode(input);
            var decoded = HorizonCompressor.DecodeToFloat(encoded);

            decoded.Length.Should().Be(1440);
            for (int i = 0; i < decoded.Length; i++)
            {
                float expected = Dequantize(Quantize(input[i]));
                decoded[i].Should().BeApproximately(expected, 1e-6f);
            }
        }

        [TestMethod]
        public void DecodeToFloat_Span_ShouldMatchArrayOverload()
        {
            var input = new float[1440];
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = -49f + (i % 98);
            }

            var encoded = HorizonCompressor.Encode(input);
            var byArray = HorizonCompressor.DecodeToFloat(encoded);

            Span<float> bySpan = new float[1440];
            int written = HorizonCompressor.Decode(encoded, bySpan);
            written.Should().Be(1440);
            bySpan.ToArray().Should().Equal(byArray);
        }

        [TestMethod]
        public void DecodeToFloat_Array_NonHorizonLength_ShouldThrow()
        {
            var encoded = HorizonCompressor.Encode(new short[] { 1, 2, 3, 4 });
            Action act = () => HorizonCompressor.DecodeToFloat(encoded);
            act.Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void DecodeToFloat_Span_OutputTooSmall_ShouldThrow()
        {
            var input = new float[1440];
            var encoded = HorizonCompressor.Encode(input);
            var output = new float[1000];
            Action act = () => HorizonCompressor.Decode(encoded, output);
            act.Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void EncodeFloat_Array_DeltaOutOfRange_ShouldClampAndSucceed()
        {
            var input = new float[1440];
            input[0] = -50f;
            input[1] = 50f;
            
            var encoded = HorizonCompressor.Encode(input);
            var decoded = HorizonCompressor.DecodeToFloat(encoded);

            // -50 to 50 is 100 deg jump.
            // Clamped to 16383 units (~25 deg).
            float expectedClamped = -50f + (16383 * (50f / short.MaxValue));
            decoded[1].Should().BeApproximately(expectedClamped, 0.01f);
        }

        [TestMethod]
        public void EncodeFloat_Span_DeltaOutOfRange_ShouldClampAndSucceed()
        {
            var input = new float[1440];
            input[0] = -50f;
            input[1] = 50f;
            var output = new byte[2880];
            int written = HorizonCompressor.Encode(input, output);

            var decoded = new float[1440];
            HorizonCompressor.Decode(output.AsSpan(0, written), decoded);

            float expectedClamped = -50f + (16383 * (50f / short.MaxValue));
            decoded[1].Should().BeApproximately(expectedClamped, 0.01f);
        }

        private static void RoundTrip(short[] input)
        {
            Span<byte> encoded = new byte[64];
            Span<short> decoded = new short[input.Length];
            int written = HorizonCompressor.Encode(input, encoded);
            int read = HorizonCompressor.Decode(encoded.Slice(0, written), decoded);
            decoded.Slice(0, read).ToArray().Should().Equal(input);
        }

        private static short Quantize(float elevationDeg)
        {
            float clamped = Math.Clamp(elevationDeg, -50f, 50f);
            int quantized = (int)MathF.Round(clamped * (short.MaxValue / 50f), MidpointRounding.AwayFromZero);
            quantized = Math.Clamp(quantized, -short.MaxValue, short.MaxValue);
            return (short)quantized;
        }

        private static float Dequantize(short value)
        {
            return value * (50f / short.MaxValue);
        }
    }
}
