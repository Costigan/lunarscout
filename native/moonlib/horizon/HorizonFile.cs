using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Diagnostics;
using System.IO;

namespace moonlib.horizon
{
    public static class HorizonFile
    {
        // Horizon files must be named "horizon_*.bin" or "horizon_*.cbin".
        private const int HorizonSamples = 1440;
        private const int HorizonRows = 128;
        private const int HorizonCols = 128;
        private const int HorizonCount = HorizonRows * HorizonCols;
        private const int TotalSamples = HorizonCount * HorizonSamples;
        private const int MaxCompressedBytes = 2 * HorizonSamples;
        private const int LengthPrefixBytes = sizeof(ushort);
        private const long LegacyRawFileBytes = (long)TotalSamples * sizeof(float);
        private const string RequiredFileNamePrefix = "horizon_";

        public static float[] ReadHorizonFile(string path)
        {
            ArgumentNullException.ThrowIfNull(path);
            ValidateHorizonFileName(path);
            var extension = Path.GetExtension(path);
            if (string.IsNullOrWhiteSpace(extension))
                throw new ArgumentException("File path must have an extension to determine format.", nameof(path));
            else if (extension.Equals(".bin", StringComparison.OrdinalIgnoreCase))
                return ReadUncompressedHorizonFile(path);
            else if (extension.Equals(".cbin", StringComparison.OrdinalIgnoreCase))
                return ReadCompressedHorizonFile(path);
            else
                throw new NotSupportedException($"Unsupported horizon file extension: {extension}");
        }

        public static void WriteHorizonFile(string path, float[] data)
        {
            ArgumentNullException.ThrowIfNull(path);
            ArgumentNullException.ThrowIfNull(data);
            ValidateHorizonFileName(path);
            var extension = Path.GetExtension(path);
            if (string.IsNullOrWhiteSpace(extension))
                throw new ArgumentException("File path must have an extension to determine format.", nameof(path));
            else if (extension.Equals(".bin", StringComparison.OrdinalIgnoreCase))
                WriteUncompressedHorizonFile(path, data);
            else if (extension.Equals(".cbin", StringComparison.OrdinalIgnoreCase))
                WriteCompressedHorizonFile(path, data);
            else
                throw new NotSupportedException($"Unsupported horizon file extension: {extension}");
        }

        public static void WriteHorizonFile(string path, ReadOnlySpan<float> data)
        {
            ArgumentNullException.ThrowIfNull(path);
            ValidateHorizonFileName(path);
            var extension = Path.GetExtension(path);
            if (string.IsNullOrWhiteSpace(extension))
                throw new ArgumentException("File path must have an extension to determine format.", nameof(path));
            else if (extension.Equals(".bin", StringComparison.OrdinalIgnoreCase))
                WriteUncompressedHorizonFile(path, data);
            else if (extension.Equals(".cbin", StringComparison.OrdinalIgnoreCase))
                WriteCompressedHorizonFile(path, data);
            else
                throw new NotSupportedException($"Unsupported horizon file extension: {extension}");
        }

        public static float[] ReadUncompressedHorizonFile(string path)
        {
            ArgumentNullException.ThrowIfNull(path);
            ValidateHorizonFileName(path);
            var fi = new FileInfo(path);
            if (fi.Length == LegacyRawFileBytes) 
                return Utilities.LoadBinaryArray<float>(path);
            else if (fi.Length == LegacyRawFileBytes+ 7 * sizeof(float))
            {
                var result = new float[TotalSamples];
                using var fs = File.OpenRead(path);
                using var br = new BinaryReader(fs);
                for (int i = 0; i < TotalSamples; i++)
                    result[i] = br.ReadSingle();
                return result;
            }
            else
                throw new InvalidDataException($"Unexpected uncompressed horizon file size: {fi.Length} bytes; expected {LegacyRawFileBytes} or {LegacyRawFileBytes + 7 * sizeof(float)} bytes.");
        }

        public static float[] ReadCompressedHorizonFile(string path)
        {
            Debug.Assert(path != null && Path.GetExtension(path)?.Equals(".cbin", StringComparison.OrdinalIgnoreCase) == true,
                "ReadCompressedHorizonFile should only be called for .cbin files.");
            ValidateHorizonFileName(path);
            var fi = new FileInfo(path);
            if (fi.Length <= 0)
                throw new InvalidDataException($"Unexpected compressed horizon size: {fi.Length} bytes.");

            var result = new float[TotalSamples];
            using var fs = File.OpenRead(path);
            Span<byte> lengthBuf = stackalloc byte[LengthPrefixBytes];
            Span<byte> encoded = stackalloc byte[MaxCompressedBytes];

            for (int horizonIdx = 0; horizonIdx < HorizonCount; horizonIdx++)
            {
                try
                {
                    fs.ReadExactly(lengthBuf);
                }
                catch (EndOfStreamException ex)
                {
                    throw new InvalidDataException(
                        $"Compressed horizon file ended while reading block length at horizon index {horizonIdx}.", ex);
                }

                ushort encodedLen = BinaryPrimitives.ReadUInt16LittleEndian(lengthBuf);
                if (encodedLen == 0 || encodedLen > MaxCompressedBytes)
                {
                    throw new InvalidDataException(
                        $"Invalid compressed horizon block length: {encodedLen}.");
                }

                try
                {
                    fs.ReadExactly(encoded.Slice(0, encodedLen));
                }
                catch (EndOfStreamException ex)
                {
                    throw new InvalidDataException(
                        $"Compressed horizon file ended while reading encoded data for horizon index {horizonIdx}.", ex);
                }

                int dstOffset = horizonIdx * HorizonSamples;
                int written = HorizonCompressor.Decode(
                    encoded.Slice(0, encodedLen),
                    result.AsSpan(dstOffset, HorizonSamples));
                if (written != HorizonSamples)
                {
                    throw new InvalidDataException($"Decoded {written} samples; expected {HorizonSamples}.");
                }
            }

            if (fs.Position != fs.Length)
            {
                throw new InvalidDataException("Compressed horizon file contains trailing bytes.");
            }

            return result;
        }

        public static void WriteUncompressedHorizonFile(string path, float[] data)
        {
            ArgumentNullException.ThrowIfNull(data);
            ValidateHorizonFileName(path);
            if (path == null || Path.GetExtension(path)?.Equals(".bin", StringComparison.OrdinalIgnoreCase) != true)
                throw new ArgumentException("WriteUncompressedHorizonFile should only be called for .bin files.", nameof(path));
            if (data.Length != TotalSamples)
            {
                throw new ArgumentException(
                    $"Expected {TotalSamples} float samples ({HorizonRows}x{HorizonCols} horizons of {HorizonSamples}).",
                    nameof(data));
            }
            Utilities.WriteBinaryArray(path, data);
        }

        public static void WriteUncompressedHorizonFile(string path, ReadOnlySpan<float> data)
        {
            ArgumentNullException.ThrowIfNull(path);
            ValidateHorizonFileName(path);
            if (Path.GetExtension(path)?.Equals(".bin", StringComparison.OrdinalIgnoreCase) != true)
                throw new ArgumentException("WriteUncompressedHorizonFile should only be called for .bin files.", nameof(path));
            if (data.Length != TotalSamples)
            {
                throw new ArgumentException(
                    $"Expected {TotalSamples} float samples ({HorizonRows}x{HorizonCols} horizons of {HorizonSamples}).",
                    nameof(data));
            }

            using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
            using var bw = new BinaryWriter(fs);
            for (int i = 0; i < data.Length; i++)
                bw.Write(data[i]);
        }

        public static void WriteCompressedHorizonFile(string path, float[] data)
        {
            ArgumentNullException.ThrowIfNull(data);
            ValidateHorizonFileName(path);
            WriteCompressedHorizonFile(path, data.AsSpan());
        }

        public static void WriteCompressedHorizonFile(string path, ReadOnlySpan<float> data)
        {
            ArgumentNullException.ThrowIfNull(path);
            ValidateHorizonFileName(path);
            if (Path.GetExtension(path)?.Equals(".cbin", StringComparison.OrdinalIgnoreCase) != true)
                throw new ArgumentException("WriteCompressedHorizonFile should only be called for .cbin files.", nameof(path));
            if (data.Length != TotalSamples)
            {
                throw new ArgumentException(
                    $"Expected {TotalSamples} float samples ({HorizonRows}x{HorizonCols} horizons of {HorizonSamples}).",
                    nameof(data));
            }

            int maxFileBytes = checked(HorizonCount * (LengthPrefixBytes + MaxCompressedBytes));
            byte[] buffer = ArrayPool<byte>.Shared.Rent(maxFileBytes);

            try
            {
                int offset = 0;
                for (int horizonIdx = 0; horizonIdx < HorizonCount; horizonIdx++)
                {
                    int srcOffset = horizonIdx * HorizonSamples;
                    int lengthOffset = offset;
                    offset += LengthPrefixBytes;

                    int written = HorizonCompressor.Encode(
                        data.Slice(srcOffset, HorizonSamples),
                        buffer.AsSpan(offset, MaxCompressedBytes));

                    BinaryPrimitives.WriteUInt16LittleEndian(
                        buffer.AsSpan(lengthOffset, LengthPrefixBytes),
                        (ushort)written);
                    offset += written;
                }

                using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
                fs.Write(buffer.AsSpan(0, offset));
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(buffer);
            }
        }

        public static int CompressDirectory(
            string directoryPath,
            bool deleteUncompressed = true,
            bool verbose = false)
        {
            ArgumentNullException.ThrowIfNull(directoryPath);
            if (!Directory.Exists(directoryPath))
                throw new DirectoryNotFoundException($"Directory not found: {directoryPath}");

            int converted = 0;
            var filenames = Directory.GetFiles(directoryPath, $"{RequiredFileNamePrefix}*.bin", SearchOption.TopDirectoryOnly);
            foreach (var path in filenames)
            {
                if (Path.GetExtension(path)?.Equals(".bin", StringComparison.OrdinalIgnoreCase) != true)
                    continue;
                string compressedPath = Path.ChangeExtension(path, ".cbin");
                try
                {
                    var data = ReadUncompressedHorizonFile(path);
                    WriteCompressedHorizonFile(compressedPath, data);

                    if (deleteUncompressed)
                        File.Delete(path);

                    converted++;

                    if (verbose)
                        Console.WriteLine($"Compressed ({converted:D3}/{filenames.Length:D3}) {compressedPath}");
                }
                catch (Exception ex)
                {
                    if (File.Exists(compressedPath))
                    {
                        try
                        {
                            File.Delete(compressedPath);
                        }
                        catch (Exception deleteEx)
                        {
                            Console.Error.WriteLine($"Error deleting incomplete compressed file '{compressedPath}': {deleteEx.Message}");
                        }
                    }
                    Console.Error.WriteLine($"Error converting '{path}': {ex.Message}");
                }
            }

            if (verbose)
            {
                Console.WriteLine(
                    $"Converted {converted} uncompressed horizon file(s) in '{directoryPath}' (deleteUncompressed={deleteUncompressed}).");
            }

            return converted;
        }

        private static void ValidateHorizonFileName(string path)
        {
            ArgumentNullException.ThrowIfNull(path);
            string fileName = Path.GetFileName(path);
            if (!fileName.StartsWith(RequiredFileNamePrefix, StringComparison.OrdinalIgnoreCase))
            {
                throw new ArgumentException(
                    $"Horizon file name must start with \"{RequiredFileNamePrefix}\".",
                    nameof(path));
            }
        }
    }
}
