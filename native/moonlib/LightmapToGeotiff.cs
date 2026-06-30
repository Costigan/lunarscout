using HDF.PInvoke;
using OSGeo.GDAL;
using Serilog;
using System.Runtime.InteropServices;
using System.Text;

namespace moonlib
{
    public sealed class LightmapToGeotiff
    {
        private const string DatasetName = "light_curves";
        private const int DefaultTileSize = 128;

        public static void Convert(
            string hdf5Path,
            string geotiffPath,
            int tileSize = DefaultTileSize,
            bool buildPyramids = true,
            string overviewResampling = "NEAREST")
        {
            if (string.IsNullOrWhiteSpace(hdf5Path))
                throw new ArgumentException("Input HDF5 path must be non-empty.", nameof(hdf5Path));
            if (string.IsNullOrWhiteSpace(geotiffPath))
                throw new ArgumentException("Output GeoTIFF path must be non-empty.", nameof(geotiffPath));
            if (tileSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(tileSize), "Tile size must be positive.");

            string inputPath = Path.GetFullPath(hdf5Path);
            string outputPath = Path.GetFullPath(geotiffPath);
            if (!File.Exists(inputPath))
                throw new FileNotFoundException("Input HDF5 file does not exist.", inputPath);

            string? outputDirectory = Path.GetDirectoryName(outputPath);
            if (string.IsNullOrWhiteSpace(outputDirectory))
                throw new InvalidOperationException("GeoTIFF output path has no parent directory.");
            Directory.CreateDirectory(outputDirectory);

            string temporaryPath = Path.Combine(
                outputDirectory,
                $".{Path.GetFileName(outputPath)}.{Guid.NewGuid():N}.tmp.tif");

            long file = -1;
            long dataset = -1;
            long fileSpace = -1;
            Dataset? geotiff = null;
            try
            {
                file = H5F.open(inputPath, H5F.ACC_RDONLY);
                CheckId(file, "H5F.open");
                dataset = H5D.open(file, DatasetName);
                CheckId(dataset, "H5D.open");

                fileSpace = H5D.get_space(dataset);
                CheckId(fileSpace, "H5D.get_space");
                ulong[] dims = GetDatasetDimensions(fileSpace);
                int height = CheckedInt(dims[0], "height");
                int width = CheckedInt(dims[1], "width");
                int bandCount = CheckedInt(dims[2], "time/band count");

                long hdf5Type = H5D.get_type(dataset);
                CheckId(hdf5Type, "H5D.get_type");
                try
                {
                    DataType gdalType = GetGdalDataType(hdf5Type);
                    string compression = ChooseCompression(gdalType);
                    string predictor = ChoosePredictor(gdalType);
                    string projection = ReadStringAttribute(dataset, "projection_wkt") ?? string.Empty;
                    double[] geoTransform = ReadDoubleArrayAttribute(dataset, "geo_transform", expectedLength: 6);
                    string[] timestamps = ReadStringArrayAttribute(dataset, "timestamps_utc") ?? Array.Empty<string>();

                    geotiff = CreateGeoTiff(
                        temporaryPath,
                        width,
                        height,
                        bandCount,
                        gdalType,
                        projection,
                        geoTransform,
                        tileSize,
                        compression,
                        predictor);

                    WriteBandMetadata(geotiff, timestamps);
                    CopyTiles(dataset, geotiff, hdf5Type, gdalType, width, height, bandCount, tileSize);

                    geotiff.FlushCache();
                    if (buildPyramids)
                    {
                        int[] levels = BuildOverviewLevels(width, height);
                        if (levels.Length > 0)
                        {
                            Log.Information(
                                "Building GeoTIFF overviews for {Path}: {Levels}",
                                temporaryPath,
                                string.Join(",", levels));
                            geotiff.BuildOverviews(overviewResampling, levels, null, null);
                        }
                    }
                    geotiff.FlushCache();
                    geotiff.Dispose();
                    geotiff = null;

                    File.Move(temporaryPath, outputPath, overwrite: true);
                }
                finally
                {
                    H5T.close(hdf5Type);
                }
            }
            finally
            {
                geotiff?.Dispose();
                if (fileSpace >= 0)
                    H5S.close(fileSpace);
                if (dataset >= 0)
                    H5D.close(dataset);
                if (file >= 0)
                    H5F.close(file);
                if (File.Exists(temporaryPath))
                {
                    try { File.Delete(temporaryPath); }
                    catch { /* best-effort scratch cleanup */ }
                }
            }
        }

        private static Dataset CreateGeoTiff(
            string path,
            int width,
            int height,
            int bands,
            DataType dataType,
            string projection,
            double[] geoTransform,
            int tileSize,
            string compression,
            string? predictor)
        {
            var options = new List<string>
            {
                "TILED=YES",
                $"BLOCKXSIZE={tileSize}",
                $"BLOCKYSIZE={tileSize}",
                $"COMPRESS={compression}",
                "INTERLEAVE=BAND",
                "BIGTIFF=YES"
            };
            if (!string.IsNullOrWhiteSpace(predictor))
                options.Add($"PREDICTOR={predictor}");
            if (compression == "DEFLATE")
                options.Add("ZLEVEL=6");
            else if (compression == "ZSTD")
                options.Add("ZSTD_LEVEL=9");

            var driver = Gdal.GetDriverByName("GTiff")
                ?? throw new InvalidOperationException("GDAL GTiff driver is unavailable.");
            Dataset dataset = driver.Create(path, width, height, bands, dataType, options.ToArray())
                ?? throw new InvalidOperationException($"Failed to create GeoTIFF '{path}'.");
            dataset.SetProjection(projection);
            dataset.SetGeoTransform(geoTransform);
            return dataset;
        }

        private static void CopyTiles(
            long hdf5Dataset,
            Dataset geotiff,
            long hdf5Type,
            DataType gdalType,
            int width,
            int height,
            int bandCount,
            int tileSize)
        {
            int bytesPerSample = DataTypeSizeBytes(gdalType);
            for (int y = 0; y < height; y += tileSize)
            {
                int tileHeight = Math.Min(tileSize, height - y);
                for (int x = 0; x < width; x += tileSize)
                {
                    int tileWidth = Math.Min(tileSize, width - x);
                    byte[] tile = ReadHdf5Tile(
                        hdf5Dataset,
                        hdf5Type,
                        y,
                        x,
                        tileHeight,
                        tileWidth,
                        bandCount,
                        bytesPerSample);

                    WriteTileToBands(geotiff, gdalType, tile, x, y, tileWidth, tileHeight, bandCount, bytesPerSample);
                }
                Log.Information(
                    "Converted HDF5 lightmap rows {StartRow}-{EndRow} of {Height}",
                    y,
                    y + tileHeight - 1,
                    height);
            }
        }

        private static byte[] ReadHdf5Tile(
            long dataset,
            long hdf5Type,
            int y,
            int x,
            int tileHeight,
            int tileWidth,
            int bandCount,
            int bytesPerSample)
        {
            long fileSpace = -1;
            long memorySpace = -1;
            GCHandle handle = default;
            byte[] buffer = new byte[checked(tileHeight * tileWidth * bandCount * bytesPerSample)];
            try
            {
                fileSpace = H5D.get_space(dataset);
                CheckId(fileSpace, "H5D.get_space(tile)");
                ulong[] start = { (ulong)y, (ulong)x, 0 };
                ulong[] count = { (ulong)tileHeight, (ulong)tileWidth, (ulong)bandCount };
                CheckNonNegative(
                    H5S.select_hyperslab(fileSpace, H5S.seloper_t.SET, start, null, count, null),
                    "H5S.select_hyperslab(file)");

                ulong[] dims = { (ulong)tileHeight, (ulong)tileWidth, (ulong)bandCount };
                memorySpace = H5S.create_simple(3, dims, null);
                CheckId(memorySpace, "H5S.create_simple(memory)");

                handle = GCHandle.Alloc(buffer, GCHandleType.Pinned);
                CheckNonNegative(
                    H5D.read(dataset, hdf5Type, memorySpace, fileSpace, H5P.DEFAULT, handle.AddrOfPinnedObject()),
                    "H5D.read");
                return buffer;
            }
            finally
            {
                if (handle.IsAllocated)
                    handle.Free();
                if (memorySpace >= 0)
                    H5S.close(memorySpace);
                if (fileSpace >= 0)
                    H5S.close(fileSpace);
            }
        }

        private static void WriteTileToBands(
            Dataset geotiff,
            DataType gdalType,
            byte[] interleavedTile,
            int x,
            int y,
            int tileWidth,
            int tileHeight,
            int bandCount,
            int bytesPerSample)
        {
            int pixelCount = checked(tileWidth * tileHeight);
            byte[] bandBuffer = new byte[checked(pixelCount * bytesPerSample)];
            for (int bandIndex = 0; bandIndex < bandCount; bandIndex++)
            {
                for (int pixelIndex = 0; pixelIndex < pixelCount; pixelIndex++)
                {
                    int sourceOffset = checked((pixelIndex * bandCount + bandIndex) * bytesPerSample);
                    int destinationOffset = pixelIndex * bytesPerSample;
                    Buffer.BlockCopy(interleavedTile, sourceOffset, bandBuffer, destinationOffset, bytesPerSample);
                }

                Band band = geotiff.GetRasterBand(bandIndex + 1);
                GCHandle handle = GCHandle.Alloc(bandBuffer, GCHandleType.Pinned);
                try
                {
                    CPLErr result = band.WriteRaster(
                        x,
                        y,
                        tileWidth,
                        tileHeight,
                        handle.AddrOfPinnedObject(),
                        tileWidth,
                        tileHeight,
                        gdalType,
                        0,
                        0);
                    if (result != CPLErr.CE_None)
                        throw new InvalidOperationException($"GDAL WriteRaster failed for band {bandIndex + 1} with {result}.");
                }
                finally
                {
                    handle.Free();
                }
            }
        }

        private static void WriteBandMetadata(Dataset geotiff, IReadOnlyList<string> timestamps)
        {
            int count = Math.Min(geotiff.RasterCount, timestamps.Count);
            for (int index = 0; index < count; index++)
            {
                Band band = geotiff.GetRasterBand(index + 1);
                band.SetDescription(timestamps[index]);
                band.SetMetadataItem("TIMESTAMP_UTC", timestamps[index], null);
            }
        }

        private static int[] BuildOverviewLevels(int width, int height)
        {
            var levels = new List<int>();
            int factor = 2;
            int minDimension = Math.Min(width, height);
            while (minDimension / factor >= 128)
            {
                levels.Add(factor);
                factor *= 2;
            }
            return levels.ToArray();
        }

        private static ulong[] GetDatasetDimensions(long fileSpace)
        {
            int rank = H5S.get_simple_extent_ndims(fileSpace);
            if (rank != 3)
                throw new InvalidOperationException($"Expected HDF5 dataset rank 3, got {rank}.");
            ulong[] dims = new ulong[rank];
            CheckNonNegative(
                H5S.get_simple_extent_dims(fileSpace, dims, null),
                "H5S.get_simple_extent_dims");
            return dims;
        }

        private static DataType GetGdalDataType(long hdf5Type)
        {
            if (H5T.equal(hdf5Type, H5T.STD_U8LE) > 0 || H5T.equal(hdf5Type, H5T.NATIVE_UINT8) > 0)
                return DataType.GDT_Byte;
            if (H5T.equal(hdf5Type, H5T.STD_I16LE) > 0 || H5T.equal(hdf5Type, H5T.NATIVE_INT16) > 0)
                return DataType.GDT_Int16;
            if (H5T.equal(hdf5Type, H5T.STD_U16LE) > 0 || H5T.equal(hdf5Type, H5T.NATIVE_UINT16) > 0)
                return DataType.GDT_UInt16;
            if (H5T.equal(hdf5Type, H5T.STD_I32LE) > 0 || H5T.equal(hdf5Type, H5T.NATIVE_INT32) > 0)
                return DataType.GDT_Int32;
            if (H5T.equal(hdf5Type, H5T.STD_U32LE) > 0 || H5T.equal(hdf5Type, H5T.NATIVE_UINT32) > 0)
                return DataType.GDT_UInt32;
            if (H5T.equal(hdf5Type, H5T.IEEE_F32LE) > 0 || H5T.equal(hdf5Type, H5T.NATIVE_FLOAT) > 0)
                return DataType.GDT_Float32;
            if (H5T.equal(hdf5Type, H5T.IEEE_F64LE) > 0 || H5T.equal(hdf5Type, H5T.NATIVE_DOUBLE) > 0)
                return DataType.GDT_Float64;

            throw new NotSupportedException("Unsupported HDF5 lightmap dataset datatype.");
        }

        private static int DataTypeSizeBytes(DataType dataType)
        {
            return dataType switch
            {
                DataType.GDT_Byte => 1,
                DataType.GDT_Int16 or DataType.GDT_UInt16 => 2,
                DataType.GDT_Int32 or DataType.GDT_UInt32 or DataType.GDT_Float32 => 4,
                DataType.GDT_Float64 => 8,
                _ => throw new NotSupportedException($"Unsupported GDAL datatype {dataType}.")
            };
        }

        private static string ChooseCompression(DataType dataType)
        {
            return dataType switch
            {
                DataType.GDT_Byte => "LZW",
                DataType.GDT_Int16 or DataType.GDT_UInt16 or DataType.GDT_Int32 or DataType.GDT_UInt32 => "DEFLATE",
                DataType.GDT_Float32 or DataType.GDT_Float64 => "DEFLATE",
                _ => "LZW"
            };
        }

        private static string? ChoosePredictor(DataType dataType)
        {
            return dataType switch
            {
                DataType.GDT_Int16 or DataType.GDT_UInt16 or DataType.GDT_Int32 or DataType.GDT_UInt32 => "2",
                DataType.GDT_Float32 or DataType.GDT_Float64 => "3",
                _ => null
            };
        }

        private static string? ReadStringAttribute(long target, string name)
        {
            long attr = H5A.open(target, name);
            if (attr < 0)
                return null;
            long type = -1;
            GCHandle handle = default;
            try
            {
                type = H5A.get_type(attr);
                CheckId(type, $"H5A.get_type({name})");
                int size = H5T.get_size(type).ToInt32();
                if (size <= 0)
                    return string.Empty;
                byte[] bytes = new byte[size];
                handle = GCHandle.Alloc(bytes, GCHandleType.Pinned);
                CheckNonNegative(H5A.read(attr, type, handle.AddrOfPinnedObject()), $"H5A.read({name})");
                return DecodeFixedString(bytes);
            }
            finally
            {
                if (handle.IsAllocated)
                    handle.Free();
                if (type >= 0)
                    H5T.close(type);
                H5A.close(attr);
            }
        }

        private static double[] ReadDoubleArrayAttribute(long target, string name, int expectedLength)
        {
            long attr = H5A.open(target, name);
            CheckId(attr, $"H5A.open({name})");
            long space = -1;
            GCHandle handle = default;
            try
            {
                space = H5A.get_space(attr);
                CheckId(space, $"H5A.get_space({name})");
                ulong[] dims = new ulong[1];
                CheckNonNegative(H5S.get_simple_extent_dims(space, dims, null), $"H5S.get_simple_extent_dims({name})");
                int count = CheckedInt(dims[0], name);
                if (count != expectedLength)
                    throw new InvalidOperationException($"Attribute '{name}' length is {count}; expected {expectedLength}.");
                double[] values = new double[count];
                handle = GCHandle.Alloc(values, GCHandleType.Pinned);
                CheckNonNegative(H5A.read(attr, H5T.NATIVE_DOUBLE, handle.AddrOfPinnedObject()), $"H5A.read({name})");
                return values;
            }
            finally
            {
                if (handle.IsAllocated)
                    handle.Free();
                if (space >= 0)
                    H5S.close(space);
                H5A.close(attr);
            }
        }

        private static string[]? ReadStringArrayAttribute(long target, string name)
        {
            long attr = H5A.open(target, name);
            if (attr < 0)
                return null;
            long type = -1;
            long space = -1;
            GCHandle handle = default;
            try
            {
                type = H5A.get_type(attr);
                CheckId(type, $"H5A.get_type({name})");
                int width = H5T.get_size(type).ToInt32();
                space = H5A.get_space(attr);
                CheckId(space, $"H5A.get_space({name})");
                ulong[] dims = new ulong[1];
                CheckNonNegative(H5S.get_simple_extent_dims(space, dims, null), $"H5S.get_simple_extent_dims({name})");
                int count = CheckedInt(dims[0], name);
                byte[] bytes = new byte[checked(count * width)];
                handle = GCHandle.Alloc(bytes, GCHandleType.Pinned);
                CheckNonNegative(H5A.read(attr, type, handle.AddrOfPinnedObject()), $"H5A.read({name})");

                string[] values = new string[count];
                for (int index = 0; index < count; index++)
                {
                    byte[] slice = new byte[width];
                    Buffer.BlockCopy(bytes, index * width, slice, 0, width);
                    values[index] = DecodeFixedString(slice);
                }
                return values;
            }
            finally
            {
                if (handle.IsAllocated)
                    handle.Free();
                if (space >= 0)
                    H5S.close(space);
                if (type >= 0)
                    H5T.close(type);
                H5A.close(attr);
            }
        }

        private static string DecodeFixedString(byte[] bytes)
        {
            int length = Array.IndexOf(bytes, (byte)0);
            if (length < 0)
                length = bytes.Length;
            while (length > 0 && bytes[length - 1] == 0)
                length--;
            return Encoding.UTF8.GetString(bytes, 0, length);
        }

        private static int CheckedInt(ulong value, string name)
        {
            if (value > int.MaxValue)
                throw new InvalidOperationException($"HDF5 {name} dimension {value} exceeds Int32.MaxValue.");
            return (int)value;
        }

        private static void CheckId(long id, string operation)
        {
            if (id < 0)
                throw new InvalidOperationException($"{operation} failed with id={id}.");
        }

        private static void CheckNonNegative(int status, string operation)
        {
            if (status < 0)
                throw new InvalidOperationException($"{operation} failed with status={status}.");
        }
    }
}
