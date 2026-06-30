using OSGeo.GDAL;
using System.Runtime.InteropServices;

namespace moonlib.pipeline.streaming
{
    internal sealed class NativeReduceGeoTiffWriter : IDisposable
    {
        private const int BlockSize = 128;
        private readonly object _writeGate = new();
        private readonly string _outputPath;
        private readonly string _temporaryPath;
        private readonly int _width;
        private readonly int _height;
        private readonly int _bandCount;
        private readonly int _selectedBandIndex;
        private Dataset? _dataset;
        private int _tilesWritten;
        private double? _valueMin;
        private double? _valueMax;
        private bool _committed;

        public NativeReduceGeoTiffWriter(
            string scenarioRootDir,
            string demPath,
            NativeReduceRasterOutputSpec output,
            int bandCount)
        {
            if (bandCount < 1)
                throw new ArgumentOutOfRangeException(nameof(bandCount));
            if (output.SelectedBandIndex < 1 || output.SelectedBandIndex > bandCount)
                throw new ArgumentOutOfRangeException(nameof(output), "Selected band index must be within the reducer band count.");

            string scenarioRoot = Path.GetFullPath(scenarioRootDir);
            _outputPath = Path.GetFullPath(output.OutputPath);
            if (string.Equals(_outputPath, Path.GetFullPath(demPath), StringComparison.Ordinal))
                throw new InvalidOperationException("NativeReduce raster output must not replace its DEM input.");
            string relativeOutput = Path.GetRelativePath(scenarioRoot, _outputPath);
            if (Path.IsPathRooted(relativeOutput) || relativeOutput == ".." || relativeOutput.StartsWith($"..{Path.DirectorySeparatorChar}", StringComparison.Ordinal))
                throw new InvalidOperationException("NativeReduce raster output must remain within the scenario root.");

            string? outputDirectory = Path.GetDirectoryName(_outputPath);
            if (string.IsNullOrWhiteSpace(outputDirectory))
                throw new InvalidOperationException("NativeReduce raster output has no parent directory.");
            Directory.CreateDirectory(outputDirectory);
            _temporaryPath = Path.Combine(
                outputDirectory,
                $".{Path.GetFileName(_outputPath)}.{Guid.NewGuid():N}.tmp");

            using var dem = Gdal.Open(Path.GetFullPath(demPath), Access.GA_ReadOnly)
                ?? throw new InvalidOperationException($"Failed to open DEM '{demPath}'.");
            _width = dem.RasterXSize;
            _height = dem.RasterYSize;
            _bandCount = bandCount;
            _selectedBandIndex = output.SelectedBandIndex;

            var geoTransform = new double[6];
            dem.GetGeoTransform(geoTransform);
            string projection = dem.GetProjectionRef() ?? string.Empty;

            string compression = output.Compression.Trim().ToUpperInvariant();
            var creationOptions = new List<string>
            {
                "TILED=YES",
                $"BLOCKXSIZE={BlockSize}",
                $"BLOCKYSIZE={BlockSize}",
                $"COMPRESS={compression}",
                "PREDICTOR=3",
                "BIGTIFF=IF_SAFER"
            };
            if (compression == "DEFLATE")
                creationOptions.Add("ZLEVEL=6");
            else if (compression == "ZSTD")
                creationOptions.Add("ZSTD_LEVEL=9");

            var driver = Gdal.GetDriverByName("GTiff")
                ?? throw new InvalidOperationException("GDAL GTiff driver is unavailable.");
            _dataset = driver.Create(
                _temporaryPath,
                _width,
                _height,
                _bandCount,
                DataType.GDT_Float32,
                creationOptions.ToArray())
                ?? throw new InvalidOperationException($"Failed to create temporary GeoTIFF '{_temporaryPath}'.");
            _dataset.SetProjection(projection);
            _dataset.SetGeoTransform(geoTransform);
            for (int bandIndex = 1; bandIndex <= _bandCount; bandIndex++)
            {
                using var band = _dataset.GetRasterBand(bandIndex);
                band.SetNoDataValue(output.NoDataValue);
                band.Fill(output.NoDataValue, 0.0);
            }
        }

        public void WriteTile(int xOffset, int yOffset, int tileWidth, int tileHeight, float[] values)
        {
            if (values.Length != checked(_bandCount * tileWidth * tileHeight))
                throw new ArgumentException("NativeReduce tile buffer length does not match its dimensions.", nameof(values));

            int writeWidth = Math.Min(tileWidth, _width - xOffset);
            int writeHeight = Math.Min(tileHeight, _height - yOffset);
            if (xOffset < 0 || yOffset < 0 || writeWidth <= 0 || writeHeight <= 0)
                throw new ArgumentOutOfRangeException(nameof(xOffset), "NativeReduce tile lies outside the output raster.");

            lock (_writeGate)
            {
                var dataset = _dataset ?? throw new ObjectDisposedException(nameof(NativeReduceGeoTiffWriter));
                int sourceBandSize = checked(tileWidth * tileHeight);
                int writeBandSize = checked(writeWidth * writeHeight);
                for (int bandIndex = 0; bandIndex < _bandCount; bandIndex++)
                {
                    float[] bandValues;
                    int sourceBase = bandIndex * sourceBandSize;
                    if (writeWidth == tileWidth && writeHeight == tileHeight)
                    {
                        bandValues = new float[sourceBandSize];
                        Array.Copy(values, sourceBase, bandValues, 0, sourceBandSize);
                    }
                    else
                    {
                        bandValues = new float[writeBandSize];
                        for (int row = 0; row < writeHeight; row++)
                            Array.Copy(values, sourceBase + row * tileWidth, bandValues, row * writeWidth, writeWidth);
                    }

                    using var band = dataset.GetRasterBand(bandIndex + 1);
                    var handle = GCHandle.Alloc(bandValues, GCHandleType.Pinned);
                    try
                    {
                        var error = band.WriteRaster(
                            xOffset,
                            yOffset,
                            writeWidth,
                            writeHeight,
                            handle.AddrOfPinnedObject(),
                            writeWidth,
                            writeHeight,
                            DataType.GDT_Float32,
                            0,
                            0);
                        if (error != CPLErr.CE_None)
                            throw new InvalidOperationException($"GDAL failed writing NativeReduce band {bandIndex + 1} tile at ({xOffset}, {yOffset}).");
                    }
                    finally
                    {
                        handle.Free();
                    }

                    if (bandIndex + 1 == _selectedBandIndex)
                    {
                        foreach (float value in bandValues)
                        {
                            _valueMin = !_valueMin.HasValue ? value : Math.Min(_valueMin.Value, value);
                            _valueMax = !_valueMax.HasValue ? value : Math.Max(_valueMax.Value, value);
                        }
                    }
                }
                _tilesWritten++;
            }
        }

        public NativeReduceRasterResult Commit()
        {
            lock (_writeGate)
            {
                if (_committed)
                    throw new InvalidOperationException("NativeReduce GeoTIFF has already been committed.");
                var dataset = _dataset ?? throw new ObjectDisposedException(nameof(NativeReduceGeoTiffWriter));
                dataset.FlushCache();
                dataset.Dispose();
                _dataset = null;
                File.Move(_temporaryPath, _outputPath, overwrite: true);
                _committed = true;
                return new NativeReduceRasterResult(
                    _outputPath,
                    _tilesWritten,
                    _valueMin,
                    _valueMax,
                    new FileInfo(_outputPath).Length);
            }
        }

        public void Dispose()
        {
            lock (_writeGate)
            {
                _dataset?.Dispose();
                _dataset = null;
                if (!_committed && File.Exists(_temporaryPath))
                {
                    try { File.Delete(_temporaryPath); }
                    catch { /* best-effort scratch cleanup */ }
                }
            }
        }
    }
}
