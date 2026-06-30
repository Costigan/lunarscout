using OSGeo.GDAL;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace moonlib.horizon
{
    public static class TiledGeotiffWriter
    {
        public const int PatchSize = 128;

        private static readonly string[] _creationOptions =
            { "TILED=YES", "BLOCKXSIZE=128", "BLOCKYSIZE=128", "COMPRESS=LZW", "BIGTIFF=YES", "SPARSE_OK=TRUE" };

        // =================================================================
        // DataType mapping
        // =================================================================

        private static DataType GetGdalType<T>() where T : unmanaged =>
            typeof(T) switch
            {
                Type t when t == typeof(byte)   => DataType.GDT_Byte,
                Type t when t == typeof(float)  => DataType.GDT_Float32,
                Type t when t == typeof(double) => DataType.GDT_Float64,
                Type t when t == typeof(int)    => DataType.GDT_Int32,
                Type t when t == typeof(short)  => DataType.GDT_Int16,
                Type t when t == typeof(uint)   => DataType.GDT_UInt32,
                Type t when t == typeof(ushort) => DataType.GDT_UInt16,
                _ => throw new ArgumentException($"Unsupported pixel type '{typeof(T).Name}'.")
            };

        // =================================================================
        // Single‑dataset pattern
        // =================================================================

        /// <summary>
        /// Open an existing tiled GeoTIFF for update, or create a new one.
        /// Idempotent — if a file at <paramref name="path"/> already matches
        /// the requested size, band count, and pixel type it is reopened.
        /// </summary>
        /// <typeparam name="T">Pixel type (byte, float, double, …).</typeparam>
        /// <param name="bands">Number of raster bands (usually 1).</param>
        public static Dataset OpenTiled<T>(
            string path,
            int width,
            int height,
            int bands,
            double noDataValue,
            string projection,
            double[] geoTransform) where T : unmanaged
        {
            var dataType = GetGdalType<T>();

            if (System.IO.File.Exists(path))
            {
                Dataset? existing = null;
                try { existing = Gdal.Open(path, Access.GA_Update); }
                catch { existing = null; }

                if (existing is not null)
                {
                    bool match = existing.RasterXSize == width
                              && existing.RasterYSize == height
                              && existing.RasterCount == bands
                              && existing.GetRasterBand(1).DataType == dataType;
                    if (match)
                    {
                        for (int b = 1; b <= bands; b++)
                            existing.GetRasterBand(b).SetNoDataValue(noDataValue);
                        return existing;
                    }
                    existing.Dispose();
                }

                try { System.IO.File.Delete(path); }
                catch { /* best-effort */ }
            }

            System.IO.Directory.CreateDirectory(System.IO.Path.GetDirectoryName(path)!);

            var driver = Gdal.GetDriverByName("GTiff")
                         ?? throw new InvalidOperationException("GDAL GTiff driver is unavailable.");

            var ds = driver.Create(path, width, height, bands, dataType, _creationOptions);
            if (ds is null)
                throw new InvalidOperationException($"Failed to create dataset '{path}'.");

            for (int b = 1; b <= bands; b++)
                ds.GetRasterBand(b).SetNoDataValue(noDataValue);

            ds.SetProjection(projection);
            ds.SetGeoTransform(geoTransform);

            return ds;
        }

        /// <summary>
        /// Write a 128×128 patch of flat pixel data into band 1 of <paramref name="dataset"/>
        /// at the given upper‑left column / row.
        /// </summary>
        public static void WritePatch<T>(this Dataset dataset, int col, int row, T[] data) where T : unmanaged
        {
            var band = dataset.GetRasterBand(1);
            var handle = GCHandle.Alloc(data, GCHandleType.Pinned);
            try
            {
                band.WriteRaster(col, row, PatchSize, PatchSize, handle.AddrOfPinnedObject(),
                    PatchSize, PatchSize, GetGdalType<T>(), 0, 0);
            }
            finally
            {
                handle.Free();
            }
        }

        /// <summary>
        /// Extract one time‑step slice from a flat per‑patch results buffer
        /// (layout: [pixel * timeCount + t]) and write it to <paramref name="band"/>
        /// of the dataset.
        /// </summary>
        public static void WriteTimeStep<T>(
            this Dataset dataset,
            int band,
            int col,
            int row,
            T[] flatResults,
            int timeStep,
            int timeCount) where T : unmanaged
        {
            int pixelCount = PatchSize * PatchSize;
            var slice = new T[pixelCount];
            for (int i = 0; i < pixelCount; i++)
                slice[i] = flatResults[i * timeCount + timeStep];

            var rasterBand = dataset.GetRasterBand(band);
            var handle = GCHandle.Alloc(slice, GCHandleType.Pinned);
            try
            {
                rasterBand.WriteRaster(col, row, PatchSize, PatchSize, handle.AddrOfPinnedObject(),
                    PatchSize, PatchSize, GetGdalType<T>(), 0, 0);
            }
            finally
            {
                handle.Free();
            }
        }

        // =================================================================
        // Keyed‑dataset pattern (one file per key — e.g. per time‑step)
        // =================================================================

        /// <summary>
        /// Lazily create or reopen a single‑band tiled GeoTIFF for <paramref name="key"/>,
        /// stored in <paramref name="cache"/>.  If no entry exists yet the file is
        /// created via <paramref name="filenameFunc"/> inside <paramref name="directory"/>.
        /// </summary>
        public static Dataset GetOrCreateTiled<TKey, T>(
            Dictionary<TKey, Dataset> cache,
            TKey key,
            string directory,
            Func<TKey, string> filenameFunc,
            int width,
            int height,
            double noDataValue,
            string projection,
            double[] geoTransform) where TKey : notnull where T : unmanaged
        {
            lock (cache)
            {
                if (cache.TryGetValue(key, out var existing))
                    return existing;

                var path = System.IO.Path.Combine(directory, filenameFunc(key));
                var ds = OpenTiled<T>(path, width, height, bands: 1, noDataValue, projection, geoTransform);
                cache[key] = ds;
                return ds;
            }
        }

        // =================================================================
        // Flat‑index helpers
        // =================================================================

        public static int FlatIndex(int x, int y) => y * PatchSize + x;

        /// <summary>
        /// Copy one time‑step slice from a flat per‑patch buffer into <paramref name="dest"/>.
        /// <paramref name="dest"/> must have length PatchSize².
        /// </summary>
        public static void SliceTimeStep<T>(
            T[] flatResults,
            int timeStep,
            int timeCount,
            Span<T> dest)
        {
            int pixelCount = PatchSize * PatchSize;
            for (int i = 0; i < pixelCount; i++)
                dest[i] = flatResults[i * timeCount + timeStep];
        }

        public static float[] ExtractPatchDem(float[,] elev, int col, int row)
        {
            var patch = new float[PatchSize * PatchSize];
            for (int y = 0; y < PatchSize; y++)
            {
                int srcRow = row + y;
                int dstOff = y * PatchSize;
                for (int x = 0; x < PatchSize; x++)
                    patch[dstOff + x] = elev[srcRow, col + x];
            }
            return patch;
        }
    }
}
