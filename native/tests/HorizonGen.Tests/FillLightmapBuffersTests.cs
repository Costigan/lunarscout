using FluentAssertions;
using moonlib.horizon;
using moonlib.pipeline;
using OSGeo.GDAL;
using OSGeo.OSR;
using System.Runtime.InteropServices;

namespace moonlib.tests
{
    [TestClass]
    public sealed class FillLightmapBuffersTests
    {
        [TestMethod]
        public void Poll_FillsPythonOwnedBufferAndForgetsItWhenReturned()
        {
            string root = CreateTempDir();
            try
            {
                string demPath = Path.Combine(root, "dem.tif");
                string horizonDir = Path.Combine(root, "horizons");
                Directory.CreateDirectory(horizonDir);
                WriteSyntheticDem(demPath, 128, 128);

                string horizonPath = Path.Combine(horizonDir, "horizon_00000_00000_000.cbin");
                WriteSyntheticHorizonTile(horizonPath, fillValue: 0f);

                using var filler = new FillLightmapBuffers();
                var request = new FillLightmapBuffersRequest(
                    DemPath: demPath,
                    HorizonDir: horizonDir,
                    TimestampsUtc: new[] { new DateTime(2027, 1, 1, 0, 0, 0, DateTimeKind.Utc) },
                    MaxReadParallelism: 1,
                    MaxComputeParallelism: 1,
                    QueueCapacity: 2,
                    UseSpiceSunVectors: false);

                filler.Start(request);

                var buffer = new byte[FillLightmapBuffers.Width * FillLightmapBuffers.Height];
                Array.Fill(buffer, (byte)123);
                GCHandle handle = GCHandle.Alloc(buffer, GCHandleType.Pinned);
                try
                {
                    var offered = new[]
                    {
                        new FillLightmapAvailableBuffer(17, handle.AddrOfPinnedObject().ToInt64(), buffer.Length)
                    };
                    FillLightmapFilledBuffer[] filled = Array.Empty<FillLightmapFilledBuffer>();
                    FillLightmapPollResult result = filler.GetStatus();

                    var deadline = DateTime.UtcNow.AddSeconds(20);
                    while (DateTime.UtcNow < deadline)
                    {
                        result = filler.Poll(offered, timeoutMs: 250);
                        offered = Array.Empty<FillLightmapAvailableBuffer>();
                        if (result.FilledBuffers.Length > 0)
                        {
                            filled = result.FilledBuffers;
                            break;
                        }
                    }

                    filled.Should().ContainSingle();
                    filled[0].BufferId.Should().Be(17);
                    filled[0].PatchRow.Should().Be(0);
                    filled[0].PatchCol.Should().Be(0);
                    filled[0].TimeCount.Should().Be(1);
                    filled[0].State.Should().Be(FillLightmapBufferState.Filled);
                    buffer.Should().OnlyContain(value => value == 0);

                    FillLightmapPollResult final = result;
                    deadline = DateTime.UtcNow.AddSeconds(20);
                    while (DateTime.UtcNow < deadline && final.State == FillLightmapRunState.Running)
                        final = filler.Poll(Array.Empty<FillLightmapAvailableBuffer>(), timeoutMs: 250);

                    final.State.Should().Be(FillLightmapRunState.Completed);
                    final.OwnedBufferCount.Should().Be(0);
                }
                finally
                {
                    handle.Free();
                }
            }
            finally
            {
                DeleteDir(root);
            }
        }

        private static string CreateTempDir()
        {
            string path = Path.Combine(Path.GetTempPath(), "FillLightmapBuffersTests_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(path);
            return path;
        }

        private static void DeleteDir(string path)
        {
            if (!Directory.Exists(path))
                return;
            try { Directory.Delete(path, recursive: true); }
            catch { }
        }

        private static void WriteSyntheticDem(string path, int width, int height)
        {
            Driver? driver = Gdal.GetDriverByName("GTiff");
            driver.Should().NotBeNull();

            using var ds = driver!.Create(path, width, height, 1, DataType.GDT_Float32, null);
            ds.Should().NotBeNull();

            using var srs = new SpatialReference(null);
            srs.ImportFromProj4(ElevationMap.LongLatProj).Should().Be(0);
            srs.ExportToWkt(out string? wkt, Array.Empty<string>());
            ds.SetProjection(wkt ?? string.Empty);
            ds.SetGeoTransform(new[] { 0.0, 0.01, 0.0, 0.0, 0.0, -0.01 });

            var data = new float[width * height];
            Array.Fill(data, 1000f);
            ds.GetRasterBand(1).WriteRaster(0, 0, width, height, data, width, height, 0, 0);
            ds.FlushCache();
        }

        private static void WriteSyntheticHorizonTile(string path, float fillValue)
        {
            int total = FillLightmapBuffers.Width * FillLightmapBuffers.Height * FillLightmapBuffers.HorizonSamples;
            var values = new float[total];
            if (Math.Abs(fillValue) > 0f)
                Array.Fill(values, fillValue);
            HorizonFile.WriteHorizonFile(path, values);
        }
    }
}
