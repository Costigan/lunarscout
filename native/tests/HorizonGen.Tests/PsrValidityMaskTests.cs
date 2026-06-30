using FluentAssertions;
using moonlib.mapops;
using moonlib.pipeline;
using OSGeo.GDAL;

namespace moonlib.tests
{
    [TestClass]
    [TestCategory("Fast")]
    public class PsrValidityMaskTests
    {
        [TestMethod]
        public void CompleteCoverageUsesVirtualAllValidMaskWithoutNodata()
        {
            string path = TempTiffPath();
            try
            {
                using (var dataset = CreatePsrDataset(path, width: 256, height: 128))
                {
                    bool storedMask = MapOperations.FinalizePsrValidityMask(
                        dataset,
                        width: 256,
                        height: 128,
                        new[] { (Col: 0, Row: 0), (Col: 128, Row: 0) });
                    storedMask.Should().BeFalse();
                }

                using var reopened = Gdal.Open(path, Access.GA_ReadOnly);
                var band = reopened.GetRasterBand(1);
                band.GetNoDataValue(out _, out int hasNoData);
                hasNoData.Should().Be(0);
                band.GetMaskFlags().Should().Be(GdalConst.GMF_ALL_VALID);
                File.Exists(path + ".msk").Should().BeFalse();
            }
            finally
            {
                DeleteTiff(path);
            }
        }

        [TestMethod]
        public void PartialCoverageUsesInternalValidityMaskWithoutNodata()
        {
            string path = TempTiffPath();
            try
            {
                using (var dataset = CreatePsrDataset(path, width: 256, height: 128))
                {
                    var data = Enumerable.Repeat(byte.MaxValue, 256 * 128).ToArray();
                    dataset.GetRasterBand(1).WriteRaster(0, 0, 256, 128, data, 256, 128, 0, 0);
                    bool storedMask = MapOperations.FinalizePsrValidityMask(
                        dataset,
                        width: 256,
                        height: 128,
                        new[] { (Col: 0, Row: 0) });
                    storedMask.Should().BeTrue();
                }

                using var reopened = Gdal.Open(path, Access.GA_ReadOnly);
                var band = reopened.GetRasterBand(1);
                band.GetNoDataValue(out _, out int hasNoData);
                hasNoData.Should().Be(0);
                (band.GetMaskFlags() & GdalConst.GMF_PER_DATASET).Should().NotBe(0);
                File.Exists(path + ".msk").Should().BeFalse();

                var mask = new byte[256 * 128];
                band.GetMaskBand().ReadRaster(0, 0, 256, 128, mask, 256, 128, 0, 0);
                for (int row = 0; row < 128; row++)
                {
                    mask.Skip(row * 256).Take(128)
                        .Should().OnlyContain(value => value == byte.MaxValue);
                    mask.Skip(row * 256 + 128).Take(128)
                        .Should().OnlyContain(value => value == 0);
                }

                var values = new byte[256 * 128];
                band.ReadRaster(0, 0, 256, 128, values, 256, 128, 0, 0);
                for (int row = 0; row < 128; row++)
                {
                    values.Skip(row * 256 + 128).Take(128)
                        .Should().OnlyContain(value => value == 0);
                }
            }
            finally
            {
                DeleteTiff(path);
            }
        }

        private static Dataset CreatePsrDataset(string path, int width, int height)
        {
            Driver driver = Gdal.GetDriverByName("GTiff");
            return LightmapPipeline.OpenDataset(
                driver,
                path,
                DataType.GDT_Byte,
                no_data_value: null,
                width,
                height,
                projection: string.Empty,
                geoTransform: new[] { 0.0, 1.0, 0.0, 0.0, 0.0, -1.0 });
        }

        private static string TempTiffPath() =>
            Path.Combine(Path.GetTempPath(), $"psr-mask-{Guid.NewGuid():N}.tif");

        private static void DeleteTiff(string path)
        {
            File.Delete(path);
            File.Delete(path + ".msk");
            File.Delete(path + ".aux.xml");
        }
    }
}
