using FluentAssertions;
using moonlib.horizon;

namespace moonlib.tests
{
    [TestClass]
    [TestCategory("Fast")]
    public class HorizonFileTests
    {
        private const int HorizonSamples = 1440;
        private const int HorizonRows = 128;
        private const int HorizonCols = 128;
        private const int TotalSamples = HorizonSamples * HorizonRows * HorizonCols;

        private static readonly float[] ZeroData = new float[TotalSamples];

        [TestMethod]
        public void WriteHorizonFile_Bin_DispatchWritesUncompressed()
        {
            string dir = CreateTempDir();
            try
            {
                string path = Path.Combine(dir, "horizon_dispatch.bin");
                HorizonFile.WriteHorizonFile(path, ZeroData);

                var fi = new FileInfo(path);
                fi.Exists.Should().BeTrue();
                fi.Length.Should().Be((long)TotalSamples * sizeof(float));
            }
            finally
            {
                DeleteDir(dir);
            }
        }

        [TestMethod]
        public void WriteHorizonFile_Cbin_DispatchWritesCompressedAndReadable()
        {
            string dir = CreateTempDir();
            try
            {
                string path = Path.Combine(dir, "horizon_dispatch.cbin");
                HorizonFile.WriteHorizonFile(path, ZeroData);

                var fi = new FileInfo(path);
                fi.Exists.Should().BeTrue();
                fi.Length.Should().BeGreaterThan(0);
                fi.Length.Should().BeLessThan((long)TotalSamples * sizeof(float));

                var read = HorizonFile.ReadHorizonFile(path);
                read.Length.Should().Be(TotalSamples);
                read[0].Should().Be(0f);
                read[^1].Should().Be(0f);
            }
            finally
            {
                DeleteDir(dir);
            }
        }

        [TestMethod]
        public void ConvertUncompressedFilesInDirectory_DefaultDelete_RemovesBin()
        {
            string dir = CreateTempDir();
            try
            {
                string binPath = Path.Combine(dir, "horizon_convert.bin");
                string cbinPath = Path.Combine(dir, "horizon_convert.cbin");
                HorizonFile.WriteUncompressedHorizonFile(binPath, ZeroData);

                int converted = HorizonFile.CompressDirectory(dir);

                converted.Should().Be(1);
                File.Exists(binPath).Should().BeFalse();
                File.Exists(cbinPath).Should().BeTrue();

                var read = HorizonFile.ReadCompressedHorizonFile(cbinPath);
                read.Length.Should().Be(TotalSamples);
                read[0].Should().Be(0f);
                read[^1].Should().Be(0f);
            }
            finally
            {
                DeleteDir(dir);
            }
        }

        [TestMethod]
        public void ConvertUncompressedFilesInDirectory_DeleteFalse_KeepsBin()
        {
            string dir = CreateTempDir();
            try
            {
                string binPath = Path.Combine(dir, "horizon_keep.bin");
                string cbinPath = Path.Combine(dir, "horizon_keep.cbin");
                HorizonFile.WriteUncompressedHorizonFile(binPath, ZeroData);

                int converted = HorizonFile.CompressDirectory(dir, deleteUncompressed: false);

                converted.Should().Be(1);
                File.Exists(binPath).Should().BeTrue();
                File.Exists(cbinPath).Should().BeTrue();
            }
            finally
            {
                DeleteDir(dir);
            }
        }

        [TestMethod]
        public void WriteHorizonFile_InvalidPrefix_ShouldThrow()
        {
            string dir = CreateTempDir();
            try
            {
                string path = Path.Combine(dir, "not_horizon.bin");
                Action act = () => HorizonFile.WriteHorizonFile(path, ZeroData);
                act.Should().Throw<ArgumentException>();
            }
            finally
            {
                DeleteDir(dir);
            }
        }

        private static string CreateTempDir()
        {
            string dir = Path.Combine(Path.GetTempPath(), "HorizonFileTests_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            return dir;
        }

        private static void DeleteDir(string dir)
        {
            if (!Directory.Exists(dir)) return;
            try
            {
                Directory.Delete(dir, recursive: true);
            }
            catch
            {
                // Best-effort test cleanup.
            }
        }
    }
}
