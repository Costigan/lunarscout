using FluentAssertions;
using moonlib.horizon;

namespace moonlib.tests
{
    [TestClass]
    [TestCategory("Fast")]
    public class MoonlibBridgeTests
    {
        private const int HorizonSamples = 1440;
        private const int HorizonRows = 128;
        private const int HorizonCols = 128;
        private const int TotalSamples = HorizonSamples * HorizonRows * HorizonCols;

        private static readonly float[] ZeroData = new float[TotalSamples];

        [TestMethod]
        public void CompressHorizonsDirectory_DelegatesToHorizonFileCompression()
        {
            string dir = CreateTempDir();
            try
            {
                string binPath = Path.Combine(dir, "horizon_bridge.bin");
                string cbinPath = Path.Combine(dir, "horizon_bridge.cbin");
                HorizonFile.WriteUncompressedHorizonFile(binPath, ZeroData);

                var bridge = new MoonlibBridge();
                int converted = bridge.CompressHorizonsDirectory(dir, deleteUncompressed: true, useParallel: false);

                converted.Should().Be(1);
                File.Exists(binPath).Should().BeFalse();
                File.Exists(cbinPath).Should().BeTrue();
            }
            finally
            {
                DeleteDir(dir);
            }
        }

        [TestMethod]
        public void TemporalStreamingMethods_DelegateThroughMoonlibBridge()
        {
            var bridge = new MoonlibBridge();

            bridge.RegisterOutputBuffer("missing", 0, 0, 0).Should().BeFalse();
            bridge.RegisterOutputBufferV2("missing", 0, 0, 0).Should().BeFalse();
            bridge.ReleaseBuffer("missing", 0).Should().BeFalse();
            bridge.CancelJob("missing").Should().BeFalse();
            bridge.DisposeJob("missing").Should().BeFalse();
            bridge.TryGetNextTile("missing", 0).Should().BeNull();
            bridge.TryGetNextTileV2("missing", 0).Should().BeNull();
            bridge.GetNativeReduceRasterResult("missing").Should().BeNull();
        }

        private static string CreateTempDir()
        {
            string dir = Path.Combine(Path.GetTempPath(), "MoonlibBridgeTests_" + Guid.NewGuid().ToString("N"));
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
