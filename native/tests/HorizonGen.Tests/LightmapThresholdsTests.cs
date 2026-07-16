using FluentAssertions;
using moonlib.horizon;
using moonlib.pipeline;

namespace moonlib.tests
{
    [TestClass]
    [TestCategory("Fast")]
    public sealed class LightmapThresholdsTests
    {
        [TestMethod]
        public void CountPatchBits_CountsEachBitByTimeIndex()
        {
            const int timeCount = 2;
            byte[] data = new byte[LightmapThresholds.Width * LightmapThresholds.Height * timeCount];
            data[0] = 0b0001_0001;
            data[1] = 0b1000_0000;
            data[2] = 0b1111_1111;
            long[,] counts = new long[timeCount, 8];

            LightmapThresholds.CountPatchBits(
                new PatchThresholdResult { PatchCol = 0, PatchRow = 0, Data = data },
                timeCount,
                counts);

            counts[0, 0].Should().Be(2);
            counts[0, 1].Should().Be(1);
            counts[0, 2].Should().Be(1);
            counts[0, 3].Should().Be(1);
            counts[0, 4].Should().Be(2);
            counts[0, 5].Should().Be(1);
            counts[0, 6].Should().Be(1);
            counts[0, 7].Should().Be(1);

            counts[1, 0].Should().Be(0);
            counts[1, 7].Should().Be(1);
        }

        [TestMethod]
        public void CountPatchBits_RejectsUnexpectedPatchLength()
        {
            long[,] counts = new long[1, 8];

            Action act = () => LightmapThresholds.CountPatchBits(
                new PatchThresholdResult { PatchCol = 0, PatchRow = 0, Data = new byte[1] },
                timeCount: 1,
                counts);

            act.Should().Throw<ArgumentException>()
                .WithMessage("*Patch buffer length does not match threshold output dimensions*");
        }
    }
}
