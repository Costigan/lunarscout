using moonlib.horizon;
using System.Drawing;

namespace moonlib.tests
{
    [TestClass]
    /// <summary>
    /// Tests the <see cref="Utilities.EnumeratePatchLocationsInSpiralPattern"/> iterator, which yields
    /// patch coordinates in a spiral order starting from a center point. This pattern is used to prioritize
    /// processing closer to the observer/center.
    /// </summary>
    public class HorizonGeneratorSpiralPatternTests
    {
        /// <summary>
        /// Verifies that the first patch yielded is the one immediately to the right of the center.
        /// </summary>
        [TestMethod]
        [TestCategory("Fast")]
        public void SpiralPattern_FirstPatch_IsRightOfCenter()
        {
            var patchSize = TerrainPatch.PatchSize;
            var center = new Point(10 * patchSize.Width, 10 * patchSize.Height);
            var spiral = Utilities.EnumeratePatchLocationsInSpiralPattern(center).Take(1).ToList();
            Assert.AreEqual(1, spiral.Count);
            // First patch is to the right of center
            Assert.AreEqual(new Point((10 + 1) * patchSize.Width, 10 * patchSize.Height), spiral[0]);
        }

        /// <summary>
        /// Verifies the sequence of the first few patches matches a hardcoded clockwise spiral pattern.
        /// </summary>
        [TestMethod]
        [TestCategory("Fast")]
        public void SpiralPattern_Sequence_IsClockwiseSpiral()
        {
            var patchSize = TerrainPatch.PatchSize;
            var center =new Point(0, 0);
            var expectedGrid = new[]
            {
                (1, 0), // right
                (1, 1), // down
                (0, 1), // left
                (-1, 1), // left
                (-1, 0), // up
                (-1, -1), // up
                (0, -1), // right
                (1, -1), // right
                (2, -1), // right (spiral expands)
            };
            var expected = expectedGrid.Select(g => new Point(g.Item1 * patchSize.Width, g.Item2 * patchSize.Height)).ToArray();
            var spiral = Utilities.EnumeratePatchLocationsInSpiralPattern(center).Take(expected.Length).ToList();
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], spiral[i], $"Mismatch at index {i}");
            }
        }

        /// <summary>
        /// Verifies that the spiral pattern does not yield the center point itself (assumed processed separately or implicit).
        /// </summary>
        [TestMethod]
        [TestCategory("Fast")]
        public void SpiralPattern_DoesNotIncludeCenter()
        {
            var patchSize = TerrainPatch.PatchSize;
            var center = new Point(5 * patchSize.Width, 5 * patchSize.Height);
            var spiral = Utilities.EnumeratePatchLocationsInSpiralPattern(center).Take(20).ToList();
            Assert.IsFalse(spiral.Any(p => p == center));
        }

        /// <summary>
        /// Verifies that the generated points are unique within the tested range.
        /// </summary>
        [TestMethod]
        [TestCategory("Fast")]
        public void SpiralPattern_UniquePatches_ForGivenCount()
        {
            var patchSize = TerrainPatch.PatchSize;
            var center =new Point(0, 0);
            int count = 100;
            var spiral = Utilities.EnumeratePatchLocationsInSpiralPattern(center).Take(count).ToList();
            var unique = new HashSet<Point>(spiral);
            Assert.AreEqual(count, unique.Count);
        }
    }
}
