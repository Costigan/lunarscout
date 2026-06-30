using System;
using System.Drawing;
using moonlib;

namespace moonlib.tests
{
    [TestClass]
    /// <summary>
    /// Tests the <see cref="Utilities.RoundBoundingBox"/> method, which ensures that processing regions
    /// are aligned to patch boundaries (e.g., 128x128 pixels) to facilitate efficient blocked processing.
    /// </summary>
    public class HorizonGeneratorRoundBoundingBoxTests
    {
        /// <summary>
        /// Verifies that a bounding box with positive coordinates is expanded outward to the nearest multiple
        /// of the patch size (default 128).
        /// </summary>
        [TestMethod]
        public void RoundBoundingBox_PositiveCoordinates_ExpandsToMultiples()
        {
            var bbox = Rectangle.FromLTRB(10, 20, 250, 300);

            Rectangle rounded = Utilities.RoundBoundingBox(bbox);

            Assert.AreEqual(Rectangle.FromLTRB(0, 0, 256, 384), rounded);
        }

        /// <summary>
        /// Verifies that negative coordinates are correctly rounded "down" (more negative) to the nearest
        /// patch boundary, ensuring the box expands outward.
        /// </summary>
        [TestMethod]
        public void RoundBoundingBox_NegativeCoordinates_ExpandsOutward()
        {
            var bbox = Rectangle.FromLTRB(-10, -130, -1, 50);

            Rectangle rounded = Utilities.RoundBoundingBox(bbox);

            Assert.AreEqual(Rectangle.FromLTRB(-128, -256, 0, 128), rounded);
        }

        /// <summary>
        /// Verifies that a custom patch size can be provided and is respected by the rounding logic.
        /// </summary>
        [TestMethod]
        public void RoundBoundingBox_CustomPatchSize_UsesProvidedStep()
        {
            var bbox = Rectangle.FromLTRB(33, 65, 127, 191);
            int patchSize = 64;

            Rectangle rounded = Utilities.RoundBoundingBox(bbox, patchSize);

            Assert.AreEqual(Rectangle.FromLTRB(0, 64, 128, 192), rounded);
        }

        /// <summary>
        /// Verifies that invalid patch sizes (zero or negative) throw an <see cref="ArgumentOutOfRangeException"/>.
        /// </summary>
        [TestMethod]
        public void RoundBoundingBox_InvalidPatchSize_Throws()
        {
            var bbox = Rectangle.FromLTRB(0, 0, 10, 10);

            Assert.ThrowsException<ArgumentOutOfRangeException>(() => Utilities.RoundBoundingBox(bbox, 0));
            Assert.ThrowsException<ArgumentOutOfRangeException>(() => Utilities.RoundBoundingBox(bbox, -5));
        }
    }
}
