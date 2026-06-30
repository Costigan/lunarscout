using FluentAssertions;
using moonlib.horizon;

namespace moonlib.tests
{
    [TestClass]
    [TestCategory("Fast")]
    public class HorizonTileStoreTests
    {
        [TestMethod]
        public void BuildPath_DefaultsToPartitionedCompressedLayout()
        {
            string dir = CreateTempDir();
            try
            {
                var store = new HorizonTileStore(dir);

                string path = store.BuildPath(tileY: 21504, tileX: 20480, observerElevationMeters: 0f);

                path.Should().Be(Path.Combine(dir, "21504", "horizon_21504_20480_000.cbin"));
            }
            finally
            {
                DeleteDir(dir);
            }
        }

        [TestMethod]
        public void TryParseFileName_ParsesCanonicalCoordinateFileName()
        {
            bool ok = HorizonTileStore.TryParseFileName("horizon_21504_20480_000.cbin", out var key);

            ok.Should().BeTrue();
            key.TileY.Should().Be(21504);
            key.TileX.Should().Be(20480);
            key.ObserverElevationDecimeters.Should().Be(0);
        }

        [TestMethod]
        public void FindExistingPath_PrefersPartitionedCompressedFile()
        {
            string dir = CreateTempDir();
            try
            {
                var store = new HorizonTileStore(dir);
                string flatPath = Path.Combine(dir, "horizon_21504_20480_000.cbin");
                string partitionedPath = store.BuildPath(21504, 20480, 0f, compress: true);

                File.WriteAllText(flatPath, "flat");
                Directory.CreateDirectory(Path.GetDirectoryName(partitionedPath)!);
                File.WriteAllText(partitionedPath, "partitioned");

                store.FindExistingPath(21504, 20480, 0f).Should().Be(partitionedPath);
            }
            finally
            {
                DeleteDir(dir);
            }
        }

        [TestMethod]
        public void FindExistingPath_FallsBackToLegacyFlatFile()
        {
            string dir = CreateTempDir();
            try
            {
                var store = new HorizonTileStore(dir);
                string flatPath = Path.Combine(dir, "horizon_21504_20480_000.bin");
                File.WriteAllText(flatPath, "flat");

                store.FindExistingPath(21504, 20480, 0f).Should().Be(flatPath);
            }
            finally
            {
                DeleteDir(dir);
            }
        }

        [TestMethod]
        public void EnumerateTiles_ReturnsPartitionedAndFlatFilesOnceWithCompressedPreference()
        {
            string dir = CreateTempDir();
            try
            {
                var store = new HorizonTileStore(dir);
                string flatBin = Path.Combine(dir, "horizon_00000_00128_000.bin");
                string partitionedCbin = store.BuildPath(0, 128, 0f, compress: true);
                string otherElevation = store.BuildPath(128, 128, 2f, compress: true);

                File.WriteAllText(flatBin, "flat-bin");
                Directory.CreateDirectory(Path.GetDirectoryName(partitionedCbin)!);
                File.WriteAllText(partitionedCbin, "partitioned-cbin");
                Directory.CreateDirectory(Path.GetDirectoryName(otherElevation)!);
                File.WriteAllText(otherElevation, "other-elevation");

                var tiles = store.EnumerateTiles(observerElevationMeters: 0f).ToList();

                tiles.Should().ContainSingle();
                tiles[0].Key.TileY.Should().Be(0);
                tiles[0].Key.TileX.Should().Be(128);
                tiles[0].Path.Should().Be(partitionedCbin);
            }
            finally
            {
                DeleteDir(dir);
            }
        }

        [TestMethod]
        public void PartitionFlatDirectory_MovesValidTopLevelFilesIntoYDirectories()
        {
            string dir = CreateTempDir();
            try
            {
                string source = Path.Combine(dir, "horizon_21504_20480_000.cbin");
                File.WriteAllText(source, "tile");
                File.WriteAllText(Path.Combine(dir, "horizon_invalid.cbin"), "invalid");

                var result = HorizonTileStore.PartitionFlatDirectory(dir);

                result.Moved.Should().Be(1);
                result.Invalid.Should().Be(1);
                File.Exists(source).Should().BeFalse();
                File.Exists(Path.Combine(dir, "21504", "horizon_21504_20480_000.cbin")).Should().BeTrue();
            }
            finally
            {
                DeleteDir(dir);
            }
        }

        [TestMethod]
        public void PartitionFlatDirectory_ReportsConflictWithoutOverwrite()
        {
            string dir = CreateTempDir();
            try
            {
                string source = Path.Combine(dir, "horizon_21504_20480_000.cbin");
                string destination = Path.Combine(dir, "21504", "horizon_21504_20480_000.cbin");
                File.WriteAllText(source, "source");
                Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
                File.WriteAllText(destination, "different");

                var result = HorizonTileStore.PartitionFlatDirectory(dir);

                result.Moved.Should().Be(0);
                result.Conflicted.Should().Be(1);
                File.ReadAllText(source).Should().Be("source");
                File.ReadAllText(destination).Should().Be("different");
            }
            finally
            {
                DeleteDir(dir);
            }
        }

        private static string CreateTempDir()
        {
            string dir = Path.Combine(Path.GetTempPath(), "HorizonTileStoreTests_" + Guid.NewGuid().ToString("N"));
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
