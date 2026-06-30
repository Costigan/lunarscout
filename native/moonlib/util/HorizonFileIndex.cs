using moonlib.horizon;

namespace moonlib.util
{
    public class HorizonFileIndex
    {
        public const int PatchSize = 128;
        private Dictionary<(int, int), string> _index = new();
        private string? _lastAccessedFile;
        float[]? cached_horizon = null;
        int cached_observer_x = -1, cached_observer_y = -1;

        public HorizonFileIndex() { }
        public HorizonFileIndex(string dir, float targetElevation)
        {
            Scan(dir, targetElevation);
        }

        public void Clear()
        {
            _index.Clear();
            cached_horizon = null;
        }

        public void Scan(string dir, float targetElevation)
        {
            _index.Clear();
            var store = new HorizonTileStore(dir);
            foreach (var tile in store.EnumerateTiles(targetElevation))
            {
                _index[(tile.Key.TileX, tile.Key.TileY)] = tile.Path;
            }
            Console.WriteLine($"Index Scanned: {_index.Count} matching tiles found for elevation {targetElevation}m.");
        }

        public float[]? LoadHorizon(int x, int y)
        {
            if (cached_observer_x == x && cached_observer_y == y && cached_horizon != null)
            {
                //Console.WriteLine($"Using Cached Horizon for Pixel ({x}, {y})");
                return cached_horizon;
            }
            int c = (x / PatchSize) * PatchSize, r = (y / PatchSize) * PatchSize;
            if (!_index.TryGetValue((c, r), out var f))
            {
                // Only print mismatch if we actually have an index (to avoid spam before any file is loaded)
                if (_index.Count > 0)
                    Console.WriteLine($"Tile Index Mismatch: No horizon tile found for pixel ({x}, {y}) [Target Tile: {c}, {r}]");
                return null;
            }

            if (f != _lastAccessedFile)
            {
                Console.WriteLine($"Accessing Horizon Tile: {Path.GetFileName(f)}");
                _lastAccessedFile = f;
            }

            int row_in_tile = y % PatchSize, col_in_tile = x % PatchSize;
            var h = new float[1440];
            if (Path.GetExtension(f).Equals(".cbin", StringComparison.OrdinalIgnoreCase))
            {
                var tile = HorizonFile.ReadCompressedHorizonFile(f);
                Array.Copy(tile, (row_in_tile * PatchSize + col_in_tile) * 1440, h, 0, 1440);
            }
            else
            {
                using var fs = File.OpenRead(f);
                var base_of_horizon_in_file = (row_in_tile * PatchSize + col_in_tile) * 1440L * 4L;
                fs.Seek(base_of_horizon_in_file, SeekOrigin.Begin);
                var b = new BinaryReader(fs);
                for (int i = 0; i < 1440; i++)
                    h[i] = b.ReadSingle();
            }

            Console.WriteLine($"Horizon Loaded for Pixel ({x}, {y}) from Tile ({c}, {r}) offsets ({col_in_tile}, {row_in_tile})");
            (cached_horizon, cached_observer_x, cached_observer_y) = (h, x, y);

            return h;
        }
    }
}
