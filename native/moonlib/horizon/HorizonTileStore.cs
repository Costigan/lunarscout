using System.Globalization;

namespace moonlib.horizon
{
    public enum HorizonTileLayout
    {
        Flat,
        PartitionedByY
    }

    public readonly record struct HorizonTileKey(
        int TileY,
        int TileX,
        int ObserverElevationDecimeters)
    {
        public float ObserverElevationMeters => ObserverElevationDecimeters / 10f;
    }

    public readonly record struct HorizonPartitionResult(
        int Moved,
        int Skipped,
        int Invalid,
        int Conflicted);

    public sealed class HorizonTileStore
    {
        private const string Prefix = "horizon_";

        public HorizonTileStore(
            string rootDirectory,
            HorizonTileLayout writeLayout = HorizonTileLayout.PartitionedByY,
            bool readLegacyFlatFiles = true)
        {
            if (string.IsNullOrWhiteSpace(rootDirectory))
                throw new ArgumentException("Horizon root directory must be provided.", nameof(rootDirectory));

            RootDirectory = rootDirectory;
            WriteLayout = writeLayout;
            ReadLegacyFlatFiles = readLegacyFlatFiles;
        }

        public bool IsPartitioned => WriteLayout == HorizonTileLayout.PartitionedByY;
        public string RootDirectory { get; }
        public HorizonTileLayout WriteLayout { get; }
        public bool ReadLegacyFlatFiles { get; }

        public string BuildFileName(int tileY, int tileX, float observerElevationMeters, bool compress = true)
        {
            ValidateCoordinate(tileY, nameof(tileY));
            ValidateCoordinate(tileX, nameof(tileX));
            int elevationDm = ToElevationDecimeters(observerElevationMeters);
            string extension = compress ? ".cbin" : ".bin";
            return $"{Prefix}{tileY:D5}_{tileX:D5}_{elevationDm:D3}{extension}";
        }

        public string BuildRelativePath(int tileY, int tileX, float observerElevationMeters, bool compress = true)
        {
            string fileName = BuildFileName(tileY, tileX, observerElevationMeters, compress);
            return WriteLayout == HorizonTileLayout.PartitionedByY
                ? Path.Combine(tileY.ToString("D5", CultureInfo.InvariantCulture), fileName)
                : fileName;
        }

        public string BuildPath(int tileY, int tileX, float observerElevationMeters, bool compress = true) =>
            Path.Combine(RootDirectory, BuildRelativePath(tileY, tileX, observerElevationMeters, compress));

        public string? FindExistingPath(int tileY, int tileX, float observerElevationMeters)
        {
            foreach (var path in CandidateReadPaths(tileY, tileX, observerElevationMeters))
            {
                if (File.Exists(path))
                    return path;
            }

            return null;
        }

        public bool Exists(int tileY, int tileX, float observerElevationMeters) =>
            FindExistingPath(tileY, tileX, observerElevationMeters) != null;

        public void Write(int tileY, int tileX, float observerElevationMeters, ReadOnlySpan<float> data, bool compress = true)
        {
            string finalPath = BuildPath(tileY, tileX, observerElevationMeters, compress);
            string? directory = Path.GetDirectoryName(finalPath);
            if (!string.IsNullOrWhiteSpace(directory))
                Directory.CreateDirectory(directory);

            string extension = compress ? ".cbin" : ".bin";
            string tempPath = Path.Combine(
                directory ?? RootDirectory,
                $"{Path.GetFileNameWithoutExtension(finalPath)}.{Guid.NewGuid():N}.tmp{extension}");

            try
            {
                if (compress)
                    HorizonFile.WriteCompressedHorizonFile(tempPath, data);
                else
                    HorizonFile.WriteUncompressedHorizonFile(tempPath, data);

                File.Move(tempPath, finalPath, overwrite: true);
            }
            catch
            {
                TryDelete(tempPath);
                throw;
            }
        }

        public float[] Read(int tileY, int tileX, float observerElevationMeters)
        {
            string? path = FindExistingPath(tileY, tileX, observerElevationMeters);
            if (path == null)
            {
                throw new FileNotFoundException(
                    $"No horizon tile found for tileY={tileY}, tileX={tileX}, observerElevationMeters={observerElevationMeters.ToString(CultureInfo.InvariantCulture)}.",
                    BuildPath(tileY, tileX, observerElevationMeters));
            }

            return HorizonFile.ReadHorizonFile(path);
        }

        public IEnumerable<string> EnumerateFiles(float? observerElevationMeters = null) =>
            EnumerateTiles(observerElevationMeters).Select(tile => tile.Path);

        public IEnumerable<(HorizonTileKey Key, string Path)> EnumerateTiles(float? observerElevationMeters = null)
        {
            if (!Directory.Exists(RootDirectory))
                yield break;

            var seen = new HashSet<HorizonTileKey>();
            int? elevationDm = observerElevationMeters.HasValue
                ? ToElevationDecimeters(observerElevationMeters.Value)
                : null;

            foreach (var candidate in EnumerateCandidateFiles())
            {
                // Debugging
                //if (!"/media/mhs/BEB8-5B41/new_datasets/polar_mosaic/horizons/03328/horizon_03328_10624_000.cbin".Equals(candidate, StringComparison.OrdinalIgnoreCase))
                //    continue;

                if (!TryParseFileName(candidate, out var key))
                    continue;
                if (elevationDm.HasValue && key.ObserverElevationDecimeters != elevationDm.Value)
                    continue;
                if (!seen.Add(key))
                    continue;
                yield return (key, candidate);
            }
        }

        public static bool TryParseFileName(string path, out HorizonTileKey key)
        {
            key = default;
            string name = Path.GetFileNameWithoutExtension(path);
            if (string.IsNullOrWhiteSpace(name))
                return false;

            var parts = name.Split('_');
            if (parts.Length != 4 || !parts[0].Equals("horizon", StringComparison.OrdinalIgnoreCase))
                return false;

            if (!TryParseFixedDigits(parts[1], 5, out int tileY))
                return false;
            if (!TryParseFixedDigits(parts[2], 5, out int tileX))
                return false;
            if (!TryParseFixedDigits(parts[3], 3, out int elevationDm))
                return false;

            key = new HorizonTileKey(tileY, tileX, elevationDm);
            return true;
        }

        public static HorizonPartitionResult PartitionFlatDirectory(string rootDirectory)
        {
            if (string.IsNullOrWhiteSpace(rootDirectory))
                throw new ArgumentException("Horizon root directory must be provided.", nameof(rootDirectory));
            if (!Directory.Exists(rootDirectory))
                throw new DirectoryNotFoundException($"Directory not found: {rootDirectory}");

            int moved = 0;
            int skipped = 0;
            int invalid = 0;
            int conflicted = 0;

            foreach (var sourcePath in Directory.EnumerateFiles(rootDirectory, $"{Prefix}*.*", SearchOption.TopDirectoryOnly))
            {
                string extension = Path.GetExtension(sourcePath);
                if (!extension.Equals(".bin", StringComparison.OrdinalIgnoreCase) &&
                    !extension.Equals(".cbin", StringComparison.OrdinalIgnoreCase))
                {
                    invalid++;
                    continue;
                }

                if (!TryParseFileName(sourcePath, out var key))
                {
                    invalid++;
                    continue;
                }

                string targetDirectory = Path.Combine(rootDirectory, key.TileY.ToString("D5", CultureInfo.InvariantCulture));
                string targetPath = Path.Combine(targetDirectory, Path.GetFileName(sourcePath));

                if (Path.GetFullPath(sourcePath).Equals(Path.GetFullPath(targetPath), StringComparison.Ordinal))
                {
                    skipped++;
                    continue;
                }

                if (File.Exists(targetPath))
                {
                    long sourceLength = new FileInfo(sourcePath).Length;
                    long targetLength = new FileInfo(targetPath).Length;
                    if (sourceLength == targetLength)
                        skipped++;
                    else
                        conflicted++;
                    continue;
                }

                Directory.CreateDirectory(targetDirectory);
                File.Move(sourcePath, targetPath);
                moved++;
            }

            return new HorizonPartitionResult(moved, skipped, invalid, conflicted);
        }

        private IEnumerable<string> EnumerateCandidateFiles()
        {
            foreach (var path in EnumeratePartitionedByExtension(".cbin"))
                yield return path;
            foreach (var path in EnumeratePartitionedByExtension(".bin"))
                yield return path;

            if (!ReadLegacyFlatFiles)
                yield break;

            foreach (var path in Directory.EnumerateFiles(RootDirectory, $"{Prefix}*.cbin", SearchOption.TopDirectoryOnly)
                         .OrderBy(path => path, StringComparer.OrdinalIgnoreCase))
                yield return path;
            foreach (var path in Directory.EnumerateFiles(RootDirectory, $"{Prefix}*.bin", SearchOption.TopDirectoryOnly)
                         .OrderBy(path => path, StringComparer.OrdinalIgnoreCase))
                yield return path;
        }

        private IEnumerable<string> EnumeratePartitionedByExtension(string extension)
        {
            if (!Directory.Exists(RootDirectory))
                yield break;

            foreach (var directory in Directory.EnumerateDirectories(RootDirectory, "*", SearchOption.TopDirectoryOnly)
                         .Where(IsYDirectory)
                         .OrderBy(path => path, StringComparer.OrdinalIgnoreCase))
            {
                foreach (var path in Directory.EnumerateFiles(directory, $"{Prefix}*{extension}", SearchOption.TopDirectoryOnly)
                             .OrderBy(path => path, StringComparer.OrdinalIgnoreCase))
                    yield return path;
            }
        }

        private IEnumerable<string> CandidateReadPaths(int tileY, int tileX, float observerElevationMeters)
        {
            string fileNameCbin = BuildFileName(tileY, tileX, observerElevationMeters, compress: true);
            string fileNameBin = BuildFileName(tileY, tileX, observerElevationMeters, compress: false);
            string yDir = tileY.ToString("D5", CultureInfo.InvariantCulture);

            yield return Path.Combine(RootDirectory, yDir, fileNameCbin);
            yield return Path.Combine(RootDirectory, yDir, fileNameBin);

            if (ReadLegacyFlatFiles)
            {
                yield return Path.Combine(RootDirectory, fileNameCbin);
                yield return Path.Combine(RootDirectory, fileNameBin);
            }
        }

        private static int ToElevationDecimeters(float observerElevationMeters) =>
            (int)(observerElevationMeters * 10);

        private static void ValidateCoordinate(int value, string name)
        {
            if (value < 0 || value > 99999)
                throw new ArgumentOutOfRangeException(name, value, "Horizon tile coordinates must fit the D5 filename field.");
        }

        private static bool TryParseFixedDigits(string value, int digits, out int parsed)
        {
            parsed = default;
            if (value.Length != digits)
                return false;
            for (int i = 0; i < value.Length; i++)
            {
                if (!char.IsDigit(value[i]))
                    return false;
            }
            return int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out parsed);
        }

        private static bool IsYDirectory(string path)
        {
            string name = Path.GetFileName(path);
            return TryParseFixedDigits(name, 5, out _);
        }

        private static void TryDelete(string path)
        {
            try
            {
                if (File.Exists(path))
                    File.Delete(path);
            }
            catch
            {
                // Best-effort cleanup after failed writes.
            }
        }
    }
}
