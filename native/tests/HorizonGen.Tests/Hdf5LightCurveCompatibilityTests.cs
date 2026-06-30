using FluentAssertions;
using PureHDF;
using PureHDF.Filters;
using PureHDF.Selections;
using PureHDF.VOL.Native;
using System.Diagnostics;
using System.Text.Json;

namespace moonlib.tests
{
    [TestClass]
    [TestCategory("Fast")]
    public sealed class Hdf5LightCurveCompatibilityTests
    {
        [TestMethod]
        [Ignore("PureHDF 2.1.2 writes chunked byte datasets that h5py 3.16 rejects with a chunk-layout datatype-size mismatch.")]
        public void PureHdfChunkedDeflateDataset_IsReadableByPythonH5py()
        {
            string dir = CreateTempDir();
            try
            {
                string path = Path.Combine(dir, "light_curves.h5");
                WriteChunkedLightCurveHdf5(path, selectedPatchOnly: false);

                using JsonDocument report = ReadWithPythonH5py(path);
                JsonElement root = report.RootElement;
                root.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32())
                    .Should().Equal(16, 16, 20);
                root.GetProperty("chunks").EnumerateArray().Select(v => v.GetInt32())
                    .Should().Equal(8, 8, 20);
                root.GetProperty("dtype").GetString().Should().Be("uint8");
                root.GetProperty("compression").GetString().Should().Be("gzip");
                root.GetProperty("sample").EnumerateArray().Select(v => v.GetInt32())
                    .Should().Equal(35, 42, 49, 56);
                root.GetProperty("patch_checksum").GetInt32().Should().Be(ExpectedPatchChecksum());
                root.GetProperty("axis_order").GetString().Should().Be("y,x,time");
            }
            finally
            {
                Directory.Delete(dir, recursive: true);
            }
        }

        private static void WriteChunkedLightCurveHdf5(string path, bool selectedPatchOnly)
        {
            const int height = 16;
            const int width = 16;
            const int timeCount = 20;
            const int patchY = 4;
            const int patchX = 5;
            const int patchHeight = 8;
            const int patchWidth = 8;

            var filters = new List<H5Filter>
            {
                new(DeflateFilter.Id, new Dictionary<string, object>
                {
                    [DeflateFilter.COMPRESSION_LEVEL] = 1,
                }),
            };
            var datasetCreation = new H5DatasetCreation(ChunkCache: null, Filters: filters);
            H5Dataset dataset;
            if (selectedPatchOnly)
            {
                byte[] patch = CreatePatch(patchHeight, patchWidth, timeCount, patchY, patchX);
                var fileSelection = new HyperslabSelection(
                    rank: 3,
                    starts: new ulong[] { patchY, patchX, 0 },
                    blocks: new ulong[] { patchHeight, patchWidth, timeCount }
                );
                dataset = new H5Dataset(
                    data: patch,
                    chunks: new uint[] { 8, 8, timeCount },
                    memorySelection: new AllSelection(),
                    fileSelection: fileSelection,
                    fileDims: new ulong[] { height, width, timeCount },
                    datasetCreation: datasetCreation,
                    opaqueInfo: null
                );
            }
            else
            {
                dataset = new H5Dataset(
                    data: CreateFullCube(height, width, timeCount, patchY, patchX, patchHeight, patchWidth),
                    chunks: new uint[] { 8, 8, timeCount },
                    memorySelection: null,
                    fileSelection: null,
                    fileDims: new ulong[] { height, width, timeCount },
                    datasetCreation: datasetCreation,
                    opaqueInfo: null
                );
            }

            dataset.Attributes = new()
            {
                ["axis_order"] = "y,x,time",
                ["signal_name"] = "synthetic_shadow_value",
            };
            var file = new H5File
            {
                ["light_curves"] = dataset,
            };
            file.Write(path);
        }

        private static JsonDocument ReadWithPythonH5py(string path)
        {
            string repoRoot = FindRepoRoot();
            string python = Path.Combine(repoRoot, ".venv", "bin", "python");
            File.Exists(python).Should().BeTrue("the repo-managed Python environment is required for h5py compatibility checks");

            string script = string.Join(
                "\n",
                "import json, sys",
                "import h5py",
                "path = sys.argv[1]",
                "with h5py.File(path, 'r') as f:",
                "    d = f['light_curves']",
                "    sample = d[4, 5, 0:4].astype('int64').tolist()",
                "    patch_checksum = int(d[4:12, 5:13, :].astype('int64').sum())",
                "    axis_order = d.attrs.get('axis_order')",
                "    if isinstance(axis_order, bytes):",
                "        axis_order = axis_order.decode('utf-8')",
                "    print(json.dumps({",
                "        'shape': list(d.shape),",
                "        'chunks': list(d.chunks),",
                "        'dtype': str(d.dtype),",
                "        'compression': d.compression,",
                "        'sample': sample,",
                "        'patch_checksum': patch_checksum,",
                "        'axis_order': axis_order,",
                "    }))"
            );

            var start = new ProcessStartInfo
            {
                FileName = python,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            };
            start.ArgumentList.Add("-c");
            start.ArgumentList.Add(script);
            start.ArgumentList.Add(path);

            using Process process = Process.Start(start)
                ?? throw new InvalidOperationException("Failed to start Python h5py verification.");
            string stdout = process.StandardOutput.ReadToEnd();
            string stderr = process.StandardError.ReadToEnd();
            process.WaitForExit(milliseconds: 30_000).Should().BeTrue("Python h5py verification should finish quickly");
            process.ExitCode.Should().Be(0, stderr);
            return JsonDocument.Parse(stdout);
        }

        private static byte[] CreatePatch(int patchHeight, int patchWidth, int timeCount, int y0, int x0)
        {
            byte[] values = new byte[patchHeight * patchWidth * timeCount];
            int index = 0;
            for (int y = 0; y < patchHeight; y++)
            {
                for (int x = 0; x < patchWidth; x++)
                {
                    for (int t = 0; t < timeCount; t++)
                    {
                        values[index++] = (byte)(((y0 + y) * 5 + (x0 + x) * 3 + t * 7) % 256);
                    }
                }
            }
            return values;
        }

        private static byte[] CreateFullCube(
            int height,
            int width,
            int timeCount,
            int patchY,
            int patchX,
            int patchHeight,
            int patchWidth)
        {
            byte[] values = new byte[height * width * timeCount];
            for (int y = 0; y < patchHeight; y++)
            {
                for (int x = 0; x < patchWidth; x++)
                {
                    for (int t = 0; t < timeCount; t++)
                    {
                        int index = ((patchY + y) * width + (patchX + x)) * timeCount + t;
                        values[index] = (byte)(((patchY + y) * 5 + (patchX + x) * 3 + t * 7) % 256);
                    }
                }
            }
            return values;
        }

        private static int ExpectedPatchChecksum()
        {
            return CreatePatch(8, 8, 20, y0: 4, x0: 5).Select(value => (int)value).Sum();
        }

        private static string FindRepoRoot()
        {
            DirectoryInfo? current = new(AppContext.BaseDirectory);
            while (current is not null)
            {
                if (File.Exists(Path.Combine(current.FullName, "requirements.txt"))
                    && Directory.Exists(Path.Combine(current.FullName, "native")))
                {
                    return current.FullName;
                }
                current = current.Parent;
            }
            throw new InvalidOperationException("Could not locate repository root.");
        }

        private static string CreateTempDir()
        {
            string dir = Path.Combine(Path.GetTempPath(), "Hdf5LightCurveCompatibilityTests_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            return dir;
        }

    }
}
