using FluentAssertions;
using moonlib;
using moonlib.horizon;
using moonlib.pipeline.streaming;
using OSGeo.GDAL;
using OSGeo.OSR;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace moonlib.tests
{
    [TestClass]
    [TestCategory("Fast")]
    public class LightmapArrayStreamingBridgeTests
    {
        [TestMethod]
        public void StartLightmapArrayStreaming_NoHorizonTiles_EmitsTerminalAndCompletes()
        {
            string root = CreateTempDir();
            try
            {
                string demPath = Path.Combine(root, "dem.tif");
                string horizonDir = Path.Combine(root, "horizons");
                Directory.CreateDirectory(horizonDir);
                WriteSyntheticDem(demPath, 128, 128);

                var bridge = new LightmapArrayStreamingBridge();
                string jobId = bridge.StartLightmapArrayStreaming(new LightmapArrayStreamRequest(
                    ScenarioRootDir: root,
                    DemPath: demPath,
                    SurroundingDemPaths: new List<string>(),
                    HorizonDir: horizonDir,
                    StartUtc: new DateTime(2024, 01, 01, 0, 0, 0, DateTimeKind.Utc),
                    StopUtc: new DateTime(2024, 01, 01, 0, 0, 0, DateTimeKind.Utc),
                    TimeStepHours: 1.0,
                    ObserverElevationMeters: 0f,
                    UseSpiceSunVectors: false
                ));

                var terminalStatus = WaitForTerminalState(bridge, jobId);
                terminalStatus.State.Should().Be(StreamJobState.Completed);
                terminalStatus.Progress01.Should().BeApproximately(1.0, 1e-6);

                var terminalEnvelope = bridge.TryGetNextTile(jobId, 2000);
                terminalEnvelope.Should().NotBeNull();
                terminalEnvelope!.State.Should().Be(StreamTileState.Terminal);
                terminalEnvelope.Message.Should().NotBeNullOrWhiteSpace();

                bridge.DisposeJob(jobId).Should().BeTrue();
            }
            finally
            {
                DeleteDir(root);
            }
        }

        [TestMethod]
        public void StreamingSingleTile_WritesIntoRegisteredBuffer_AndTracksConsumption()
        {
            string root = CreateTempDir();
            IntPtr ptr = IntPtr.Zero;
            try
            {
                string demPath = Path.Combine(root, "dem.tif");
                string horizonDir = Path.Combine(root, "horizons");
                Directory.CreateDirectory(horizonDir);
                WriteSyntheticDem(demPath, 128, 128);
                WriteSyntheticHorizonTile(Path.Combine(horizonDir, "horizon_00000_00000_000.bin"), -90f);

                var bridge = new LightmapArrayStreamingBridge();
                string jobId = bridge.StartLightmapArrayStreaming(new LightmapArrayStreamRequest(
                    ScenarioRootDir: root,
                    DemPath: demPath,
                    SurroundingDemPaths: new List<string>(),
                    HorizonDir: horizonDir,
                    StartUtc: new DateTime(2024, 01, 01, 0, 0, 0, DateTimeKind.Utc),
                    StopUtc: new DateTime(2024, 01, 01, 0, 0, 0, DateTimeKind.Utc),
                    TimeStepHours: 1.0,
                    ObserverElevationMeters: 0f,
                    UseSpiceSunVectors: false
                ));

                int expectedBytes = 128 * 128;
                ptr = Marshal.AllocHGlobal(expectedBytes);
                var sentinel = Enumerable.Repeat((byte)0xAA, expectedBytes).ToArray();
                Marshal.Copy(sentinel, 0, ptr, expectedBytes);

                bridge.RegisterOutputBuffer(jobId, 99, ptr.ToInt64(), expectedBytes - 1).Should().BeFalse();
                bridge.RegisterOutputBuffer(jobId, 1, ptr.ToInt64(), expectedBytes).Should().BeTrue();

                TileEnvelope? ready = null;
                var pollStopwatch = Stopwatch.StartNew();
                while (pollStopwatch.Elapsed < TimeSpan.FromSeconds(45))
                {
                    var item = bridge.TryGetNextTile(jobId, 500);
                    if (item == null)
                        continue;
                    if (item.State == StreamTileState.Ready)
                    {
                        ready = item;
                        break;
                    }
                    if (item.State == StreamTileState.Error)
                    {
                        Assert.Fail($"Streaming tile reported error: {item.Message}");
                    }
                }

                ready.Should().NotBeNull("at least one ready tile must be produced");
                ready!.BufferId.Should().Be(1);
                ready.TimeCount.Should().Be(1);
                ready.Width.Should().Be(128);
                ready.Height.Should().Be(128);

                var written = new byte[expectedBytes];
                Marshal.Copy(ptr, written, 0, expectedBytes);
                written.Should().Contain(b => b != 0xAA);

                bridge.ReleaseBuffer(jobId, ready.BufferId).Should().BeTrue();

                TileEnvelope? terminal = null;
                var terminalStopwatch = Stopwatch.StartNew();
                while (terminalStopwatch.Elapsed < TimeSpan.FromSeconds(20))
                {
                    var item = bridge.TryGetNextTile(jobId, 500);
                    if (item == null)
                        continue;
                    if (item.State == StreamTileState.Terminal)
                    {
                        terminal = item;
                        break;
                    }
                }

                terminal.Should().NotBeNull("stream should emit a terminal envelope");
                var finalStatus = WaitForTerminalState(bridge, jobId);
                finalStatus.State.Should().Be(StreamJobState.Completed);
                finalStatus.TilesProduced.Should().BeGreaterThanOrEqualTo(1);
                finalStatus.TilesConsumed.Should().BeGreaterThanOrEqualTo(1);

                bridge.DisposeJob(jobId).Should().BeTrue();
            }
            finally
            {
                if (ptr != IntPtr.Zero)
                    Marshal.FreeHGlobal(ptr);
                DeleteDir(root);
            }
        }

        [TestMethod]
        public void StreamingSingleTile_V2ChunkedSunFraction_ReassemblesToV1Output()
        {
            string root = CreateTempDir();
            IntPtr ptrV1 = IntPtr.Zero;
            IntPtr ptrV2 = IntPtr.Zero;
            try
            {
                string demPath = Path.Combine(root, "dem.tif");
                string horizonDir = Path.Combine(root, "horizons");
                Directory.CreateDirectory(horizonDir);
                WriteSyntheticDem(demPath, 128, 128);
                WriteSyntheticHorizonTile(Path.Combine(horizonDir, "horizon_00000_00000_000.bin"), -90f);

                var bridge = new LightmapArrayStreamingBridge();
                var startUtc = new DateTime(2024, 01, 01, 0, 0, 0, DateTimeKind.Utc);
                var stopUtc = new DateTime(2024, 01, 01, 2, 0, 0, DateTimeKind.Utc);

                string jobIdV1 = bridge.StartLightmapArrayStreaming(new LightmapArrayStreamRequest(
                    ScenarioRootDir: root,
                    DemPath: demPath,
                    SurroundingDemPaths: new List<string>(),
                    HorizonDir: horizonDir,
                    StartUtc: startUtc,
                    StopUtc: stopUtc,
                    TimeStepHours: 1.0,
                    ObserverElevationMeters: 0f,
                    UseSpiceSunVectors: false
                ));

                string jobIdV2 = bridge.StartLightmapArrayStreamingV2(new LightmapArrayStreamRequestV2(
                    ScenarioRootDir: root,
                    DemPath: demPath,
                    SurroundingDemPaths: new List<string>(),
                    HorizonDir: horizonDir,
                    StartUtc: startUtc,
                    StopUtc: stopUtc,
                    TimeStepHours: 1.0,
                    ObserverElevationMeters: 0f,
                    UseSpiceSunVectors: false,
                    Mode: LightmapStreamMode.SignalStream,
                    Signals: new List<TemporalSignalSpec> { new(TemporalSignalKind.SunFractionU8, 0, true) },
                    SignalLayout: new TemporalSignalStreamLayout(ChunkTimeCount: 2, InterleaveChannels: false)
                ));

                int timeCount = 3;
                int fullBytes = timeCount * 128 * 128;
                int chunkBytes = 2 * 1 * 128 * 128;
                ptrV1 = Marshal.AllocHGlobal(fullBytes);
                ptrV2 = Marshal.AllocHGlobal(chunkBytes);

                bridge.RegisterOutputBuffer(jobIdV1, 1, ptrV1.ToInt64(), fullBytes).Should().BeTrue();
                bridge.RegisterOutputBufferV2(jobIdV2, 2, ptrV2.ToInt64(), chunkBytes).Should().BeTrue();

                byte[] v1Output = WaitAndCopyFirstReadyV1(bridge, jobIdV1, ptrV1, fullBytes);
                byte[] v2Output = new byte[fullBytes];
                int chunksSeen = 0;
                bool v2TerminalSeen = false;
                var sw = Stopwatch.StartNew();
                while (sw.Elapsed < TimeSpan.FromSeconds(45))
                {
                    var item = bridge.TryGetNextTileV2(jobIdV2, 500);
                    if (item is null)
                        continue;
                    if (item.State == StreamTileState.Terminal)
                    {
                        v2TerminalSeen = true;
                        break;
                    }
                    if (item.State == StreamTileState.Error)
                        Assert.Fail($"V2 streaming tile reported error: {item.Message}");

                    item.ScalarType.Should().Be(StreamScalarType.UInt8);
                    item.Rank.Should().Be(4);
                    item.Dim1.Should().Be(1);
                    item.Dim2.Should().Be(128);
                    item.Dim3.Should().Be(128);

                    int actualBytes = item.Dim0 * item.Dim1 * item.Dim2 * item.Dim3;
                    var chunk = new byte[actualBytes];
                    Marshal.Copy(ptrV2, chunk, 0, actualBytes);
                    Buffer.BlockCopy(chunk, 0, v2Output, item.TimeOffset * 128 * 128, actualBytes);
                    bridge.ReleaseBuffer(jobIdV2, item.BufferId).Should().BeTrue();
                    chunksSeen++;
                }

                chunksSeen.Should().Be(2);
                v2TerminalSeen.Should().BeTrue();
                v2Output.Should().Equal(v1Output);

                bridge.DisposeJob(jobIdV1).Should().BeTrue();
                bridge.DisposeJob(jobIdV2).Should().BeTrue();
            }
            finally
            {
                if (ptrV1 != IntPtr.Zero)
                    Marshal.FreeHGlobal(ptrV1);
                if (ptrV2 != IntPtr.Zero)
                    Marshal.FreeHGlobal(ptrV2);
                DeleteDir(root);
            }
        }

        [TestMethod]
        public void StreamingSingleTile_V2Float32SunCenterMargin_EmitsFloat32Chunk()
        {
            string root = CreateTempDir();
            IntPtr ptr = IntPtr.Zero;
            try
            {
                string demPath = Path.Combine(root, "dem.tif");
                string horizonDir = Path.Combine(root, "horizons");
                Directory.CreateDirectory(horizonDir);
                WriteSyntheticDem(demPath, 128, 128);
                WriteSyntheticHorizonTile(Path.Combine(horizonDir, "horizon_00000_00000_000.bin"), -90f);

                var bridge = new LightmapArrayStreamingBridge();
                string jobId = bridge.StartLightmapArrayStreamingV2(new LightmapArrayStreamRequestV2(
                    ScenarioRootDir: root,
                    DemPath: demPath,
                    SurroundingDemPaths: new List<string>(),
                    HorizonDir: horizonDir,
                    StartUtc: new DateTime(2024, 01, 01, 0, 0, 0, DateTimeKind.Utc),
                    StopUtc: new DateTime(2024, 01, 01, 0, 0, 0, DateTimeKind.Utc),
                    TimeStepHours: 1.0,
                    ObserverElevationMeters: 0f,
                    UseSpiceSunVectors: false,
                    Mode: LightmapStreamMode.SignalStream,
                    Signals: new List<TemporalSignalSpec> { new(TemporalSignalKind.SunCenterMarginDegF32, 0, true) },
                    SignalLayout: new TemporalSignalStreamLayout(ChunkTimeCount: 2, InterleaveChannels: false)
                ));

                int byteLength = 2 * 1 * 128 * 128 * sizeof(float);
                ptr = Marshal.AllocHGlobal(byteLength);
                var sentinelBytes = Enumerable.Repeat((byte)0xCD, byteLength).ToArray();
                Marshal.Copy(sentinelBytes, 0, ptr, byteLength);

                bridge.RegisterOutputBufferV2(jobId, 7, ptr.ToInt64(), byteLength).Should().BeTrue();

                TileEnvelopeV2? ready = null;
                var sw = Stopwatch.StartNew();
                while (sw.Elapsed < TimeSpan.FromSeconds(45))
                {
                    var item = bridge.TryGetNextTileV2(jobId, 500);
                    if (item is null)
                        continue;
                    if (item.State == StreamTileState.Ready)
                    {
                        ready = item;
                        break;
                    }
                    if (item.State == StreamTileState.Error)
                        Assert.Fail($"V2 float32 streaming tile reported error: {item.Message}");
                }

                ready.Should().NotBeNull();
                ready!.ScalarType.Should().Be(StreamScalarType.Float32);
                ready.Rank.Should().Be(4);
                ready.Dim0.Should().Be(1);
                ready.Dim1.Should().Be(1);
                ready.Dim2.Should().Be(128);
                ready.Dim3.Should().Be(128);

                var values = new float[ready.Dim0 * ready.Dim1 * ready.Dim2 * ready.Dim3];
                Marshal.Copy(ptr, values, 0, values.Length);
                values.Should().OnlyContain(v => !float.IsNaN(v) && !float.IsInfinity(v));

                bridge.ReleaseBuffer(jobId, ready.BufferId).Should().BeTrue();

                TileEnvelopeV2? terminal = null;
                sw.Restart();
                while (sw.Elapsed < TimeSpan.FromSeconds(20))
                {
                    var item = bridge.TryGetNextTileV2(jobId, 500);
                    if (item is null)
                        continue;
                    if (item.State == StreamTileState.Terminal)
                    {
                        terminal = item;
                        break;
                    }
                }
                terminal.Should().NotBeNull();
                bridge.DisposeJob(jobId).Should().BeTrue();
            }
            finally
            {
                if (ptr != IntPtr.Zero)
                    Marshal.FreeHGlobal(ptr);
                DeleteDir(root);
            }
        }

        [TestMethod]
        public void StreamingSingleTile_V2NativeReduceAverageSunFraction_MatchesV1Mean()
        {
            string root = CreateTempDir();
            IntPtr ptrV1 = IntPtr.Zero;
            IntPtr ptrV2 = IntPtr.Zero;
            try
            {
                string demPath = Path.Combine(root, "dem.tif");
                string horizonDir = Path.Combine(root, "horizons");
                Directory.CreateDirectory(horizonDir);
                WriteSyntheticDem(demPath, 128, 128);
                WriteSyntheticHorizonTile(Path.Combine(horizonDir, "horizon_00000_00000_000.bin"), -90f);

                var bridge = new LightmapArrayStreamingBridge();
                var startUtc = new DateTime(2024, 01, 01, 0, 0, 0, DateTimeKind.Utc);
                var stopUtc = new DateTime(2024, 01, 01, 2, 0, 0, DateTimeKind.Utc);

                string jobIdV1 = bridge.StartLightmapArrayStreaming(new LightmapArrayStreamRequest(
                    ScenarioRootDir: root,
                    DemPath: demPath,
                    SurroundingDemPaths: new List<string>(),
                    HorizonDir: horizonDir,
                    StartUtc: startUtc,
                    StopUtc: stopUtc,
                    TimeStepHours: 1.0,
                    ObserverElevationMeters: 0f,
                    UseSpiceSunVectors: false
                ));

                string jobIdV2 = bridge.StartLightmapArrayStreamingV2(new LightmapArrayStreamRequestV2(
                    ScenarioRootDir: root,
                    DemPath: demPath,
                    SurroundingDemPaths: new List<string>(),
                    HorizonDir: horizonDir,
                    StartUtc: startUtc,
                    StopUtc: stopUtc,
                    TimeStepHours: 1.0,
                    ObserverElevationMeters: 0f,
                    UseSpiceSunVectors: false,
                    Mode: LightmapStreamMode.NativeReduce,
                    Reducers: new List<NativeReducerSpec>
                    {
                        new AverageSunFractionReducerSpec(OutputNormalized01: true, OutputType: ReducedTileOutputType.Float32)
                    }
                ));

                int timeCount = 3;
                int fullBytesV1 = timeCount * 128 * 128;
                int bytesV2 = 1 * 128 * 128 * sizeof(float);
                ptrV1 = Marshal.AllocHGlobal(fullBytesV1);
                ptrV2 = Marshal.AllocHGlobal(bytesV2);

                bridge.RegisterOutputBuffer(jobIdV1, 1, ptrV1.ToInt64(), fullBytesV1).Should().BeTrue();
                bridge.RegisterOutputBufferV2(jobIdV2, 2, ptrV2.ToInt64(), bytesV2).Should().BeTrue();

                byte[] v1Output = WaitAndCopyFirstReadyV1(bridge, jobIdV1, ptrV1, fullBytesV1);

                TileEnvelopeV2? v2Ready = null;
                var sw = Stopwatch.StartNew();
                while (sw.Elapsed < TimeSpan.FromSeconds(45))
                {
                    var item = bridge.TryGetNextTileV2(jobIdV2, 500);
                    if (item is null)
                        continue;
                    if (item.State == StreamTileState.Ready)
                    {
                        v2Ready = item;
                        break;
                    }
                    if (item.State == StreamTileState.Error)
                        Assert.Fail($"V2 native reduce reported error: {item.Message}");
                }

                v2Ready.Should().NotBeNull();
                v2Ready!.Rank.Should().Be(3);
                v2Ready.ScalarType.Should().Be(StreamScalarType.Float32);
                v2Ready.Dim0.Should().Be(1);
                v2Ready.Dim1.Should().Be(128);
                v2Ready.Dim2.Should().Be(128);

                var reduced = new float[128 * 128];
                Marshal.Copy(ptrV2, reduced, 0, reduced.Length);
                for (int pixel = 0; pixel < reduced.Length; pixel++)
                {
                    float expected = 0f;
                    for (int t = 0; t < timeCount; t++)
                        expected += v1Output[t * reduced.Length + pixel];
                    expected = (expected / timeCount) / 255f;
                    if (Math.Abs(reduced[pixel] - expected) > 1e-6f)
                        Assert.Fail($"Reducer mismatch at pixel {pixel}: got {reduced[pixel]}, expected {expected}");
                }

                bridge.ReleaseBuffer(jobIdV2, v2Ready.BufferId).Should().BeTrue();
                WaitForTerminalEnvelopeV2(bridge, jobIdV2).Should().NotBeNull();

                bridge.DisposeJob(jobIdV1).Should().BeTrue();
                bridge.DisposeJob(jobIdV2).Should().BeTrue();
            }
            finally
            {
                if (ptrV1 != IntPtr.Zero)
                    Marshal.FreeHGlobal(ptrV1);
                if (ptrV2 != IntPtr.Zero)
                    Marshal.FreeHGlobal(ptrV2);
                DeleteDir(root);
            }
        }

        [TestMethod]
        public void NativeReduceAverageSunFraction_WritesAtomicTiledCompressedGeoTiff()
        {
            string root = CreateTempDir();
            try
            {
                string demPath = Path.Combine(root, "dem.tif");
                string horizonDir = Path.Combine(root, "horizons");
                string outputPath = Path.Combine(root, "analysis", "average_sun_fraction.tif");
                Directory.CreateDirectory(horizonDir);
                WriteSyntheticDem(demPath, 256, 128);
                WriteSyntheticHorizonTile(Path.Combine(horizonDir, "horizon_00000_00000_000.bin"), -90f);

                var bridge = new LightmapArrayStreamingBridge();
                string jobId = bridge.StartLightmapArrayStreamingV2(new LightmapArrayStreamRequestV2(
                    ScenarioRootDir: root,
                    DemPath: demPath,
                    SurroundingDemPaths: new List<string>(),
                    HorizonDir: horizonDir,
                    StartUtc: new DateTime(2024, 01, 01, 0, 0, 0, DateTimeKind.Utc),
                    StopUtc: new DateTime(2024, 01, 01, 2, 0, 0, DateTimeKind.Utc),
                    TimeStepHours: 1.0,
                    ObserverElevationMeters: 0f,
                    UseSpiceSunVectors: false,
                    Mode: LightmapStreamMode.NativeReduce,
                    Reducers: new List<NativeReducerSpec>
                    {
                        new AverageSunFractionReducerSpec(OutputNormalized01: true)
                    },
                    RasterOutput: new NativeReduceRasterOutputSpec(outputPath)
                ));

                var status = WaitForTerminalState(bridge, jobId);
                status.State.Should().Be(StreamJobState.Completed);
                var result = bridge.GetNativeReduceRasterResult(jobId);
                result.Should().NotBeNull();
                result!.OutputPath.Should().Be(Path.GetFullPath(outputPath));
                result.TilesWritten.Should().Be(1);
                result.ValueMin.Should().BeInRange(0.0, 1.0);
                result.ValueMax.Should().BeInRange(0.0, 1.0);

                using var output = Gdal.Open(outputPath, Access.GA_ReadOnly);
                output.Should().NotBeNull();
                output.RasterXSize.Should().Be(256);
                output.RasterYSize.Should().Be(128);
                output.RasterCount.Should().Be(1);
                using var band = output.GetRasterBand(1);
                band.DataType.Should().Be(DataType.GDT_Float32);
                band.GetBlockSize(out int blockWidth, out int blockHeight);
                blockWidth.Should().Be(128);
                blockHeight.Should().Be(128);
                output.GetMetadataItem("COMPRESSION", "IMAGE_STRUCTURE").Should().Be("DEFLATE");
                band.GetNoDataValue(out double noData, out int hasNoData);
                hasNoData.Should().Be(1);
                noData.Should().Be(-9999.0);

                var pixels = new float[256 * 128];
                band.ReadRaster(0, 0, 256, 128, pixels, 256, 128, 0, 0);
                for (int row = 0; row < 128; row++)
                {
                    pixels.Skip(row * 256).Take(128).Should().OnlyContain(value => value >= 0f && value <= 1f);
                    pixels.Skip(row * 256 + 128).Take(128).Should().OnlyContain(value => value == -9999f);
                }

                bridge.DisposeJob(jobId).Should().BeTrue();
                Directory.EnumerateFiles(Path.GetDirectoryName(outputPath)!, ".*.tmp").Should().BeEmpty();
            }
            finally
            {
                DeleteDir(root);
            }
        }

        [TestMethod]
        public void NativeReduceRasterFailure_PreservesExistingOutputAndCleansScratch()
        {
            string root = CreateTempDir();
            try
            {
                string demPath = Path.Combine(root, "dem.tif");
                string horizonDir = Path.Combine(root, "horizons");
                string outputPath = Path.Combine(root, "analysis", "average_sun_fraction.tif");
                Directory.CreateDirectory(horizonDir);
                Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
                WriteSyntheticDem(demPath, 128, 128);
                File.WriteAllBytes(Path.Combine(horizonDir, "horizon_00000_00000_000.bin"), new byte[] { 1, 2, 3 });
                byte[] original = "existing-output"u8.ToArray();
                File.WriteAllBytes(outputPath, original);

                var bridge = new LightmapArrayStreamingBridge();
                string jobId = bridge.StartLightmapArrayStreamingV2(new LightmapArrayStreamRequestV2(
                    ScenarioRootDir: root,
                    DemPath: demPath,
                    SurroundingDemPaths: new List<string>(),
                    HorizonDir: horizonDir,
                    StartUtc: new DateTime(2024, 01, 01, 0, 0, 0, DateTimeKind.Utc),
                    StopUtc: new DateTime(2024, 01, 01, 0, 0, 0, DateTimeKind.Utc),
                    TimeStepHours: 1.0,
                    ObserverElevationMeters: 0f,
                    UseSpiceSunVectors: false,
                    Mode: LightmapStreamMode.NativeReduce,
                    Reducers: new List<NativeReducerSpec> { new AverageSunFractionReducerSpec() },
                    RasterOutput: new NativeReduceRasterOutputSpec(outputPath)
                ));

                var status = WaitForTerminalState(bridge, jobId);
                status.State.Should().Be(StreamJobState.Failed);
                bridge.GetNativeReduceRasterResult(jobId).Should().BeNull();
                File.ReadAllBytes(outputPath).Should().Equal(original);
                Directory.EnumerateFiles(Path.GetDirectoryName(outputPath)!, ".*.tmp").Should().BeEmpty();
                bridge.DisposeJob(jobId).Should().BeTrue();
            }
            finally
            {
                DeleteDir(root);
            }
        }

        [TestMethod]
        public void NativeReduceRaster_NoHorizonTiles_WritesAllNoDataOutput()
        {
            string root = CreateTempDir();
            try
            {
                string demPath = Path.Combine(root, "dem.tif");
                string horizonDir = Path.Combine(root, "horizons");
                string outputPath = Path.Combine(root, "analysis", "average_sun_fraction.tif");
                Directory.CreateDirectory(horizonDir);
                WriteSyntheticDem(demPath, 128, 128);

                var bridge = new LightmapArrayStreamingBridge();
                string jobId = bridge.StartLightmapArrayStreamingV2(new LightmapArrayStreamRequestV2(
                    ScenarioRootDir: root,
                    DemPath: demPath,
                    SurroundingDemPaths: new List<string>(),
                    HorizonDir: horizonDir,
                    StartUtc: new DateTime(2024, 01, 01, 0, 0, 0, DateTimeKind.Utc),
                    StopUtc: new DateTime(2024, 01, 01, 0, 0, 0, DateTimeKind.Utc),
                    TimeStepHours: 1.0,
                    ObserverElevationMeters: 0f,
                    UseSpiceSunVectors: false,
                    Mode: LightmapStreamMode.NativeReduce,
                    Reducers: new List<NativeReducerSpec> { new AverageSunFractionReducerSpec() },
                    RasterOutput: new NativeReduceRasterOutputSpec(outputPath)
                ));

                WaitForTerminalState(bridge, jobId).State.Should().Be(StreamJobState.Completed);
                bridge.GetNativeReduceRasterResult(jobId)!.TilesWritten.Should().Be(0);
                using var output = Gdal.Open(outputPath, Access.GA_ReadOnly);
                var pixels = new float[128 * 128];
                output.GetRasterBand(1).ReadRaster(0, 0, 128, 128, pixels, 128, 128, 0, 0);
                pixels.Should().OnlyContain(value => value == -9999f);
                bridge.DisposeJob(jobId).Should().BeTrue();
            }
            finally
            {
                DeleteDir(root);
            }
        }

        [TestMethod]
        public void NativeReduceCumulativeDuration_WritesDurationGeoTiff()
        {
            string root = CreateTempDir();
            try
            {
                string demPath = Path.Combine(root, "dem.tif");
                string horizonDir = Path.Combine(root, "horizons");
                string outputPath = Path.Combine(root, "analysis", "cumulative_duration.tif");
                Directory.CreateDirectory(horizonDir);
                WriteSyntheticDem(demPath, 256, 128);
                WriteSyntheticHorizonTile(Path.Combine(horizonDir, "horizon_00000_00000_000.bin"), -90f);

                var bridge = new LightmapArrayStreamingBridge();
                string jobId = bridge.StartLightmapArrayStreamingV2(new LightmapArrayStreamRequestV2(
                    ScenarioRootDir: root,
                    DemPath: demPath,
                    SurroundingDemPaths: new List<string>(),
                    HorizonDir: horizonDir,
                    StartUtc: new DateTime(2024, 01, 01, 0, 0, 0, DateTimeKind.Utc),
                    StopUtc: new DateTime(2024, 01, 01, 2, 0, 0, DateTimeKind.Utc),
                    TimeStepHours: 1.0,
                    ObserverElevationMeters: 0f,
                    UseSpiceSunVectors: false,
                    Mode: LightmapStreamMode.NativeReduce,
                    Reducers: new List<NativeReducerSpec>
                    {
                        new CumulativeDurationWhereReducerSpec(
                            SunPredicate: new SunFractionPredicateSpec(MinSunFractionU8: 0, GreaterThanOrEqual: true),
                            Unit: DurationOutputUnit.Hours)
                    },
                    RasterOutput: new NativeReduceRasterOutputSpec(outputPath)
                ));

                WaitForTerminalState(bridge, jobId).State.Should().Be(StreamJobState.Completed);
                var result = bridge.GetNativeReduceRasterResult(jobId);
                result.Should().NotBeNull();
                result!.TilesWritten.Should().Be(1);
                // Predicate holds for all three hourly samples, so covered pixels
                // accumulate exactly 3.0 hours and the value range is not [0, 1].
                result.ValueMin.Should().BeApproximately(3.0, 1e-6);
                result.ValueMax.Should().BeApproximately(3.0, 1e-6);

                using var output = Gdal.Open(outputPath, Access.GA_ReadOnly);
                output.Should().NotBeNull();
                output.RasterXSize.Should().Be(256);
                output.RasterYSize.Should().Be(128);
                using var band = output.GetRasterBand(1);
                band.DataType.Should().Be(DataType.GDT_Float32);
                band.GetBlockSize(out int blockWidth, out int blockHeight);
                blockWidth.Should().Be(128);
                blockHeight.Should().Be(128);
                output.GetMetadataItem("COMPRESSION", "IMAGE_STRUCTURE").Should().Be("DEFLATE");
                band.GetNoDataValue(out double noData, out int hasNoData);
                hasNoData.Should().Be(1);
                noData.Should().Be(-9999.0);

                var pixels = new float[256 * 128];
                band.ReadRaster(0, 0, 256, 128, pixels, 256, 128, 0, 0);
                for (int row = 0; row < 128; row++)
                {
                    pixels.Skip(row * 256).Take(128).Should().OnlyContain(value => value == 3f);
                    pixels.Skip(row * 256 + 128).Take(128).Should().OnlyContain(value => value == -9999f);
                }

                bridge.DisposeJob(jobId).Should().BeTrue();
                Directory.EnumerateFiles(Path.GetDirectoryName(outputPath)!, ".*.tmp").Should().BeEmpty();
            }
            finally
            {
                DeleteDir(root);
            }
        }

        private static LightmapArrayStreamStatus WaitForTerminalState(LightmapArrayStreamingBridge bridge, string jobId)
        {
            var sw = Stopwatch.StartNew();
            while (sw.Elapsed < TimeSpan.FromSeconds(30))
            {
                var status = bridge.GetJobStatus(jobId);
                if (status.State is StreamJobState.Completed or StreamJobState.Cancelled or StreamJobState.Failed)
                    return status;
                Thread.Sleep(50);
            }

            Assert.Fail("Timed out waiting for streaming job terminal state.");
            return bridge.GetJobStatus(jobId);
        }

        private static byte[] WaitAndCopyFirstReadyV1(LightmapArrayStreamingBridge bridge, string jobId, IntPtr ptr, int byteLength)
        {
            TileEnvelope? ready = null;
            var sw = Stopwatch.StartNew();
            while (sw.Elapsed < TimeSpan.FromSeconds(45))
            {
                var item = bridge.TryGetNextTile(jobId, 500);
                if (item is null)
                    continue;
                if (item.State == StreamTileState.Ready)
                {
                    ready = item;
                    break;
                }
                if (item.State == StreamTileState.Error)
                    Assert.Fail($"V1 streaming tile reported error: {item.Message}");
            }

            ready.Should().NotBeNull();
            var output = new byte[byteLength];
            Marshal.Copy(ptr, output, 0, byteLength);
            bridge.ReleaseBuffer(jobId, ready!.BufferId).Should().BeTrue();

            var terminalSeen = false;
            sw.Restart();
            while (sw.Elapsed < TimeSpan.FromSeconds(20))
            {
                var item = bridge.TryGetNextTile(jobId, 500);
                if (item is null)
                    continue;
                if (item.State == StreamTileState.Terminal)
                {
                    terminalSeen = true;
                    break;
                }
            }
            terminalSeen.Should().BeTrue();
            return output;
        }

        private static TileEnvelopeV2? WaitForTerminalEnvelopeV2(LightmapArrayStreamingBridge bridge, string jobId)
        {
            var sw = Stopwatch.StartNew();
            while (sw.Elapsed < TimeSpan.FromSeconds(20))
            {
                var item = bridge.TryGetNextTileV2(jobId, 500);
                if (item is null)
                    continue;
                if (item.State == StreamTileState.Terminal)
                    return item;
                if (item.State == StreamTileState.Error)
                    Assert.Fail($"V2 streaming tile reported error while waiting for terminal: {item.Message}");
            }
            return null;
        }

        private static string CreateTempDir()
        {
            string path = Path.Combine(Path.GetTempPath(), "LightmapArrayStreamingBridgeTests_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(path);
            return path;
        }

        private static void DeleteDir(string path)
        {
            if (!Directory.Exists(path))
                return;
            try
            {
                Directory.Delete(path, recursive: true);
            }
            catch
            {
                // Best-effort test cleanup.
            }
        }

        private static void WriteSyntheticDem(string path, int width, int height)
        {
            Driver? driver = Gdal.GetDriverByName("GTiff");
            driver.Should().NotBeNull();

            using var ds = driver!.Create(path, width, height, 1, DataType.GDT_Float32, null);
            ds.Should().NotBeNull();

            using var srs = new SpatialReference(null);
            int importResult = srs.ImportFromProj4(ElevationMap.LongLatProj);
            importResult.Should().Be(0);
            srs.ExportToWkt(out string? wkt, new string[] { });
            ds.SetProjection(wkt ?? string.Empty);
            ds.SetGeoTransform(new double[] { 0.0, 0.01, 0.0, 0.0, 0.0, -0.01 });

            var data = new float[width * height];
            for (int idx = 0; idx < data.Length; idx++)
                data[idx] = 1000f;
            ds.GetRasterBand(1).WriteRaster(0, 0, width, height, data, width, height, 0, 0);
            ds.FlushCache();
        }

        private static void WriteSyntheticHorizonTile(string path, float fillValue)
        {
            int total = 128 * 128 * LightmapGenerator.HorizonSamples;
            var values = new float[total];
            if (Math.Abs(fillValue) > 0f)
                Array.Fill(values, fillValue);
            HorizonFile.WriteHorizonFile(path, values);
        }
    }
}
