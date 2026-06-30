using moonlib.math;
using Serilog;
using System.Drawing;
using System.Drawing.Imaging;

namespace moonlib.horizon
{
    public class HorizonComparator
    {
        // Set N to 5 as requested
        private const int NumTestPoints = 10;

        public void CompareAndPlot(string outputDir)
        {
            if (!Directory.Exists(outputDir)) Directory.CreateDirectory(outputDir);

            Log.Information("Starting Comparison for {NumPoints} points...", NumTestPoints);

            // 1. Setup Reference Generator (Global Config)
            ReferenceHorizonGenerator.DEM_names = ReferenceHorizonGenerator.DEM_names
                .Where(n => !n.Equals("south", StringComparison.OrdinalIgnoreCase))
                .ToArray();

            var refGen = ReferenceHorizonGenerator.Singleton;
            var dems = ReferenceHorizonGenerator.LoadDEMs();

            if (dems.Count == 0)
            {
                Log.Error("No DEMs loaded.");
                return;
            }

            var mainDem = dems[0];
            int width = mainDem.Width;
            int height = mainDem.Height;
            Log.Information("DEM Size: {Width}x{Height}", width, height);

            // 2. Generate Test Points
            var rand = new Random(54321); // Fixed seed for reproducibility, was 12345
            var testPoints = new List<Point>();
            for (int i = 0; i < NumTestPoints; i++)
            {
                // Avoid extreme edges (100px padding)
                int x = rand.Next(100, width - 100);
                int y = rand.Next(100, height - 100);
                testPoints.Add(new Point(x, y));
            }

            testPoints = testPoints.Skip(1).Take(1).ToList();

            var observer_m = 2f; // Default observer height

            // 3. Prepare Table Header
            Log.Information("| {Col1,-15} | {Col2,-10} | {Col3,-10} | {Col4,-15} | {Col5,-15} |", "Point (X, Y)", "Max Diff", "MSE", "Ref Range", "QT Range");
            Log.Information("|{Sep1}|{Sep2}|{Sep3}|{Sep4}|{Sep5}|", new string('-', 17), new string('-', 12), new string('-', 12), new string('-', 17), new string('-', 17));

            // 4. Run Tests
            string qtOutputDir = Path.Combine(outputDir, "qt_out");
            if (Directory.Exists(qtOutputDir)) Directory.Delete(qtOutputDir, true);

            using (var qtGen = new QuadTreeHorizonGenerator(disableHierarchy: true, enableNearFieldReferenceMerge: true))
            {
                foreach (var point in testPoints)
                {
                    var origin = new PixelOrigin { X = point.X, Y = point.Y, Z = observer_m };
                    RunComparisonForPoint(origin, refGen, dems, qtGen, qtOutputDir, outputDir);
                }
            }
        }

        private void RunComparisonForPoint(PixelOrigin origin, ReferenceHorizonGenerator refGen, List<ElevationMap> dems, QuadTreeHorizonGenerator qtGen, string qtOutputDir, string mainOutputDir)
        {
            var point = new Point((int)origin.X, (int)origin.Y);
            var observer_m = origin.Z;
            int testX = point.X;
            int testY = point.Y;

            var observer_dec = (int)Math.Round(observer_m * 10);

            // --- A. Reference Generator ---
            ViewshedHorizon refResult;
            try
            {
                var (obs_lat, obs_lon) = dems[0].Point2LatLonDeg(origin.X, origin.Y);
                var latlon_origin = new LatLonOrigin { Latitude = obs_lat, Longitude = obs_lon, Z = observer_m };
                refResult = refGen.GenerateFromLatLon(latlon_origin, dems);
            }
            catch (Exception ex)
            {
                Log.Information("| {TestX,5}, {TestY,5}   | Error Ref: {ErrorMessage}", testX, testY, ex.Message);
                return;
            }

            // --- B. QuadTree Generator (with diagnostics callback) ---
            var diagBuffers = new Dictionary<HorizonBufferType, HorizonAngles>();
            qtGen.DiagnosticsCallback = (bufferType, angles) =>
            {
                if (!diagBuffers.ContainsKey(bufferType))
                    diagBuffers[bufferType] = angles;
            };
            try
            {
                qtGen.GenerateHorizons(qtOutputDir, dems, testX, testY, 1, 1, observer_m);
            }
            catch (Exception ex)
            {
                Log.Information("| {TestX,5}, {TestY,5}   | Error QT: {ErrorMessage}", testX, testY, ex.Message);
                qtGen.DiagnosticsCallback = null;
                return;
            }
            qtGen.DiagnosticsCallback = null;

            var refDegs = refResult.Elevations;

            // Extract series for plotting and comparison
            float[]? farFieldDegs = null;
            float[]? nearFieldDegs = null;
            var demSeries = new List<(string label, float[] data, Color color)>();
            foreach (var kvp in diagBuffers)
            {
                var degs = kvp.Value.Degrees;
                switch (kvp.Key)
                {
                    case HorizonBufferType.FarField:
                        farFieldDegs = degs;
                        break;
                    case HorizonBufferType.NearField:
                        nearFieldDegs = degs;
                        break;
                    case HorizonBufferType.DEM1:
                        demSeries.Add(("DEM1", degs, Color.Purple));
                        break;
                    case HorizonBufferType.DEM2:
                        demSeries.Add(("DEM2", degs, Color.Brown));
                        break;
                    case HorizonBufferType.DEM3:
                        demSeries.Add(("DEM3", degs, Color.DarkCyan));
                        break;
                    case HorizonBufferType.DEM4:
                        demSeries.Add(("DEM4", degs, Color.DarkMagenta));
                        break;
                    case HorizonBufferType.DEM5:
                        demSeries.Add(("DEM5", degs, Color.DarkOrange));
                        break;
                    case HorizonBufferType.DEMN:
                        demSeries.Add(("DEMN", degs, Color.Gray));
                        break;
                }
            }

            if (farFieldDegs == null)
            {
                Log.Information("| {TestX,5}, {TestY,5}   | Error: FarField buffer missing from diagnostics callback", testX, testY);
                return;
            }

            // --- C. Compare ---
            float maxDiff = 0;
            float mse = 0;
            int validCount = 0;

            float minRef = float.MaxValue, maxRef = float.MinValue;
            float minQt = float.MaxValue, maxQt = float.MinValue;

            for (int i = 0; i < 1440; i++)
            {
                var refindex = ReferenceHorizonGenerator.ConvertHorizonIndexToQuadTreeIndex(i);
                float r = refDegs[refindex];
                float q = farFieldDegs[i];

                bool rValid = IsValid(r);
                bool qValid = IsValid(q);

                if (rValid) { if (r < minRef) minRef = r; if (r > maxRef) maxRef = r; }
                if (qValid) { if (q < minQt) minQt = q; if (q > maxQt) maxQt = q; }

                if (!rValid && !qValid) continue;

                if (rValid != qValid)
                {
                    maxDiff = Math.Max(maxDiff, 999);
                    continue;
                }

                float diff = Math.Abs(r - q);
                if (diff > maxDiff) maxDiff = diff;
                mse += diff * diff;
                validCount++;
            }
            if (validCount > 0) mse /= validCount;

            string refRange = validCount > 0 ? $"[{minRef:F1}, {maxRef:F1}]" : "N/A";
            string qtRange = validCount > 0 ? $"[{minQt:F1}, {maxQt:F1}]" : "N/A";

            Log.Information("| {TestX,5}, {TestY,5}   | {MaxDiff,10:F4} | {MSE,10:F4} | {RefRange,-15} | {QtRange,-15} |", testX, testY, maxDiff, mse, refRange, qtRange);

            // --- D. Plot ---
            PlotHorizons(refDegs, farFieldDegs, Path.Combine(mainOutputDir, $"comparison_{testX}_{testY}.png"), demSeries, nearFieldDegs);
        }

        public static bool IsValid(float f)
        {
            return !float.IsNegativeInfinity(f) && !float.IsNaN(f) && f > -1.0e30f;
        }

        private static T[] LoadBinaryArray<T>(string path) where T : struct
        {
            var type = typeof(T);
            int typeSize = System.Runtime.InteropServices.Marshal.SizeOf<T>();
            byte[] bytes = File.ReadAllBytes(path);
            int length = bytes.Length / typeSize;
            T[] arr = new T[length];
            Buffer.BlockCopy(bytes, 0, arr, 0, bytes.Length);
            return arr;
        }

        public static void PlotHorizons(float[] refH, float[] qtH, string path, List<(string label, float[] data, Color color)>? extraSeries = null, float[]? nearOnly = null)
        {
            if (refH == null) throw new ArgumentNullException(nameof(refH));
            if (qtH == null) throw new ArgumentNullException(nameof(qtH));
            if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("Output path is required.", nameof(path));

            var plotSeries = new List<HorizonPlotSeries>
            {
                new HorizonPlotSeries("Reference", RemapReferenceSeries(refH), Color.Red, 2f),
                new HorizonPlotSeries("QuadTree", qtH, Color.Blue, 2f, 1f)
            };

            if (nearOnly != null)
                plotSeries.Add(new HorizonPlotSeries("Near Field", nearOnly, Color.DarkGoldenrod, 1f, -1f));

            if (extraSeries != null)
            {
                for (int s = 0; s < extraSeries.Count; s++)
                {
                    var (label, data, color) = extraSeries[s];
                    plotSeries.Add(new HorizonPlotSeries(label, data, color, 1f, s + 1));
                }
            }

            PlotHorizons(path, plotSeries);
        }

        public static void PlotHorizons(string path, IEnumerable<HorizonPlotSeries> series)
        {
            if (string.IsNullOrWhiteSpace(path))
                throw new ArgumentException("Output path is required.", nameof(path));
            if (series == null)
                throw new ArgumentNullException(nameof(series));

            var seriesList = series.ToList();
            if (seriesList.Count == 0)
            {
                Log.Error("No horizon series provided for plotting.");
                return;
            }

            int sampleCount = seriesList.Max(s => s.Data?.Length ?? 0);
            if (sampleCount < 2)
            {
                Log.Error("Insufficient data points to plot horizons.");
                return;
            }

            int w = 1600;
            int h = 600;
            using var bmp = new Bitmap(w, h);
            using var g = Graphics.FromImage(bmp);
            g.Clear(Color.White);

            var penGrid = new Pen(Color.LightGray, 1);

            float minEl = float.MaxValue;
            float maxEl = float.MinValue;
            void ConsiderRange(float[]? data)
            {
                if (data == null) return;
                foreach (var val in data)
                {
                    if (!IsValid(val)) continue;
                    if (val < minEl) minEl = val;
                    if (val > maxEl) maxEl = val;
                }
            }

            foreach (var s in seriesList)
                ConsiderRange(s.Data);

            if (minEl == float.MaxValue || maxEl == float.MinValue)
            {
                Log.Error("No valid data to plot.");
                return;
            }

            float range = maxEl - minEl;
            if (range < 1) range = 1;
            minEl -= range * 0.1f;
            maxEl += range * 0.1f;

            float ScaleX(int i) => (float)i / sampleCount * w;
            float ScaleY(float val)
            {
                if (!IsValid(val)) return h;
                return h - ((val - minEl) / (maxEl - minEl) * h);
            }

            using (var font = new Font(FontFamily.GenericSansSerif, 10))
            {
                int azimuthStep = Math.Max(sampleCount / 8, 1);
                for (int i = 0; i <= sampleCount; i += azimuthStep)
                {
                    float x = ScaleX(Math.Min(i, sampleCount));
                    g.DrawLine(penGrid, x, 0, x, h);
                    float degrees = sampleCount == 0 ? 0f : (float)i / sampleCount * 360f;
                    g.DrawString($"{degrees:F0}°", font, Brushes.Gray, x + 2, h - 20);
                }

                float step = range / 10.0f;
                float magnitude = (float)Math.Pow(10, Math.Floor(Math.Log10(step)));
                float normalizedStep = step / magnitude;
                if (normalizedStep < 2) step = 1 * magnitude;
                else if (normalizedStep < 5) step = 2 * magnitude;
                else step = 5 * magnitude;

                float startY = (float)Math.Floor(minEl / step) * step;
                for (float val = startY; val <= maxEl; val += step)
                {
                    float y = ScaleY(val);
                    g.DrawLine(penGrid, 0, y, w, y);
                    g.DrawString($"{val:F1}°", font, Brushes.Gray, 5, y - 15);
                }
            }

            var resources = seriesList.Select(s => GetPenAndBrush((s.Color, s.Label, s.PenWidth))).ToList();

            for (int s = 0; s < seriesList.Count; s++)
            {
                var data = seriesList[s].Data;
                if (data == null || data.Length < 2) continue;
                var pen = resources[s].pen;
                float offset = seriesList[s].YOffset;
                for (int i = 0; i < data.Length - 1; i++)
                {
                    if (IsValid(data[i]) && IsValid(data[i + 1]))
                        g.DrawLine(pen, ScaleX(i), ScaleY(data[i]) + offset, ScaleX(i + 1), ScaleY(data[i + 1]) + offset);
                }
            }

            using (var f = new Font(FontFamily.GenericSansSerif, 12))
            {
                float y = 10;
                for (int i = 0; i < seriesList.Count; i++)
                {
                    g.DrawString(seriesList[i].Label, f, resources[i].brush, 60, y);
                    y += 20;
                }
            }

            path = Path.GetFullPath(path);
            bmp.Save(path, ImageFormat.Png);
            Log.Information("Comparison plot saved to {OutputPath}", path);
        }

        private static float[] RemapReferenceSeries(float[] refH)
        {
            var result = new float[refH.Length];
            for (int i = 0; i < refH.Length; i++)
            {
                int sourceIndex = ReferenceHorizonGenerator.ConvertHorizonIndexToQuadTreeIndex(i);
                if (sourceIndex >= 0 && sourceIndex < refH.Length)
                    result[i] = refH[sourceIndex];
                else
                    result[i] = float.NegativeInfinity;
            }
            return result;
        }

        private static (Pen pen, Brush brush, string label) GetPenAndBrush((Color color, string label, float penWidth) colors)
        {
            var pen = new Pen(colors.color, colors.penWidth);
            var brush = new SolidBrush(colors.color);
            return (pen, brush, colors.label);
        }
    }

    public class HorizonPlotSeries
    {
        public HorizonPlotSeries(string label, float[] data, Color color, float penWidth = 1f, float yOffset = 0f)
        {
            Label = label ?? string.Empty;
            Data = data ?? throw new ArgumentNullException(nameof(data));
            Color = color;
            PenWidth = penWidth <= 0 ? 1f : penWidth;
            YOffset = yOffset;
        }

        public string Label { get; }
        public float[] Data { get; }
        public Color Color { get; set; }
        public float PenWidth { get; set; }
        public float YOffset { get; set; }
    }
}
