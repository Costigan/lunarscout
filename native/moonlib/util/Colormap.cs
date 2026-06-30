using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;

namespace moonlib.util
{
    public interface IColormap
    {
        string Name { get; }
        float Low { get; }
        float High { get; }
        float Range { get; }
        Color this[int index] { get; }
        Color this[float level] { get; }
    }

    public abstract class Colormap
    {
        public required string Name { get; set; }
        public float Low { get; set; } = 0f;
        public float High { get; set; } = 1f;
        public float Range => High - Low;

        float? _Step;
        public float Step { get => _Step.HasValue ? _Step.Value : (Range / 10f); set { _Step = value; } }

        public abstract Color this[int index] { get; }
        public abstract Color this[float value] { get; }

        public static float Clamp(float value, float low, float high) => Math.Max(low, Math.Min(high, value));
        public static int Clamp(int value, int low, int high) => Math.Max(low, Math.Min(high, value));

        public static IEnumerable<float> EnumerateRange(float low, float high, int count) => Enumerable.Range(0, count).Select(i => i / (float)(count - 1));

        public virtual Func<Colormap, int, List<(float, string)>> DefaultPrinter => (Colormap m, int steps) =>
            Enumerable.Range(0, steps).Select(i => ((float)i / (steps - 1), ((float)i / (steps - 1)).ToString())).ToList();

        public virtual Func<Colormap, int, List<(float, string)>> PercentPrinter => (Colormap m, int steps) =>
            Enumerable.Range(0, steps).Select(i =>
                {
                    var frac = (float)i / (steps - 1);
                    var percent = (int)Math.Round(frac * 100f);
                    return (frac, percent.ToString() + "%");
                }).ToList();

        public void LoadPalette(Bitmap bmp)
        {
            if (bmp.PixelFormat != PixelFormat.Format8bppIndexed)
                return;
            var p = bmp.Palette;
            LoadPalette(p);
            bmp.Palette = p;
        }

        public virtual void LoadPalette(ColorPalette p)
        {
            var (low, range) = (Low, Range);
            for (var i = 0; i < 256; i++)
            {
                var f = low + (i / 255f) * range;
                p.Entries[i] = this[f];
            }
        }

        public virtual Bitmap MakeLegend(int width = 60, int color_width = 30, int height = 512, int steps = 11, int y_border = 8, Func<Colormap, int, List<(float, string)>>? printer = null)
        {
            var bmp = new Bitmap(width, height, PixelFormat.Format24bppRgb);
            using (var g = Graphics.FromImage(bmp))
                DrawLegend(g, width, color_width, height, steps, y_border, printer);
            return bmp;
        }

        public virtual void DrawLegend(Graphics g, int width = 60, int color_width = 30, int height = 512, int steps = 11, int y_border = 8, Func<Colormap, int, List<(float, string)>>? printer = null)
        {
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
            printer = printer ?? DefaultPrinter;
            var inner_height = height - y_border * 2;
            // Draw background
            g.FillRectangle(Brushes.White, 0, 0, width, height);
            // Draw colors
            //for (var row = 0; row < inner_height; row++)
            //    using (var brush = new SolidBrush(this[row / (float)(inner_height - 1)]))
            //        g.FillRectangle(brush, 0, height - (row + y_border), color_width, 1);

            // Draw colors
            for (var row = 0; row < inner_height; row++)
            {
                var frac = row / (float)(inner_height - 1);
                using (var brush = new SolidBrush(this[frac]))
                    g.FillRectangle(brush, 0, height - (row + y_border), color_width, 1);
            }

            g.DrawRectangle(Pens.Black, 0, y_border, color_width, inner_height);
            // Draw steps
            var tick_len = 3;
            var step_x = color_width + 1;
            foreach (var (frac, label) in printer(this, steps))
            {
                var y = height - (y_border + (int)Math.Round(inner_height * (frac / 1f)));
                g.DrawLine(Pens.Black, 0, y, tick_len, y);
                g.DrawLine(Pens.Black, color_width - tick_len, y, color_width, y);
                g.DrawString(label, SystemFonts.SmallCaptionFont!, Brushes.Black, color_width + 2, y - 6);
            }
        }

        public virtual void DrawLegend(Graphics g, Point offset, int width = 60, int color_width = 30, int height = 512, int steps = 11, int y_border = 8, Func<Colormap, int, List<(float, string)>>? printer = null)
        {
            var gstate = g.Save();
            g.TranslateTransform(offset.X, offset.Y);
            DrawLegend(g, width, color_width, height, steps, y_border, printer);
            g.Restore(gstate);
        }

        public virtual Bitmap AddLegend(Bitmap source, string? title = null, int legend_width = 60, int color_width = 30, int steps = 11, Func<Colormap, int, List<(float, string)>>? printer = null)
        {
            int top_border = 12, bottom_border = 12, right_border = 12, left_border = 6;
            if (title != null)
                top_border += SystemFonts.SmallCaptionFont!.Height;
            var width = source.Width + left_border + right_border + legend_width;
            var height = source.Height + top_border + bottom_border;
            var target = new Bitmap(width, height, PixelFormat.Format32bppArgb);
            using (var g = Graphics.FromImage(target))
            {
                g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
                g.TextRenderingHint = System.Drawing.Text.TextRenderingHint.AntiAlias;

                // Draw the legend
                DrawLegend(g, new Point(left_border + source.Width + right_border, top_border), y_border: 0, height: source.Height, printer: printer);

                // Draw the title
                if (title != null)
                {
                    var sizef = g.MeasureString(title, SystemFonts.SmallCaptionFont!);
                    g.DrawString(title, SystemFonts.SmallCaptionFont!, Brushes.Black, left_border + source.Width / 2 - sizef.Width, 0);
                }

                g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.Default;
                g.DrawImageUnscaled(source, left_border, top_border);
            }
            return target;
        }
    }

    public class ListColormap : Colormap, IColormap
    {
        protected List<Color>? _colors;

        public virtual List<Color> Colors { get => _colors!; set { _colors = value; } }

        public int Count => Colors.Count;
        public override Color this[int index] => Colors[Clamp(index, 0, Count - 1)];
        public int IndexOf(float level) => (int)Math.Round(((Clamp(level, Low, High) - Low) / Range) * (Count - 1));
        public override Color this[float value] => Colors[IndexOf(value)];

        public ListColormap() { }
        public ListColormap(IEnumerable<Color> colors) { this.Colors = colors.ToList(); }

        public static IEnumerable<Color> EnumerateGrayscale() => Enumerable.Range(0, 256).Select(i => Color.FromArgb(i, i, i));
        public static IEnumerable<Color> EnumerateJet() => Enumerable.Range(0, 256).Select(i => ColormapData.ColorFromData(ColormapData.Jet, i / 256f));
        public static IEnumerable<Color> EnumerateHelix() => Enumerable.Range(0, 256).Select(i => ColormapData.ColorFromData(ColormapData.Helix, i / 256f));
        public static IEnumerable<Color> EnumerateHelix2() => Enumerable.Range(0, 256).Select(i => ColormapData.ColorFromData(ColormapData.Helix2, i / 256f));
        public static IEnumerable<Color> EnumerateHelix3() => Enumerable.Range(0, 256).Select(i => ColormapData.ColorFromData(ColormapData.Helix3, i / 256f));
        public static IEnumerable<Color> EnumerateHelix4() => Enumerable.Range(0, 256).Select(i => ColormapData.ColorFromData(ColormapData.Helix4, i / 256f));
        public static IEnumerable<Color> EnumerateHelix5() => Enumerable.Range(0, 256).Select(i => ColormapData.ColorFromData(ColormapData.Helix5, i / 256f));
        public static IEnumerable<Color> EnumerateHelix6() => Enumerable.Range(0, 256).Select(i => ColormapData.ColorFromData(ColormapData.Helix6, i / 256f));
        public static IEnumerable<Color> EnumerateViridis() => Enumerable.Range(0, 256).Select(i => ColormapData.ColorFromData(ColormapData.Viridis, i / 256f));

        public static IEnumerable<Color> EnumerateLightingLevelMap100()
        {
            for (var i = 0; i < 10; i++) yield return Color.Black;

            var color1 = ColorUtilities.ToColor(0, 0, .1);
            for (var i = 10; i < 20; i++) yield return color1;

            var color2 = ColorUtilities.ToColor(0, 0, .2);
            for (var i = 20; i < 30; i++) yield return color2;

            var color3 = ColorUtilities.ToColor(0, 0, .3);
            for (var i = 30; i < 40; i++) yield return color3;

            var color4 = ColorUtilities.ToColor(0, 0, .4);
            for (var i = 40; i < 50; i++) yield return color4;

            var red = ColorUtilities.ToColor(0, 1, .5);
            for (var i = 50; i < 60; i++) yield return red;

            var orange = ColorUtilities.ToColor(30, 1, .6);
            for (var i = 60; i < 70; i++) yield return orange;

            var yellow = ColorUtilities.ToColor(60, 1, .7);
            for (var i = 70; i < 80; i++) yield return yellow;

            var green = ColorUtilities.ToColor(120, 1, .8);
            for (var i = 80; i < 90; i++) yield return green;

            var blue = ColorUtilities.ToColor(240, 1, .9);
            for (var i = 90; i < 95; i++) yield return blue;

            for (var i = 95; i < 99; i++) yield return Color.FromArgb(244, 244, 244);

            yield return Color.White;
        }

        public static IEnumerable<Color> EnumerateLightingLevelMap256() => Map100To256(EnumerateLightingLevelMap100());

        static IEnumerable<Color> Map100To256(IEnumerable<Color> source)
        {
            var ary = source.ToArray();

            if (OldUtilities.debug_false)
            {
                var c1 = ary[0];
                for (var i = 0; i < 100; i++)
                {
                    var c2 = ary[i];
                    if (c1 != c2)
                    {
                        Console.WriteLine($"Step100 at {i}");
                        c1 = c2;
                    }
                }

                var ary2 = Enumerable.Range(0, 256).Select(i => ary[(int)(i * (99f / 255f))]).ToList();
                Console.WriteLine($"ary2.Count={ary2.Count}");
                c1 = ary2[0];
                for (var i = 0; i < 256; i++)
                {
                    var c2 = ary2[i];
                    if (c1 != c2)
                    {
                        Console.WriteLine($"Step256 at {i}");
                        c1 = c2;
                    }
                }
            }

            for (var i = 0; i < 256; i++)
                yield return ary[(int)(i * (99f / 255f))];
        }

        public void Print(TextWriter sw)
        {
            foreach (var c in Colors)
                Console.WriteLine($"{c.R} {c.G} {c.B}");
        }

        public static IEnumerable<Color> EnumerateLightingLevelMap10()
        {
            yield return Color.Black;

            var color1 = ColorUtilities.ToColor(0, 0, .1);
            yield return color1;

            var color2 = ColorUtilities.ToColor(0, 0, .2);
            yield return color2;

            var color3 = ColorUtilities.ToColor(0, 0, .3);
            yield return color3;

            var color4 = ColorUtilities.ToColor(0, 0, .4);
            yield return color4;

            var red = ColorUtilities.ToColor(0, 1, .5);
            yield return red;

            var orange = ColorUtilities.ToColor(30, 1, .6);
            yield return orange;

            var yellow = ColorUtilities.ToColor(60, 1, .7);
            yield return yellow;

            var green = ColorUtilities.ToColor(120, 1, .8);
            yield return green;

            var blue = ColorUtilities.ToColor(240, 1, .9);
            yield return blue;

            yield return Color.FromArgb(244, 244, 244);

            yield return Color.White;
        }

        public static IEnumerable<Color> EnumerateLongestNightv1() => EnumerateLightingLevelMap10().Reverse();
    }

    public class GrayscaleColormap : ListColormap { public GrayscaleColormap() : base(EnumerateGrayscale().ToList()) { } }
    public class JetColormap : ListColormap { public JetColormap() : base(EnumerateJet()) { } }
    public class ViridisColormap : ListColormap { public ViridisColormap() : base(EnumerateViridis()) { } }
    public class LightingColormap256 : ListColormap
    {
        public LightingColormap256() : base(EnumerateLightingLevelMap256().ToList()) { }
        public override Func<Colormap, int, List<(float, string)>> DefaultPrinter => PercentPrinter;
    }

    // See https://colorcet.com/
    public class CsvColormap : ListColormap
    {
        public static string Directory = @"csv/CETperceptual_csv_0_255";
        public required string Filename { get; set; }

        public override List<Color> Colors
        {
            get
            {
                if (_colors == null)
                    LoadColors(Filename);
                return _colors!;
            }
            set { _colors = value; }
        }

        public void LoadColors(string Filename)
        {
            if (Filename == null)
                throw new Exception("Cannot load colors from a null filename");
            var path = GetPath();
            if (string.Empty.Equals(Path.GetExtension(path)))
                path += ".csv";
            _colors = new List<Color>(256);
            using (var sr = new StreamReader(path))
            {
                string line = null;
                while ((line = sr.ReadLine()) != null)
                {
                    var tokens = line.Split(',');
                    if (tokens.Length == 3)
                    {
                        if (!int.TryParse(tokens[0], out int red) ||
                            !int.TryParse(tokens[1], out int green) ||
                            !int.TryParse(tokens[2], out int blue))
                            Quit();
                        _colors.Add(Color.FromArgb(red, green, blue));
                    }
                    if (tokens.Length == 4)
                    {
                        if (!int.TryParse(tokens[0], out int alpha) ||
                            !int.TryParse(tokens[1], out int red) ||
                            !int.TryParse(tokens[2], out int green) ||
                            !int.TryParse(tokens[3], out int blue))
                            Quit();
                        _colors.Add(Color.FromArgb(alpha, red, green, blue));
                    }
                    if (tokens.Length != 3 && tokens.Length != 4)
                        Quit();
                }
                if (_colors.Count != 256)
                    Quit();
            }

            void Quit() => throw new Exception($"malformed colormap csv file: {path}");
        }

        public string GetPath()
        {
            if (!string.Empty.Equals(Path.GetDirectoryName(Filename)))
                return Filename;
            return Path.Combine(Directory, Filename);
        }
    }

    public class ColormapFromPalette : ListColormap
    {
        public ColormapFromPalette(ColorPalette p)
        {
            Debug.Assert(p != null);
            _colors = new List<Color>(p.Entries.Length);
            for (var i = 0; i < p.Entries.Length; i++)
                _colors.Add(p.Entries[i]);
        }
    }

    public class IceStabilityDepthColormap : ListColormap
    {
        public IceStabilityDepthColormap() : base(EnumerateJet())
        {
            Colors[0] = Color.Black;
            for (var i = 250; i < 256; i++)
                Colors[i] = Color.White;
        }
    }

    public class IceStabilityDepthZoneColormap : ListColormap
    {
        public IceStabilityDepthZoneColormap() : base(Enumerable.Empty<Color>())
        {
            var red = Color.Red; // Color.FromArgb(255, 128, 128);
            var green = Color.Green; // Color.FromArgb(179, 255, 179);
            var yellow = Color.Yellow; // Color.FromArgb(255, 255, 179);
            var transparent = Color.FromArgb(0, 0, 0, 0);
            for (var i = 0; i < 256; i++)
                Colors.Add(Color.Black);
            Colors[0] = red;
            for (var i = 1; i <= 50; i++)
                Colors[i] = green;
            for (var i = 51; i <= 100; i++)
                Colors[i] = yellow;
            for (var i = 101; i < 256; i++)
                Colors[i] = transparent;
        }
    }

    public class RampColormap : ListColormap
    {
        public RampColormap(byte low_value, byte high_value, int low_index, int high_index) : base(EnumerateRamp(low_value, high_value, low_index, high_index)){}

        static IEnumerable<Color> EnumerateRamp(byte low_value, byte high_value, int low_index, int high_index)
        {
            for (var i = 0; i <= low_index; i++)
                yield return Color.FromArgb(low_value, low_value, low_value);
            var a = (high_value - low_value) / (float)(high_index - low_index);
            for (var i = low_index + 1; i <= high_index; i++)
            {
                var v = (byte)((i - low_index) * a + low_value);
                yield return Color.FromArgb(v, v, v);
            }
        }
}

    public class LightingColormap100 : ListColormap
    {
        public LightingColormap100() : base(EnumerateLightingLevelMap100().ToList()) { }
        public override Func<Colormap, int, List<(float, string)>> DefaultPrinter => PercentPrinter;
    }

    public class LongestNightColormapV1 : ListColormap {
        public LongestNightColormapV1() : base(Colors().ToList()) { }
        public override void LoadPalette(ColorPalette p)
        {
            for (var i = 0; i < base.Colors.Count; i++)
                p.Entries[i] = base.Colors[i];
        }

        public new static IEnumerable<Color> Colors()
        {   // 4 hour steps; white is 0 hours (longest night)
            for (var i = 0; i <= 6; i++) yield return Color.Yellow; // <1 day
            for (var i = 7; i <= 12; i++) yield return Color.Orange;  // <2 days
            for (var i = 13; i <= 18; i++) yield return Color.Red; // <3 days
            for (var i = 19; i <= 24; i++) yield return Color.Brown;  // <4 days
            for (var i = 25; i < 256; i++)
            {
                var frac = 0.5d - 0.5d * (i - 25d) / (256d - 25d);
                yield return ColorUtilities.ToColor(0, 0, frac);
            }
        }

        public override Func<Colormap, int, List<(float, string)>> DefaultPrinter => (Colormap m, int steps) =>
            Enumerable.Range(0, 256 / 6).Select(i =>
                {
                    var hours = i * 6 * 4;
                    var days = hours / 24;
                    var frac = (float)hours / (256*4);
                    return (frac, days.ToString());
                }).ToList();
    }

    public class Prime1MissionDurationColormap : ListColormap
    {
        public Prime1MissionDurationColormap() : base(Colors().ToList()) { }
        public override void LoadPalette(ColorPalette p)
        {
            for (var i = 0; i < base.Colors.Count; i++)
                p.Entries[i] = base.Colors[i];
        }

        public new static IEnumerable<Color> Colors()
        {   // 1 day steps
            if (OldUtilities.debug_false)
            {
                for (var i = 0; i <= 3; i++) yield return Color.FromArgb(0, Color.Brown);    // <3 days
                for (var i = 4; i <= 4; i++) yield return Color.Brown;     // <4 days
                for (var i = 5; i <= 5; i++) yield return Color.Red;  // <5 days
                for (var i = 6; i <= 6; i++) yield return Color.Orange;  // <6 days
                for (var i = 7; i <= 10; i++) yield return Color.Gold;
                for (var i = 11; i <= 15; i++) yield return Color.Green;
                for (var i = 16; i < 256; i++) yield return Color.LimeGreen;
            }
            else
            {
                for (var i = 0; i <= 9; i++) yield return Color.FromArgb(110 + 2 * i, 100 + 2 * i, 100 + 2 * i);
                for (var i = 10; i <= 10; i++) yield return Color.FromArgb(68, 191, 135);
                for (var i = 11; i <= 11; i++) yield return Color.FromArgb(251, 210, 88);
                for (var i = 12; i <= 12; i++) yield return Color.DodgerBlue;
                for (var i = 13; i <= 13; i++) yield return Color.FromArgb(255, 255, 0);
                for (var i = 14; i < 256; i++) yield return Color.FromArgb(255, 0, 255);
            }
        }

        public override Func<Colormap, int, List<(float, string)>> DefaultPrinter => (Colormap m, int steps) =>
            Enumerable.Range(0, 256 / 6).Select(i =>
            {
                var hours = i * 6 * 4;
                var days = hours / 24;
                var frac = (float)hours / (256 * 4);
                return (frac, days.ToString());
            }).ToList();
    }

    public class SafeHavenColormap : ListColormap
    {
        public SafeHavenColormap() : base(Colors().ToList()) { }
        public override void LoadPalette(ColorPalette p)
        {
            for (var i = 0; i < base.Colors.Count; i++)
                p.Entries[i] = base.Colors[i];
        }

        public new static IEnumerable<Color> Colors()
        {   // 1 hour steps; 
            //yield return Color.Yellow;                              // 0 hours (longest night)
            for (var i = 1; i <= 24; i++) yield return Color.Yellow;  // <1 day
            for (var i = 25; i <= 48; i++) yield return Color.Gold; // <2 days
            for (var i = 49; i <= 70; i++) yield return Color.Orange;    // <3 days
            for (var i = 71; i < 256; i++)
            {
                //var frac = 0.5d - 0.5d * (i - 97d) / (256d - 97d);
                //yield return ColorUtilities.ToColor(0, 0, frac);
                yield return Color.FromArgb(0, 0, 0, 0);  // Color.FromArgb(0);
            }
        }

        public override Func<Colormap, int, List<(float, string)>> DefaultPrinter => (Colormap m, int steps) =>
            Enumerable.Range(0, 256 / 6).Select(i =>
            {
                var hours = i * 6;
                var days = hours / 24;
                var frac = (float)hours / 256f;
                return (frac, days.ToString());
            }).ToList();
    }

    public class LookupColormap : Colormap, IColormap
    {
        protected List<float> values = new List<float>();
        protected List<Color> colors = new List<Color>();
        public int Count => values.Count;
        public override Color this[int index] => colors[Clamp(index, 0, Count - 1)];
        public override Color this[float value]
        {
            get
            {
                var index = values.BinarySearch(value);
                return colors[index < 0 ? ~index : index];
            }
        }

        public LookupColormap(List<Color> colors) { this.colors = colors; values = EnumerateRange(0f, 1f, colors.Count).ToList(); }
        public LookupColormap(List<Color> colors, Func<float, float> func) { this.colors = colors; values = EnumerateRange(0f, 1f, colors.Count).Select(v => func(v)).ToList(); }
        public LookupColormap(List<Color> colors, List<float> values) { this.colors = colors; this.values = values; }
    }

    public static class ColormapData
    {
        public static Color ColorFromData(double[,] data, int i, int alpha = 255) => Color.FromArgb(alpha, (int)(255d * data[i, 0]), (int)(255d * data[i, 1]), (int)(255d * data[i, 2]));
        public static Color ColorFromData(double[,] data, float frac, int alpha = 255) => ColorFromData(data, Math.Min(255, (int)(frac * data.GetLength(0))), alpha);

        public static Color ColorFromData(float[,] data, int i, int alpha = 255) => Color.FromArgb(alpha, (int)(255f * data[i, 0]), (int)(255f * data[i, 1]), (int)(255f * data[i, 2]));
        public static Color ColorFromData(float[,] data, float frac, int alpha = 255) => ColorFromData(data, Math.Min(255, (int)Math.Round(frac * data.GetLength(0))), alpha);

        public static double[,] Paige = {
            {1.0000000e+00, 1.0000000e+00, 1.0000000e+00},
            {5.3125000e-01, 0.0000000e+00, 0.0000000e+00},
            {5.6250000e-01, 0.0000000e+00, 0.0000000e+00},
            {5.9375000e-01, 0.0000000e+00, 0.0000000e+00},
            {6.2500000e-01, 0.0000000e+00, 0.0000000e+00},
            {6.5625000e-01, 0.0000000e+00, 0.0000000e+00},
            {6.8750000e-01, 0.0000000e+00, 0.0000000e+00},
            {7.1875000e-01, 0.0000000e+00, 0.0000000e+00},
            {7.5000000e-01, 0.0000000e+00, 0.0000000e+00},
            {7.8125000e-01, 0.0000000e+00, 0.0000000e+00},
            {8.1250000e-01, 0.0000000e+00, 0.0000000e+00},
            {8.4375000e-01, 0.0000000e+00, 0.0000000e+00},
            {8.7500000e-01, 0.0000000e+00, 0.0000000e+00},
            {9.0625000e-01, 0.0000000e+00, 0.0000000e+00},
            {9.3750000e-01, 0.0000000e+00, 0.0000000e+00},
            {9.6875000e-01, 0.0000000e+00, 0.0000000e+00},
            {1.0000000e+00, 0.0000000e+00, 0.0000000e+00},
            {1.0000000e+00, 3.1250000e-02, 0.0000000e+00},
            {1.0000000e+00, 6.2500000e-02, 0.0000000e+00},
            {1.0000000e+00, 9.3750000e-02, 0.0000000e+00},
            {1.0000000e+00, 1.2500000e-01, 0.0000000e+00},
            {1.0000000e+00, 1.5625000e-01, 0.0000000e+00},
            {1.0000000e+00, 1.8750000e-01, 0.0000000e+00},
            {1.0000000e+00, 2.1875000e-01, 0.0000000e+00},
            {1.0000000e+00, 2.5000000e-01, 0.0000000e+00},
            {1.0000000e+00, 2.8125000e-01, 0.0000000e+00},
            {1.0000000e+00, 3.1250000e-01, 0.0000000e+00},
            {1.0000000e+00, 3.4375000e-01, 0.0000000e+00},
            {1.0000000e+00, 3.7500000e-01, 0.0000000e+00},
            {1.0000000e+00, 4.0625000e-01, 0.0000000e+00},
            {1.0000000e+00, 4.3750000e-01, 0.0000000e+00},
            {1.0000000e+00, 4.6875000e-01, 0.0000000e+00},
            {1.0000000e+00, 5.0000000e-01, 0.0000000e+00},
            {1.0000000e+00, 5.3125000e-01, 0.0000000e+00},
            {1.0000000e+00, 5.6250000e-01, 0.0000000e+00},
            {1.0000000e+00, 5.9375000e-01, 0.0000000e+00},
            {1.0000000e+00, 6.2500000e-01, 0.0000000e+00},
            {1.0000000e+00, 6.5625000e-01, 0.0000000e+00},
            {1.0000000e+00, 6.8750000e-01, 0.0000000e+00},
            {1.0000000e+00, 7.1875000e-01, 0.0000000e+00},
            {1.0000000e+00, 7.5000000e-01, 0.0000000e+00},
            {1.0000000e+00, 7.8125000e-01, 0.0000000e+00},
            {1.0000000e+00, 8.1250000e-01, 0.0000000e+00},
            {1.0000000e+00, 8.4375000e-01, 0.0000000e+00},
            {1.0000000e+00, 8.7500000e-01, 0.0000000e+00},
            {1.0000000e+00, 9.0625000e-01, 0.0000000e+00},
            {1.0000000e+00, 9.3750000e-01, 0.0000000e+00},
            {1.0000000e+00, 9.6875000e-01, 0.0000000e+00},
            {1.0000000e+00, 1.0000000e+00, 0.0000000e+00},
            {9.6875000e-01, 1.0000000e+00, 3.1250000e-02},
            {9.3750000e-01, 1.0000000e+00, 6.2500000e-02},
            {9.0625000e-01, 1.0000000e+00, 9.3750000e-02},
            {8.7500000e-01, 1.0000000e+00, 1.2500000e-01},
            {8.4375000e-01, 1.0000000e+00, 1.5625000e-01},
            {8.1250000e-01, 1.0000000e+00, 1.8750000e-01},
            {7.8125000e-01, 1.0000000e+00, 2.1875000e-01},
            {7.5000000e-01, 1.0000000e+00, 2.5000000e-01},
            {7.1875000e-01, 1.0000000e+00, 2.8125000e-01},
            {6.8750000e-01, 1.0000000e+00, 3.1250000e-01},
            {6.5625000e-01, 1.0000000e+00, 3.4375000e-01},
            {6.2500000e-01, 1.0000000e+00, 3.7500000e-01},
            {5.9375000e-01, 1.0000000e+00, 4.0625000e-01},
            {5.6250000e-01, 1.0000000e+00, 4.3750000e-01},
            {5.3125000e-01, 1.0000000e+00, 4.6875000e-01},
            {5.0000000e-01, 1.0000000e+00, 5.0000000e-01},
            {4.6875000e-01, 1.0000000e+00, 5.3125000e-01},
            {4.3750000e-01, 1.0000000e+00, 5.6250000e-01},
            {4.0625000e-01, 1.0000000e+00, 5.9375000e-01},
            {3.7500000e-01, 1.0000000e+00, 6.2500000e-01},
            {3.4375000e-01, 1.0000000e+00, 6.5625000e-01},
            {3.1250000e-01, 1.0000000e+00, 6.8750000e-01},
            {2.8125000e-01, 1.0000000e+00, 7.1875000e-01},
            {2.5000000e-01, 1.0000000e+00, 7.5000000e-01},
            {2.1875000e-01, 1.0000000e+00, 7.8125000e-01},
            {1.8750000e-01, 1.0000000e+00, 8.1250000e-01},
            {1.5625000e-01, 1.0000000e+00, 8.4375000e-01},
            {1.2500000e-01, 1.0000000e+00, 8.7500000e-01},
            {9.3750000e-02, 1.0000000e+00, 9.0625000e-01},
            {6.2500000e-02, 1.0000000e+00, 9.3750000e-01},
            {3.1250000e-02, 1.0000000e+00, 9.6875000e-01},
            {0.0000000e+00, 1.0000000e+00, 1.0000000e+00},
            {0.0000000e+00, 9.6875000e-01, 1.0000000e+00},
            {0.0000000e+00, 9.3750000e-01, 1.0000000e+00},
            {0.0000000e+00, 9.0625000e-01, 1.0000000e+00},
            {0.0000000e+00, 8.7500000e-01, 1.0000000e+00},
            {0.0000000e+00, 8.4375000e-01, 1.0000000e+00},
            {0.0000000e+00, 8.1250000e-01, 1.0000000e+00},
            {0.0000000e+00, 7.8125000e-01, 1.0000000e+00},
            {0.0000000e+00, 7.5000000e-01, 1.0000000e+00},
            {0.0000000e+00, 7.1875000e-01, 1.0000000e+00},
            {0.0000000e+00, 6.8750000e-01, 1.0000000e+00},
            {0.0000000e+00, 6.5625000e-01, 1.0000000e+00},
            {0.0000000e+00, 6.2500000e-01, 1.0000000e+00},
            {0.0000000e+00, 5.9375000e-01, 1.0000000e+00},
            {0.0000000e+00, 5.6250000e-01, 1.0000000e+00},
            {0.0000000e+00, 5.3125000e-01, 1.0000000e+00},
            {0.0000000e+00, 5.0000000e-01, 1.0000000e+00},
            {0.0000000e+00, 4.6875000e-01, 1.0000000e+00},
            {0.0000000e+00, 4.3750000e-01, 1.0000000e+00},
            {0.0000000e+00, 4.0625000e-01, 1.0000000e+00},
            {0.0000000e+00, 3.7500000e-01, 1.0000000e+00},
            {0.0000000e+00, 3.4375000e-01, 1.0000000e+00},
            {0.0000000e+00, 3.1250000e-01, 1.0000000e+00},
            {0.0000000e+00, 2.8125000e-01, 1.0000000e+00},
            {0.0000000e+00, 2.5000000e-01, 1.0000000e+00},
            {0.0000000e+00, 2.1875000e-01, 1.0000000e+00},
            {0.0000000e+00, 1.8750000e-01, 1.0000000e+00},
            {0.0000000e+00, 1.5625000e-01, 1.0000000e+00},
            {0.0000000e+00, 1.2500000e-01, 1.0000000e+00},
            {0.0000000e+00, 9.3750000e-02, 1.0000000e+00},
            {0.0000000e+00, 6.2500000e-02, 1.0000000e+00},
            {0.0000000e+00, 3.1250000e-02, 1.0000000e+00},
            {0.0000000e+00, 0.0000000e+00, 1.0000000e+00},
            {0.0000000e+00, 0.0000000e+00, 9.6875000e-01},
            {0.0000000e+00, 0.0000000e+00, 9.3750000e-01},
            {0.0000000e+00, 0.0000000e+00, 9.0625000e-01},
            {0.0000000e+00, 0.0000000e+00, 8.7500000e-01},
            {0.0000000e+00, 0.0000000e+00, 8.4375000e-01},
            {0.0000000e+00, 0.0000000e+00, 8.1250000e-01},
            {0.0000000e+00, 0.0000000e+00, 7.8125000e-01},
            {0.0000000e+00, 0.0000000e+00, 7.5000000e-01},
            {0.0000000e+00, 0.0000000e+00, 7.1875000e-01},
            {0.0000000e+00, 0.0000000e+00, 6.8750000e-01},
            {0.0000000e+00, 0.0000000e+00, 6.5625000e-01},
            {0.0000000e+00, 0.0000000e+00, 6.2500000e-01},
            {0.0000000e+00, 0.0000000e+00, 5.9375000e-01},
            {0.0000000e+00, 0.0000000e+00, 5.6250000e-01},
            {8.5000000e-01, 8.5000000e-01, 7.5000000e-01},
        };

        public static double[,] Jet = {
            {0.0000000e+00, 0.0000000e+00, 5.3125000e-01},
            {0.0000000e+00, 0.0000000e+00, 5.6250000e-01},
            {0.0000000e+00, 0.0000000e+00, 5.9375000e-01},
            {0.0000000e+00, 0.0000000e+00, 6.2500000e-01},
            {0.0000000e+00, 0.0000000e+00, 6.5625000e-01},
            {0.0000000e+00, 0.0000000e+00, 6.8750000e-01},
            {0.0000000e+00, 0.0000000e+00, 7.1875000e-01},
            {0.0000000e+00, 0.0000000e+00, 7.5000000e-01},
            {0.0000000e+00, 0.0000000e+00, 7.8125000e-01},
            {0.0000000e+00, 0.0000000e+00, 8.1250000e-01},
            {0.0000000e+00, 0.0000000e+00, 8.4375000e-01},
            {0.0000000e+00, 0.0000000e+00, 8.7500000e-01},
            {0.0000000e+00, 0.0000000e+00, 9.0625000e-01},
            {0.0000000e+00, 0.0000000e+00, 9.3750000e-01},
            {0.0000000e+00, 0.0000000e+00, 9.6875000e-01},
            {0.0000000e+00, 0.0000000e+00, 1.0000000e+00},
            {0.0000000e+00, 3.1250000e-02, 1.0000000e+00},
            {0.0000000e+00, 6.2500000e-02, 1.0000000e+00},
            {0.0000000e+00, 9.3750000e-02, 1.0000000e+00},
            {0.0000000e+00, 1.2500000e-01, 1.0000000e+00},
            {0.0000000e+00, 1.5625000e-01, 1.0000000e+00},
            {0.0000000e+00, 1.8750000e-01, 1.0000000e+00},
            {0.0000000e+00, 2.1875000e-01, 1.0000000e+00},
            {0.0000000e+00, 2.5000000e-01, 1.0000000e+00},
            {0.0000000e+00, 2.8125000e-01, 1.0000000e+00},
            {0.0000000e+00, 3.1250000e-01, 1.0000000e+00},
            {0.0000000e+00, 3.4375000e-01, 1.0000000e+00},
            {0.0000000e+00, 3.7500000e-01, 1.0000000e+00},
            {0.0000000e+00, 4.0625000e-01, 1.0000000e+00},
            {0.0000000e+00, 4.3750000e-01, 1.0000000e+00},
            {0.0000000e+00, 4.6875000e-01, 1.0000000e+00},
            {0.0000000e+00, 5.0000000e-01, 1.0000000e+00},
            {0.0000000e+00, 5.3125000e-01, 1.0000000e+00},
            {0.0000000e+00, 5.6250000e-01, 1.0000000e+00},
            {0.0000000e+00, 5.9375000e-01, 1.0000000e+00},
            {0.0000000e+00, 6.2500000e-01, 1.0000000e+00},
            {0.0000000e+00, 6.5625000e-01, 1.0000000e+00},
            {0.0000000e+00, 6.8750000e-01, 1.0000000e+00},
            {0.0000000e+00, 7.1875000e-01, 1.0000000e+00},
            {0.0000000e+00, 7.5000000e-01, 1.0000000e+00},
            {0.0000000e+00, 7.8125000e-01, 1.0000000e+00},
            {0.0000000e+00, 8.1250000e-01, 1.0000000e+00},
            {0.0000000e+00, 8.4375000e-01, 1.0000000e+00},
            {0.0000000e+00, 8.7500000e-01, 1.0000000e+00},
            {0.0000000e+00, 9.0625000e-01, 1.0000000e+00},
            {0.0000000e+00, 9.3750000e-01, 1.0000000e+00},
            {0.0000000e+00, 9.6875000e-01, 1.0000000e+00},
            {0.0000000e+00, 1.0000000e+00, 1.0000000e+00},
            {3.1250000e-02, 1.0000000e+00, 9.6875000e-01},
            {6.2500000e-02, 1.0000000e+00, 9.3750000e-01},
            {9.3750000e-02, 1.0000000e+00, 9.0625000e-01},
            {1.2500000e-01, 1.0000000e+00, 8.7500000e-01},
            {1.5625000e-01, 1.0000000e+00, 8.4375000e-01},
            {1.8750000e-01, 1.0000000e+00, 8.1250000e-01},
            {2.1875000e-01, 1.0000000e+00, 7.8125000e-01},
            {2.5000000e-01, 1.0000000e+00, 7.5000000e-01},
            {2.8125000e-01, 1.0000000e+00, 7.1875000e-01},
            {3.1250000e-01, 1.0000000e+00, 6.8750000e-01},
            {3.4375000e-01, 1.0000000e+00, 6.5625000e-01},
            {3.7500000e-01, 1.0000000e+00, 6.2500000e-01},
            {4.0625000e-01, 1.0000000e+00, 5.9375000e-01},
            {4.3750000e-01, 1.0000000e+00, 5.6250000e-01},
            {4.6875000e-01, 1.0000000e+00, 5.3125000e-01},
            {5.0000000e-01, 1.0000000e+00, 5.0000000e-01},
            {5.3125000e-01, 1.0000000e+00, 4.6875000e-01},
            {5.6250000e-01, 1.0000000e+00, 4.3750000e-01},
            {5.9375000e-01, 1.0000000e+00, 4.0625000e-01},
            {6.2500000e-01, 1.0000000e+00, 3.7500000e-01},
            {6.5625000e-01, 1.0000000e+00, 3.4375000e-01},
            {6.8750000e-01, 1.0000000e+00, 3.1250000e-01},
            {7.1875000e-01, 1.0000000e+00, 2.8125000e-01},
            {7.5000000e-01, 1.0000000e+00, 2.5000000e-01},
            {7.8125000e-01, 1.0000000e+00, 2.1875000e-01},
            {8.1250000e-01, 1.0000000e+00, 1.8750000e-01},
            {8.4375000e-01, 1.0000000e+00, 1.5625000e-01},
            {8.7500000e-01, 1.0000000e+00, 1.2500000e-01},
            {9.0625000e-01, 1.0000000e+00, 9.3750000e-02},
            {9.3750000e-01, 1.0000000e+00, 6.2500000e-02},
            {9.6875000e-01, 1.0000000e+00, 3.1250000e-02},
            {1.0000000e+00, 1.0000000e+00, 0.0000000e+00},
            {1.0000000e+00, 9.6875000e-01, 0.0000000e+00},
            {1.0000000e+00, 9.3750000e-01, 0.0000000e+00},
            {1.0000000e+00, 9.0625000e-01, 0.0000000e+00},
            {1.0000000e+00, 8.7500000e-01, 0.0000000e+00},
            {1.0000000e+00, 8.4375000e-01, 0.0000000e+00},
            {1.0000000e+00, 8.1250000e-01, 0.0000000e+00},
            {1.0000000e+00, 7.8125000e-01, 0.0000000e+00},
            {1.0000000e+00, 7.5000000e-01, 0.0000000e+00},
            {1.0000000e+00, 7.1875000e-01, 0.0000000e+00},
            {1.0000000e+00, 6.8750000e-01, 0.0000000e+00},
            {1.0000000e+00, 6.5625000e-01, 0.0000000e+00},
            {1.0000000e+00, 6.2500000e-01, 0.0000000e+00},
            {1.0000000e+00, 5.9375000e-01, 0.0000000e+00},
            {1.0000000e+00, 5.6250000e-01, 0.0000000e+00},
            {1.0000000e+00, 5.3125000e-01, 0.0000000e+00},
            {1.0000000e+00, 5.0000000e-01, 0.0000000e+00},
            {1.0000000e+00, 4.6875000e-01, 0.0000000e+00},
            {1.0000000e+00, 4.3750000e-01, 0.0000000e+00},
            {1.0000000e+00, 4.0625000e-01, 0.0000000e+00},
            {1.0000000e+00, 3.7500000e-01, 0.0000000e+00},
            {1.0000000e+00, 3.4375000e-01, 0.0000000e+00},
            {1.0000000e+00, 3.1250000e-01, 0.0000000e+00},
            {1.0000000e+00, 2.8125000e-01, 0.0000000e+00},
            {1.0000000e+00, 2.5000000e-01, 0.0000000e+00},
            {1.0000000e+00, 2.1875000e-01, 0.0000000e+00},
            {1.0000000e+00, 1.8750000e-01, 0.0000000e+00},
            {1.0000000e+00, 1.5625000e-01, 0.0000000e+00},
            {1.0000000e+00, 1.2500000e-01, 0.0000000e+00},
            {1.0000000e+00, 9.3750000e-02, 0.0000000e+00},
            {1.0000000e+00, 6.2500000e-02, 0.0000000e+00},
            {1.0000000e+00, 3.1250000e-02, 0.0000000e+00},
            {1.0000000e+00, 0.0000000e+00, 0.0000000e+00},
            {9.6875000e-01, 0.0000000e+00, 0.0000000e+00},
            {9.3750000e-01, 0.0000000e+00, 0.0000000e+00},
            {9.0625000e-01, 0.0000000e+00, 0.0000000e+00},
            {8.7500000e-01, 0.0000000e+00, 0.0000000e+00},
            {8.4375000e-01, 0.0000000e+00, 0.0000000e+00},
            {8.1250000e-01, 0.0000000e+00, 0.0000000e+00},
            {7.8125000e-01, 0.0000000e+00, 0.0000000e+00},
            {7.5000000e-01, 0.0000000e+00, 0.0000000e+00},
            {7.1875000e-01, 0.0000000e+00, 0.0000000e+00},
            {6.8750000e-01, 0.0000000e+00, 0.0000000e+00},
            {6.5625000e-01, 0.0000000e+00, 0.0000000e+00},
            {6.2500000e-01, 0.0000000e+00, 0.0000000e+00},
            {5.9375000e-01, 0.0000000e+00, 0.0000000e+00},
            {5.6250000e-01, 0.0000000e+00, 0.0000000e+00},
            {5.3125000e-01, 0.0000000e+00, 0.0000000e+00},
            {5.0000000e-01, 0.0000000e+00, 0.0000000e+00},
        };

        //# Mon Apr 15 2019 19:20:37 GMT-0700 (Pacific Daylight Time)
        //# ---------------------------------------------
        //# R/G/B cubehelix colour scheme 
        //#
        //# see http://www.mrao.cam.ac.uk/~dag/CUBEHELIX/
        //#----------------------------------------------
        //# see Green (2011), BASI, 39, 289. 
        //#
        //# start............: 0.5
        //# rotations........: -1.5
        //# hue..............: 1.0
        //# gamma............: 1.0
        //# number of levels.: 255
        //#----------------------------------------------
        //# Dave Green: dag @ mrao.cam.ac.uk 
        //#----------------------------------------------
        //# faction and R/G/B values
        //#
        public static float[,] Helix = {{0.000f, 0.000f, 0.000f},
            {0.007f, 0.002f, 0.006f},
            {0.013f, 0.004f, 0.012f},
            {0.020f, 0.007f, 0.019f},
            {0.026f, 0.009f, 0.025f},
            {0.032f, 0.011f, 0.032f},
            {0.038f, 0.014f, 0.039f},
            {0.043f, 0.016f, 0.046f},
            {0.048f, 0.019f, 0.054f},
            {0.053f, 0.022f, 0.061f},
            {0.058f, 0.024f, 0.068f},
            {0.063f, 0.027f, 0.076f},
            {0.067f, 0.030f, 0.084f},
            {0.071f, 0.033f, 0.092f},
            {0.075f, 0.037f, 0.100f},
            {0.079f, 0.040f, 0.107f},
            {0.082f, 0.044f, 0.115f},
            {0.085f, 0.047f, 0.123f},
            {0.088f, 0.051f, 0.131f},
            {0.091f, 0.055f, 0.139f},
            {0.093f, 0.059f, 0.147f},
            {0.095f, 0.063f, 0.155f},
            {0.097f, 0.067f, 0.163f},
            {0.099f, 0.072f, 0.170f},
            {0.100f, 0.076f, 0.178f},
            {0.101f, 0.081f, 0.185f},
            {0.102f, 0.086f, 0.193f},
            {0.103f, 0.090f, 0.200f},
            {0.104f, 0.096f, 0.207f},
            {0.104f, 0.101f, 0.214f},
            {0.104f, 0.106f, 0.221f},
            {0.104f, 0.111f, 0.227f},
            {0.104f, 0.117f, 0.234f},
            {0.104f, 0.123f, 0.240f},
            {0.103f, 0.128f, 0.246f},
            {0.103f, 0.134f, 0.251f},
            {0.102f, 0.140f, 0.257f},
            {0.101f, 0.146f, 0.262f},
            {0.101f, 0.153f, 0.267f},
            {0.100f, 0.159f, 0.272f},
            {0.098f, 0.165f, 0.276f},
            {0.097f, 0.172f, 0.280f},
            {0.096f, 0.178f, 0.284f},
            {0.095f, 0.185f, 0.288f},
            {0.094f, 0.192f, 0.291f},
            {0.093f, 0.198f, 0.294f},
            {0.091f, 0.205f, 0.297f},
            {0.090f, 0.212f, 0.299f},
            {0.089f, 0.219f, 0.301f},
            {0.088f, 0.226f, 0.303f},
            {0.087f, 0.233f, 0.305f},
            {0.086f, 0.240f, 0.306f},
            {0.085f, 0.247f, 0.307f},
            {0.084f, 0.254f, 0.307f},
            {0.083f, 0.261f, 0.308f},
            {0.083f, 0.267f, 0.308f},
            {0.083f, 0.274f, 0.308f},
            {0.082f, 0.281f, 0.307f},
            {0.082f, 0.288f, 0.306f},
            {0.082f, 0.295f, 0.305f},
            {0.082f, 0.302f, 0.304f},
            {0.083f, 0.308f, 0.303f},
            {0.084f, 0.315f, 0.301f},
            {0.084f, 0.322f, 0.299f},
            {0.086f, 0.328f, 0.297f},
            {0.087f, 0.335f, 0.294f},
            {0.088f, 0.341f, 0.292f},
            {0.090f, 0.347f, 0.289f},
            {0.092f, 0.353f, 0.286f},
            {0.095f, 0.360f, 0.283f},
            {0.097f, 0.365f, 0.280f},
            {0.100f, 0.371f, 0.276f},
            {0.103f, 0.377f, 0.273f},
            {0.107f, 0.383f, 0.269f},
            {0.111f, 0.388f, 0.265f},
            {0.115f, 0.393f, 0.262f},
            {0.119f, 0.398f, 0.258f},
            {0.124f, 0.403f, 0.254f},
            {0.129f, 0.408f, 0.250f},
            {0.134f, 0.413f, 0.246f},
            {0.140f, 0.418f, 0.242f},
            {0.146f, 0.422f, 0.238f},
            {0.152f, 0.426f, 0.234f},
            {0.158f, 0.430f, 0.230f},
            {0.165f, 0.434f, 0.226f},
            {0.172f, 0.438f, 0.222f},
            {0.180f, 0.442f, 0.219f},
            {0.188f, 0.445f, 0.215f},
            {0.196f, 0.448f, 0.212f},
            {0.204f, 0.451f, 0.208f},
            {0.212f, 0.454f, 0.205f},
            {0.221f, 0.457f, 0.202f},
            {0.230f, 0.460f, 0.199f},
            {0.240f, 0.462f, 0.197f},
            {0.249f, 0.464f, 0.194f},
            {0.259f, 0.466f, 0.192f},
            {0.269f, 0.468f, 0.190f},
            {0.279f, 0.470f, 0.188f},
            {0.290f, 0.472f, 0.187f},
            {0.301f, 0.473f, 0.186f},
            {0.312f, 0.474f, 0.185f},
            {0.323f, 0.476f, 0.184f},
            {0.334f, 0.477f, 0.184f},
            {0.345f, 0.478f, 0.184f},
            {0.357f, 0.478f, 0.184f},
            {0.368f, 0.479f, 0.184f},
            {0.380f, 0.480f, 0.185f},
            {0.392f, 0.480f, 0.186f},
            {0.404f, 0.480f, 0.188f},
            {0.416f, 0.481f, 0.190f},
            {0.428f, 0.481f, 0.192f},
            {0.440f, 0.481f, 0.195f},
            {0.452f, 0.481f, 0.198f},
            {0.464f, 0.481f, 0.201f},
            {0.476f, 0.480f, 0.205f},
            {0.488f, 0.480f, 0.209f},
            {0.500f, 0.480f, 0.214f},
            {0.513f, 0.479f, 0.218f},
            {0.525f, 0.479f, 0.224f},
            {0.536f, 0.479f, 0.229f},
            {0.548f, 0.478f, 0.235f},
            {0.560f, 0.478f, 0.241f},
            {0.572f, 0.477f, 0.248f},
            {0.583f, 0.477f, 0.255f},
            {0.594f, 0.476f, 0.262f},
            {0.606f, 0.476f, 0.270f},
            {0.617f, 0.475f, 0.278f},
            {0.628f, 0.475f, 0.286f},
            {0.638f, 0.475f, 0.295f},
            {0.649f, 0.474f, 0.304f},
            {0.659f, 0.474f, 0.313f},
            {0.669f, 0.474f, 0.323f},
            {0.679f, 0.474f, 0.333f},
            {0.688f, 0.474f, 0.343f},
            {0.697f, 0.474f, 0.353f},
            {0.706f, 0.474f, 0.364f},
            {0.715f, 0.474f, 0.375f},
            {0.724f, 0.474f, 0.386f},
            {0.732f, 0.475f, 0.397f},
            {0.740f, 0.475f, 0.409f},
            {0.747f, 0.476f, 0.421f},
            {0.755f, 0.477f, 0.432f},
            {0.762f, 0.477f, 0.444f},
            {0.768f, 0.479f, 0.457f},
            {0.774f, 0.480f, 0.469f},
            {0.780f, 0.481f, 0.481f},
            {0.786f, 0.482f, 0.494f},
            {0.791f, 0.484f, 0.506f},
            {0.796f, 0.486f, 0.519f},
            {0.801f, 0.488f, 0.532f},
            {0.806f, 0.490f, 0.545f},
            {0.810f, 0.492f, 0.557f},
            {0.813f, 0.494f, 0.570f},
            {0.817f, 0.497f, 0.583f},
            {0.820f, 0.500f, 0.596f},
            {0.822f, 0.503f, 0.608f},
            {0.825f, 0.506f, 0.621f},
            {0.827f, 0.509f, 0.633f},
            {0.829f, 0.513f, 0.646f},
            {0.830f, 0.516f, 0.658f},
            {0.831f, 0.520f, 0.670f},
            {0.832f, 0.524f, 0.683f},
            {0.833f, 0.528f, 0.695f},
            {0.833f, 0.532f, 0.706f},
            {0.833f, 0.537f, 0.718f},
            {0.833f, 0.541f, 0.729f},
            {0.833f, 0.546f, 0.741f},
            {0.832f, 0.551f, 0.752f},
            {0.831f, 0.556f, 0.762f},
            {0.830f, 0.561f, 0.773f},
            {0.829f, 0.567f, 0.783f},
            {0.828f, 0.572f, 0.793f},
            {0.826f, 0.578f, 0.803f},
            {0.824f, 0.584f, 0.812f},
            {0.822f, 0.590f, 0.822f},
            {0.820f, 0.596f, 0.831f},
            {0.818f, 0.602f, 0.839f},
            {0.816f, 0.608f, 0.848f},
            {0.813f, 0.615f, 0.856f},
            {0.811f, 0.621f, 0.863f},
            {0.808f, 0.628f, 0.871f},
            {0.805f, 0.635f, 0.878f},
            {0.803f, 0.641f, 0.884f},
            {0.800f, 0.648f, 0.891f},
            {0.797f, 0.655f, 0.897f},
            {0.794f, 0.662f, 0.903f},
            {0.792f, 0.669f, 0.908f},
            {0.789f, 0.676f, 0.913f},
            {0.786f, 0.684f, 0.918f},
            {0.784f, 0.691f, 0.923f},
            {0.781f, 0.698f, 0.927f},
            {0.779f, 0.705f, 0.931f},
            {0.776f, 0.712f, 0.934f},
            {0.774f, 0.720f, 0.937f},
            {0.772f, 0.727f, 0.940f},
            {0.770f, 0.734f, 0.943f},
            {0.768f, 0.741f, 0.945f},
            {0.766f, 0.749f, 0.947f},
            {0.764f, 0.756f, 0.949f},
            {0.763f, 0.763f, 0.951f},
            {0.761f, 0.770f, 0.952f},
            {0.760f, 0.777f, 0.953f},
            {0.760f, 0.784f, 0.954f},
            {0.759f, 0.791f, 0.954f},
            {0.758f, 0.798f, 0.955f},
            {0.758f, 0.804f, 0.955f},
            {0.758f, 0.811f, 0.955f},
            {0.758f, 0.818f, 0.955f},
            {0.759f, 0.824f, 0.954f},
            {0.759f, 0.831f, 0.954f},
            {0.760f, 0.837f, 0.953f},
            {0.762f, 0.843f, 0.953f},
            {0.763f, 0.849f, 0.952f},
            {0.765f, 0.855f, 0.951f},
            {0.767f, 0.861f, 0.950f},
            {0.769f, 0.867f, 0.949f},
            {0.771f, 0.872f, 0.948f},
            {0.774f, 0.878f, 0.947f},
            {0.777f, 0.883f, 0.945f},
            {0.780f, 0.889f, 0.944f},
            {0.784f, 0.894f, 0.943f},
            {0.788f, 0.899f, 0.942f},
            {0.792f, 0.903f, 0.941f},
            {0.796f, 0.908f, 0.940f},
            {0.800f, 0.913f, 0.939f},
            {0.805f, 0.917f, 0.938f},
            {0.810f, 0.921f, 0.937f},
            {0.815f, 0.926f, 0.937f},
            {0.821f, 0.930f, 0.936f},
            {0.826f, 0.933f, 0.936f},
            {0.832f, 0.937f, 0.936f},
            {0.838f, 0.941f, 0.936f},
            {0.844f, 0.944f, 0.936f},
            {0.850f, 0.948f, 0.936f},
            {0.857f, 0.951f, 0.937f},
            {0.864f, 0.954f, 0.938f},
            {0.870f, 0.957f, 0.938f},
            {0.877f, 0.960f, 0.940f},
            {0.884f, 0.963f, 0.941f},
            {0.891f, 0.966f, 0.943f},
            {0.898f, 0.969f, 0.945f},
            {0.906f, 0.971f, 0.947f},
            {0.913f, 0.974f, 0.949f},
            {0.920f, 0.976f, 0.952f},
            {0.928f, 0.978f, 0.955f},
            {0.935f, 0.981f, 0.958f},
            {0.943f, 0.983f, 0.962f},
            {0.950f, 0.985f, 0.966f},
            {0.957f, 0.987f, 0.970f},
            {0.965f, 0.990f, 0.974f},
            {0.972f, 0.992f, 0.979f},
            {0.979f, 0.994f, 0.984f},
            {0.986f, 0.996f, 0.989f},
            {0.993f, 0.998f, 0.994f},
            {1.000f, 1.000f, 1.000f}};

        //# Mon Apr 15 2019 19:20:37 GMT-0700 (Pacific Daylight Time)
        //# ---------------------------------------------
        //# R/G/B cubehelix colour scheme 
        //#
        //# see http://www.mrao.cam.ac.uk/~dag/CUBEHELIX/
        //#----------------------------------------------
        //# see Green (2011), BASI, 39, 289. 
        //#
        //# start............: 0.5
        //# rotations........: -1.5
        //# hue..............: 1.0
        //# gamma............: 1.0
        //# number of levels.: 255
        //#----------------------------------------------
        //# Dave Green: dag @ mrao.cam.ac.uk 
        //#----------------------------------------------
        //# faction and R/G/B values
        public static float[,] Helix2 = {{0.000f, 0.000f, 0.000f},
            {0.004f, 0.003f, 0.008f},
            {0.007f, 0.007f, 0.016f},
            {0.010f, 0.011f, 0.023f},
            {0.013f, 0.014f, 0.031f},
            {0.016f, 0.018f, 0.039f},
            {0.019f, 0.022f, 0.046f},
            {0.021f, 0.026f, 0.054f},
            {0.024f, 0.030f, 0.061f},
            {0.026f, 0.034f, 0.068f},
            {0.028f, 0.038f, 0.076f},
            {0.030f, 0.043f, 0.083f},
            {0.032f, 0.047f, 0.090f},
            {0.034f, 0.051f, 0.097f},
            {0.036f, 0.056f, 0.103f},
            {0.037f, 0.061f, 0.110f},
            {0.039f, 0.065f, 0.117f},
            {0.040f, 0.070f, 0.123f},
            {0.041f, 0.075f, 0.129f},
            {0.042f, 0.080f, 0.136f},
            {0.043f, 0.085f, 0.142f},
            {0.044f, 0.090f, 0.148f},
            {0.045f, 0.095f, 0.153f},
            {0.046f, 0.101f, 0.159f},
            {0.047f, 0.106f, 0.164f},
            {0.047f, 0.111f, 0.170f},
            {0.048f, 0.117f, 0.175f},
            {0.048f, 0.122f, 0.180f},
            {0.049f, 0.128f, 0.185f},
            {0.049f, 0.133f, 0.189f},
            {0.050f, 0.139f, 0.194f},
            {0.050f, 0.144f, 0.198f},
            {0.050f, 0.150f, 0.202f},
            {0.051f, 0.156f, 0.206f},
            {0.051f, 0.162f, 0.210f},
            {0.051f, 0.168f, 0.214f},
            {0.052f, 0.173f, 0.217f},
            {0.052f, 0.179f, 0.221f},
            {0.052f, 0.185f, 0.224f},
            {0.053f, 0.191f, 0.227f},
            {0.053f, 0.197f, 0.229f},
            {0.054f, 0.203f, 0.232f},
            {0.054f, 0.209f, 0.234f},
            {0.055f, 0.215f, 0.237f},
            {0.055f, 0.221f, 0.239f},
            {0.056f, 0.227f, 0.241f},
            {0.056f, 0.233f, 0.242f},
            {0.057f, 0.239f, 0.244f},
            {0.058f, 0.245f, 0.246f},
            {0.059f, 0.251f, 0.247f},
            {0.060f, 0.257f, 0.248f},
            {0.061f, 0.263f, 0.249f},
            {0.062f, 0.269f, 0.250f},
            {0.063f, 0.275f, 0.250f},
            {0.065f, 0.281f, 0.251f},
            {0.066f, 0.287f, 0.251f},
            {0.068f, 0.292f, 0.252f},
            {0.069f, 0.298f, 0.252f},
            {0.071f, 0.304f, 0.252f},
            {0.073f, 0.310f, 0.252f},
            {0.075f, 0.315f, 0.252f},
            {0.077f, 0.321f, 0.251f},
            {0.079f, 0.327f, 0.251f},
            {0.082f, 0.332f, 0.250f},
            {0.084f, 0.338f, 0.250f},
            {0.087f, 0.343f, 0.249f},
            {0.090f, 0.349f, 0.248f},
            {0.093f, 0.354f, 0.247f},
            {0.096f, 0.359f, 0.246f},
            {0.099f, 0.364f, 0.245f},
            {0.102f, 0.370f, 0.244f},
            {0.106f, 0.375f, 0.243f},
            {0.110f, 0.380f, 0.242f},
            {0.113f, 0.385f, 0.240f},
            {0.117f, 0.390f, 0.239f},
            {0.122f, 0.394f, 0.238f},
            {0.126f, 0.399f, 0.236f},
            {0.130f, 0.404f, 0.235f},
            {0.135f, 0.408f, 0.233f},
            {0.140f, 0.413f, 0.232f},
            {0.145f, 0.417f, 0.230f},
            {0.150f, 0.422f, 0.229f},
            {0.155f, 0.426f, 0.228f},
            {0.161f, 0.430f, 0.226f},
            {0.166f, 0.434f, 0.225f},
            {0.172f, 0.438f, 0.223f},
            {0.178f, 0.442f, 0.222f},
            {0.184f, 0.446f, 0.221f},
            {0.190f, 0.450f, 0.219f},
            {0.197f, 0.453f, 0.218f},
            {0.203f, 0.457f, 0.217f},
            {0.210f, 0.460f, 0.216f},
            {0.216f, 0.464f, 0.215f},
            {0.223f, 0.467f, 0.214f},
            {0.231f, 0.470f, 0.213f},
            {0.238f, 0.474f, 0.212f},
            {0.245f, 0.477f, 0.211f},
            {0.253f, 0.480f, 0.210f},
            {0.260f, 0.483f, 0.210f},
            {0.268f, 0.485f, 0.209f},
            {0.276f, 0.488f, 0.209f},
            {0.284f, 0.491f, 0.209f},
            {0.292f, 0.493f, 0.208f},
            {0.300f, 0.496f, 0.208f},
            {0.308f, 0.498f, 0.208f},
            {0.317f, 0.501f, 0.209f},
            {0.325f, 0.503f, 0.209f},
            {0.334f, 0.505f, 0.210f},
            {0.343f, 0.507f, 0.210f},
            {0.352f, 0.509f, 0.211f},
            {0.360f, 0.511f, 0.212f},
            {0.369f, 0.513f, 0.213f},
            {0.378f, 0.515f, 0.214f},
            {0.387f, 0.517f, 0.216f},
            {0.397f, 0.519f, 0.217f},
            {0.406f, 0.520f, 0.219f},
            {0.415f, 0.522f, 0.221f},
            {0.424f, 0.523f, 0.223f},
            {0.434f, 0.525f, 0.225f},
            {0.443f, 0.526f, 0.228f},
            {0.452f, 0.528f, 0.230f},
            {0.462f, 0.529f, 0.233f},
            {0.471f, 0.530f, 0.236f},
            {0.481f, 0.532f, 0.239f},
            {0.490f, 0.533f, 0.242f},
            {0.500f, 0.534f, 0.246f},
            {0.509f, 0.535f, 0.250f},
            {0.519f, 0.537f, 0.253f},
            {0.528f, 0.538f, 0.257f},
            {0.537f, 0.539f, 0.262f},
            {0.547f, 0.540f, 0.266f},
            {0.556f, 0.541f, 0.271f},
            {0.566f, 0.542f, 0.275f},
            {0.575f, 0.543f, 0.280f},
            {0.584f, 0.544f, 0.285f},
            {0.593f, 0.545f, 0.291f},
            {0.602f, 0.546f, 0.296f},
            {0.611f, 0.547f, 0.302f},
            {0.621f, 0.548f, 0.308f},
            {0.629f, 0.549f, 0.314f},
            {0.638f, 0.550f, 0.320f},
            {0.647f, 0.551f, 0.326f},
            {0.656f, 0.552f, 0.332f},
            {0.664f, 0.553f, 0.339f},
            {0.673f, 0.554f, 0.346f},
            {0.681f, 0.555f, 0.353f},
            {0.690f, 0.556f, 0.360f},
            {0.698f, 0.558f, 0.367f},
            {0.706f, 0.559f, 0.374f},
            {0.714f, 0.560f, 0.382f},
            {0.722f, 0.561f, 0.390f},
            {0.730f, 0.563f, 0.397f},
            {0.737f, 0.564f, 0.405f},
            {0.745f, 0.565f, 0.413f},
            {0.752f, 0.567f, 0.421f},
            {0.759f, 0.568f, 0.430f},
            {0.766f, 0.570f, 0.438f},
            {0.773f, 0.571f, 0.446f},
            {0.780f, 0.573f, 0.455f},
            {0.787f, 0.575f, 0.464f},
            {0.793f, 0.576f, 0.472f},
            {0.800f, 0.578f, 0.481f},
            {0.806f, 0.580f, 0.490f},
            {0.812f, 0.582f, 0.499f},
            {0.818f, 0.584f, 0.508f},
            {0.823f, 0.586f, 0.517f},
            {0.829f, 0.588f, 0.526f},
            {0.834f, 0.590f, 0.535f},
            {0.840f, 0.593f, 0.545f},
            {0.845f, 0.595f, 0.554f},
            {0.850f, 0.597f, 0.563f},
            {0.855f, 0.600f, 0.573f},
            {0.859f, 0.602f, 0.582f},
            {0.864f, 0.605f, 0.591f},
            {0.868f, 0.608f, 0.601f},
            {0.872f, 0.611f, 0.610f},
            {0.876f, 0.614f, 0.619f},
            {0.880f, 0.617f, 0.629f},
            {0.884f, 0.620f, 0.638f},
            {0.887f, 0.623f, 0.647f},
            {0.890f, 0.626f, 0.656f},
            {0.894f, 0.629f, 0.666f},
            {0.897f, 0.633f, 0.675f},
            {0.900f, 0.636f, 0.684f},
            {0.902f, 0.640f, 0.693f},
            {0.905f, 0.643f, 0.702f},
            {0.908f, 0.647f, 0.711f},
            {0.910f, 0.651f, 0.720f},
            {0.912f, 0.655f, 0.728f},
            {0.914f, 0.659f, 0.737f},
            {0.916f, 0.663f, 0.746f},
            {0.918f, 0.667f, 0.754f},
            {0.920f, 0.671f, 0.763f},
            {0.921f, 0.676f, 0.771f},
            {0.923f, 0.680f, 0.779f},
            {0.924f, 0.684f, 0.787f},
            {0.925f, 0.689f, 0.795f},
            {0.927f, 0.694f, 0.803f},
            {0.928f, 0.698f, 0.811f},
            {0.929f, 0.703f, 0.818f},
            {0.930f, 0.708f, 0.826f},
            {0.930f, 0.713f, 0.833f},
            {0.931f, 0.718f, 0.840f},
            {0.932f, 0.723f, 0.847f},
            {0.932f, 0.728f, 0.854f},
            {0.933f, 0.733f, 0.861f},
            {0.933f, 0.738f, 0.868f},
            {0.934f, 0.743f, 0.874f},
            {0.934f, 0.749f, 0.880f},
            {0.935f, 0.754f, 0.886f},
            {0.935f, 0.760f, 0.892f},
            {0.935f, 0.765f, 0.898f},
            {0.935f, 0.771f, 0.904f},
            {0.936f, 0.776f, 0.909f},
            {0.936f, 0.782f, 0.914f},
            {0.936f, 0.787f, 0.919f},
            {0.936f, 0.793f, 0.924f},
            {0.937f, 0.799f, 0.929f},
            {0.937f, 0.804f, 0.934f},
            {0.937f, 0.810f, 0.938f},
            {0.937f, 0.816f, 0.942f},
            {0.938f, 0.821f, 0.946f},
            {0.938f, 0.827f, 0.950f},
            {0.938f, 0.833f, 0.954f},
            {0.939f, 0.839f, 0.958f},
            {0.939f, 0.845f, 0.961f},
            {0.940f, 0.850f, 0.964f},
            {0.941f, 0.856f, 0.967f},
            {0.941f, 0.862f, 0.970f},
            {0.942f, 0.868f, 0.973f},
            {0.943f, 0.873f, 0.975f},
            {0.944f, 0.879f, 0.978f},
            {0.945f, 0.885f, 0.980f},
            {0.946f, 0.891f, 0.982f},
            {0.947f, 0.896f, 0.984f},
            {0.949f, 0.902f, 0.986f},
            {0.950f, 0.908f, 0.988f},
            {0.952f, 0.913f, 0.989f},
            {0.953f, 0.919f, 0.991f},
            {0.955f, 0.924f, 0.992f},
            {0.957f, 0.930f, 0.993f},
            {0.959f, 0.935f, 0.994f},
            {0.961f, 0.940f, 0.995f},
            {0.964f, 0.946f, 0.996f},
            {0.966f, 0.951f, 0.997f},
            {0.969f, 0.956f, 0.997f},
            {0.972f, 0.961f, 0.998f},
            {0.975f, 0.966f, 0.998f},
            {0.978f, 0.971f, 0.999f},
            {0.981f, 0.976f, 0.999f},
            {0.984f, 0.981f, 0.999f},
            {0.988f, 0.986f, 1.000f},
            {0.992f, 0.991f, 1.000f},
            {0.996f, 0.995f, 1.000f},
            {1.000f, 1.000f, 1.000f}};

        //# Mon Apr 15 2019 20:12:38 GMT-0700 (Pacific Daylight Time)
        //# ---------------------------------------------
        //# R/G/B cubehelix colour scheme 
        //#
        //# see http://www.mrao.cam.ac.uk/~dag/CUBEHELIX/
        //#----------------------------------------------
        //# see Green (2011), BASI, 39, 289. 
        //#
        //# start............: 0.5
        //# rotations........: -1.5
        //# hue..............: 1.0
        //# gamma............: 0.4
        //# number of levels.: 255
        //#----------------------------------------------
        //# Dave Green: dag @ mrao.cam.ac.uk 
        //#----------------------------------------------
        //# faction and R/G/B values
        //#
        public static float[,] Helix3 ={{0.000f, 0.000f, 0.000f},
            {0.179f, 0.064f, 0.160f},
            {0.230f, 0.088f, 0.212f},
            {0.264f, 0.106f, 0.252f},
            {0.290f, 0.121f, 0.285f},
            {0.311f, 0.136f, 0.314f},
            {0.328f, 0.149f, 0.340f},
            {0.343f, 0.161f, 0.364f},
            {0.355f, 0.173f, 0.386f},
            {0.365f, 0.184f, 0.407f},
            {0.374f, 0.195f, 0.427f},
            {0.381f, 0.206f, 0.446f},
            {0.387f, 0.217f, 0.464f},
            {0.392f, 0.227f, 0.481f},
            {0.397f, 0.237f, 0.497f},
            {0.400f, 0.248f, 0.513f},
            {0.403f, 0.258f, 0.527f},
            {0.405f, 0.268f, 0.541f},
            {0.406f, 0.278f, 0.554f},
            {0.407f, 0.288f, 0.567f},
            {0.407f, 0.298f, 0.579f},
            {0.407f, 0.308f, 0.590f},
            {0.407f, 0.318f, 0.601f},
            {0.406f, 0.328f, 0.611f},
            {0.405f, 0.338f, 0.621f},
            {0.403f, 0.348f, 0.630f},
            {0.402f, 0.358f, 0.638f},
            {0.400f, 0.368f, 0.646f},
            {0.398f, 0.378f, 0.653f},
            {0.395f, 0.387f, 0.660f},
            {0.393f, 0.397f, 0.666f},
            {0.391f, 0.407f, 0.672f},
            {0.388f, 0.417f, 0.677f},
            {0.385f, 0.426f, 0.682f},
            {0.383f, 0.436f, 0.686f},
            {0.380f, 0.445f, 0.689f},
            {0.377f, 0.455f, 0.692f},
            {0.374f, 0.464f, 0.695f},
            {0.372f, 0.474f, 0.697f},
            {0.369f, 0.483f, 0.699f},
            {0.366f, 0.492f, 0.700f},
            {0.364f, 0.501f, 0.701f},
            {0.362f, 0.510f, 0.702f},
            {0.359f, 0.519f, 0.702f},
            {0.357f, 0.528f, 0.701f},
            {0.355f, 0.537f, 0.701f},
            {0.354f, 0.545f, 0.700f},
            {0.352f, 0.554f, 0.698f},
            {0.351f, 0.562f, 0.697f},
            {0.349f, 0.570f, 0.694f},
            {0.348f, 0.579f, 0.692f},
            {0.348f, 0.586f, 0.689f},
            {0.347f, 0.594f, 0.686f},
            {0.347f, 0.602f, 0.683f},
            {0.347f, 0.609f, 0.680f},
            {0.347f, 0.617f, 0.676f},
            {0.347f, 0.624f, 0.672f},
            {0.348f, 0.631f, 0.668f},
            {0.349f, 0.638f, 0.663f},
            {0.350f, 0.644f, 0.659f},
            {0.352f, 0.651f, 0.654f},
            {0.353f, 0.657f, 0.649f},
            {0.356f, 0.663f, 0.644f},
            {0.358f, 0.669f, 0.639f},
            {0.361f, 0.675f, 0.634f},
            {0.363f, 0.681f, 0.629f},
            {0.367f, 0.686f, 0.624f},
            {0.370f, 0.691f, 0.618f},
            {0.374f, 0.696f, 0.613f},
            {0.378f, 0.701f, 0.607f},
            {0.382f, 0.705f, 0.602f},
            {0.387f, 0.710f, 0.597f},
            {0.392f, 0.714f, 0.591f},
            {0.397f, 0.718f, 0.586f},
            {0.403f, 0.722f, 0.581f},
            {0.408f, 0.726f, 0.576f},
            {0.414f, 0.729f, 0.570f},
            {0.421f, 0.732f, 0.565f},
            {0.427f, 0.735f, 0.560f},
            {0.434f, 0.738f, 0.556f},
            {0.441f, 0.741f, 0.551f},
            {0.448f, 0.743f, 0.546f},
            {0.455f, 0.746f, 0.542f},
            {0.463f, 0.748f, 0.538f},
            {0.471f, 0.750f, 0.534f},
            {0.479f, 0.752f, 0.530f},
            {0.487f, 0.753f, 0.526f},
            {0.495f, 0.755f, 0.523f},
            {0.504f, 0.756f, 0.520f},
            {0.512f, 0.757f, 0.517f},
            {0.521f, 0.758f, 0.514f},
            {0.530f, 0.759f, 0.512f},
            {0.539f, 0.760f, 0.509f},
            {0.548f, 0.761f, 0.507f},
            {0.558f, 0.761f, 0.506f},
            {0.567f, 0.761f, 0.504f},
            {0.577f, 0.762f, 0.503f},
            {0.586f, 0.762f, 0.502f},
            {0.596f, 0.762f, 0.501f},
            {0.605f, 0.762f, 0.501f},
            {0.615f, 0.761f, 0.501f},
            {0.625f, 0.761f, 0.501f},
            {0.634f, 0.761f, 0.502f},
            {0.644f, 0.760f, 0.502f},
            {0.654f, 0.760f, 0.504f},
            {0.663f, 0.759f, 0.505f},
            {0.673f, 0.758f, 0.507f},
            {0.683f, 0.757f, 0.508f},
            {0.692f, 0.757f, 0.511f},
            {0.702f, 0.756f, 0.513f},
            {0.711f, 0.755f, 0.516f},
            {0.720f, 0.754f, 0.519f},
            {0.730f, 0.753f, 0.522f},
            {0.739f, 0.752f, 0.526f},
            {0.748f, 0.751f, 0.530f},
            {0.757f, 0.750f, 0.534f},
            {0.766f, 0.749f, 0.538f},
            {0.774f, 0.748f, 0.543f},
            {0.783f, 0.747f, 0.548f},
            {0.791f, 0.746f, 0.553f},
            {0.799f, 0.745f, 0.558f},
            {0.807f, 0.744f, 0.564f},
            {0.815f, 0.743f, 0.569f},
            {0.823f, 0.743f, 0.575f},
            {0.830f, 0.742f, 0.581f},
            {0.838f, 0.741f, 0.588f},
            {0.845f, 0.740f, 0.594f},
            {0.851f, 0.739f, 0.601f},
            {0.858f, 0.739f, 0.608f},
            {0.865f, 0.738f, 0.615f},
            {0.871f, 0.738f, 0.622f},
            {0.877f, 0.737f, 0.629f},
            {0.883f, 0.737f, 0.637f},
            {0.888f, 0.737f, 0.644f},
            {0.893f, 0.737f, 0.652f},
            {0.899f, 0.736f, 0.660f},
            {0.903f, 0.736f, 0.668f},
            {0.908f, 0.736f, 0.676f},
            {0.912f, 0.737f, 0.684f},
            {0.917f, 0.737f, 0.692f},
            {0.920f, 0.737f, 0.700f},
            {0.924f, 0.738f, 0.708f},
            {0.928f, 0.738f, 0.716f},
            {0.931f, 0.739f, 0.724f},
            {0.934f, 0.739f, 0.732f},
            {0.936f, 0.740f, 0.740f},
            {0.939f, 0.741f, 0.749f},
            {0.941f, 0.742f, 0.757f},
            {0.943f, 0.743f, 0.765f},
            {0.945f, 0.745f, 0.773f},
            {0.947f, 0.746f, 0.781f},
            {0.948f, 0.747f, 0.789f},
            {0.949f, 0.749f, 0.797f},
            {0.951f, 0.751f, 0.804f},
            {0.951f, 0.752f, 0.812f},
            {0.952f, 0.754f, 0.820f},
            {0.952f, 0.756f, 0.827f},
            {0.953f, 0.758f, 0.834f},
            {0.953f, 0.760f, 0.842f},
            {0.953f, 0.763f, 0.849f},
            {0.952f, 0.765f, 0.856f},
            {0.952f, 0.767f, 0.862f},
            {0.952f, 0.770f, 0.869f},
            {0.951f, 0.773f, 0.876f},
            {0.950f, 0.775f, 0.882f},
            {0.949f, 0.778f, 0.888f},
            {0.948f, 0.781f, 0.894f},
            {0.947f, 0.784f, 0.900f},
            {0.946f, 0.787f, 0.906f},
            {0.944f, 0.790f, 0.911f},
            {0.943f, 0.793f, 0.917f},
            {0.941f, 0.796f, 0.922f},
            {0.940f, 0.800f, 0.927f},
            {0.938f, 0.803f, 0.931f},
            {0.936f, 0.806f, 0.936f},
            {0.935f, 0.810f, 0.940f},
            {0.933f, 0.813f, 0.945f},
            {0.931f, 0.817f, 0.949f},
            {0.929f, 0.820f, 0.952f},
            {0.927f, 0.824f, 0.956f},
            {0.925f, 0.827f, 0.959f},
            {0.923f, 0.831f, 0.962f},
            {0.922f, 0.835f, 0.965f},
            {0.920f, 0.838f, 0.968f},
            {0.918f, 0.842f, 0.971f},
            {0.916f, 0.846f, 0.973f},
            {0.914f, 0.850f, 0.976f},
            {0.912f, 0.853f, 0.978f},
            {0.911f, 0.857f, 0.980f},
            {0.909f, 0.861f, 0.981f},
            {0.907f, 0.864f, 0.983f},
            {0.906f, 0.868f, 0.984f},
            {0.904f, 0.872f, 0.986f},
            {0.903f, 0.875f, 0.987f},
            {0.902f, 0.879f, 0.988f},
            {0.901f, 0.883f, 0.988f},
            {0.899f, 0.886f, 0.989f},
            {0.898f, 0.890f, 0.989f},
            {0.898f, 0.893f, 0.990f},
            {0.897f, 0.897f, 0.990f},
            {0.896f, 0.900f, 0.990f},
            {0.895f, 0.904f, 0.990f},
            {0.895f, 0.907f, 0.990f},
            {0.895f, 0.910f, 0.990f},
            {0.894f, 0.913f, 0.990f},
            {0.894f, 0.917f, 0.989f},
            {0.894f, 0.920f, 0.989f},
            {0.894f, 0.923f, 0.989f},
            {0.894f, 0.926f, 0.988f},
            {0.895f, 0.929f, 0.987f},
            {0.895f, 0.932f, 0.987f},
            {0.896f, 0.934f, 0.986f},
            {0.897f, 0.937f, 0.985f},
            {0.897f, 0.940f, 0.985f},
            {0.898f, 0.942f, 0.984f},
            {0.899f, 0.945f, 0.983f},
            {0.901f, 0.947f, 0.982f},
            {0.902f, 0.950f, 0.981f},
            {0.903f, 0.952f, 0.981f},
            {0.905f, 0.954f, 0.980f},
            {0.907f, 0.957f, 0.979f},
            {0.909f, 0.959f, 0.978f},
            {0.910f, 0.961f, 0.978f},
            {0.912f, 0.963f, 0.977f},
            {0.914f, 0.965f, 0.977f},
            {0.917f, 0.967f, 0.976f},
            {0.919f, 0.968f, 0.976f},
            {0.921f, 0.970f, 0.975f},
            {0.924f, 0.972f, 0.975f},
            {0.926f, 0.973f, 0.975f},
            {0.929f, 0.975f, 0.974f},
            {0.932f, 0.976f, 0.974f},
            {0.934f, 0.978f, 0.974f},
            {0.937f, 0.979f, 0.974f},
            {0.940f, 0.981f, 0.974f},
            {0.943f, 0.982f, 0.975f},
            {0.946f, 0.983f, 0.975f},
            {0.949f, 0.984f, 0.975f},
            {0.952f, 0.985f, 0.976f},
            {0.955f, 0.986f, 0.977f},
            {0.958f, 0.988f, 0.977f},
            {0.961f, 0.989f, 0.978f},
            {0.964f, 0.990f, 0.979f},
            {0.967f, 0.991f, 0.980f},
            {0.970f, 0.991f, 0.982f},
            {0.974f, 0.992f, 0.983f},
            {0.977f, 0.993f, 0.984f},
            {0.980f, 0.994f, 0.986f},
            {0.983f, 0.995f, 0.988f},
            {0.986f, 0.996f, 0.990f},
            {0.989f, 0.997f, 0.991f},
            {0.992f, 0.997f, 0.993f},
            {0.994f, 0.998f, 0.995f},
            {0.997f, 0.999f, 0.998f},
            {1.000f, 1.000f, 1.000f}};

        //# Mon Apr 15 2019 20:40:32 GMT-0700 (Pacific Daylight Time)
        //# ---------------------------------------------
        //# R/G/B cubehelix colour scheme 
        //#
        //# see http://www.mrao.cam.ac.uk/~dag/CUBEHELIX/
        //#----------------------------------------------
        //# see Green (2011), BASI, 39, 289. 
        //#
        //# start............: 0.5
        //# rotations........: 1.5
        //# hue..............: 1.0
        //# gamma............: 1.0
        //# number of levels.: 255
        //#----------------------------------------------
        //# Dave Green: dag @ mrao.cam.ac.uk 
        //#----------------------------------------------
        //# faction and R/G/B values
        //#
        public static float[,] Helix4 ={{0.000f, 0.000f, 0.000f},
            {0.007f, 0.002f, 0.006f},
            {0.014f, 0.004f, 0.011f},
            {0.021f, 0.006f, 0.016f},
            {0.028f, 0.008f, 0.021f},
            {0.035f, 0.010f, 0.026f},
            {0.043f, 0.013f, 0.030f},
            {0.050f, 0.015f, 0.034f},
            {0.057f, 0.017f, 0.038f},
            {0.065f, 0.019f, 0.042f},
            {0.072f, 0.022f, 0.045f},
            {0.080f, 0.024f, 0.048f},
            {0.087f, 0.026f, 0.051f},
            {0.094f, 0.029f, 0.053f},
            {0.102f, 0.031f, 0.055f},
            {0.109f, 0.034f, 0.057f},
            {0.116f, 0.037f, 0.059f},
            {0.123f, 0.040f, 0.060f},
            {0.130f, 0.043f, 0.062f},
            {0.136f, 0.046f, 0.062f},
            {0.143f, 0.049f, 0.063f},
            {0.150f, 0.052f, 0.064f},
            {0.156f, 0.056f, 0.064f},
            {0.162f, 0.059f, 0.064f},
            {0.168f, 0.063f, 0.064f},
            {0.174f, 0.067f, 0.064f},
            {0.179f, 0.070f, 0.064f},
            {0.185f, 0.074f, 0.063f},
            {0.190f, 0.079f, 0.063f},
            {0.195f, 0.083f, 0.062f},
            {0.200f, 0.087f, 0.061f},
            {0.204f, 0.092f, 0.060f},
            {0.208f, 0.097f, 0.059f},
            {0.212f, 0.101f, 0.058f},
            {0.216f, 0.106f, 0.057f},
            {0.220f, 0.111f, 0.056f},
            {0.223f, 0.117f, 0.055f},
            {0.226f, 0.122f, 0.053f},
            {0.229f, 0.128f, 0.052f},
            {0.231f, 0.133f, 0.051f},
            {0.233f, 0.139f, 0.050f},
            {0.235f, 0.145f, 0.049f},
            {0.237f, 0.151f, 0.048f},
            {0.238f, 0.157f, 0.047f},
            {0.240f, 0.163f, 0.047f},
            {0.241f, 0.169f, 0.046f},
            {0.241f, 0.176f, 0.046f},
            {0.242f, 0.182f, 0.045f},
            {0.242f, 0.189f, 0.045f},
            {0.242f, 0.196f, 0.045f},
            {0.242f, 0.202f, 0.045f},
            {0.241f, 0.209f, 0.046f},
            {0.240f, 0.216f, 0.046f},
            {0.240f, 0.223f, 0.047f},
            {0.239f, 0.230f, 0.048f},
            {0.237f, 0.237f, 0.049f},
            {0.236f, 0.244f, 0.051f},
            {0.234f, 0.251f, 0.053f},
            {0.232f, 0.259f, 0.055f},
            {0.230f, 0.266f, 0.057f},
            {0.228f, 0.273f, 0.060f},
            {0.226f, 0.280f, 0.063f},
            {0.224f, 0.288f, 0.066f},
            {0.221f, 0.295f, 0.069f},
            {0.219f, 0.302f, 0.073f},
            {0.216f, 0.309f, 0.077f},
            {0.214f, 0.316f, 0.082f},
            {0.211f, 0.324f, 0.087f},
            {0.208f, 0.331f, 0.092f},
            {0.206f, 0.338f, 0.097f},
            {0.203f, 0.345f, 0.103f},
            {0.200f, 0.352f, 0.109f},
            {0.197f, 0.359f, 0.116f},
            {0.195f, 0.365f, 0.122f},
            {0.192f, 0.372f, 0.129f},
            {0.189f, 0.379f, 0.137f},
            {0.187f, 0.385f, 0.144f},
            {0.184f, 0.392f, 0.152f},
            {0.182f, 0.398f, 0.161f},
            {0.180f, 0.404f, 0.169f},
            {0.178f, 0.410f, 0.178f},
            {0.176f, 0.416f, 0.188f},
            {0.174f, 0.422f, 0.197f},
            {0.172f, 0.428f, 0.207f},
            {0.171f, 0.433f, 0.217f},
            {0.170f, 0.439f, 0.227f},
            {0.169f, 0.444f, 0.238f},
            {0.168f, 0.449f, 0.248f},
            {0.167f, 0.454f, 0.259f},
            {0.167f, 0.459f, 0.271f},
            {0.167f, 0.463f, 0.282f},
            {0.167f, 0.468f, 0.294f},
            {0.167f, 0.472f, 0.305f},
            {0.168f, 0.476f, 0.317f},
            {0.169f, 0.480f, 0.330f},
            {0.170f, 0.484f, 0.342f},
            {0.171f, 0.487f, 0.354f},
            {0.173f, 0.491f, 0.367f},
            {0.175f, 0.494f, 0.379f},
            {0.178f, 0.497f, 0.392f},
            {0.180f, 0.500f, 0.404f},
            {0.183f, 0.503f, 0.417f},
            {0.187f, 0.506f, 0.430f},
            {0.190f, 0.508f, 0.443f},
            {0.194f, 0.510f, 0.455f},
            {0.199f, 0.512f, 0.468f},
            {0.204f, 0.514f, 0.481f},
            {0.209f, 0.516f, 0.494f},
            {0.214f, 0.518f, 0.506f},
            {0.220f, 0.519f, 0.519f},
            {0.226f, 0.520f, 0.531f},
            {0.232f, 0.521f, 0.543f},
            {0.238f, 0.523f, 0.556f},
            {0.245f, 0.523f, 0.568f},
            {0.253f, 0.524f, 0.579f},
            {0.260f, 0.525f, 0.591f},
            {0.268f, 0.525f, 0.603f},
            {0.276f, 0.526f, 0.614f},
            {0.285f, 0.526f, 0.625f},
            {0.294f, 0.526f, 0.636f},
            {0.303f, 0.526f, 0.647f},
            {0.312f, 0.526f, 0.657f},
            {0.321f, 0.526f, 0.667f},
            {0.331f, 0.526f, 0.677f},
            {0.341f, 0.526f, 0.687f},
            {0.351f, 0.526f, 0.696f},
            {0.362f, 0.525f, 0.705f},
            {0.372f, 0.525f, 0.714f},
            {0.383f, 0.525f, 0.722f},
            {0.394f, 0.524f, 0.730f},
            {0.406f, 0.524f, 0.738f},
            {0.417f, 0.523f, 0.745f},
            {0.428f, 0.523f, 0.752f},
            {0.440f, 0.522f, 0.759f},
            {0.452f, 0.522f, 0.765f},
            {0.464f, 0.521f, 0.771f},
            {0.475f, 0.521f, 0.776f},
            {0.487f, 0.521f, 0.782f},
            {0.500f, 0.520f, 0.786f},
            {0.512f, 0.520f, 0.791f},
            {0.524f, 0.520f, 0.795f},
            {0.536f, 0.519f, 0.799f},
            {0.548f, 0.519f, 0.802f},
            {0.560f, 0.519f, 0.805f},
            {0.572f, 0.519f, 0.808f},
            {0.584f, 0.519f, 0.810f},
            {0.596f, 0.520f, 0.812f},
            {0.608f, 0.520f, 0.814f},
            {0.620f, 0.520f, 0.815f},
            {0.632f, 0.521f, 0.816f},
            {0.643f, 0.522f, 0.816f},
            {0.655f, 0.522f, 0.816f},
            {0.666f, 0.523f, 0.816f},
            {0.677f, 0.524f, 0.816f},
            {0.688f, 0.526f, 0.815f},
            {0.699f, 0.527f, 0.814f},
            {0.710f, 0.528f, 0.813f},
            {0.721f, 0.530f, 0.812f},
            {0.731f, 0.532f, 0.810f},
            {0.741f, 0.534f, 0.808f},
            {0.751f, 0.536f, 0.806f},
            {0.760f, 0.538f, 0.803f},
            {0.770f, 0.540f, 0.801f},
            {0.779f, 0.543f, 0.798f},
            {0.788f, 0.546f, 0.795f},
            {0.796f, 0.549f, 0.792f},
            {0.804f, 0.552f, 0.788f},
            {0.812f, 0.555f, 0.785f},
            {0.820f, 0.558f, 0.781f},
            {0.828f, 0.562f, 0.778f},
            {0.835f, 0.566f, 0.774f},
            {0.842f, 0.570f, 0.770f},
            {0.848f, 0.574f, 0.766f},
            {0.854f, 0.578f, 0.762f},
            {0.860f, 0.582f, 0.758f},
            {0.866f, 0.587f, 0.754f},
            {0.871f, 0.592f, 0.750f},
            {0.876f, 0.597f, 0.746f},
            {0.881f, 0.602f, 0.742f},
            {0.885f, 0.607f, 0.738f},
            {0.889f, 0.612f, 0.735f},
            {0.893f, 0.617f, 0.731f},
            {0.897f, 0.623f, 0.727f},
            {0.900f, 0.629f, 0.724f},
            {0.903f, 0.635f, 0.720f},
            {0.905f, 0.640f, 0.717f},
            {0.908f, 0.647f, 0.714f},
            {0.910f, 0.653f, 0.711f},
            {0.912f, 0.659f, 0.708f},
            {0.913f, 0.665f, 0.706f},
            {0.914f, 0.672f, 0.703f},
            {0.916f, 0.678f, 0.701f},
            {0.916f, 0.685f, 0.699f},
            {0.917f, 0.692f, 0.697f},
            {0.918f, 0.698f, 0.696f},
            {0.918f, 0.705f, 0.695f},
            {0.918f, 0.712f, 0.694f},
            {0.918f, 0.719f, 0.693f},
            {0.917f, 0.726f, 0.692f},
            {0.917f, 0.733f, 0.692f},
            {0.917f, 0.739f, 0.692f},
            {0.916f, 0.746f, 0.693f},
            {0.915f, 0.753f, 0.693f},
            {0.914f, 0.760f, 0.694f},
            {0.913f, 0.767f, 0.695f},
            {0.912f, 0.774f, 0.697f},
            {0.911f, 0.781f, 0.699f},
            {0.910f, 0.788f, 0.701f},
            {0.909f, 0.795f, 0.703f},
            {0.907f, 0.802f, 0.706f},
            {0.906f, 0.808f, 0.709f},
            {0.905f, 0.815f, 0.712f},
            {0.904f, 0.822f, 0.716f},
            {0.903f, 0.828f, 0.720f},
            {0.902f, 0.835f, 0.724f},
            {0.900f, 0.841f, 0.728f},
            {0.899f, 0.847f, 0.733f},
            {0.899f, 0.854f, 0.738f},
            {0.898f, 0.860f, 0.743f},
            {0.897f, 0.866f, 0.749f},
            {0.897f, 0.872f, 0.754f},
            {0.896f, 0.877f, 0.760f},
            {0.896f, 0.883f, 0.766f},
            {0.896f, 0.889f, 0.773f},
            {0.896f, 0.894f, 0.779f},
            {0.896f, 0.899f, 0.786f},
            {0.896f, 0.904f, 0.793f},
            {0.897f, 0.910f, 0.800f},
            {0.898f, 0.914f, 0.807f},
            {0.899f, 0.919f, 0.815f},
            {0.900f, 0.924f, 0.822f},
            {0.901f, 0.928f, 0.830f},
            {0.903f, 0.933f, 0.837f},
            {0.905f, 0.937f, 0.845f},
            {0.907f, 0.941f, 0.853f},
            {0.909f, 0.945f, 0.861f},
            {0.912f, 0.949f, 0.869f},
            {0.915f, 0.953f, 0.877f},
            {0.918f, 0.956f, 0.885f},
            {0.921f, 0.960f, 0.893f},
            {0.925f, 0.963f, 0.900f},
            {0.929f, 0.967f, 0.908f},
            {0.933f, 0.970f, 0.916f},
            {0.937f, 0.973f, 0.924f},
            {0.942f, 0.976f, 0.932f},
            {0.947f, 0.978f, 0.939f},
            {0.952f, 0.981f, 0.946f},
            {0.957f, 0.984f, 0.954f},
            {0.962f, 0.986f, 0.961f},
            {0.968f, 0.989f, 0.968f},
            {0.974f, 0.991f, 0.975f},
            {0.980f, 0.993f, 0.981f},
            {0.987f, 0.996f, 0.988f},
            {0.993f, 0.998f, 0.994f},
            {1.000f, 1.000f, 1.000f}};

        //# Tue Apr 16 2019 21:40:37 GMT-0700 (Pacific Daylight Time)
        //# ---------------------------------------------
        //# R/G/B cubehelix colour scheme 
        //#
        //# see http://www.mrao.cam.ac.uk/~dag/CUBEHELIX/
        //#----------------------------------------------
        //# see Green (2011), BASI, 39, 289. 
        //#
        //# start............: 0.5
        //# rotations........: 1.5
        //# hue..............: 1.6
        //# gamma............: 1.0
        //# number of levels.: 255
        //#----------------------------------------------
        //# Dave Green: dag @ mrao.cam.ac.uk 
        //#----------------------------------------------
        //# faction and R/G/B values
        //#
        public static float[,] Helix5 = {{0.000f, 0.000f, 0.000f},
            {0.009f, 0.001f, 0.007f},
            {0.018f, 0.002f, 0.013f},
            {0.027f, 0.003f, 0.019f},
            {0.036f, 0.004f, 0.025f},
            {0.045f, 0.005f, 0.030f},
            {0.054f, 0.006f, 0.034f},
            {0.064f, 0.007f, 0.039f},
            {0.073f, 0.008f, 0.042f},
            {0.083f, 0.010f, 0.046f},
            {0.092f, 0.011f, 0.048f},
            {0.101f, 0.012f, 0.051f},
            {0.111f, 0.014f, 0.053f},
            {0.120f, 0.016f, 0.054f},
            {0.129f, 0.017f, 0.055f},
            {0.138f, 0.019f, 0.056f},
            {0.147f, 0.021f, 0.056f},
            {0.156f, 0.023f, 0.056f},
            {0.165f, 0.026f, 0.056f},
            {0.173f, 0.028f, 0.055f},
            {0.182f, 0.031f, 0.054f},
            {0.190f, 0.034f, 0.052f},
            {0.197f, 0.037f, 0.051f},
            {0.205f, 0.040f, 0.048f},
            {0.212f, 0.044f, 0.046f},
            {0.219f, 0.047f, 0.043f},
            {0.226f, 0.051f, 0.040f},
            {0.232f, 0.055f, 0.037f},
            {0.238f, 0.060f, 0.034f},
            {0.243f, 0.064f, 0.030f},
            {0.248f, 0.069f, 0.027f},
            {0.253f, 0.074f, 0.023f},
            {0.258f, 0.079f, 0.019f},
            {0.262f, 0.084f, 0.015f},
            {0.265f, 0.090f, 0.011f},
            {0.269f, 0.096f, 0.007f},
            {0.272f, 0.102f, 0.002f},
            {0.274f, 0.108f, 0.000f},
            {0.276f, 0.114f, 0.000f},
            {0.278f, 0.121f, 0.000f},
            {0.279f, 0.128f, 0.000f},
            {0.280f, 0.135f, 0.000f},
            {0.280f, 0.142f, 0.000f},
            {0.280f, 0.149f, 0.000f},
            {0.279f, 0.157f, 0.000f},
            {0.279f, 0.165f, 0.000f},
            {0.277f, 0.173f, 0.000f},
            {0.276f, 0.181f, 0.000f},
            {0.274f, 0.189f, 0.000f},
            {0.271f, 0.197f, 0.000f},
            {0.269f, 0.206f, 0.000f},
            {0.265f, 0.214f, 0.000f},
            {0.262f, 0.223f, 0.000f},
            {0.258f, 0.232f, 0.000f},
            {0.254f, 0.241f, 0.000f},
            {0.250f, 0.250f, 0.000f},
            {0.245f, 0.259f, 0.000f},
            {0.240f, 0.268f, 0.000f},
            {0.235f, 0.277f, 0.000f},
            {0.229f, 0.286f, 0.000f},
            {0.224f, 0.295f, 0.000f},
            {0.218f, 0.304f, 0.000f},
            {0.212f, 0.314f, 0.000f},
            {0.205f, 0.323f, 0.000f},
            {0.199f, 0.332f, 0.000f},
            {0.193f, 0.341f, 0.000f},
            {0.186f, 0.350f, 0.000f},
            {0.179f, 0.360f, 0.000f},
            {0.173f, 0.369f, 0.000f},
            {0.166f, 0.377f, 0.000f},
            {0.159f, 0.386f, 0.000f},
            {0.152f, 0.395f, 0.007f},
            {0.146f, 0.404f, 0.015f},
            {0.139f, 0.412f, 0.023f},
            {0.132f, 0.421f, 0.032f},
            {0.126f, 0.429f, 0.042f},
            {0.119f, 0.437f, 0.052f},
            {0.113f, 0.445f, 0.062f},
            {0.107f, 0.452f, 0.073f},
            {0.101f, 0.460f, 0.085f},
            {0.095f, 0.467f, 0.096f},
            {0.090f, 0.474f, 0.109f},
            {0.085f, 0.481f, 0.122f},
            {0.080f, 0.488f, 0.135f},
            {0.075f, 0.495f, 0.149f},
            {0.071f, 0.501f, 0.163f},
            {0.067f, 0.507f, 0.177f},
            {0.063f, 0.513f, 0.192f},
            {0.059f, 0.518f, 0.207f},
            {0.056f, 0.524f, 0.223f},
            {0.054f, 0.529f, 0.239f},
            {0.052f, 0.533f, 0.255f},
            {0.050f, 0.538f, 0.271f},
            {0.049f, 0.542f, 0.288f},
            {0.048f, 0.546f, 0.305f},
            {0.047f, 0.550f, 0.322f},
            {0.047f, 0.553f, 0.340f},
            {0.048f, 0.556f, 0.357f},
            {0.049f, 0.559f, 0.375f},
            {0.050f, 0.562f, 0.393f},
            {0.052f, 0.564f, 0.411f},
            {0.055f, 0.566f, 0.429f},
            {0.058f, 0.568f, 0.447f},
            {0.061f, 0.569f, 0.465f},
            {0.066f, 0.571f, 0.483f},
            {0.070f, 0.572f, 0.501f},
            {0.075f, 0.572f, 0.519f},
            {0.081f, 0.573f, 0.537f},
            {0.087f, 0.573f, 0.555f},
            {0.094f, 0.573f, 0.572f},
            {0.101f, 0.573f, 0.590f},
            {0.109f, 0.572f, 0.607f},
            {0.117f, 0.571f, 0.624f},
            {0.126f, 0.571f, 0.641f},
            {0.135f, 0.569f, 0.658f},
            {0.145f, 0.568f, 0.674f},
            {0.155f, 0.567f, 0.690f},
            {0.166f, 0.565f, 0.706f},
            {0.177f, 0.563f, 0.721f},
            {0.189f, 0.561f, 0.737f},
            {0.201f, 0.559f, 0.751f},
            {0.213f, 0.556f, 0.765f},
            {0.226f, 0.554f, 0.779f},
            {0.239f, 0.551f, 0.793f},
            {0.253f, 0.549f, 0.806f},
            {0.267f, 0.546f, 0.818f},
            {0.281f, 0.543f, 0.830f},
            {0.296f, 0.540f, 0.842f},
            {0.311f, 0.537f, 0.853f},
            {0.326f, 0.534f, 0.863f},
            {0.342f, 0.531f, 0.873f},
            {0.358f, 0.528f, 0.883f},
            {0.374f, 0.525f, 0.891f},
            {0.390f, 0.522f, 0.900f},
            {0.406f, 0.518f, 0.907f},
            {0.423f, 0.515f, 0.914f},
            {0.440f, 0.512f, 0.921f},
            {0.456f, 0.509f, 0.927f},
            {0.473f, 0.506f, 0.932f},
            {0.490f, 0.504f, 0.937f},
            {0.507f, 0.501f, 0.941f},
            {0.524f, 0.498f, 0.945f},
            {0.541f, 0.496f, 0.948f},
            {0.558f, 0.493f, 0.950f},
            {0.575f, 0.491f, 0.952f},
            {0.592f, 0.489f, 0.954f},
            {0.609f, 0.487f, 0.954f},
            {0.626f, 0.485f, 0.954f},
            {0.642f, 0.483f, 0.954f},
            {0.659f, 0.482f, 0.953f},
            {0.675f, 0.480f, 0.952f},
            {0.691f, 0.479f, 0.950f},
            {0.707f, 0.478f, 0.947f},
            {0.722f, 0.478f, 0.944f},
            {0.738f, 0.477f, 0.941f},
            {0.753f, 0.477f, 0.937f},
            {0.768f, 0.477f, 0.933f},
            {0.782f, 0.477f, 0.928f},
            {0.796f, 0.478f, 0.923f},
            {0.810f, 0.478f, 0.917f},
            {0.823f, 0.479f, 0.911f},
            {0.836f, 0.480f, 0.905f},
            {0.849f, 0.482f, 0.898f},
            {0.861f, 0.484f, 0.891f},
            {0.873f, 0.486f, 0.884f},
            {0.884f, 0.488f, 0.877f},
            {0.895f, 0.491f, 0.869f},
            {0.905f, 0.493f, 0.861f},
            {0.915f, 0.497f, 0.853f},
            {0.925f, 0.500f, 0.845f},
            {0.934f, 0.504f, 0.836f},
            {0.942f, 0.507f, 0.828f},
            {0.951f, 0.512f, 0.819f},
            {0.958f, 0.516f, 0.811f},
            {0.965f, 0.521f, 0.802f},
            {0.972f, 0.526f, 0.793f},
            {0.978f, 0.531f, 0.785f},
            {0.984f, 0.536f, 0.776f},
            {0.989f, 0.542f, 0.767f},
            {0.993f, 0.548f, 0.759f},
            {0.998f, 0.554f, 0.750f},
            {1.000f, 0.560f, 0.742f},
            {1.000f, 0.567f, 0.734f},
            {1.000f, 0.574f, 0.726f},
            {1.000f, 0.581f, 0.718f},
            {1.000f, 0.588f, 0.710f},
            {1.000f, 0.595f, 0.703f},
            {1.000f, 0.603f, 0.696f},
            {1.000f, 0.610f, 0.689f},
            {1.000f, 0.618f, 0.683f},
            {1.000f, 0.626f, 0.676f},
            {1.000f, 0.634f, 0.671f},
            {1.000f, 0.642f, 0.665f},
            {1.000f, 0.651f, 0.660f},
            {1.000f, 0.659f, 0.655f},
            {1.000f, 0.667f, 0.651f},
            {1.000f, 0.676f, 0.647f},
            {1.000f, 0.685f, 0.643f},
            {1.000f, 0.693f, 0.640f},
            {0.997f, 0.702f, 0.637f},
            {0.994f, 0.711f, 0.635f},
            {0.991f, 0.720f, 0.633f},
            {0.987f, 0.728f, 0.632f},
            {0.983f, 0.737f, 0.631f},
            {0.979f, 0.746f, 0.631f},
            {0.975f, 0.755f, 0.631f},
            {0.971f, 0.763f, 0.631f},
            {0.967f, 0.772f, 0.632f},
            {0.962f, 0.780f, 0.634f},
            {0.958f, 0.789f, 0.636f},
            {0.954f, 0.797f, 0.638f},
            {0.950f, 0.806f, 0.641f},
            {0.945f, 0.814f, 0.645f},
            {0.941f, 0.822f, 0.649f},
            {0.937f, 0.830f, 0.653f},
            {0.933f, 0.838f, 0.658f},
            {0.929f, 0.845f, 0.663f},
            {0.925f, 0.853f, 0.668f},
            {0.922f, 0.860f, 0.674f},
            {0.918f, 0.868f, 0.681f},
            {0.915f, 0.875f, 0.687f},
            {0.912f, 0.882f, 0.694f},
            {0.909f, 0.888f, 0.702f},
            {0.906f, 0.895f, 0.710f},
            {0.904f, 0.901f, 0.718f},
            {0.902f, 0.907f, 0.726f},
            {0.900f, 0.913f, 0.735f},
            {0.899f, 0.919f, 0.744f},
            {0.898f, 0.925f, 0.753f},
            {0.897f, 0.930f, 0.763f},
            {0.897f, 0.935f, 0.772f},
            {0.896f, 0.940f, 0.782f},
            {0.897f, 0.945f, 0.792f},
            {0.897f, 0.949f, 0.802f},
            {0.898f, 0.953f, 0.812f},
            {0.900f, 0.957f, 0.822f},
            {0.902f, 0.961f, 0.833f},
            {0.904f, 0.965f, 0.843f},
            {0.906f, 0.968f, 0.853f},
            {0.909f, 0.971f, 0.864f},
            {0.913f, 0.974f, 0.874f},
            {0.917f, 0.977f, 0.884f},
            {0.921f, 0.980f, 0.894f},
            {0.925f, 0.982f, 0.904f},
            {0.930f, 0.985f, 0.914f},
            {0.936f, 0.987f, 0.924f},
            {0.941f, 0.989f, 0.933f},
            {0.948f, 0.991f, 0.943f},
            {0.954f, 0.992f, 0.952f},
            {0.961f, 0.994f, 0.960f},
            {0.968f, 0.995f, 0.969f},
            {0.976f, 0.997f, 0.977f},
            {0.983f, 0.998f, 0.985f},
            {0.992f, 0.999f, 0.993f},
            {1.000f, 1.000f, 1.000f}};

        //# Tue Apr 16 2019 21:52:01 GMT-0700 (Pacific Daylight Time)
        //# ---------------------------------------------
        //# R/G/B cubehelix colour scheme 
        //#
        //# see http://www.mrao.cam.ac.uk/~dag/CUBEHELIX/
        //#----------------------------------------------
        //# see Green (2011), BASI, 39, 289. 
        //#
        //# start............: 0.5
        //# rotations........: 1.4
        //# hue..............: 1.7
        //# gamma............: 1.6
        //# number of levels.: 255
        //#----------------------------------------------
        //# Dave Green: dag @ mrao.cam.ac.uk 
        //#----------------------------------------------
        //# faction and R/G/B values
        //#
        public static float[,] Helix6 = {{0.000f, 0.000f, 0.000f},
            {0.000f, 0.000f, 0.000f},
            {0.001f, 0.000f, 0.001f},
            {0.002f, 0.000f, 0.001f},
            {0.003f, 0.000f, 0.002f},
            {0.004f, 0.000f, 0.003f},
            {0.006f, 0.000f, 0.004f},
            {0.008f, 0.001f, 0.005f},
            {0.010f, 0.001f, 0.006f},
            {0.012f, 0.001f, 0.006f},
            {0.014f, 0.001f, 0.007f},
            {0.016f, 0.001f, 0.008f},
            {0.019f, 0.002f, 0.009f},
            {0.021f, 0.002f, 0.010f},
            {0.024f, 0.002f, 0.010f},
            {0.027f, 0.003f, 0.011f},
            {0.030f, 0.003f, 0.011f},
            {0.033f, 0.003f, 0.012f},
            {0.036f, 0.004f, 0.012f},
            {0.039f, 0.004f, 0.012f},
            {0.043f, 0.005f, 0.012f},
            {0.046f, 0.006f, 0.012f},
            {0.049f, 0.006f, 0.012f},
            {0.053f, 0.007f, 0.012f},
            {0.056f, 0.008f, 0.012f},
            {0.060f, 0.009f, 0.011f},
            {0.063f, 0.010f, 0.010f},
            {0.066f, 0.011f, 0.010f},
            {0.070f, 0.013f, 0.009f},
            {0.073f, 0.014f, 0.008f},
            {0.077f, 0.015f, 0.007f},
            {0.080f, 0.017f, 0.005f},
            {0.083f, 0.019f, 0.004f},
            {0.086f, 0.020f, 0.002f},
            {0.089f, 0.022f, 0.001f},
            {0.092f, 0.024f, 0.000f},
            {0.095f, 0.027f, 0.000f},
            {0.098f, 0.029f, 0.000f},
            {0.100f, 0.031f, 0.000f},
            {0.103f, 0.034f, 0.000f},
            {0.105f, 0.037f, 0.000f},
            {0.107f, 0.039f, 0.000f},
            {0.110f, 0.042f, 0.000f},
            {0.111f, 0.045f, 0.000f},
            {0.113f, 0.049f, 0.000f},
            {0.115f, 0.052f, 0.000f},
            {0.116f, 0.056f, 0.000f},
            {0.117f, 0.059f, 0.000f},
            {0.118f, 0.063f, 0.000f},
            {0.119f, 0.067f, 0.000f},
            {0.119f, 0.071f, 0.000f},
            {0.120f, 0.076f, 0.000f},
            {0.120f, 0.080f, 0.000f},
            {0.120f, 0.085f, 0.000f},
            {0.119f, 0.089f, 0.000f},
            {0.119f, 0.094f, 0.000f},
            {0.118f, 0.099f, 0.000f},
            {0.117f, 0.104f, 0.000f},
            {0.116f, 0.110f, 0.000f},
            {0.115f, 0.115f, 0.000f},
            {0.113f, 0.120f, 0.000f},
            {0.111f, 0.126f, 0.000f},
            {0.109f, 0.132f, 0.000f},
            {0.107f, 0.138f, 0.000f},
            {0.104f, 0.144f, 0.000f},
            {0.102f, 0.150f, 0.000f},
            {0.099f, 0.156f, 0.000f},
            {0.096f, 0.162f, 0.000f},
            {0.093f, 0.168f, 0.000f},
            {0.089f, 0.175f, 0.000f},
            {0.086f, 0.181f, 0.000f},
            {0.082f, 0.187f, 0.000f},
            {0.079f, 0.194f, 0.000f},
            {0.075f, 0.201f, 0.000f},
            {0.071f, 0.207f, 0.000f},
            {0.067f, 0.214f, 0.000f},
            {0.062f, 0.220f, 0.000f},
            {0.058f, 0.227f, 0.000f},
            {0.054f, 0.234f, 0.000f},
            {0.049f, 0.240f, 0.000f},
            {0.045f, 0.247f, 0.000f},
            {0.040f, 0.254f, 0.000f},
            {0.036f, 0.260f, 0.000f},
            {0.031f, 0.267f, 0.001f},
            {0.027f, 0.274f, 0.008f},
            {0.022f, 0.280f, 0.015f},
            {0.018f, 0.286f, 0.022f},
            {0.014f, 0.293f, 0.030f},
            {0.009f, 0.299f, 0.038f},
            {0.005f, 0.305f, 0.047f},
            {0.001f, 0.311f, 0.056f},
            {0.000f, 0.317f, 0.065f},
            {0.000f, 0.323f, 0.075f},
            {0.000f, 0.329f, 0.085f},
            {0.000f, 0.335f, 0.095f},
            {0.000f, 0.340f, 0.106f},
            {0.000f, 0.345f, 0.118f},
            {0.000f, 0.351f, 0.129f},
            {0.000f, 0.356f, 0.141f},
            {0.000f, 0.361f, 0.154f},
            {0.000f, 0.365f, 0.166f},
            {0.000f, 0.370f, 0.179f},
            {0.000f, 0.374f, 0.193f},
            {0.000f, 0.379f, 0.206f},
            {0.000f, 0.383f, 0.220f},
            {0.000f, 0.386f, 0.234f},
            {0.000f, 0.390f, 0.249f},
            {0.000f, 0.394f, 0.263f},
            {0.000f, 0.397f, 0.278f},
            {0.000f, 0.400f, 0.293f},
            {0.000f, 0.403f, 0.309f},
            {0.000f, 0.405f, 0.324f},
            {0.000f, 0.408f, 0.340f},
            {0.000f, 0.410f, 0.356f},
            {0.000f, 0.412f, 0.371f},
            {0.000f, 0.414f, 0.387f},
            {0.000f, 0.415f, 0.403f},
            {0.000f, 0.417f, 0.419f},
            {0.000f, 0.418f, 0.435f},
            {0.001f, 0.419f, 0.452f},
            {0.007f, 0.420f, 0.468f},
            {0.014f, 0.420f, 0.484f},
            {0.021f, 0.421f, 0.500f},
            {0.028f, 0.421f, 0.516f},
            {0.036f, 0.421f, 0.531f},
            {0.044f, 0.421f, 0.547f},
            {0.053f, 0.420f, 0.563f},
            {0.062f, 0.420f, 0.578f},
            {0.072f, 0.419f, 0.593f},
            {0.082f, 0.418f, 0.608f},
            {0.093f, 0.417f, 0.623f},
            {0.104f, 0.416f, 0.637f},
            {0.115f, 0.415f, 0.652f},
            {0.127f, 0.413f, 0.666f},
            {0.140f, 0.412f, 0.679f},
            {0.152f, 0.410f, 0.692f},
            {0.166f, 0.408f, 0.705f},
            {0.179f, 0.406f, 0.718f},
            {0.193f, 0.404f, 0.730f},
            {0.208f, 0.402f, 0.742f},
            {0.222f, 0.400f, 0.753f},
            {0.237f, 0.398f, 0.764f},
            {0.252f, 0.396f, 0.775f},
            {0.268f, 0.393f, 0.785f},
            {0.284f, 0.391f, 0.794f},
            {0.300f, 0.389f, 0.804f},
            {0.317f, 0.386f, 0.812f},
            {0.333f, 0.384f, 0.820f},
            {0.350f, 0.382f, 0.828f},
            {0.367f, 0.380f, 0.835f},
            {0.384f, 0.377f, 0.841f},
            {0.402f, 0.375f, 0.847f},
            {0.419f, 0.373f, 0.853f},
            {0.437f, 0.371f, 0.858f},
            {0.454f, 0.369f, 0.862f},
            {0.472f, 0.368f, 0.866f},
            {0.490f, 0.366f, 0.869f},
            {0.508f, 0.364f, 0.872f},
            {0.525f, 0.363f, 0.874f},
            {0.543f, 0.362f, 0.875f},
            {0.561f, 0.361f, 0.876f},
            {0.578f, 0.360f, 0.877f},
            {0.596f, 0.359f, 0.877f},
            {0.613f, 0.358f, 0.876f},
            {0.631f, 0.358f, 0.875f},
            {0.648f, 0.358f, 0.874f},
            {0.665f, 0.358f, 0.872f},
            {0.681f, 0.358f, 0.869f},
            {0.698f, 0.359f, 0.866f},
            {0.714f, 0.359f, 0.862f},
            {0.730f, 0.360f, 0.858f},
            {0.746f, 0.362f, 0.854f},
            {0.761f, 0.363f, 0.849f},
            {0.776f, 0.365f, 0.844f},
            {0.791f, 0.367f, 0.839f},
            {0.805f, 0.369f, 0.833f},
            {0.819f, 0.372f, 0.826f},
            {0.832f, 0.375f, 0.820f},
            {0.846f, 0.378f, 0.813f},
            {0.858f, 0.382f, 0.806f},
            {0.871f, 0.385f, 0.798f},
            {0.882f, 0.390f, 0.791f},
            {0.894f, 0.394f, 0.783f},
            {0.905f, 0.399f, 0.775f},
            {0.915f, 0.404f, 0.767f},
            {0.925f, 0.409f, 0.759f},
            {0.934f, 0.415f, 0.750f},
            {0.943f, 0.421f, 0.742f},
            {0.951f, 0.427f, 0.733f},
            {0.959f, 0.433f, 0.725f},
            {0.967f, 0.440f, 0.716f},
            {0.973f, 0.447f, 0.708f},
            {0.980f, 0.455f, 0.700f},
            {0.985f, 0.462f, 0.691f},
            {0.991f, 0.470f, 0.683f},
            {0.995f, 0.478f, 0.675f},
            {1.000f, 0.487f, 0.668f},
            {1.000f, 0.495f, 0.660f},
            {1.000f, 0.504f, 0.653f},
            {1.000f, 0.513f, 0.646f},
            {1.000f, 0.523f, 0.639f},
            {1.000f, 0.532f, 0.632f},
            {1.000f, 0.542f, 0.626f},
            {1.000f, 0.552f, 0.620f},
            {1.000f, 0.562f, 0.615f},
            {1.000f, 0.572f, 0.610f},
            {1.000f, 0.583f, 0.605f},
            {1.000f, 0.593f, 0.601f},
            {1.000f, 0.604f, 0.598f},
            {1.000f, 0.615f, 0.594f},
            {1.000f, 0.626f, 0.592f},
            {1.000f, 0.637f, 0.590f},
            {1.000f, 0.648f, 0.588f},
            {1.000f, 0.659f, 0.587f},
            {1.000f, 0.670f, 0.586f},
            {0.997f, 0.682f, 0.587f},
            {0.994f, 0.693f, 0.587f},
            {0.991f, 0.704f, 0.588f},
            {0.987f, 0.715f, 0.590f},
            {0.983f, 0.726f, 0.593f},
            {0.980f, 0.738f, 0.596f},
            {0.976f, 0.749f, 0.600f},
            {0.972f, 0.760f, 0.604f},
            {0.968f, 0.770f, 0.609f},
            {0.964f, 0.781f, 0.614f},
            {0.961f, 0.792f, 0.621f},
            {0.957f, 0.802f, 0.627f},
            {0.953f, 0.813f, 0.635f},
            {0.950f, 0.823f, 0.643f},
            {0.947f, 0.833f, 0.651f},
            {0.944f, 0.843f, 0.661f},
            {0.941f, 0.853f, 0.670f},
            {0.938f, 0.862f, 0.681f},
            {0.936f, 0.871f, 0.692f},
            {0.934f, 0.880f, 0.703f},
            {0.933f, 0.889f, 0.715f},
            {0.932f, 0.898f, 0.727f},
            {0.931f, 0.906f, 0.740f},
            {0.930f, 0.914f, 0.753f},
            {0.931f, 0.921f, 0.767f},
            {0.931f, 0.929f, 0.781f},
            {0.932f, 0.936f, 0.795f},
            {0.934f, 0.943f, 0.810f},
            {0.936f, 0.949f, 0.825f},
            {0.939f, 0.955f, 0.840f},
            {0.942f, 0.961f, 0.856f},
            {0.946f, 0.967f, 0.871f},
            {0.950f, 0.972f, 0.887f},
            {0.955f, 0.977f, 0.903f},
            {0.961f, 0.982f, 0.919f},
            {0.968f, 0.986f, 0.936f},
            {0.975f, 0.990f, 0.952f},
            {0.982f, 0.994f, 0.968f},
            {0.991f, 0.997f, 0.984f},
            {1.000f, 1.000f, 1.000f}};

        // https://www.mathworks.com/matlabcentral/fileexchange/51986-perceptually-uniform-colormaps
        public static float[,] Viridis = {{ 0.26700401f,  0.00487433f,  0.32941519f},
           { 0.26851048f,  0.00960483f,  0.33542652f},
           { 0.26994384f,  0.01462494f,  0.34137895f},
           { 0.27130489f,  0.01994186f,  0.34726862f},
           { 0.27259384f,  0.02556309f,  0.35309303f},
           { 0.27380934f,  0.03149748f,  0.35885256f},
           { 0.27495242f,  0.03775181f,  0.36454323f},
           { 0.27602238f,  0.04416723f,  0.37016418f},
           { 0.2770184f,   0.05034437f,  0.37571452f},
           { 0.27794143f,  0.05632444f,  0.38119074f},
           { 0.27879067f,  0.06214536f,  0.38659204f},
           { 0.2795655f,   0.06783587f,  0.39191723f},
           { 0.28026658f,  0.07341724f,  0.39716349f},
           { 0.28089358f,  0.07890703f,  0.40232944f},
           { 0.28144581f,  0.0843197f,   0.40741404f},
           { 0.28192358f,  0.08966622f,  0.41241521f},
           { 0.28232739f,  0.09495545f,  0.41733086f},
           { 0.28265633f,  0.10019576f,  0.42216032f},
           { 0.28291049f,  0.10539345f,  0.42690202f},
           { 0.28309095f,  0.11055307f,  0.43155375f},
           { 0.28319704f,  0.11567966f,  0.43611482f},
           { 0.28322882f,  0.12077701f,  0.44058404f},
           { 0.28318684f,  0.12584799f,  0.44496f},
           { 0.283072f,    0.13089477f,  0.44924127f},
           { 0.28288389f,  0.13592005f,  0.45342734f},
           { 0.28262297f,  0.14092556f,  0.45751726f},
           { 0.28229037f,  0.14591233f,  0.46150995f},
           { 0.28188676f,  0.15088147f,  0.46540474f},
           { 0.28141228f,  0.15583425f,  0.46920128f},
           { 0.28086773f,  0.16077132f,  0.47289909f},
           { 0.28025468f,  0.16569272f,  0.47649762f},
           { 0.27957399f,  0.17059884f,  0.47999675f},
           { 0.27882618f,  0.1754902f,  0.48339654f},
           { 0.27801236f,  0.18036684f,  0.48669702f},
           { 0.27713437f,  0.18522836f,  0.48989831f},
           { 0.27619376f,  0.19007447f,  0.49300074f},
           { 0.27519116f,  0.1949054f,  0.49600488f},
           { 0.27412802f,  0.19972086f,  0.49891131f},
           { 0.27300596f,  0.20452049f,  0.50172076f},
           { 0.27182812f,  0.20930306f,  0.50443413f},
           { 0.27059473f,  0.21406899f,  0.50705243f},
           { 0.26930756f,  0.21881782f,  0.50957678f},
           { 0.26796846f,  0.22354911f,  0.5120084f},
           { 0.26657984f,  0.2282621f,  0.5143487f},
           { 0.2651445f,  0.23295593f,  0.5165993f},
           { 0.2636632f,  0.23763078f,  0.51876163f},
           { 0.26213801f,  0.24228619f,  0.52083736f},
           { 0.26057103f,  0.2469217f,  0.52282822f},
           { 0.25896451f,  0.25153685f,  0.52473609f},
           { 0.25732244f,  0.2561304f,  0.52656332f},
           { 0.25564519f,  0.26070284f,  0.52831152f},
           { 0.25393498f,  0.26525384f,  0.52998273f},
           { 0.25219404f,  0.26978306f,  0.53157905f},
           { 0.25042462f,  0.27429024f,  0.53310261f},
           { 0.24862899f,  0.27877509f,  0.53455561f},
           { 0.2468114f,  0.28323662f,  0.53594093f},
           { 0.24497208f,  0.28767547f,  0.53726018f},
           { 0.24311324f,  0.29209154f,  0.53851561f},
           { 0.24123708f,  0.29648471f,  0.53970946f},
           { 0.23934575f,  0.30085494f,  0.54084398f},
           { 0.23744138f,  0.30520222f,  0.5419214f},
           { 0.23552606f,  0.30952657f,  0.54294396f},
           { 0.23360277f,  0.31382773f,  0.54391424f},
           { 0.2316735f,  0.3181058f,  0.54483444f},
           { 0.22973926f,  0.32236127f,  0.54570633f},
           { 0.22780192f,  0.32659432f,  0.546532f},
           { 0.2258633f,  0.33080515f,  0.54731353f},
           { 0.22392515f,  0.334994f,  0.54805291f},
           { 0.22198915f,  0.33916114f,  0.54875211f},
           { 0.22005691f,  0.34330688f,  0.54941304f},
           { 0.21812995f,  0.34743154f,  0.55003755f},
           { 0.21620971f,  0.35153548f,  0.55062743f},
           { 0.21429757f,  0.35561907f,  0.5511844f},
           { 0.21239477f,  0.35968273f,  0.55171011f},
           { 0.2105031f,  0.36372671f,  0.55220646f},
           { 0.20862342f,  0.36775151f,  0.55267486f},
           { 0.20675628f,  0.37175775f,  0.55311653f},
           { 0.20490257f,  0.37574589f,  0.55353282f},
           { 0.20306309f,  0.37971644f,  0.55392505f},
           { 0.20123854f,  0.38366989f,  0.55429441f},
           { 0.1994295f,  0.38760678f,  0.55464205f},
           { 0.1976365f,  0.39152762f,  0.55496905f},
           { 0.19585993f,  0.39543297f,  0.55527637f},
           { 0.19410009f,  0.39932336f,  0.55556494f},
           { 0.19235719f,  0.40319934f,  0.55583559f},
           { 0.19063135f,  0.40706148f,  0.55608907f},
           { 0.18892259f,  0.41091033f,  0.55632606f},
           { 0.18723083f,  0.41474645f,  0.55654717f},
           { 0.18555593f,  0.4185704f,  0.55675292f},
           { 0.18389763f,  0.42238275f,  0.55694377f},
           { 0.18225561f,  0.42618405f,  0.5571201f},
           { 0.18062949f,  0.42997486f,  0.55728221f},
           { 0.17901879f,  0.43375572f,  0.55743035f},
           { 0.17742298f,  0.4375272f,  0.55756466f},
           { 0.17584148f,  0.44128981f,  0.55768526f},
           { 0.17427363f,  0.4450441f,  0.55779216f},
           { 0.17271876f,  0.4487906f,  0.55788532f},
           { 0.17117615f,  0.4525298f,  0.55796464f},
           { 0.16964573f,  0.45626209f,  0.55803034f},
           { 0.16812641f,  0.45998802f,  0.55808199f},
           { 0.1666171f,  0.46370813f,  0.55811913f},
           { 0.16511703f,  0.4674229f,  0.55814141f},
           { 0.16362543f,  0.47113278f,  0.55814842f},
           { 0.16214155f,  0.47483821f,  0.55813967f},
           { 0.16066467f,  0.47853961f,  0.55811466f},
           { 0.15919413f,  0.4822374f,  0.5580728f},
           { 0.15772933f,  0.48593197f,  0.55801347f},
           { 0.15626973f,  0.4896237f,  0.557936f},
           { 0.15481488f,  0.49331293f,  0.55783967f},
           { 0.15336445f,  0.49700003f,  0.55772371f},
           { 0.1519182f,  0.50068529f,  0.55758733f},
           { 0.15047605f,  0.50436904f,  0.55742968f},
           { 0.14903918f,  0.50805136f,  0.5572505f},
           { 0.14760731f,  0.51173263f,  0.55704861f},
           { 0.14618026f,  0.51541316f,  0.55682271f},
           { 0.14475863f,  0.51909319f,  0.55657181f},
           { 0.14334327f,  0.52277292f,  0.55629491f},
           { 0.14193527f,  0.52645254f,  0.55599097f},
           { 0.14053599f,  0.53013219f,  0.55565893f},
           { 0.13914708f,  0.53381201f,  0.55529773f},
           { 0.13777048f,  0.53749213f,  0.55490625f},
           { 0.1364085f,  0.54117264f,  0.55448339f},
           { 0.13506561f,  0.54485335f,  0.55402906f},
           { 0.13374299f,  0.54853458f,  0.55354108f},
           { 0.13244401f,  0.55221637f,  0.55301828f},
           { 0.13117249f,  0.55589872f,  0.55245948f},
           { 0.1299327f,  0.55958162f,  0.55186354f},
           { 0.12872938f,  0.56326503f,  0.55122927f},
           { 0.12756771f,  0.56694891f,  0.55055551f},
           { 0.12645338f,  0.57063316f,  0.5498411f},
           { 0.12539383f,  0.57431754f,  0.54908564f},
           { 0.12439474f,  0.57800205f,  0.5482874f},
           { 0.12346281f,  0.58168661f,  0.54744498f},
           { 0.12260562f,  0.58537105f,  0.54655722f},
           { 0.12183122f,  0.58905521f,  0.54562298f},
           { 0.12114807f,  0.59273889f,  0.54464114f},
           { 0.12056501f,  0.59642187f,  0.54361058f},
           { 0.12009154f,  0.60010387f,  0.54253043f},
           { 0.11973756f,  0.60378459f,  0.54139999f},
           { 0.11951163f,  0.60746388f,  0.54021751f},
           { 0.11942341f,  0.61114146f,  0.53898192f},
           { 0.11948255f,  0.61481702f,  0.53769219f},
           { 0.11969858f,  0.61849025f,  0.53634733f},
           { 0.12008079f,  0.62216081f,  0.53494633f},
           { 0.12063824f,  0.62582833f,  0.53348834f},
           { 0.12137972f,  0.62949242f,  0.53197275f},
           { 0.12231244f,  0.63315277f,  0.53039808f},
           { 0.12344358f,  0.63680899f,  0.52876343f},
           { 0.12477953f,  0.64046069f,  0.52706792f},
           { 0.12632581f,  0.64410744f,  0.52531069f},
           { 0.12808703f,  0.64774881f,  0.52349092f},
           { 0.13006688f,  0.65138436f,  0.52160791f},
           { 0.13226797f,  0.65501363f,  0.51966086f},
           { 0.13469183f,  0.65863619f,  0.5176488f},
           { 0.13733921f,  0.66225157f,  0.51557101f},
           { 0.14020991f,  0.66585927f,  0.5134268f},
           { 0.14330291f,  0.66945881f,  0.51121549f},
           { 0.1466164f,  0.67304968f,  0.50893644f},
           { 0.15014782f,  0.67663139f,  0.5065889f},
           { 0.15389405f,  0.68020343f,  0.50417217f},
           { 0.15785146f,  0.68376525f,  0.50168574f},
           { 0.16201598f,  0.68731632f,  0.49912906f},
           { 0.1663832f,  0.69085611f,  0.49650163f},
           { 0.1709484f,  0.69438405f,  0.49380294f},
           { 0.17570671f,  0.6978996f,  0.49103252f},
           { 0.18065314f,  0.70140222f,  0.48818938f},
           { 0.18578266f,  0.70489133f,  0.48527326f},
           { 0.19109018f,  0.70836635f,  0.48228395f},
           { 0.19657063f,  0.71182668f,  0.47922108f},
           { 0.20221902f,  0.71527175f,  0.47608431f},
           { 0.20803045f,  0.71870095f,  0.4728733f},
           { 0.21400015f,  0.72211371f,  0.46958774f},
           { 0.22012381f,  0.72550945f,  0.46622638f},
           { 0.2263969f,  0.72888753f,  0.46278934f},
           { 0.23281498f,  0.73224735f,  0.45927675f},
           { 0.2393739f,  0.73558828f,  0.45568838f},
           { 0.24606968f,  0.73890972f,  0.45202405f},
           { 0.25289851f,  0.74221104f,  0.44828355f},
           { 0.25985676f,  0.74549162f,  0.44446673f},
           { 0.26694127f,  0.74875084f,  0.44057284f},
           { 0.27414922f,  0.75198807f,  0.4366009f},
           { 0.28147681f,  0.75520266f,  0.43255207f},
           { 0.28892102f,  0.75839399f,  0.42842626f},
           { 0.29647899f,  0.76156142f,  0.42422341f},
           { 0.30414796f,  0.76470433f,  0.41994346f},
           { 0.31192534f,  0.76782207f,  0.41558638f},
           { 0.3198086f,  0.77091403f,  0.41115215f},
           { 0.3277958f,  0.77397953f,  0.40664011f},
           { 0.33588539f,  0.7770179f,  0.40204917f},
           { 0.34407411f,  0.78002855f,  0.39738103f},
           { 0.35235985f,  0.78301086f,  0.39263579f},
           { 0.36074053f,  0.78596419f,  0.38781353f},
           { 0.3692142f,  0.78888793f,  0.38291438f},
           { 0.37777892f,  0.79178146f,  0.3779385f},
           { 0.38643282f,  0.79464415f,  0.37288606f},
           { 0.39517408f,  0.79747541f,  0.36775726f},
           { 0.40400101f,  0.80027461f,  0.36255223f},
           { 0.4129135f,  0.80304099f,  0.35726893f},
           { 0.42190813f,  0.80577412f,  0.35191009f},
           { 0.43098317f,  0.80847343f,  0.34647607f},
           { 0.44013691f,  0.81113836f,  0.3409673f},
           { 0.44936763f,  0.81376835f,  0.33538426f},
           { 0.45867362f,  0.81636288f,  0.32972749f},
           { 0.46805314f,  0.81892143f,  0.32399761f},
           { 0.47750446f,  0.82144351f,  0.31819529f},
           { 0.4870258f,  0.82392862f,  0.31232133f},
           { 0.49661536f,  0.82637633f,  0.30637661f},
           { 0.5062713f,  0.82878621f,  0.30036211f},
           { 0.51599182f,  0.83115784f,  0.29427888f},
           { 0.52577622f,  0.83349064f,  0.2881265f},
           { 0.5356211f,  0.83578452f,  0.28190832f},
           { 0.5455244f,  0.83803918f,  0.27562602f},
           { 0.55548397f,  0.84025437f,  0.26928147f},
           { 0.5654976f,  0.8424299f,  0.26287683f},
           { 0.57556297f,  0.84456561f,  0.25641457f},
           { 0.58567772f,  0.84666139f,  0.24989748f},
           { 0.59583934f,  0.84871722f,  0.24332878f},
           { 0.60604528f,  0.8507331f,  0.23671214f},
           { 0.61629283f,  0.85270912f,  0.23005179f},
           { 0.62657923f,  0.85464543f,  0.22335258f},
           { 0.63690157f,  0.85654226f,  0.21662012f},
           { 0.64725685f,  0.85839991f,  0.20986086f},
           { 0.65764197f,  0.86021878f,  0.20308229f},
           { 0.66805369f,  0.86199932f,  0.19629307f},
           { 0.67848868f,  0.86374211f,  0.18950326f},
           { 0.68894351f,  0.86544779f,  0.18272455f},
           { 0.69941463f,  0.86711711f,  0.17597055f},
           { 0.70989842f,  0.86875092f,  0.16925712f},
           { 0.72039115f,  0.87035015f,  0.16260273f},
           { 0.73088902f,  0.87191584f,  0.15602894f},
           { 0.74138803f,  0.87344918f,  0.14956101f},
           { 0.75188414f,  0.87495143f,  0.14322828f},
           { 0.76237342f,  0.87642392f,  0.13706449f},
           { 0.77285183f,  0.87786808f,  0.13110864f},
           { 0.78331535f,  0.87928545f,  0.12540538f},
           { 0.79375994f,  0.88067763f,  0.12000532f},
           { 0.80418159f,  0.88204632f,  0.11496505f},
           { 0.81457634f,  0.88339329f,  0.11034678f},
           { 0.82494028f,  0.88472036f,  0.10621724f},
           { 0.83526959f,  0.88602943f,  0.1026459f},
           { 0.84556056f,  0.88732243f,  0.09970219f},
           { 0.8558096f,  0.88860134f,  0.09745186f},
           { 0.86601325f,  0.88986815f,  0.09595277f},
           { 0.87616824f,  0.89112487f,  0.09525046f},
           { 0.88627146f,  0.89237353f,  0.09537439f},
           { 0.89632002f,  0.89361614f,  0.09633538f},
           { 0.90631121f,  0.89485467f,  0.09812496f},
           { 0.91624212f,  0.89609127f,  0.1007168f},
           { 0.92610579f,  0.89732977f,  0.10407067f},
           { 0.93590444f,  0.8985704f,  0.10813094f},
           { 0.94563626f,  0.899815f,  0.11283773f},
           { 0.95529972f,  0.90106534f,  0.11812832f},
           { 0.96489353f,  0.90232311f,  0.12394051f},
           { 0.97441665f,  0.90358991f,  0.13021494f},
           { 0.98386829f,  0.90486726f,  0.13689671f},
           { 0.99324789f,  0.90615657f,  0.1439362f}};

    }
}
