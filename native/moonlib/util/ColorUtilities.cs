using System;
using System.Drawing;

namespace moonlib.util
{
    public static class ColorUtilities
    {
        /// <summary>
        /// Converts RGB to HSB.
        /// </summary>
        public static HSB RGBtoHSB(int red, int green, int blue)
        {
            // normalize red, green and blue values
            double r = ((double)red / 255.0);
            double g = ((double)green / 255.0);
            double b = ((double)blue / 255.0);

            // conversion start
            double max = Math.Max(r, Math.Max(g, b));
            double min = Math.Min(r, Math.Min(g, b));

            double h = 0.0;
            if (max == r && g >= b)
            {
                h = 60 * (g - b) / (max - min);
            }
            else if (max == r && g < b)
            {
                h = 60 * (g - b) / (max - min) + 360;
            }
            else if (max == g)
            {
                h = 60 * (b - r) / (max - min) + 120;
            }
            else if (max == b)
            {
                h = 60 * (r - g) / (max - min) + 240;
            }

            double s = (max == 0) ? 0.0 : (1.0 - (min / max));

            return new HSB(h, s, (double)max);
        }

        public static HSB RGBtoHSB(RGB rgb) => RGBtoHSB(rgb.Red, rgb.Green, rgb.Blue);

        /// <summary>
        /// Converts HSB to RGB.
        /// </summary>
        public static RGB HSBtoRGB(double hue, double sat, double bri)
        {
            double r = 0;
            double g = 0;
            double b = 0;

            if (sat == 0)
            {
                r = g = b = bri;
            }
            else
            {
                // the color wheel consists of 6 sectors. Figure out which sector
                // you're in.
                double sectorPos = hue / 60.0;
                int sectorNumber = (int)(Math.Floor(sectorPos));
                // get the fractional part of the sector
                double fractionalSector = sectorPos - sectorNumber;

                // calculate values for the three axes of the color.
                double p = bri * (1.0 - sat);
                double q = bri * (1.0 - (sat * fractionalSector));
                double t = bri * (1.0 - (sat * (1 - fractionalSector)));

                // assign the fractional colors to r, g, and b based on the sector
                // the angle is in.
                switch (sectorNumber)
                {
                    case 0:
                        r = bri;
                        g = t;
                        b = p;
                        break;
                    case 1:
                        r = q;
                        g = bri;
                        b = p;
                        break;
                    case 2:
                        r = p;
                        g = bri;
                        b = t;
                        break;
                    case 3:
                        r = p;
                        g = q;
                        b = bri;
                        break;
                    case 4:
                        r = t;
                        g = p;
                        b = bri;
                        break;
                    case 5:
                        r = bri;
                        g = p;
                        b = q;
                        break;
                }
            }

            return new RGB(
                Convert.ToInt32(r * 255.0),
                Convert.ToInt32(g * 255.0),
                Convert.ToInt32(b * 255.0)
            );
        }

        public static RGB HSBtoRGB(HSB hsb) => HSBtoRGB(hsb.Hue, hsb.Saturation, hsb.Brightness);

        public static Color ToColor(double hue, double sat, double bri) => ToColor(HSBtoRGB(hue, sat, bri));
        public static Color ToColor(RGB c) => Color.FromArgb(c.Red, c.Green, c.Blue);

        public static Color MultiplyBrightness(Color c, double factor)
        {
            var hsb = RGBtoHSB(new RGB(c));
            hsb.Brightness *= factor;
            return hsb.ToColor();
        }

        public static Color MultiplySaturation(Color c, double factor)
        {
            var hsb = RGBtoHSB(new RGB(c));
            hsb.Saturation *= factor;
            return hsb.ToColor();
        }

        public static Color SetAlpha(Color c, int alpha) => Color.FromArgb(alpha, c);

        public static Color SetBrightness(Color c, double brightness)
        {
            var hsb = RGBtoHSB(new RGB(c));
            hsb.Brightness = brightness;
            return hsb.ToColor();
        }

        public static Color SetSaturation(Color c, double saturation)
        {
            var hsb = RGBtoHSB(new RGB(c));
            hsb.Saturation = saturation;
            return hsb.ToColor();
        }
    }

    /// <summary>
    /// RGB structure.
    /// </summary>
    public struct RGB
    {
        /// <summary>
        /// Gets an empty RGB structure;
        /// </summary>
        public static readonly RGB Empty = new RGB();

        private int red;
        private int green;
        private int blue;

        public static bool operator ==(RGB item1, RGB item2)
        {
            return (
                item1.Red == item2.Red
                && item1.Green == item2.Green
                && item1.Blue == item2.Blue
                );
        }

        public static bool operator !=(RGB item1, RGB item2)
        {
            return (
                item1.Red != item2.Red
                || item1.Green != item2.Green
                || item1.Blue != item2.Blue
                );
        }

        /// <summary>
        /// Gets or sets red value.
        /// </summary>
        public int Red
        {
            get => red;
            set => red = (value > 255) ? 255 : ((value < 0) ? 0 : value);
        }

        /// <summary>
        /// Gets or sets red value.
        /// </summary>
        public int Green
        {
            get => green;
            set => green = (value > 255) ? 255 : ((value < 0) ? 0 : value);
        }

        /// <summary>
        /// Gets or sets red value.
        /// </summary>
        public int Blue
        {
            get => blue;
            set => blue = (value > 255) ? 255 : ((value < 0) ? 0 : value);
        }

        public RGB(int R, int G, int B)
        {
            this.red = (R > 255) ? 255 : ((R < 0) ? 0 : R);
            this.green = (G > 255) ? 255 : ((G < 0) ? 0 : G);
            this.blue = (B > 255) ? 255 : ((B < 0) ? 0 : B);
        }

        public RGB(Color c) => (this.red, this.green, this.blue) = (c.R, c.G, c.B);

        public Color ToColor() => Color.FromArgb(this.red, this.green, this.blue);
        public Color ToColor(int alpha) => Color.FromArgb(alpha, this.red, this.green, this.blue);

        public override bool Equals(Object? obj)
        {
            if (obj == null || GetType() != obj.GetType()) return false;

            return (this == (RGB)obj);
        }

        public override int GetHashCode()
        {
            return Red.GetHashCode() ^ Green.GetHashCode() ^ Blue.GetHashCode();
        }
    }

    /// <summary>
    /// Structure to define HSB.
    /// </summary>
    public struct HSB
    {
        /// <summary>
        /// Gets an empty HSB structure;
        /// </summary>
        public static readonly HSB Empty = new HSB();

        private double hue;
        private double saturation;
        private double brightness;

        public static bool operator ==(HSB item1, HSB item2)
        {
            return (
                item1.Hue == item2.Hue
                && item1.Saturation == item2.Saturation
                && item1.Brightness == item2.Brightness
                );
        }

        public static bool operator !=(HSB item1, HSB item2)
        {
            return (
                item1.Hue != item2.Hue
                || item1.Saturation != item2.Saturation
                || item1.Brightness != item2.Brightness
                );
        }

        /// <summary>
        /// Gets or sets the hue component.
        /// </summary>
        public double Hue
        {
            get => hue;
            set => hue = (value > 360) ? 360 : ((value < 0) ? 0 : value);
        }

        /// <summary>
        /// Gets or sets saturation component.
        /// </summary>
        public double Saturation
        {
            get => saturation;
            set => saturation = (value > 1) ? 1 : ((value < 0) ? 0 : value);
        }

        /// <summary>
        /// Gets or sets the brightness component.
        /// </summary>
        public double Brightness
        {
            get => brightness;
            set => brightness = (value > 1) ? 1 : ((value < 0) ? 0 : value);
        }

        /// <summary>
        /// Creates an instance of a HSB structure.
        /// </summary>
        /// <param name="h">Hue value.</param>
        /// <param name="s">Saturation value.</param>
        /// <param name="b">Brightness value.</param>
        public HSB(double h, double s, double b)
        {
            hue = (h > 360) ? 360 : ((h < 0) ? 0 : h);
            saturation = (s > 1) ? 1 : ((s < 0) ? 0 : s);
            brightness = (b > 1) ? 1 : ((b < 0) ? 0 : b);
        }

        public Color ToColor() => ColorUtilities.HSBtoRGB(this).ToColor();
        public Color ToColor(int alpha) => ColorUtilities.HSBtoRGB(this).ToColor(alpha);

        public override bool Equals(Object? obj)
        {
            if (obj == null || GetType() != obj.GetType()) return false;

            return (this == (HSB)obj);
        }

        public override int GetHashCode()
        {
            return Hue.GetHashCode() ^ Saturation.GetHashCode() ^
                Brightness.GetHashCode();
        }
    }

    public static class ColorExtensions
    {
        public static Color SetAlpha(this Color c, int alpha) => ColorUtilities.SetAlpha(c, alpha);
        public static Color SetBrightness(this Color c, double brightness) => ColorUtilities.SetBrightness(c, brightness);
        public static Color SetSaturation(this Color c, double saturation) => ColorUtilities.SetSaturation(c, saturation);
        public static Color MultiplyBrightness(this Color c, double brightness) => ColorUtilities.MultiplyBrightness(c, brightness);
        public static Color MultiplySaturation(this Color c, double saturation) => ColorUtilities.MultiplySaturation(c, saturation);

        public static Color Interpolate(this Color c1, Color c2, float factor)
        {
            return Color.FromArgb(
                (byte)Interpolate(c1.R, c2.R, factor),
                (byte)Interpolate(c1.G, c2.G, factor),
                (byte)Interpolate(c1.B, c2.B, factor));
            float Interpolate(float a, float b, float f) => a + (b - a) * f;
        }


        public static void WriteToCSV(this System.Drawing.Imaging.ColorPalette palette, string path)
        {
            using (var sw = new System.IO.StreamWriter(path))
                foreach (var c in palette.Entries)
                    sw.WriteLine($"{c.A},{c.R},{c.G},{c.B}");
        }
    }

}
