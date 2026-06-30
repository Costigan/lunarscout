using System.Drawing;

namespace moonlib.util
{
    public static class Constants
    {
        public const double MissionSiteCenterLatitude = -85.42088;
        public const double MissionSiteCenterLongitude = 31.6218;

        public const string DefaultDurationTimeFormat = @"d\:hh\:mm\:ss";
        public const string DefaultDateFormat = "yyyy-MM-ddTHH:mm:ss.fffZ";
        public const string ExcelExportDateFormat = "yyyy-MM-ddTHH:mm:ssZ";
        public const string DateToFilenameFormat = "yyyy-MM-ddTHH-mm-ss";
        public const string FutureDateFormat = "yyyy-MM-ddTHH:mm:ssZ";
        public const string DateFileFormat = "yyyy-MM-ddTHH-mm-ss";

        //TODO: These should be in the configuration file
        public static DateTime MissionPeriodStart = ViperDate.New(2025, 9, 1);     // Used for searches
        public static DateTime MissionPeriodStop = ViperDate.New(2026, 4, 30);
        //public static DateTime MissionPeriodStop = ViperDate.New(2025, 9, 2);

        #region Environment Constants

        public static int MaxDegreeOfParallelism = Environment.ProcessorCount - 2;

        #endregion

        #region Dataset Constants

        public const int PSRSizeThreshold = 5;          // PSRs must be at least 9 pixels; changed to 7 on 3/3/2018 to look at overnights, changed to 5 on 4/21/20
        public const float SafeHavenMaxHours = 70f;
        public static float MetersPerPixel = 20f;       // Changed with a dataset is loaded

        public const long ByteLayerDataSeparator = 0xA5A5A5A5;    // Used to read and write the Sun data file (fast access to the sun images)
        public const int ByteDataPatchSize = 128;

        /// <summary>
        /// This determines the size of terrain patches
        /// </summary>
        public const int TerrainPatchSize = 128;

        #endregion

        #region Horizon Constants

        public const int HorizonSamples = 360 * 4;
        public const float HorizonSamplesF = (float)HorizonSamples;
        public const double HorizonSamplesD = (double)HorizonSamples;

        #endregion

        public const UInt32 PsrVal = 0xFFFF0000;
        public const UInt32 PsrBackgroundVal = 0x00000000;

        #region Color Constants

        public static int AreaOuterIndexToCenterIndex(int index) => index + 1;  // This works for Area indices, not for havens (could try changing that) 

        public static Color Lighten(Color c, int d = 10) => Color.FromArgb(Clamp(c.R + d), Clamp(c.G + d), Clamp(c.B + d));
        public static Color Darken(Color c, int d = 10) => Color.FromArgb(Clamp(c.R - d), Clamp(c.G - d), Clamp(c.B - d));
        public static int Clamp(int c) => c < 0 ? 0 : c > 255 ? 255 : c;

        #endregion Color Constants
    }
}
