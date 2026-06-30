namespace corelib.Common
{
    // --------------------------------------------------------------------------------
    /// <summary>
    /// 
    /// </summary>
    // --------------------------------------------------------------------------------
    public static class CompareHelper
    {
        public const long UNIXDIFF_MS_MAX_TICKS = 10000000L;

        // ********************************************************************************
        /// <summary>
        /// Compares two unix dates.
        /// comparision 
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <returns>True if the criteriea for comparison are met.</returns>
        /// <remarks>
        /// Unix dates have the resolution of 1 second. This means 
        /// that from when we cast a datetime (UtcNow) to when we would read it back from a 
        /// database via a UnixTimeStamp cast we will have lost the sub second data. This 
        /// compariosn takes that into account and passes the comparison based on if the 
        /// absolute difference between the two UTC date time stamps are under 1 second. 
        /// </remarks>
        // ********************************************************************************
        public static bool ViperDatesAreEqual(DateTime left, DateTime right)
        {
            var delta = right - left;
            return left.Kind == right.Kind &&
                Math.Abs(delta.Ticks) < UNIXDIFF_MS_MAX_TICKS;
        }

        // ********************************************************************************
        /// <summary>
        /// This is the microsoft method of generating a version number based on date time
        /// This version number will be withing 2 seconcds accuracy of the passed date.
        /// </summary>
        /// <param name="major"></param>
        /// <param name="minor"></param>
        /// <param name="versionDate"></param>
        /// <returns></returns>
        // ********************************************************************************
        public static Version CreatePublishedVersionString(int major, int minor, DateTime versionDate)
        {
            // The third number in the version is the number of days since 1-Jan-2000,
            // and the fourth/last version number is the number of seconds since midnight
            // divided by 2.
            var epoc = new DateTime(2000, 1, 1, 0, 0, 0, DateTimeKind.Utc);
            int buildNum = Convert.ToInt32(Math.Floor((versionDate - epoc).TotalDays));
            int revNum = (Convert.ToInt32(Math.Floor(versionDate.TimeOfDay.TotalSeconds)) / 2);
            Version newv = new Version(major, minor, buildNum, revNum);
            return newv;
        }

        // ********************************************************************************
        /// <summary>
        /// This is the microsoft method of generating a date from a version number.
        /// This date will be withing 2 seconcds accuracy of the passed version.
        /// </summary>
        /// <param name="version"></param>
        /// <returns></returns>
        // ********************************************************************************
        public static DateTime FromVersion(Version version)
        {
            DateTime buildDate = new DateTime(2000, 1, 1)
                        .AddDays(version.Build).AddSeconds(version.Revision * 2);
            return buildDate;
        }

        // ********************************************************************************
        /// <summary>
        /// ObjectsAreNull checks to see if the objects sent are null and logs the 
        /// occurance if they are.  We only log the differences meaning if one object
        /// is null and the other isnt. By defintion both objects beging null results
        /// in them being identical and therefore not in need of logging the difference.
        /// </summary>
        /// <param name="leftObject"></param>
        /// <param name="rightObject"></param>
        /// <param name="log"></param>
        /// <returns></returns>
        // ********************************************************************************
        public static bool ObjectsAreNull(object leftObject, object rightObject, List<string>? log = null)
        {

            if (leftObject == null && rightObject == null)
                return true; //isnull

            if (leftObject == null)
            {
                log?.Add($"base (left) object is \"NULL\"");
                return true; //isnull
            }

            if (rightObject == null)
            {
                log?.Add($"{leftObject.GetType().Name} compare object is \"NULL\"");
                return true; //isnull
            }

            return false;
        }


        // ********************************************************************************
        /// <summary>
        /// ObjectsAreNullOrEqual quick comparison and logger to signal that the
        /// passed object to compare to is either null or 100 equal to this class.
        /// </summary>
        /// <param name="leftObject"></param>
        /// <param name="rightObject"></param>
        /// <param name="log"></param>
        /// <returns>boolean indicating that the objects are null or equal</returns>
        // ********************************************************************************
        public static bool ObjectsAreNullOrEqual(object leftObject, object rightObject, List<string>? log = null)
        {
            if (ObjectsAreNull(leftObject, rightObject, log))
                return true;

            if (object.ReferenceEquals(leftObject, rightObject))
                return true; //same exact object reference passed

            if (leftObject.Equals(rightObject))
                return true; //equals

            return false;
        }

        public static List<string> CompareStrings(string fieldname, string left, string right)
        {
            List<string> sr = new List<string>();

            if (!(string.IsNullOrEmpty(left) || string.IsNullOrEmpty(right)))
            {
                if (!left.Equals(right))
                    sr.Add($"{fieldname} has differences \"{left}\" != \"{right}\"");
            }
            else if (!string.IsNullOrEmpty(left) && string.IsNullOrEmpty(right)) sr.Add($"{fieldname} has differences \"{left}\" != \"NULL\"");
            else if (string.IsNullOrEmpty(left) && !string.IsNullOrEmpty(right)) sr.Add($"{fieldname} has differences \"NULL\" != \"{right}\"");

            return sr;
        }

        public static List<string> CompareGUIDs(string fieldname, Guid left, Guid right)
        {
            List<string> sr = new List<string>();

            if (!(Guid.Empty.Equals(left) || Guid.Empty.Equals(right)))
            {
                if (!left.Equals(right))
                    sr.Add($"{fieldname} has differences \"{left}\" != \"{right}\"");
            }
            else if (!Guid.Empty.Equals(left) && Guid.Empty.Equals(right)) sr.Add($"{fieldname} has differences \"{left}\" != \"NULL\"");
            else if (Guid.Empty.Equals(left) && !Guid.Empty.Equals(right)) sr.Add($"{fieldname} has differences \"NULL\" != \"{right}\"");

            return sr;
        }


        public static List<string> CompareISODateStrings(string fieldname, string left, string right)
        {

            List<string> sr = new List<string>();

            if (!(string.IsNullOrEmpty(left) || string.IsNullOrEmpty(right)))
            {
                if (!EquateISODateStrings(left, right))
                    sr.Add($"{fieldname} has differences \"{left}\" != \"{right}\"");
            }
            else if (!string.IsNullOrEmpty(left) && string.IsNullOrEmpty(right)) sr.Add($"{fieldname} has differences \"{left}\" != \"NULL\"");
            else if (string.IsNullOrEmpty(left) && !string.IsNullOrEmpty(right)) sr.Add($"{fieldname} has differences \"NULL\" != \"{right}\"");

            return sr;
        }

        public static bool EquateISODateStrings(string left, string right)
        {
            if (string.IsNullOrEmpty(left) && string.IsNullOrEmpty(right))
                return true;

            return TryISORoundTripParse(left, right) ||
                TryISORoundTripParse(left, right + "Z");
        }

        private static bool TryISORoundTripParse(string left, string right)
        {
            DateTime tryleft;
            DateTime tryRight;
            return DateTime.TryParse(left, null, System.Globalization.DateTimeStyles.RoundtripKind, out tryleft) &&
                DateTime.TryParse(right, null, System.Globalization.DateTimeStyles.RoundtripKind, out tryRight) &&
                tryleft == tryRight;
        }

        // ********************************************************************************
        /// <summary>
        /// 
        /// </summary>
        /// <param name="fieldname"></param>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <returns></returns>
        // ********************************************************************************
        public static List<string> CompareTwoViperDates(string fieldname, DateTime left, DateTime right)
        {
            List<string> sr = new List<string>();

            if (!CompareHelper.ViperDatesAreEqual(left, right))
                sr.Add($"{fieldname} has differences {left} != {right}");

            return sr;
        }

        // ********************************************************************************
        /// <summary>
        /// 
        /// </summary>
        /// <param name="fieldname"></param>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <returns></returns>
        // ********************************************************************************
        public static List<string> CompareTwoEnums(string fieldname, Enum left, Enum right)
        {
            List<string> sr = new List<string>();

            if (!left.Equals(right))
                sr.Add($"Enum {fieldname} has differences {left} != {right}");

            return sr;
        }


    }
}
