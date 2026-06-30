using System.Globalization;

namespace moonlib.util
{
    /// <summary>
    /// Utility class containing time-related methods
    /// </summary>
    public class ViperDate
    {
        //
        // Use these constructors to ensure UTC
        //
        public static DateTime Default = new DateTime(0, DateTimeKind.Utc);
        public static DateTime New() => new DateTime(0, DateTimeKind.Utc);
        public static DateTime Now() => new DateTime(DateTime.UtcNow.Ticks, DateTimeKind.Utc);
        public static DateTime New(DateTime d) => new DateTime(d.Year, d.Month, d.Day, d.Hour, d.Minute, d.Second, d.Millisecond, DateTimeKind.Utc);
        public static DateTime New(int year, int month, int day) => new DateTime(year, month, day, 0, 0, 0, DateTimeKind.Utc);
        public static DateTime New(int year, int month, int day, int hour, int minute, int second) => new DateTime(year, month, day, hour, minute, second, DateTimeKind.Utc);
        public static DateTime New(int year, int month, int day, int hour, int minute, int second, int millisecond) => new DateTime(year, month, day, hour, minute, second, millisecond, DateTimeKind.Utc);

        /// <summary>
        /// This will assert and make sure that we have a UTC based date time. If we are passed a non UTC date then 
        /// this class will cast it as a UTC type using the viper date new function. This way we "round up" to 
        /// being UTC.
        /// </summary>
        /// <param name="indate"></param>
        /// <returns></returns>
        public static DateTime AssertUTC(DateTime indate)
        {
            DateTime newDateVal;
            if (indate.Kind != DateTimeKind.Utc)
                newDateVal = ViperDate.New(indate);
            else
                newDateVal = indate;
            return newDateVal;
        }

        /// <summary>
        /// This will assert and make sure that we have a UTC based date time. If we are passed a non UTC date then 
        /// this class will cast it as a UTC type using the viper date new function. This way we "round up" to 
        /// being UTC.
        /// </summary>
        /// <param name="indate"></param>
        /// <returns></returns>
        public static DateTime AssertUTC(DateTime? nullableDate)
        {
            if (nullableDate is null)
                throw new NullReferenceException("Cannot Assert DateTime of type null");

            DateTime indate = (DateTime)nullableDate;

            return AssertUTC(indate);
        }

        public static DateTime New(long ticks)
        {
            long boundTicks = Math.Max(Math.Min(ticks, DateTime.MaxValue.Ticks), DateTime.MinValue.Ticks);
            return new DateTime(boundTicks, DateTimeKind.Utc);
        }

        public static DateTime Parse(string datestring) => datestring.ParseDate();

        public static bool TryParse(string datestring, out DateTime dt)
        {
            var styles = DateTimeStyles.AllowLeadingWhite | DateTimeStyles.AllowTrailingWhite;
            return DateTime.TryParseExact(datestring, Constants.FutureDateFormat, CultureInfo.InvariantCulture, styles, out dt);
        }

        public static DateTime UnixEpoch() => UNIX_EPOCH;

        public static readonly DateTime UNIX_EPOCH = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);

        /// <summary>
        /// For export dates that must be at least at the unix epoch or greater this function will
        /// make sure that the date passed is at least a recent at the unix epoch.
        /// </summary>
        /// <param name="dateTime"></param>
        /// <returns></returns>
        public static DateTime AssertDateIsAfterUnixEpoch(DateTime dateTime)
        {
            if(dateTime < UNIX_EPOCH)
                return UNIX_EPOCH;
            else
                return dateTime;
        }

        public static DateTime ParseGeneral(string str)
        {
            var d1 = DateTime.Parse(str);
            var d2 = ViperDate.New(d1.Year, d1.Month, d1.Day, d1.Hour, d1.Minute, d1.Second, 0);
            return d2;
        }

        public static List<DateTime> GetTimes(DateTime start, DateTime stop, TimeSpan step)
        {
            var times = new List<DateTime>();
            for (var t = start; t < stop; t += step)
                times.Add(t);
            return times;
        }
    }
}
