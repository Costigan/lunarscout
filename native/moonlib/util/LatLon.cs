using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using moonlib.math;

namespace moonlib.util
{
    public struct LatLon
    {
        /// <summary>
        /// The latitude in the range [-90..90].
        /// </summary>
        public readonly double Latitude; 

        /// <summary>
        /// The longitude in the range [-180..180].
        /// </summary>
        public readonly double Longitude;

        //
        // Summary:
        //     Creates a new instance of Tinman.Terrain.Georef.LatLon.
        //
        // Parameters:
        //   latitude:
        //     The latitude in the range [-90..90].
        //
        //   longitude:
        //     The longitude in the range [-180..180].
        public LatLon(double latitude, double longitude)
        {
            Latitude = latitude;
            Longitude = longitude;
        }


        public LatLon(LatLon otherlatlon)
        {
            Latitude = otherlatlon.Latitude;
            Longitude = otherlatlon.Longitude;
        }

        /// <summary>
        /// Create a new LatLon
        /// </summary>
        /// <param name="latitude">Latitude in degrees</param>
        /// <param name="longitude">Longitude in degrees</param>
        /// <returns></returns>
        // Summary:
        //     Creates a new instance of Tinman.Terrain.Georef.LatLon from the given angles.
        public static LatLon FromDegrees(double latitude, double longitude) => new LatLon(latitude, longitude);

        /// <summary>
        /// Creates a new instance of Tinman.Terrain.Georef.LatLon from the given angles.
        /// </summary>
        /// <param name="latitude">The latitude angle, in radians.</param>
        /// <param name="longitude">The longitude angle, in radians.</param>
        /// <returns>A new LatLon with values in degrees</returns>
        public static LatLon FromRadians(double latitude, double longitude)
        {
            return new LatLon(latitude * (180.0 / Math.PI), longitude * (180.0 / Math.PI));
        }

        /// <summary>
        /// Computes an angle in degrees from the given DMS tuple.
        /// </summary>
        /// <param name="degrees">The degrees part.  If this is negative, then the whole deg/min/sec is treated as negative</param>
        /// <param name="minutes">The minutes part (1/60th degree).</param>
        /// <param name="seconds">The seconds part (1/60th minute).</param>
        /// <returns>A new LatLon</returns>
        public static double FromDegreeMinuteSecond(int degrees, int minutes = 0, double seconds = 0.0)
            => (degrees < 0 ? -1d : 1d) * (Math.Abs(degrees) + minutes / 60.0 + seconds / 3600.0);

        /// <summary>
        /// Are these geographic coordinates normalized?
        /// Value:
        ///     true if the coordinates are normalized, false if not.
        ///
        /// Remarks:
        ///     Normalized latitude angles are in the range [-90..90], normalized longitude angles
        ///     are in the range [-180..180].
        /// </summary>
        public bool IsNormalized
            => -90.0 <= Latitude && Latitude <= 90.0
            && -180.0 <= Longitude && Longitude <= 180.0;

        /// <summary>
        /// Compares this LatLon value with the given one.
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public bool Equals(LatLon other)
        {
            if (!Latitude.IsSimilar(other.Latitude))        // Latitudes have to be similar
                return false;
            if (Longitude.IsSimilar(other.Longitude))       // If longitude is also similar, then we're good
                return true;                                //  otherwise, if both longitudes are close to 180 (could be either side), then we're also good.
            return Math.Abs(Longitude).IsSimilar(180.0) && Math.Abs(other.Longitude).IsSimilar(180.0);
        }

        public static bool operator ==(LatLon left, LatLon right) => left.Equals(right);

        public static bool operator !=(LatLon left, LatLon right) => !left.Equals(right);

        public override string ToString() => $"<{Latitude},{Longitude}>";

        public override bool Equals(object? obj)
        {
            return base.Equals(obj);
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }
    }
}
