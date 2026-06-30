using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace moonlib.util
{
    public interface IMapCoordinateConverter
    {
        (double easting, double northing) RowCol2EastingNorthing(double row, double col);
        (double row, double col) EastingNorthing2RowCol(double easting, double northing);

        (double lat_deg, double lon_deg) Point2LatLonDeg(Point p);
        (double lat_deg, double lon_deg) RowCol2LatLonDeg(double row, double col);
        (double lat_deg, double lon_deg) EastingNorthingToLatLon(double easting, double northing);
    }
}
