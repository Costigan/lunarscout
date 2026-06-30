using moonlib.horizon;
using OSGeo.GDAL;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace moonlib.mapops
{
    public class AnalysisContext
    {
        public string? Name { get; set; }
        public string? Directory { get; set; }

        public string? DEM_path { get; set; }
        ElevationMap? _DEM { get; set; }
        public ElevationMap? DEM
        {
            get
            {
                if (_DEM  != null)
                    return _DEM;
                if (DEM_path != null)
                    return _DEM = new ElevationMap(DEM_path, loadRaster: true);
                else
                    return null;
            }
        }

        public List<string>? SurroundingDEM_paths { get; set; }
        protected List<ElevationMap>? _surroundingDEMs { get; set; }
        public List<ElevationMap>? SurroundingDEMs
        {
            get
            {
                if (_surroundingDEMs != null)
                    return _surroundingDEMs;
                if (SurroundingDEM_paths != null)
                    return _surroundingDEMs = SurroundingDEM_paths.Select(p => new ElevationMap(p)).ToList();
                else
                    return null;
            }
        }

        public string? HorizonDirectory { get; set; }
    }
}
