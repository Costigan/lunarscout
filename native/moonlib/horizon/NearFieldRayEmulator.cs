using moonlib.math;
using OSGeo.GDAL;
using System.Globalization;

namespace moonlib.horizon
{
    public class NearFieldRayEmulator
    {
        private const float NEARFIELD_BORDER_MARGIN_METERS = 5f;

        public static double[] Run(
            List<ElevationMap> dems,
            PixelOrigin origin,
            double azimuthDeg,
            float nearFieldClampMeters,
            float observerElevation,
            string outputPath,
            bool suppressCsv = false)
        {
            if (dems == null || dems.Count == 0)
                throw new ArgumentException("At least one DEM is required.", nameof(dems));

            Gdal.AllRegister();

            var innerPath = dems[0].Path;
            if (string.IsNullOrEmpty(innerPath))
                throw new InvalidOperationException("Inner DEM path is missing.");

            using var innerDs = Gdal.Open(innerPath, Access.GA_ReadOnly);
            if (innerDs == null)
                throw new InvalidOperationException($"Failed to open inner DEM at {innerPath}");

            double[] innerGt = new double[6];
            innerDs.GetGeoTransform(innerGt);

            // Compute bordered grid
            int tileCol = (int)origin.X; // Assuming origin X is global column index if working in tile context, but here origin is likely global pixel?
            // Wait, ReferenceRayEmulator and QuadTreeRayEmulator treat origin as global pixel coordinates.
            // QuadTreeHorizonGenerator.ComputeNearFieldBlock treats tileX/Y as the top-left corner of the *tile* being processed.
            // But here we are tracing a single ray from a specific point.
            // We need to define a "tile" that covers this point or just center the grid on this point.
            // Let's assume we create a 1x1 tile at the origin point's location for the purpose of the emulator.
            // Or better, we define a small tile around the point.
            
            // To reuse ComputeBorderedGrid exactly as is:
            // It expects tileCol, tileRow which are indices of the tile in the grid system if it were tiled?
            // No, looking at QuadTreeHorizonGenerator usage: 
            // GenerateHorizons(..., tileX, tileY, width, height...) 
            // tileX/Y are PIXEL coordinates of the top-left corner of the tile.
            
            int tileX = (int)origin.X;
            int tileY = (int)origin.Y;
            int tileWidth = 1;
            int tileHeight = 1;

            double borderMeters = nearFieldClampMeters + NEARFIELD_BORDER_MARGIN_METERS;
            var (tempGt, tempCols, tempRows, borderPxX, borderPxY) = QuadTreeHorizonGenerator.ComputeBorderedGrid(innerGt, tileX, tileY, tileWidth, tileHeight, borderMeters);
            var bounds = QuadTreeHorizonGenerator.GeoTransformBounds(tempGt, tempCols, tempRows);

            var innerBand = innerDs.GetRasterBand(1);
            var targetType = innerBand.DataType;
            innerBand.GetNoDataValue(out double targetNoData, out int hasNoData);
            if (hasNoData == 0) targetNoData = -9999.0;

            var tempBuffer = new float[tempRows * tempCols];
            for (int i = 0; i < tempBuffer.Length; i++) tempBuffer[i] = (float)targetNoData;

            string targetSrs = innerDs.GetProjectionRef();

            // Perform Warp
            foreach (var dem in Enumerable.Reverse(dems))
            {
                using var ds = Gdal.Open(dem.Path, Access.GA_ReadOnly);
                if (ds == null) continue;

                var band = ds.GetRasterBand(1);
                band.GetNoDataValue(out double srcNoData, out int srcHasNoData);
                if (srcHasNoData == 0) srcNoData = targetNoData;

                var warpArgs = QuadTreeHorizonGenerator.BuildWarpArgs(targetSrs, bounds, tempCols, tempRows, srcNoData, targetNoData, targetType);
                using var warpOpts = new GDALWarpAppOptions(warpArgs);
                using var warped = Gdal.Warp(string.Empty, new Dataset[] { ds }, warpOpts, null, null);
                if (warped == null) continue;

                var warpedBand = warped.GetRasterBand(1);
                var warpedBuf = new float[tempBuffer.Length];
                warpedBand.ReadRaster(0, 0, tempCols, tempRows, warpedBuf, tempCols, tempRows, 0, 0);

                QuadTreeHorizonGenerator.MergeWarp(tempBuffer, warpedBuf, (float)targetNoData);
            }

            // Execute Ray March
            float pixelSizeMeters = (float)((Math.Abs(innerGt[1]) + Math.Abs(innerGt[5])) * 0.5);
            pixelSizeMeters = Math.Max(0.01f, pixelSizeMeters);
            float maxDistMeters = nearFieldClampMeters;
            float noDataVal = (float)targetNoData;

            // Offset of the origin pixel in the temp buffer
            // The temp buffer starts at originX - borderPxX, originY - borderPxY relative to global coords
            // origin is at global (tileX, tileY)
            // So in local grid:
            // col = (globalCol - tileX) + borderPxX
            // row = (globalRow - tileY) + borderPxY
            // Here globalCol = tileX, globalRow = tileY
            int demCol = borderPxX; // + 0
            int demRow = borderPxY; // + 0

            var slopes = new List<double>();

            using (var writer = suppressCsv ? null : new StreamWriter(outputPath))
            {
                if (!suppressCsv)
                {
                    writer!.WriteLine("step_index,dist_m,pixel_x,pixel_y,elevation_m,slope,obs_z");
                }

                // Sample Observer
                float obsH = QuadTreeHorizonGenerator.SampleBilinearFlat(tempBuffer, tempCols, tempRows, (float)demCol, (float)demRow, noDataVal);
                float eps = Math.Abs(noDataVal) * 1e-5f + 1e-3f;
                
                if (QuadTreeHorizonGenerator.IsNoDataValue(obsH, noDataVal, eps))
                {
                    // Observer is NoData
                    return new double[] { };
                }

                float obsZ = obsH + observerElevation;
                
                // Setup March
                double azRad = azimuthDeg * (Math.PI / 180.0);
                // Convert azimuth to math angle (CCW from East) if necessary?
                // In Kernel: float angleRad = (float)(azIdx * (2.0 * XMath.PI / numAz));
                // Azimuth 0 is usually North?
                // QuadTreeHorizonGenerator: "azIdx * (2.0 * PI / 1440)" -> This implies 0 to 2PI.
                // Usually Azimuth 0 = North, 90 = East.
                // Math angle 0 = East, 90 = North.
                // In Kernel: dx = Cos(angle), dy = Sin(angle).
                // If Az=0 (North), Cos(0)=1 (East??), Sin(0)=0.
                // Wait.
                // In ReferenceHorizonGenerator:
                // var theta_true = 2d * PI * ((i / HorizonSamplesD) ...
                // var angle = (PI / 2d) - theta_true;
                // var dir_obs_frame = new Vector3d(Cos(angle), Sin(angle), 0d);
                
                // In QuadTreeRayEmulator:
                // var dirMe = ComputeDirectionVector(obsToMe, az);
                // inside ComputeDirectionVector: double angle = (Math.PI / 2.0) - azimuthRad;
                
                // In NearFieldRayKernel:
                // float angleRad = (float)(azIdx * (2.0 * XMath.PI / numAz));
                // float dx = XMath.Cos(angleRad) * stepPx;
                // float dy = XMath.Sin(angleRad) * stepPx;
                // It treats the input index directly as the angle for Cos/Sin.
                // If index 0 maps to Azimuth 0 (North), then Cos(0)=1 -> X+ -> East?
                // In GDAL/GeoTIFF, X+ is East, Y+ is North (if not flipped).
                // However, usually Azimuth 0 is North.
                // If Az=0 -> dx=1, dy=0 (East).
                // If Az=90 -> dx=0, dy=1 (North or South depending on Y axis).
                // This suggests the Kernel MIGHT be interpreting Azimuth 0 as East?
                // Or maybe the Azimuth definition in `NearFieldRayKernel` matches the math convention (0 = East).
                // Let's assume input `azimuthDeg` needs to be converted to the same radian convention used in Kernel.
                // If the user passes "True North Azimuth", we might need to adjust.
                // But `QuadTreeHorizonGenerator` passes `azIdx * (2.0 * PI / 1440)` directly.
                // If `azIdx=0` corresponds to North in the output file, then the kernel calculates East rays for North azimuths?
                // Actually, `QuadTreeRayEmulator` takes `azimuthDeg` and does `PI/2 - az`.
                // Let's stick to what the Kernel does: It takes an index `azIdx`, converts to rads, uses Cos/Sin.
                // So if we pass `azimuthDeg`, we should convert it to rads: `azimuthDeg * PI / 180`.
                // Note: If `azimuthDeg` is 0 (North), and kernel treats it as 0 rads (East), there is a rotation.
                // But here we just want to emulate the kernel. If we pass "90" (East), we expect the kernel logic to use 90.
                
                float angleRad = (float)((azimuthDeg - 90.0) * (Math.PI / 180.0));
                
                float stepMeters = pixelSizeMeters;
                float stepPx = stepMeters / pixelSizeMeters; // = 1.0
                float dx = (float)Math.Cos(angleRad) * stepPx;
                float dy = (float)Math.Sin(angleRad) * stepPx;

                float px = (float)demCol;
                float py = (float)demRow;
                float traveled = 0f;
                float maxSlope = float.NegativeInfinity;
                int stepIndex = 0;

                while (traveled <= maxDistMeters)
                {
                    px += dx;
                    py += dy;
                    traveled += stepMeters;

                    if (px < 1f || py < 1f || px >= tempCols - 1 || py >= tempRows - 1)
                        break;

                    float h = QuadTreeHorizonGenerator.SampleBilinearFlat(tempBuffer, tempCols, tempRows, px, py, noDataVal);
                    if (QuadTreeHorizonGenerator.IsNoDataValue(h, noDataVal, eps))
                        continue;

                    float slope = (h - obsZ) / Math.Max(traveled, 0.01f);
                    if (slope > maxSlope)
                        maxSlope = slope;

                    slopes.Add(slope);

                    if (!suppressCsv)
                    {
                        writer!.WriteLine($"{stepIndex},{traveled},{px},{py},{h},{slope},{obsZ}");
                    }
                    stepIndex++;
                }
            }

            return slopes.ToArray();
        }
    }
}
