using ILGPU;
using ILGPU.Algorithms;
using ILGPU.IR.Values;
using ILGPU.Runtime;
using moonlib.mapops;
using moonlib.math;
using moonlib.pipeline;
using moonlib.spice;
using moonlib.util;
using Serilog;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;

namespace moonlib.horizon
{
    /// <summary>
    /// Carries the six components of a GDAL-style GeoTransform to the GPU kernel.
    /// These match GeoTransform[0..5] from ElevationMap.
    /// </summary>
    public struct GeoTransformD
    {
        public double T0; // origin X  (GeoTransform[0])
        public double T1; // pixel width  (GeoTransform[1])
        public double T2; // rotation X  (GeoTransform[2])
        public double T3; // origin Y  (GeoTransform[3])
        public double T4; // rotation Y  (GeoTransform[4])
        public double T5; // pixel height (GeoTransform[5])
    }

    /// <summary>
    /// Result of processing one 128×128 patch.
    /// </summary>
    public readonly struct PatchElevationResult
    {
        /// <summary>Zero-based column index of the patch within the DEM grid.</summary>
        public int PatchCol { get; init; }

        /// <summary>Zero-based row index of the patch within the DEM grid.</summary>
        public int PatchRow { get; init; }

        /// <summary>
        /// Elevation (degrees) for every pixel and every time step.
        /// Dimensions: [128, 128, timeCount].
        /// </summary>
        public float[] Data { get; init; }
    }

    /// <summary>
    /// Result of processing one 128×128 patch.
    /// </summary>
    public readonly struct PatchPSRResult
    {
        /// <summary>Zero-based column index of the patch within the DEM grid.</summary>
        public int PatchCol { get; init; }

        /// <summary>Zero-based row index of the patch within the DEM grid.</summary>
        public int PatchRow { get; init; }

        /// <summary>
        /// Elevation (degrees) for every pixel and every time step.
        /// Dimensions: [128, 128, timeCount].
        /// </summary>
        public byte[] Data { get; init; }
    }

    /// <summary>
    /// Result of processing one 128x128 threshold-bit patch.
    /// </summary>
    public readonly struct PatchThresholdResult
    {
        /// <summary>Absolute sample offset of the patch within the DEM grid.</summary>
        public int PatchCol { get; init; }

        /// <summary>Absolute line offset of the patch within the DEM grid.</summary>
        public int PatchRow { get; init; }

        /// <summary>
        /// Threshold bitset for every pixel and every time step.
        /// Dimensions: [128, 128, timeCount].
        /// Bits 0-3 are Sun thresholds; bits 4-7 are Earth thresholds.
        /// </summary>
        public byte[] Data { get; init; }
    }

    /// <summary>
    /// GPU-accelerated computation of solar elevation angles (and eventually
    /// Earth visibility) across patches of a lunar DEM.
    /// </summary>
    public class Lightmaps : IDisposable
    {
        private const int PatchSize = 128;
        private const int DefaultMaxConcurrentStreams = 4;
        private const float RadiansToDegrees = 57.29577951308232f;

        private Context? _context;
        private Accelerator? _accelerator;
        private ConcurrentStack<AcceleratorStream>? _streamPool;
        private ConcurrentStack<MemoryBuffer1D<float, Stride1D.Dense>>? _patchDemBuffers;
        private ConcurrentStack<MemoryBuffer1D<float, Stride1D.Dense>>? _patchHorizonBuffers;
        ConcurrentStack<MemoryBuffer1D<byte, Stride1D.Dense>>? _patchByteOutputBuffers;

        private readonly int _maxConcurrentStreams;
        private bool _disposed;

        public Exception? BackgroundTaskError { get; private set; }

        // -----------------------------------------------------------------
        // Pre-compiled kernel delegate (stream-based).
        //
        // demElevation is a per‑patch 128×128 sub‑rectangle of the full DEM.
        // tileColBase / tileRowBase give the absolute pixel origin of that
        // sub‑rectangle so the kernel can compute CRS coordinates.
        // -----------------------------------------------------------------
        private Action<
            AcceleratorStream,
            Index2D,
            ArrayView<float>,                     // demElevation  (128*128 patch)
            int,                                  // demWidth      (always 128)
            int,                                  // demHeight     (always 128)
            GeoTransformD,                        // geotransform
            ProjectionParamsDouble,               // projection parameters
            ArrayView<float>,                     // sunVectors    (flat, 3*timeCount)
            int,                                  // timeCount
            int,                                  // tileColBase  (absolute sample)
            int,                                  // tileRowBase  (absolute line)
            ArrayView<float>                      // output        (128*128*timeCount)
        >? _elevationKernel;

        // -----------------------------------------------------------------
        // Pre-compiled kernel delegate (stream-based).
        //
        // demElevation is a per‑patch 128×128 sub‑rectangle of the full DEM.
        // tileColBase / tileRowBase give the absolute pixel origin of that
        // sub‑rectangle so the kernel can compute CRS coordinates.
        // -----------------------------------------------------------------
        private Action<
            AcceleratorStream,
        Index2D,
        ArrayView<float>,                     // demElevation  (128*128 patch)
        int,                                  // demWidth      (always 128)
        int,                                  // demHeight     (always 128)
        GeoTransformD,                        // geotransform
        ProjectionParamsDouble,               // projection parameters
        ArrayView<float>,                     // sunVectors    (flat, 3*timeCount)
        ArrayView<float>,                     // horizons      (flat, 1440 * 128 * 128)
        int,                                  // timeCount
        int,                                  // tileColBase  (absolute sample)
        int,                                  // tileRowBase  (absolute line)
        ArrayView<float>                      // output        (128*128*timeCount)
        >? _elevationAboveHorizonKernel;

        private Action<
        AcceleratorStream,
        Index2D,
        ArrayView<float>,                     // demElevation  (128*128 patch)
        int,                                  // demWidth      (always 128)
        int,                                  // demHeight     (always 128)
        GeoTransformD,                        // geotransform
        ProjectionParamsDouble,               // projection parameters
        ArrayView<float>,                     // sunVectors    (flat, 3*timeCount)
        ArrayView<float>,                     // earthVectors  (flat, 3*timeCount)
        ArrayView<float>,                     // horizons      (flat, 1440 * 128 * 128)
        ArrayView<float>,                     // sunThresholds (4)
        ArrayView<float>,                     // earthThresholds (4)
        int,                                  // timeCount
        int,                                  // tileColBase  (absolute sample)
        int,                                  // tileRowBase  (absolute line)
        ArrayView<byte>                       // output        (128*128*timeCount)
        >? _thresholdBitsKernel;

        // -----------------------------------------------------------------
        // Pre-compiled kernel delegate (stream-based).
        //
        // demElevation is a per‑patch 128×128 sub‑rectangle of the full DEM.
        // tileColBase / tileRowBase give the absolute pixel origin of that
        // sub‑rectangle so the kernel can compute CRS coordinates.
        // -----------------------------------------------------------------
        private Action<
            AcceleratorStream,
        Index1D,
        ArrayView<float>,                     // demElevation  (128*128 patch)
        int,                                  // demWidth      (always 128)
        int,                                  // demHeight     (always 128)
        GeoTransformD,                        // geotransform
        ProjectionParamsDouble,               // projection parameters
        ArrayView<float>,                     // sunVectors    (flat, 3*timeCount)
        ArrayView<float>,                     // horizons      (flat, 1440 * 128 * 128)
        int,                                  // timeCount
        int,                                  // tileColBase   (absolute sample)
        int,                                  // tileRowBase   (absolute line)
        ArrayView<byte>                       // output        (128*128)
        >? _PSRKernel;

        // -----------------------------------------------------------------
        // Shared per-pixel ENU reference frame, computed once per kernel
        // invocation and re-used across all time steps.
        // -----------------------------------------------------------------
        private struct PixelEnuFrame
        {
            public int pixelIdx;   // lineInPatch * PatchSize + sampleInPatch
            public float r00, r01, r02;
            public float r10, r11, r12;
            public float r20, r21, r22;
            public float tX, tY, tZ;
        }

        // -----------------------------------------------------------------
        // Steps 1–5 common to every Lightmaps kernel: pixel → CRS → lon/lat
        // → Moon‑ME Cartesian → ENU basis → float conversion.
        // -----------------------------------------------------------------
        static PixelEnuFrame ComputePixelEnuFrame(
            int absSample, int absLine,
            int sampleInPatch, int lineInPatch,
            GeoTransformD geotransform,
            ProjectionParamsDouble proj,
            ArrayView<float> demElevation,
            int demWidth)
        {
            // ---- Step 1 — PixelToCRS (double) ----------------------------
            double crsX = geotransform.T0
                        + geotransform.T1 * (double)absSample
                        + geotransform.T2 * (double)absLine;
            double crsY = geotransform.T3
                        + geotransform.T4 * (double)absSample
                        + geotransform.T5 * (double)absLine;

            // ---- Step 2 — Stereographic → lon/lat (radians) (double) -----
            double xp = crsX - proj.FalseEasting;
            double yp = crsY - proj.FalseNorthing;
            double rho = Math.Sqrt(xp * xp + yp * yp);

            double lonRad, latRad;
            if (rho <= 1e-12)
            {
                lonRad = proj.Lon0;
                latRad = proj.Lat0;
            }
            else
            {
                double c = 2.0 * Math.Atan2(rho, 2.0 * proj.K0 * proj.R);
                double sinC = Math.Sin(c);
                double cosC = Math.Cos(c);
                double cosLat0 = Math.Cos(proj.Lat0);
                double sinLat0 = Math.Sin(proj.Lat0);

                latRad = Math.Asin(
                    cosC * sinLat0 + (yp * sinC * cosLat0) / rho);
                lonRad = proj.Lon0 + Math.Atan2(
                    xp * sinC,
                    rho * cosLat0 * cosC - yp * sinLat0 * sinC);
            }

            // ---- Step 3 — Elevation + Moon‑ME Cartesian (double) ---------
            double elev = (double)demElevation[lineInPatch * demWidth + sampleInPatch];
            double r = proj.R + elev;

            double cosLat = Math.Cos(latRad);
            double sinLat = Math.Sin(latRad);
            double cosLon = Math.Cos(lonRad);
            double sinLon = Math.Sin(lonRad);

            double moonMeX = r * cosLat * cosLon;
            double moonMeY = r * cosLat * sinLon;
            double moonMeZ = r * sinLat;

            // ---- Step 4 — GetMoonMEToENU rotation basis (double) ---------
            double upX = cosLat * cosLon;
            double upY = cosLat * sinLon;
            double upZ = sinLat;
            double eastX = -sinLon;
            double eastY = cosLon;
            double eastZ = 0.0;
            double northX = -sinLat * cosLon;
            double northY = -sinLat * sinLon;
            double northZ = cosLat;

            double transX = -(moonMeX * eastX + moonMeY * eastY + moonMeZ * eastZ);
            double transY = -(moonMeX * northX + moonMeY * northY + moonMeZ * northZ);
            double transZ = -(moonMeX * upX + moonMeY * upY + moonMeZ * upZ);

            // ---- Step 5 — double → float conversion ----------------------
            return new PixelEnuFrame
            {
                pixelIdx = lineInPatch * PatchSize + sampleInPatch,
                r00 = (float)eastX,
                r01 = (float)northX,
                r02 = (float)upX,
                r10 = (float)eastY,
                r11 = (float)northY,
                r12 = (float)upY,
                r20 = (float)eastZ,
                r21 = (float)northZ,
                r22 = (float)upZ,
                tX = (float)transX,
                tY = (float)transY,
                tZ = (float)transZ,
            };
        }
        // -----------------------------------------------------------------
        // Kernel: compute solar elevation angles for one 128×128 patch.
        // Output: elevation (degrees) per pixel per time step.
        // -----------------------------------------------------------------
        static void ComputeElevationAnglesKernel(
            Index2D index,
            ArrayView<float> demElevation,
            int demWidth,
            int demHeight,
            GeoTransformD geotransform,
            ProjectionParamsDouble proj,
            ArrayView<float> sunVectors,
            int timeCount,
            int tileColBase,
            int tileRowBase,
            ArrayView<float> output)
        {
            int sampleInPatch = index.X;
            int lineInPatch = index.Y;

            if (sampleInPatch >= PatchSize || lineInPatch >= PatchSize)
                return;

            var frame = ComputePixelEnuFrame(
                tileColBase + sampleInPatch, tileRowBase + lineInPatch,
                sampleInPatch, lineInPatch,
                geotransform, proj, demElevation, demWidth);

            for (int t = 0; t < timeCount; t++)
            {
                float svX = sunVectors[t * 3 + 0];
                float svY = sunVectors[t * 3 + 1];
                float svZ = sunVectors[t * 3 + 2];

                float enuX = svX * frame.r00 + svY * frame.r10 + svZ * frame.r20 + frame.tX;
                float enuY = svX * frame.r01 + svY * frame.r11 + svZ * frame.r21 + frame.tY;
                float enuZ = svX * frame.r02 + svY * frame.r12 + svZ * frame.r22 + frame.tZ;

                float horizontal = XMath.Sqrt(enuX * enuX + enuY * enuY);
                float elevationDeg = XMath.Atan2(enuZ, horizontal) * RadiansToDegrees;

                output[frame.pixelIdx * timeCount + t] = elevationDeg;
            }
        }

        // -----------------------------------------------------------------
        // Kernel: compute solar elevation above the local horizon.
        // Output: (solar elevation − horizon elevation) per pixel per time step.
        // -----------------------------------------------------------------
        static void ComputeElevationAnglesAboveHorizonKernel(
            Index2D index,
            ArrayView<float> demElevation,
            int demWidth,
            int demHeight,
            GeoTransformD geotransform,
            ProjectionParamsDouble proj,
            ArrayView<float> sunVectors,
            ArrayView<float> horizons,
            int timeCount,
            int tileColBase,
            int tileRowBase,
            ArrayView<float> output)
        {
            int sampleInPatch = index.X;
            int lineInPatch = index.Y;

            if (sampleInPatch >= PatchSize || lineInPatch >= PatchSize)
                return;

            var frame = ComputePixelEnuFrame(
                tileColBase + sampleInPatch, tileRowBase + lineInPatch,
                sampleInPatch, lineInPatch,
                geotransform, proj, demElevation, demWidth);

            int horizonOffset = frame.pixelIdx * 1440;

            for (int t = 0; t < timeCount; t++)
            {
                float svX = sunVectors[t * 3 + 0];
                float svY = sunVectors[t * 3 + 1];
                float svZ = sunVectors[t * 3 + 2];

                float enuX = svX * frame.r00 + svY * frame.r10 + svZ * frame.r20 + frame.tX;
                float enuY = svX * frame.r01 + svY * frame.r11 + svZ * frame.r21 + frame.tY;
                float enuZ = svX * frame.r02 + svY * frame.r12 + svZ * frame.r22 + frame.tZ;

                float horizontal = XMath.Sqrt(enuX * enuX + enuY * enuY);
                float elevationDeg = XMath.Atan2(enuZ, horizontal) * RadiansToDegrees;

                float azimuthRad = XMath.Atan2(enuX, enuY);
                if (azimuthRad < 0f) azimuthRad += XMath.PI * 2f;
                float azimuthDeg = azimuthRad * RadiansToDegrees;

                float azimuthIndexF = azimuthDeg * 4f;
                if (azimuthIndexF >= 1440f)
                    azimuthIndexF = 0f;
                int horizonIndex1 = (int)azimuthIndexF;
                float horizonFrac = azimuthIndexF - (float)horizonIndex1;
                int horizonIndex2 = horizonIndex1 + 1;
                if (horizonIndex2 >= 1440)
                    horizonIndex2 = 0;

                float h1 = horizons[horizonOffset + horizonIndex1];
                float h2 = horizons[horizonOffset + horizonIndex2];
                float horizonElev = h1 + horizonFrac * (h2 - h1);

                output[frame.pixelIdx * timeCount + t] = elevationDeg - horizonElev;
            }
        }

        // -----------------------------------------------------------------
        // Kernel: compute one threshold bitset byte per pixel per time step.
        // Bits 0-3 encode Sun margin >= sunThresholds[0..3].
        // Bits 4-7 encode Earth margin >= earthThresholds[0..3].
        // -----------------------------------------------------------------
        static void ComputeThresholdBitsKernel(
            Index2D index,
            ArrayView<float> demElevation,
            int demWidth,
            int demHeight,
            GeoTransformD geotransform,
            ProjectionParamsDouble proj,
            ArrayView<float> sunVectors,
            ArrayView<float> earthVectors,
            ArrayView<float> horizons,
            ArrayView<float> sunThresholds,
            ArrayView<float> earthThresholds,
            int timeCount,
            int tileColBase,
            int tileRowBase,
            ArrayView<byte> output)
        {
            int sampleInPatch = index.X;
            int lineInPatch = index.Y;

            if (sampleInPatch >= PatchSize || lineInPatch >= PatchSize)
                return;

            var frame = ComputePixelEnuFrame(
                tileColBase + sampleInPatch, tileRowBase + lineInPatch,
                sampleInPatch, lineInPatch,
                geotransform, proj, demElevation, demWidth);

            int horizonOffset = frame.pixelIdx * 1440;

            for (int t = 0; t < timeCount; t++)
            {
                float svX = sunVectors[t * 3 + 0];
                float svY = sunVectors[t * 3 + 1];
                float svZ = sunVectors[t * 3 + 2];

                float sunEnuX = svX * frame.r00 + svY * frame.r10 + svZ * frame.r20 + frame.tX;
                float sunEnuY = svX * frame.r01 + svY * frame.r11 + svZ * frame.r21 + frame.tY;
                float sunEnuZ = svX * frame.r02 + svY * frame.r12 + svZ * frame.r22 + frame.tZ;

                float sunHorizontal = XMath.Sqrt(sunEnuX * sunEnuX + sunEnuY * sunEnuY);
                float sunElevationDeg = XMath.Atan2(sunEnuZ, sunHorizontal) * RadiansToDegrees;

                float sunAzimuthRad = XMath.Atan2(sunEnuX, sunEnuY);
                if (sunAzimuthRad < 0f) sunAzimuthRad += XMath.PI * 2f;
                float sunAzimuthDeg = sunAzimuthRad * RadiansToDegrees;

                float sunAzimuthIndexF = sunAzimuthDeg * 4f;
                if (sunAzimuthIndexF >= 1440f)
                    sunAzimuthIndexF = 0f;
                int sunHorizonIndex1 = (int)sunAzimuthIndexF;
                float sunHorizonFrac = sunAzimuthIndexF - (float)sunHorizonIndex1;
                int sunHorizonIndex2 = sunHorizonIndex1 + 1;
                if (sunHorizonIndex2 >= 1440)
                    sunHorizonIndex2 = 0;

                float sunH1 = horizons[horizonOffset + sunHorizonIndex1];
                float sunH2 = horizons[horizonOffset + sunHorizonIndex2];
                float sunHorizonElev = sunH1 + sunHorizonFrac * (sunH2 - sunH1);
                float sunMarginDeg = sunElevationDeg - sunHorizonElev;

                float evX = earthVectors[t * 3 + 0];
                float evY = earthVectors[t * 3 + 1];
                float evZ = earthVectors[t * 3 + 2];

                float earthEnuX = evX * frame.r00 + evY * frame.r10 + evZ * frame.r20 + frame.tX;
                float earthEnuY = evX * frame.r01 + evY * frame.r11 + evZ * frame.r21 + frame.tY;
                float earthEnuZ = evX * frame.r02 + evY * frame.r12 + evZ * frame.r22 + frame.tZ;

                float earthHorizontal = XMath.Sqrt(earthEnuX * earthEnuX + earthEnuY * earthEnuY);
                float earthElevationDeg = XMath.Atan2(earthEnuZ, earthHorizontal) * RadiansToDegrees;

                float earthAzimuthRad = XMath.Atan2(earthEnuX, earthEnuY);
                if (earthAzimuthRad < 0f) earthAzimuthRad += XMath.PI * 2f;
                float earthAzimuthDeg = earthAzimuthRad * RadiansToDegrees;

                float earthAzimuthIndexF = earthAzimuthDeg * 4f;
                if (earthAzimuthIndexF >= 1440f)
                    earthAzimuthIndexF = 0f;
                int earthHorizonIndex1 = (int)earthAzimuthIndexF;
                float earthHorizonFrac = earthAzimuthIndexF - (float)earthHorizonIndex1;
                int earthHorizonIndex2 = earthHorizonIndex1 + 1;
                if (earthHorizonIndex2 >= 1440)
                    earthHorizonIndex2 = 0;

                float earthH1 = horizons[horizonOffset + earthHorizonIndex1];
                float earthH2 = horizons[horizonOffset + earthHorizonIndex2];
                float earthHorizonElev = earthH1 + earthHorizonFrac * (earthH2 - earthH1);
                float earthMarginDeg = earthElevationDeg - earthHorizonElev;

                byte bits = 0;
                if (sunMarginDeg >= sunThresholds[0]) bits |= (byte)1;
                if (sunMarginDeg >= sunThresholds[1]) bits |= (byte)2;
                if (sunMarginDeg >= sunThresholds[2]) bits |= (byte)4;
                if (sunMarginDeg >= sunThresholds[3]) bits |= (byte)8;
                if (earthMarginDeg >= earthThresholds[0]) bits |= (byte)16;
                if (earthMarginDeg >= earthThresholds[1]) bits |= (byte)32;
                if (earthMarginDeg >= earthThresholds[2]) bits |= (byte)64;
                if (earthMarginDeg >= earthThresholds[3]) bits |= (byte)128;

                output[frame.pixelIdx * timeCount + t] = bits;
            }
        }

        // -----------------------------------------------------------------
        // Kernel: one byte per pixel — 255 if never lit (PSR), 0 otherwise.
        // -----------------------------------------------------------------
        static void ComputePSRKernel(
            Index1D index,
            ArrayView<float> demElevation,
            int demWidth,
            int demHeight,
            GeoTransformD geotransform,
            ProjectionParamsDouble proj,
            ArrayView<float> sunVectors,
            ArrayView<float> horizons,
            int timeCount,
            int tileColBase,
            int tileRowBase,
            ArrayView<byte> output)
        {
            int sampleInPatch = index % PatchSize;
            int lineInPatch = index / PatchSize;

            if (sampleInPatch >= PatchSize || lineInPatch >= PatchSize)
                return;

            var frame = ComputePixelEnuFrame(
                tileColBase + sampleInPatch, tileRowBase + lineInPatch,
                sampleInPatch, lineInPatch,
                geotransform, proj, demElevation, demWidth);

            // a value of 255 indicates permanent shadow.  Initialize to that.
            byte is_psr = 255;
            int horizonOffset = frame.pixelIdx * 1440;

            for (int t = 0; t < timeCount; t++)
            {
                float svX = sunVectors[t * 3 + 0];
                float svY = sunVectors[t * 3 + 1];
                float svZ = sunVectors[t * 3 + 2];

                float enuX = svX * frame.r00 + svY * frame.r10 + svZ * frame.r20 + frame.tX;
                float enuY = svX * frame.r01 + svY * frame.r11 + svZ * frame.r21 + frame.tY;
                float enuZ = svX * frame.r02 + svY * frame.r12 + svZ * frame.r22 + frame.tZ;

                float horizontal = XMath.Sqrt(enuX * enuX + enuY * enuY);
                float elevationDeg = XMath.Atan2(enuZ, horizontal) * RadiansToDegrees;

                float azimuthRad = XMath.Atan2(enuX, enuY);
                if (azimuthRad < 0f) azimuthRad += XMath.PI * 2f;
                float azimuthDeg = azimuthRad * RadiansToDegrees;

                float azimuthIndexF = azimuthDeg * 4f;
                if (azimuthIndexF >= 1440f)
                    azimuthIndexF = 0f;
                int horizonIndex1 = (int)azimuthIndexF;
                float horizonFrac = azimuthIndexF - (float)horizonIndex1;
                int horizonIndex2 = horizonIndex1 + 1;
                if (horizonIndex2 >= 1440)
                    horizonIndex2 = 0;

                float h1 = horizons[horizonOffset + horizonIndex1];
                float h2 = horizons[horizonOffset + horizonIndex2];
                float horizonElev = h1 + horizonFrac * (h2 - h1);

                const float sun_angular_size_deg = 0.545f;
                const float limb_threshold = -sun_angular_size_deg / 2f;

                // If the sun is high enough that its upper limb is above the horizon, then
                // set is_psr to 0 indicating that this pixel is not in permanent shadow.
                if (elevationDeg - horizonElev > limb_threshold)
                    is_psr = 0;

                // Do not exit early so as to keep the threads in sync
            }

            output[frame.pixelIdx] = is_psr;
        }

        // =================================================================
        // Public API
        // =================================================================

        public Lightmaps(int maxConcurrentStreams = DefaultMaxConcurrentStreams)
        {
            if (maxConcurrentStreams <= 0)
                throw new ArgumentOutOfRangeException(
                    nameof(maxConcurrentStreams), maxConcurrentStreams,
                    "GPU concurrency must be greater than zero.");
            _maxConcurrentStreams = maxConcurrentStreams;
        }

        /// <summary>
        /// Ensure the GPU accelerator, reusable patch-DEM buffers, stream
        /// pool, and pre-compiled kernel are initialised.  Idempotent.
        /// </summary>
        void EnsureInitialized()
        {
            if (_accelerator is not null)
                return;

            Console.WriteLine("[Lightmaps] Creating ILGPU context...");
            Console.Out.Flush();
            _context = Context.Create(builder =>
                builder.Default()
                       .DebugSymbols(DebugSymbolsMode.Kernel)
                       .EnableAlgorithms());
            Console.WriteLine("[Lightmaps] ILGPU context created, enumerating devices...");
            Console.Out.Flush();

            var cudaDevice = _context.Devices.FirstOrDefault(
                d => d.AcceleratorType == AcceleratorType.Cuda);
            var oclNvidiaDevice = _context.Devices.FirstOrDefault(
                d => d.AcceleratorType == AcceleratorType.OpenCL
                  && d.Name.IndexOf("NVIDIA", StringComparison.OrdinalIgnoreCase) >= 0);
            var oclAnyDevice = _context.Devices.FirstOrDefault(
                d => d.AcceleratorType == AcceleratorType.OpenCL);
            var chosenDevice = cudaDevice ?? oclNvidiaDevice ?? oclAnyDevice
                               ?? _context.GetPreferredDevice(preferCPU: true);

            Console.WriteLine(
                $"[Lightmaps] Using device: {chosenDevice.Name} ({chosenDevice.AcceleratorType})");
            Console.Out.Flush();

            Console.WriteLine("[Lightmaps] Creating accelerator...");
            Console.Out.Flush();
            _accelerator = chosenDevice.CreateAccelerator(_context);
            Console.WriteLine("[Lightmaps] Accelerator created, compiling kernels...");
            Console.Out.Flush();

            // Pre-compile the kernel once.
            _elevationKernel = _accelerator.LoadAutoGroupedKernel<
                Index2D,
                ArrayView<float>,
                int,
                int,
                GeoTransformD,
                ProjectionParamsDouble,
                ArrayView<float>,
                int,
                int,
                int,
                ArrayView<float>>(ComputeElevationAnglesKernel);

            _elevationAboveHorizonKernel = _accelerator.LoadAutoGroupedKernel<
                Index2D,
                ArrayView<float>,
                int,
                int,
                GeoTransformD,
                ProjectionParamsDouble,
                ArrayView<float>,
                ArrayView<float>,
                int,
                int,
                int,
                ArrayView<float>>(ComputeElevationAnglesAboveHorizonKernel);

            _thresholdBitsKernel = _accelerator.LoadAutoGroupedKernel<
                Index2D,
                ArrayView<float>,
                int,
                int,
                GeoTransformD,
                ProjectionParamsDouble,
                ArrayView<float>,
                ArrayView<float>,
                ArrayView<float>,
                ArrayView<float>,
                ArrayView<float>,
                int,
                int,
                int,
                ArrayView<byte>>(ComputeThresholdBitsKernel);

            _PSRKernel = _accelerator.LoadAutoGroupedKernel<
                    Index1D,
                    ArrayView<float>,
                    int,
                    int,
                    GeoTransformD,
                    ProjectionParamsDouble,
                    ArrayView<float>,
                    ArrayView<float>,
                    int,
                    int,
                    int,
                    ArrayView<byte>>(ComputePSRKernel);

            Console.WriteLine("[Lightmaps] Kernels compiled OK.");
            Console.Out.Flush();
        }

        void EnsureStreamPool()
        {
            EnsureInitialized();
            if (_streamPool != null)
                return;
            _streamPool = new ConcurrentStack<AcceleratorStream>();
            for (int i = 0; i < _maxConcurrentStreams; i++)
                _streamPool.Push(_accelerator.CreateStream());
        }

        AcceleratorStream GetStreamFromPool()
        {
            AcceleratorStream stream;
            while (!_streamPool!.TryPop(out stream))
                Task.Delay(1).Wait();
            return stream;
        }

        void EnsureDemBuffers()
        {
            EnsureInitialized();
            if (_patchDemBuffers != null)
                return;
            int buffer_size = PatchSize * PatchSize;
            _patchDemBuffers = new ConcurrentStack<MemoryBuffer1D<float, Stride1D.Dense>>();
            for (int i = 0; i < _maxConcurrentStreams; i++)
                _patchDemBuffers.Push(_accelerator.Allocate1D<float>(buffer_size));
        }

        MemoryBuffer1D<float, Stride1D.Dense> GetDemBufferFromPool()
        {
            MemoryBuffer1D<float, Stride1D.Dense> buffer;
            while (!_patchDemBuffers!.TryPop(out buffer))
                Task.Delay(1).Wait();
            return buffer;
        }

        void EnsureHorizonBuffers()
        {
            EnsureInitialized();
            if (_patchHorizonBuffers != null)
                return;
            int buffer_size = PatchSize * PatchSize * 1440;
            _patchHorizonBuffers = new ConcurrentStack<MemoryBuffer1D<float, Stride1D.Dense>>();
            for (int i = 0; i < _maxConcurrentStreams; i++)
                _patchHorizonBuffers.Push(_accelerator.Allocate1D<float>(buffer_size));
        }

        MemoryBuffer1D<float, Stride1D.Dense> GetHorizonBufferFromPool()
        {
            MemoryBuffer1D<float, Stride1D.Dense> buffer;
            while (!_patchHorizonBuffers!.TryPop(out buffer))
                Task.Delay(1).Wait();
            return buffer;
        }

        void EnsureByteOutputBuffers(int buffer_size)
        {
            EnsureInitialized();
            if (_patchByteOutputBuffers != null)
            {
                if (_patchByteOutputBuffers.TryPeek(out var existing) && existing.Length == buffer_size)
                    return;
                while (_patchByteOutputBuffers.TryPop(out var oldBuffer))
                    oldBuffer.Dispose();
                _patchByteOutputBuffers = null;
            }
            if (_patchByteOutputBuffers != null)
                return;
            _patchByteOutputBuffers = new ConcurrentStack<MemoryBuffer1D<byte, Stride1D.Dense>>();
            for (int i = 0; i < _maxConcurrentStreams; i++)
                _patchByteOutputBuffers.Push(_accelerator.Allocate1D<byte>(buffer_size));
        }

        MemoryBuffer1D<byte, Stride1D.Dense> GetByteOutputBufferFromPool(int buffer_size)
        {
            MemoryBuffer1D<byte, Stride1D.Dense> buffer;
            while (!_patchByteOutputBuffers!.TryPop(out buffer))
                Task.Delay(1).Wait();
            if (buffer.Length != buffer_size)
                throw new Exception($"GetByteOutputBufferFromPool fetch a buffer of size {buffer.Length} that should have been {buffer_size}");
            return buffer;
        }

        /// <summary>
        /// Stream per‑pixel solar elevation angles for every 128×128 patch
        /// of the DEM via a producer–consumer BlockingCollection.
        /// </summary>
        public BlockingCollection<PatchElevationResult> StreamElevationPatches(
            string demPath,
            List<DateTime> times,
            IProgress<float>? progress = null,
            Func<bool>? isCancellationRequested = null)
        {
            if (demPath == null) throw new ArgumentNullException(nameof(demPath));
            var dem = new ElevationMap(demPath);

            var time_step_hrs = 6f;
            var all_sunvecs_me = ViperDate
                .GetTimes(ViperDate.New(1970, 1, 1), ViperDate.New(2044, 1, 1), TimeSpan.FromHours(time_step_hrs))
                .Select(t => SpiceManager.SunPosition_meters(t)).ToList();

            var sunvecs_me = MapOperations.GenerateReducedSunVectorListForPermanentShadowCalculation(dem, all_sunvecs_me);
            Console.WriteLine($"Generated reduced sun vector list for permanent shadow calculation. From {all_sunvecs_me.Count} to {sunvecs_me.Count}");

            var session = PrepareSession(dem: dem, sunVecsD: sunvecs_me);
            var queue = new BlockingCollection<PatchElevationResult>(boundedCapacity: 32);

            var workerThread = new Thread(() =>
            {
                try
                {
                    int patchesX = session.Dem.Width / PatchSize;
                    int patchesY = session.Dem.Height / PatchSize;
                    int totalPatches = patchesX * patchesY;
                    int patchesProcessed = 0;

                    var tokens = new List<LightmapProcessingToken>(totalPatches);
                    for (int py = 0; py < patchesY; py++)
                        for (int px = 0; px < patchesX; px++)
                            tokens.Add(new LightmapProcessingToken { col = px, row = py });

                    var pipeline = new Pipeline<LightmapProcessingToken>();
                    pipeline.AddStep(ProcessElevationPatch, maxDegreeOfParallelism: _maxConcurrentStreams);
                    pipeline.AddTerminalStep(async token =>
                    {
                        var data = token.float_results;
                        queue.Add(new PatchElevationResult { PatchCol = token.col, PatchRow = token.row, Data = data });

                        if (progress != null)
                        {
                            int current = Interlocked.Increment(ref patchesProcessed);
                            progress.Report((float)current / totalPatches);
                        }
                    }, maxDegreeOfParallelism: _maxConcurrentStreams);

                    pipeline.ProcessAsync(tokens).GetAwaiter().GetResult();

                    return;

                    Task<LightmapProcessingToken> ProcessElevationPatch(LightmapProcessingToken token)
                    {
                        int tileColBase = token.col * PatchSize;
                        int tileRowBase = token.row * PatchSize;

                        float[] patchDem = TiledGeotiffWriter.ExtractPatchDem(session.Dem.Elevation, tileColBase, tileRowBase);

                        var stream = GetStreamFromPool();
                        var gpuPatchDem = GetDemBufferFromPool();

                        try
                        {
                            gpuPatchDem.CopyFromCPU(patchDem);

                            using var gpuOutput = _accelerator.Allocate1D<float>(session.PerPatchOutputSize);

                            _elevationKernel!(
                                stream,
                                new Index2D(PatchSize, PatchSize),
                                gpuPatchDem.View,
                                PatchSize,
                                PatchSize,
                                session.Gt,
                                session.ProjD,
                                session.GpuSunVectors.View,
                                session.TimeCount,
                                tileColBase,
                                tileRowBase,
                                gpuOutput.View);

                            stream.Synchronize();

                            float[] flat = new float[session.PerPatchOutputSize];
                            gpuOutput.CopyToCPU(flat);
                            token.float_results = flat;

                            progress?.Report((float)patchesProcessed / totalPatches);
                            ThrowIfCancelled(isCancellationRequested);

                            return Task.FromResult(token);
                        }
                        finally
                        {
                            _streamPool!.Push(stream);
                            _patchDemBuffers!.Push(gpuPatchDem);
                        }
                    }
                }
                catch (Exception ex)
                {
                    BackgroundTaskError = ex;
                    Log.Fatal(ex, "StreamElevationPatches background task failed");
                }
                finally
                {
                    session.GpuEarthVectors.Dispose();
                    session.GpuSunVectors.Dispose();
                    queue.CompleteAdding();
                }
            });
            workerThread.Start();
            return queue;
        }

        /// <summary>
        /// Compute per‑pixel solar elevation angles for every 128×128 patch
        /// of the DEM.  Results are yielded as each batch of GPU work
        /// completes so the caller can start consuming them without waiting
        /// for every patch to finish.
        /// </summary>
        /// <param name="demPath">Path to a GeoTIFF DEM readable by ElevationMap.</param>
        /// <param name="times">UTC DateTimes for which sun positions are computed.</param>
        /// <returns>One PatchElevationResult per 128×128 patch, streamed batch‑by‑batch.</returns>
        public BlockingCollection<PatchElevationResult> StreamElevationOverTerrainPatches(
            string demPath,
            string horizonPath,
            List<DateTime> times,
            IProgress<float>? progress = null,
            Func<bool>? isCancellationRequested = null)
        {
            var queue = new BlockingCollection<PatchElevationResult>(boundedCapacity: 32);
            EnsureHorizonBuffers();
            var session = PrepareSession(demPath: demPath, times: times);

            var horizon_filenames = new HorizonTileStore(horizonPath)
                .EnumerateFiles(observerElevationMeters: 0f)
                .ToList();
            var totalCount = horizon_filenames.Count;
            Log.Information($"Found {totalCount} horizon files for lightmap generation.");

            var workerThread = new Thread(() =>
            {
                try
                {
                    int patchesProcessed = 0;

                    var pipeline = new Pipeline<LightmapProcessingToken>();
                    pipeline.AddStep(ReadHorizons, maxDegreeOfParallelism: _maxConcurrentStreams);
                    pipeline.AddStep(ProcessPatch, maxDegreeOfParallelism: _maxConcurrentStreams);
                    pipeline.AddTerminalStep(async token =>
                    {
                        var data = token.float_results;
                        queue.Add(new PatchElevationResult { PatchCol = token.col, PatchRow = token.row, Data = data });

                        if (progress != null)
                        {
                            int current = Interlocked.Increment(ref patchesProcessed);
                            progress.Report((float)current / totalCount);
                        }
                    }, maxDegreeOfParallelism: _maxConcurrentStreams);

                    pipeline.ProcessAsync(horizon_filenames.Select(f => new LightmapProcessingToken { filename = f })).GetAwaiter().GetResult();

                    return;

                    Task<LightmapProcessingToken> ProcessPatch(LightmapProcessingToken token)
                    {
                        int tileColBase = token.col;
                        int tileRowBase = token.row;

                        float[] patchDem = TiledGeotiffWriter.ExtractPatchDem(session.Dem.Elevation, tileColBase, tileRowBase);

                        var stream = GetStreamFromPool();
                        var gpuPatchDem = GetDemBufferFromPool();
                        var gpuHorizons = GetHorizonBufferFromPool();

                        try
                        {
                            gpuPatchDem.CopyFromCPU(patchDem);
                            gpuHorizons.CopyFromCPU(token.horizons);

                            using var gpuOutput = _accelerator.Allocate1D<float>(session.PerPatchOutputSize);

                            _elevationAboveHorizonKernel!(
                                stream,
                                new Index2D(PatchSize, PatchSize),
                                gpuPatchDem.View,
                                PatchSize,
                                PatchSize,
                                session.Gt,
                                session.ProjD,
                                session.GpuSunVectors.View,
                                gpuHorizons.View,
                                session.TimeCount,
                                tileColBase,
                                tileRowBase,
                                gpuOutput.View);

                            stream.Synchronize();

                            float[] flat = new float[session.PerPatchOutputSize];
                            gpuOutput.CopyToCPU(flat);
                            token.float_results = flat;

                            progress?.Report((float)patchesProcessed / totalCount);
                            ThrowIfCancelled(isCancellationRequested);

                            return Task.FromResult(token);
                        }
                        finally
                        {
                            _streamPool!.Push(stream);
                            _patchDemBuffers!.Push(gpuPatchDem);
                            _patchHorizonBuffers!.Push(gpuHorizons);
                        }
                    }
                }
                catch (Exception ex)
                {
                    BackgroundTaskError = ex;
                    Log.Fatal(ex, "StreamElevationOverTerrainPatches background task failed");
                }
                finally
                {
                    session.GpuEarthVectors.Dispose();
                    session.GpuSunVectors.Dispose();
                    queue.CompleteAdding();
                }
            });
            workerThread.Start();
            return queue;
        }

        /// <summary>
        /// Compute per‑pixel solar elevation angles for every 128×128 patch
        /// of the DEM.  Results are yielded as each batch of GPU work
        /// completes so the caller can start consuming them without waiting
        /// for every patch to finish.
        /// </summary>
        /// <param name="demPath">Path to a GeoTIFF DEM readable by ElevationMap.</param>
        /// <param name="times">UTC DateTimes for which sun positions are computed.</param>
        /// <returns>One PatchElevationResult per 128×128 patch, streamed batch‑by‑batch.</returns>
        public BlockingCollection<PatchPSRResult> StreamPSRPatches(
            string demPath,
            string horizonPath,
            IProgress<float>? progress = null,
            Func<bool>? isCancellationRequested = null)
        {
            var dem = new ElevationMap(demPath);

            var time_step_hrs = 6f;
            var all_sunvecs_me = ViperDate
                .GetTimes(ViperDate.New(1970, 1, 1), ViperDate.New(2044, 1, 1), TimeSpan.FromHours(time_step_hrs))
                .Select(t => SpiceManager.SunPosition_meters(t)).ToList();

            var sunvecs_me = MapOperations.GenerateReducedSunVectorListForPermanentShadowCalculation(dem, all_sunvecs_me);
            Console.WriteLine($"Generated reduced sun vector list for permanent shadow calculation. From {all_sunvecs_me.Count} to {sunvecs_me.Count}");

            var queue = new BlockingCollection<PatchPSRResult>(boundedCapacity: 32);

            var session = PrepareSession(dem: dem, sunVecsD: sunvecs_me);
            EnsureHorizonBuffers();
            EnsureByteOutputBuffers(PatchSize * PatchSize);

            var horizon_filenames = new HorizonTileStore(horizonPath)
                .EnumerateFiles(observerElevationMeters: 0f)
                .ToList();

            //var first_filename = horizon_filenames[0];

            // Debugging: will this change where the illegal memory accesses occur?
            //horizon_filenames.Reverse();
            //horizon_filenames = horizon_filenames.Skip(8700).ToList();

            // debugging.  The first failure occurrs with 
            //var bad_patches = new[] { (3712, 34304), (3712, 34176), (3712, 34048) };
            //horizon_filenames = horizon_filenames.Where(fn =>
            //{
            //    var (col, row, observer_elevation) = QuadTreeHorizonGenerator.ParseHorizonFilename(fn);
            //    return bad_patches.Contains((row, col));
            //}).ToList();

            //horizon_filenames.Insert(0, first_filename);

            var totalCount = horizon_filenames.Count;
            Log.Information($"Found {totalCount} horizon files for lightmap generation.");

            if (progress == null)
            {
                var stopwatch = Stopwatch.StartNew();
                progress = MapOperations.MakeProgress(stopwatch: stopwatch, stride: 50);
            }

            var workerThread = new Thread(() =>
            {
                try
                {
                    int patchesProcessed = 0;

                    var pipeline = new Pipeline<LightmapProcessingToken>();
                    pipeline.AddStep(ReadHorizons, maxDegreeOfParallelism: 8);
                    pipeline.AddStep(ProcessPatch, maxDegreeOfParallelism: _maxConcurrentStreams);
                    pipeline.AddTerminalStep(async token =>
                    {
                        var data = token.byte_results!;
                        queue.Add(new PatchPSRResult { PatchCol = token.col, PatchRow = token.row, Data = data });

                        if (progress != null)
                        {
                            int current = Interlocked.Increment(ref patchesProcessed);
                            progress.Report((float)current / totalCount);
                        }
                    }, maxDegreeOfParallelism: _maxConcurrentStreams);

                    pipeline.ProcessAsync(horizon_filenames.Select(f => new LightmapProcessingToken { filename = f })).GetAwaiter().GetResult();

                    return;

                    Task<LightmapProcessingToken> ProcessPatch(LightmapProcessingToken token)
                    {
                        if (token.horizons is null || token.horizons.Length != PatchSize * PatchSize * 1440)
                        {
                            Log.Warning($"Skipping {token.filename}: horizon data invalid (null={token.horizons is null}, len={token.horizons?.Length ?? 0})");
                            return Task.FromResult(token);
                        }

                        int tileColBase = token.col;
                        int tileRowBase = token.row;

                        float[] patchDem = TiledGeotiffWriter.ExtractPatchDem(session.Dem.Elevation, tileColBase, tileRowBase);

                        var stream = GetStreamFromPool();
                        var gpuPatchDem = GetDemBufferFromPool();
                        var gpuHorizons = GetHorizonBufferFromPool();
                        var gpuOutput = GetByteOutputBufferFromPool(PatchSize * PatchSize);

                        var location = 0;
                        try
                        {
                            gpuPatchDem.CopyFromCPU(patchDem);
                            location = 1;
                            gpuHorizons.CopyFromCPU(token.horizons);
                            location = 2;

                            _PSRKernel!(
                                stream,
                                new Index1D(PatchSize * PatchSize),
                                gpuPatchDem.View,
                                PatchSize,
                                PatchSize,
                                session.Gt,
                                session.ProjD,
                                session.GpuSunVectors.View,
                                gpuHorizons.View,
                                session.TimeCount,
                                tileColBase,
                                tileRowBase,
                                gpuOutput.View);
                            location = 3;

                            stream.Synchronize();

                            byte[] flat = new byte[PatchSize * PatchSize];
                            gpuOutput.CopyToCPU(flat);
                            token.byte_results = flat;

                            progress?.Report((float)patchesProcessed / totalCount);
                            ThrowIfCancelled(isCancellationRequested);

                            return Task.FromResult(token);
                        }
                        catch (Exception ex)
                        {
                            Console.Error.WriteLine($"Error occurred in the kernel item: {ex.Message} location={location} r={token.row} c={token.col}");
                            throw;
                        }
                        finally
                        {
                            _streamPool!.Push(stream);
                            _patchDemBuffers!.Push(gpuPatchDem);
                            _patchHorizonBuffers!.Push(gpuHorizons);
                            _patchByteOutputBuffers!.Push(gpuOutput);
                        }
                    }
                }
                catch (Exception ex)
                {
                    BackgroundTaskError = ex;
                    Log.Fatal(ex, "StreamElevationOverTerrainPatches background task failed");
                }
                finally
                {
                    session.GpuEarthVectors?.Dispose();
                    session.GpuSunVectors?.Dispose();
                    if (_patchByteOutputBuffers != null)
                        foreach (var buffer in _patchByteOutputBuffers)
                            buffer?.Dispose();
                    queue.CompleteAdding();
                }
            });
            workerThread.Start();
            return queue;
        }

        /// <summary>
        /// Compute one threshold bitset byte per pixel and time for every
        /// available 128x128 horizon patch. The supplied timestamps/vectors
        /// are assumed by callers to represent an evenly spaced time range.
        /// </summary>
        public BlockingCollection<PatchThresholdResult> StreamThresholdPatches(
            ElevationMap dem,
            string horizonPath,
            List<Vector3d> sunVectorsMe,
            List<Vector3d> earthVectorsMe,
            float[] sunThresholdsDeg,
            float[] earthThresholdsDeg,
            IProgress<float>? progress = null,
            Func<bool>? isCancellationRequested = null)
        {
            if (dem == null) throw new ArgumentNullException(nameof(dem));
            if (string.IsNullOrWhiteSpace(horizonPath))
                throw new ArgumentException("Horizon directory path must be non-empty.", nameof(horizonPath));
            if (sunVectorsMe.Count == 0)
                throw new ArgumentException("At least one Sun vector is required.", nameof(sunVectorsMe));
            if (earthVectorsMe.Count != sunVectorsMe.Count)
                throw new ArgumentException("Earth vector count must match Sun vector count.", nameof(earthVectorsMe));
            if (sunThresholdsDeg.Length != 4)
                throw new ArgumentException("Exactly four Sun thresholds are required.", nameof(sunThresholdsDeg));
            if (earthThresholdsDeg.Length != 4)
                throw new ArgumentException("Exactly four Earth thresholds are required.", nameof(earthThresholdsDeg));

            var queue = new BlockingCollection<PatchThresholdResult>(boundedCapacity: 32);

            var session = PrepareSession(
                dem: dem,
                sunVecsD: sunVectorsMe,
                earthVecsD: earthVectorsMe,
                sunThresholdsDeg: sunThresholdsDeg,
                earthThresholdsDeg: earthThresholdsDeg);
            EnsureHorizonBuffers();
            EnsureByteOutputBuffers(session.PerPatchOutputSize);

            var horizonFilenames = new HorizonTileStore(horizonPath)
                .EnumerateFiles(observerElevationMeters: 0f)
                .ToList();

            int totalCount = horizonFilenames.Count;
            Log.Information("Found {Count} horizon files for threshold-bit generation.", totalCount);

            var workerThread = new Thread(() =>
            {
                try
                {
                    int patchesProcessed = 0;

                    var pipeline = new Pipeline<LightmapProcessingToken>();
                    pipeline.AddStep(ReadHorizons, maxDegreeOfParallelism: 8);
                    pipeline.AddStep(ProcessPatch, maxDegreeOfParallelism: _maxConcurrentStreams);
                    pipeline.AddTerminalStep(token =>
                    {
                        if (token.byte_results != null)
                        {
                            queue.Add(new PatchThresholdResult
                            {
                                PatchCol = token.col,
                                PatchRow = token.row,
                                Data = token.byte_results,
                            });
                        }

                        int current = Interlocked.Increment(ref patchesProcessed);
                        progress?.Report(totalCount == 0 ? 1f : (float)current / totalCount);
                        return Task.CompletedTask;
                    }, maxDegreeOfParallelism: _maxConcurrentStreams);

                    pipeline.ProcessAsync(horizonFilenames.Select(f => new LightmapProcessingToken { filename = f })).GetAwaiter().GetResult();

                    return;

                    Task<LightmapProcessingToken> ProcessPatch(LightmapProcessingToken token)
                    {
                        if (token.horizons is null || token.horizons.Length != PatchSize * PatchSize * 1440)
                        {
                            Log.Warning(
                                "Skipping threshold generation for {Filename}: horizon data invalid (null={IsNull}, length={Length}).",
                                token.filename,
                                token.horizons is null,
                                token.horizons?.Length ?? 0);
                            return Task.FromResult(token);
                        }

                        int tileColBase = token.col;
                        int tileRowBase = token.row;
                        float[] patchDem = TiledGeotiffWriter.ExtractPatchDem(session.Dem.Elevation, tileColBase, tileRowBase);

                        var stream = GetStreamFromPool();
                        var gpuPatchDem = GetDemBufferFromPool();
                        var gpuHorizons = GetHorizonBufferFromPool();
                        var gpuOutput = GetByteOutputBufferFromPool(session.PerPatchOutputSize);

                        try
                        {
                            gpuPatchDem.CopyFromCPU(patchDem);
                            gpuHorizons.CopyFromCPU(token.horizons);

                            _thresholdBitsKernel!(
                                stream,
                                new Index2D(PatchSize, PatchSize),
                                gpuPatchDem.View,
                                PatchSize,
                                PatchSize,
                                session.Gt,
                                session.ProjD,
                                session.GpuSunVectors.View,
                                session.GpuEarthVectors.View,
                                gpuHorizons.View,
                                session.GpuSunThresholds!.View,
                                session.GpuEarthThresholds!.View,
                                session.TimeCount,
                                tileColBase,
                                tileRowBase,
                                gpuOutput.View);

                            stream.Synchronize();

                            byte[] flat = new byte[session.PerPatchOutputSize];
                            gpuOutput.CopyToCPU(flat);
                            token.byte_results = flat;

                            ThrowIfCancelled(isCancellationRequested);

                            return Task.FromResult(token);
                        }
                        finally
                        {
                            _streamPool!.Push(stream);
                            _patchDemBuffers!.Push(gpuPatchDem);
                            _patchHorizonBuffers!.Push(gpuHorizons);
                            _patchByteOutputBuffers!.Push(gpuOutput);
                        }
                    }
                }
                catch (Exception ex)
                {
                    BackgroundTaskError = ex;
                    Log.Fatal(ex, "StreamThresholdPatches background task failed");
                }
                finally
                {
                    session.GpuEarthVectors?.Dispose();
                    session.GpuSunVectors?.Dispose();
                    session.GpuEarthThresholds?.Dispose();
                    session.GpuSunThresholds?.Dispose();
                    queue.CompleteAdding();
                }
            });
            workerThread.Start();
            return queue;
        }

        public static void GeneratePSRGeotiff(string DEM_path, string HorizonDirectory, string psr_path)
        {
            var dem = new ElevationMap(DEM_path, loadRaster: false);
            using var outputDs = TiledGeotiffWriter.OpenTiled<byte>(psr_path, dem.Width, dem.Height, 1, -9999, dem.Projection, dem.GeoTransform);

            var lm = new Lightmaps(8);
            var queue = lm.StreamPSRPatches(DEM_path, HorizonDirectory);
            foreach (var r in queue.GetConsumingEnumerable())
                outputDs.WritePatch(r.PatchCol, r.PatchRow, r.Data);

            outputDs.FlushCache();

            if (lm.BackgroundTaskError is not null)
                throw new Exception("Background task failed", lm.BackgroundTaskError);
        }

        // =================================================================
        // Helpers
        // =================================================================

        static float[,,] ConvertFlatTo3D(float[] flat, int timeCount)
        {
            var data = new float[PatchSize, PatchSize, timeCount];
            for (int y = 0; y < PatchSize; y++)
            {
                for (int x = 0; x < PatchSize; x++)
                {
                    int bufOff = (y * PatchSize + x) * timeCount;
                    for (int t = 0; t < timeCount; t++)
                        data[y, x, t] = flat[bufOff + t];
                }
            }
            return data;
        }

        /// <summary>
        /// Holds the pre-computed data shared by every patch processed
        /// within a single streaming session.
        /// </summary>
        private sealed class LightmapSession
        {
            public ElevationMap Dem = null!;
            public int TimeCount;
            public int PerPatchOutputSize;
            public GeoTransformD Gt;
            public ProjectionParamsDouble ProjD;
            public MemoryBuffer1D<float, Stride1D.Dense> GpuSunVectors = null!;
            public MemoryBuffer1D<float, Stride1D.Dense> GpuEarthVectors = null!;
            public MemoryBuffer1D<float, Stride1D.Dense>? GpuSunThresholds;
            public MemoryBuffer1D<float, Stride1D.Dense>? GpuEarthThresholds;
        }

        /// <summary>
        /// Loads the DEM, generates sun/earth vectors, uploads them to the GPU,
        /// and returns a LightmapSession with everything needed by per‑patch
        /// processing.  The caller must dispose GpuSunVectors / GpuEarthVectors.
        /// </summary>
        LightmapSession PrepareSession(
            string? demPath = null,
            ElevationMap? dem = null,
            List<DateTime>? times = null,
            List<Vector3d>? sunVecsD = null,
            List<Vector3d>? earthVecsD = null,
            float[]? sunThresholdsDeg = null,
            float[]? earthThresholdsDeg = null)
        {
            EnsureInitialized();
            EnsureStreamPool();
            EnsureDemBuffers();

            if (dem == null)
            {
                if (demPath == null) throw new ArgumentNullException(nameof(demPath));
                dem = new ElevationMap(demPath, loadRaster: true);
            }

            if (dem.Elevation is null)
                throw new InvalidOperationException("DEM raster data is null.");

            if (dem.Width % PatchSize != 0 || dem.Height % PatchSize != 0)
                throw new ArgumentException(
                    $"DEM dimensions ({dem.Width}×{dem.Height}) must be multiples of {PatchSize}.");

            int timeCount = times != null ? times.Count : sunVecsD != null ? sunVecsD.Count : earthVecsD != null ? earthVecsD.Count : 0;
            if (timeCount == 0)
                throw new ArgumentException("One of times, sunVecsD or earthVecsD must be non-null");

            if (times != null)
            {
                sunVecsD = new List<Vector3d>(timeCount);
                earthVecsD = new List<Vector3d>(timeCount);
                for (int t = 0; t < timeCount; t++)
                {
                    sunVecsD.Add(SpiceManager.SunPosition_meters(times[t]));
                    earthVecsD.Add(SpiceManager.EarthPosition_meters(times[t]));
                }
            }

            float[] sunVecsFlat = null;
            float[] earthVecsFlat = null;

            MemoryBuffer1D<float, Stride1D.Dense> gpuSunVectors = null;
            MemoryBuffer1D<float, Stride1D.Dense> gpuEarthVectors = null;
            MemoryBuffer1D<float, Stride1D.Dense>? gpuSunThresholds = null;
            MemoryBuffer1D<float, Stride1D.Dense>? gpuEarthThresholds = null;

            if (sunVecsD != null)
            {
                sunVecsFlat = new float[timeCount * 3];
                for (int t = 0; t < timeCount; t++)
                {
                    int off = t * 3;
                    sunVecsFlat[off + 0] = (float)sunVecsD[t].X;
                    sunVecsFlat[off + 1] = (float)sunVecsD[t].Y;
                    sunVecsFlat[off + 2] = (float)sunVecsD[t].Z;
                }
                gpuSunVectors = _accelerator!.Allocate1D<float>(sunVecsFlat.Length);
                gpuSunVectors.CopyFromCPU(sunVecsFlat);
            }

            if (earthVecsD != null)
            {
                earthVecsFlat = new float[timeCount * 3];
                for (int t = 0; t < timeCount; t++)
                {
                    int off = t * 3;
                    earthVecsFlat[off + 0] = (float)earthVecsD[t].X;
                    earthVecsFlat[off + 1] = (float)earthVecsD[t].Y;
                    earthVecsFlat[off + 2] = (float)earthVecsD[t].Z;
                }
                gpuEarthVectors = _accelerator!.Allocate1D<float>(earthVecsFlat.Length);
                gpuEarthVectors.CopyFromCPU(earthVecsFlat);
            }

            if (sunThresholdsDeg != null)
            {
                gpuSunThresholds = _accelerator!.Allocate1D<float>(sunThresholdsDeg.Length);
                gpuSunThresholds.CopyFromCPU(sunThresholdsDeg);
            }

            if (earthThresholdsDeg != null)
            {
                gpuEarthThresholds = _accelerator!.Allocate1D<float>(earthThresholdsDeg.Length);
                gpuEarthThresholds.CopyFromCPU(earthThresholdsDeg);
            }

            return new LightmapSession
            {
                Dem = dem,
                TimeCount = timeCount,
                PerPatchOutputSize = PatchSize * PatchSize * timeCount,
                Gt = new GeoTransformD
                {
                    T0 = dem.GeoTransform[0],
                    T1 = dem.GeoTransform[1],
                    T2 = dem.GeoTransform[2],
                    T3 = dem.GeoTransform[3],
                    T4 = dem.GeoTransform[4],
                    T5 = dem.GeoTransform[5],
                },
                ProjD = new ProjectionParamsDouble
                {
                    R = dem.SrsDescriptor.R,
                    Lat0 = dem.SrsDescriptor.lat0,
                    Lon0 = dem.SrsDescriptor.lon0,
                    K0 = dem.SrsDescriptor.k0,
                    FalseEasting = dem.SrsDescriptor.FalseEasting,
                    FalseNorthing = dem.SrsDescriptor.FalseNorthing,
                },
                GpuSunVectors = gpuSunVectors,
                GpuEarthVectors = gpuEarthVectors,
                GpuSunThresholds = gpuSunThresholds,
                GpuEarthThresholds = gpuEarthThresholds,
            };
        }

        static Task<LightmapProcessingToken> ReadHorizons(LightmapProcessingToken token)
        {
            (token.col, token.row, token.observer_elevation) = QuadTreeHorizonGenerator.ParseHorizonFilename(token.filename);
            try
            {
                token.horizons = HorizonFile.ReadHorizonFile(token.filename);
            }
            catch (Exception ex)
            {
                Log.Error(ex, $"Failed to read horizon file: {token.filename}");
                token.horizons = null;
            }
            return Task.FromResult(token);
        }


        static void ThrowIfCancelled(Func<bool>? isCancellationRequested)
        {
            if (isCancellationRequested is not null && isCancellationRequested())
                throw new OperationCanceledException("Lightmaps operation was cancelled.");
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            if (_patchDemBuffers is not null)
            {
                while (_patchDemBuffers.TryPop(out var buf))
                    buf.Dispose();
                _patchDemBuffers = null;
            }

            if (_patchHorizonBuffers is not null)
            {
                while (_patchHorizonBuffers.TryPop(out var buf))
                    buf.Dispose();
                _patchHorizonBuffers = null;
            }

            if (_patchByteOutputBuffers is not null)
            {
                while (_patchByteOutputBuffers.TryPop(out var buf))
                    buf.Dispose();
                _patchByteOutputBuffers = null;
            }

            if (_streamPool is not null)
            {
                while (_streamPool.TryPop(out var stream))
                    stream.Dispose();
                _streamPool = null;
            }

            _accelerator?.Dispose();
            _accelerator = null;

            _context?.Dispose();
            _context = null;
        }
    }

    public class LightmapProcessingToken
    {
        public string? filename { get; set; }
        public int row { get; set; }
        public int col { get; set; }
        public float observer_elevation { get; set; }
        public float[]? horizons { get; set; }
        public float[]? float_results { get; set; }
        public byte[]? byte_results { get; set; }
    }
}
