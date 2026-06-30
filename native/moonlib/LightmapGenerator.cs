using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using moonlib.horizon;
using moonlib.math;

namespace moonlib
{
    public class LightmapGenerator
    {
        public const int HorizonSamples = 1440; // 360 * 4
        public const float HorizonSamplesF = (float)HorizonSamples;
        
        const float SunHalfAngle_deg = 0.54f / 2f;
        const int half_circle_ticks = 8;
        static float[] half_circle = PremultiplyHalfCircle(MakeHalfCircle());
        static float max_photons = 2f * PremultiplyHalfCircle(MakeHalfCircle()).Sum();

        static float[] MakeHalfCircle() => Enumerable.Range(0, half_circle_ticks * 2).Select(i => (float)(Math.Sqrt(64d - (half_circle_ticks - 0.5d - i) * (half_circle_ticks - 0.5d - i)) / (half_circle_ticks))).ToArray();
        static float[] PremultiplyHalfCircle(float[] h) => h.Select(v => v * SunHalfAngle_deg).ToArray();

        public static unsafe float FastSunFraction(float* _buffer, float az_deg, float el_deg)
        {
            const float BucketWidth_deg = 360f / HorizonSamplesF;
            const float BucketHalfWidth_deg = BucketWidth_deg / 2f;
            const float SunHalfAngle = 0.27f; // SunHalfAngle_deg approx
            const float FracStep = 0.135f; // SunHalfAngle_deg / BucketWidth_deg / 8

            var sun_left_deg = az_deg - SunHalfAngle - BucketHalfWidth_deg;
            var sun_left_bucket_float = sun_left_deg * (HorizonSamplesF / 360f);
            var sun_left_bucket = (int)sun_left_bucket_float;
            var frac = sun_left_bucket_float - sun_left_bucket;
            
            // Handle wrap
            if (sun_left_bucket < 0) sun_left_bucket += HorizonSamples;
            else if (sun_left_bucket >= HorizonSamples) sun_left_bucket -= HorizonSamples;

            // Quick check against horizon bounds in this sector
            // The sun spans ~2-3 buckets. Check the max horizon in these buckets.
            int b0 = sun_left_bucket;
            int b1 = b0 + 1; if (b1 >= HorizonSamples) b1 = 0;
            int b2 = b1 + 1; if (b2 >= HorizonSamples) b2 = 0;

            float h0 = _buffer[b0];
            float h1 = _buffer[b1];
            float h2 = _buffer[b2];

            float maxH = (h0 > h1) ? h0 : h1;
            if (h2 > maxH) maxH = h2;
            
            float minH = (h0 < h1) ? h0 : h1;
            if (h2 < minH) minH = h2;

            // Early Exit: Sun fully above horizon
            if ((el_deg - SunHalfAngle) > maxH) return 1.0f;
            // Early Exit: Sun fully below horizon
            if ((el_deg + SunHalfAngle) < minH) return 0.0f;

            // Detailed Integration
            var left_bucket_index = b0;
            var right_bucket_index = b1;
            
            var left_bucket_elevation_deg = h0;
            var right_bucket_elevation_deg = h1;
            var bucket_delta_deg = right_bucket_elevation_deg - left_bucket_elevation_deg;

            var photons = 0f;

            fixed (float* hc = half_circle)
            {
                for (var i = 0; i < 16; i++) // half_circle_ticks * 2
                {
                    var horizon_elevation_deg = frac * bucket_delta_deg + left_bucket_elevation_deg;
                    var sun_column_deg = hc[i];
                    var sun_top_deg = el_deg + sun_column_deg;

                    // If sun slice is below horizon, no photons
                    if (horizon_elevation_deg < sun_top_deg)
                    {
                        var angle_delta = sun_top_deg - horizon_elevation_deg;
                        var sun_column_deg2 = sun_column_deg + sun_column_deg;
                        if (angle_delta > sun_column_deg2)
                            angle_delta = sun_column_deg2;

                        photons += angle_delta;
                    }

                    frac += FracStep;
                    if (frac >= 1f)
                    {
                        left_bucket_index = right_bucket_index;
                        right_bucket_index++;
                        if (right_bucket_index >= HorizonSamples) right_bucket_index = 0;

                        left_bucket_elevation_deg = right_bucket_elevation_deg;
                        right_bucket_elevation_deg = _buffer[right_bucket_index]; // Unsafe read
                        bucket_delta_deg = right_bucket_elevation_deg - left_bucket_elevation_deg;

                        frac -= 1f;
                    }
                }
            }

            return photons / max_photons;
        }

        // This code is from builder
        public static float BuilderSunFraction(float[] _buffer, int buffer_base, float az_deg, float el_deg)
        {
            const float BucketWidth_deg = 360f / HorizonSamplesF;
            const float BucketHalfWidth_deg = BucketWidth_deg / 2f;  // Used because we're interpolating between two buckets.

            const float frac_step_per_half_circle_index = SunHalfAngle_deg / BucketWidth_deg / half_circle_ticks;

            var sun_left_deg = az_deg - SunHalfAngle_deg - BucketHalfWidth_deg;   // 
            var sun_left_bucket_float = sun_left_deg * (HorizonSamplesF / 360f);
            var sun_left_bucket = (int)sun_left_bucket_float; // [0,1440)
            var frac = sun_left_bucket_float - sun_left_bucket;
            if (sun_left_bucket < 0) sun_left_bucket += HorizonSamples;  // [0,1440)

            // We're going to interpolate the horizon between two buckets.  sun_left_bucket is the index
            // of the left most of the two buckets for the first interpolation.  frac is the fraction of the
            // way that the sun's left edge between these two buckets

            var left_bucket_index = sun_left_bucket;
            var right_bucket_index = left_bucket_index + 1;
            if (right_bucket_index >= HorizonSamples) right_bucket_index -= HorizonSamples;

            var left_bucket_elevation_deg = _buffer[buffer_base + left_bucket_index];
            var right_bucket_elevation_deg = _buffer[buffer_base + right_bucket_index];
            var bucket_delta_deg = right_bucket_elevation_deg - left_bucket_elevation_deg;

            var photons = 0f;

            for (var i = 0; i < half_circle.Length; i++)
            {
                var horizon_elevation_deg = frac * bucket_delta_deg + left_bucket_elevation_deg;
                var sun_column_deg = half_circle[i];    // This is now pre-multiplied * SunHalfAngle_deg;
                var sun_top_deg = el_deg + sun_column_deg;

                if (horizon_elevation_deg >= sun_top_deg)
                    goto continue_loop;

                var angle_delta = sun_top_deg - horizon_elevation_deg;
                var sun_column_deg2 = sun_column_deg + sun_column_deg;  // Can't have more than this much light
                if (angle_delta > sun_column_deg2)
                    angle_delta = sun_column_deg2;

                photons += angle_delta;

continue_loop:

                frac += frac_step_per_half_circle_index;
                if (frac < 1f)
                    continue;

                // Move the bucket
                left_bucket_index = right_bucket_index;
                right_bucket_index = left_bucket_index + 1;
                if (right_bucket_index >= HorizonSamples) right_bucket_index -= HorizonSamples;     // Wrap

                left_bucket_elevation_deg = right_bucket_elevation_deg;
                right_bucket_elevation_deg = _buffer[buffer_base + right_bucket_index];
                bucket_delta_deg = right_bucket_elevation_deg - left_bucket_elevation_deg;

                frac -= 1f;
            }

            var sun_fraction = photons / max_photons;

            return sun_fraction;
        }

        public static float OverHorizon(float[] _buffer, int buffer_base, float azimuth_deg, float target_elevation_deg)
        {
            const float BucketWidth_deg = 360f / HorizonSamplesF;
            const float BucketHalfWidth_deg = BucketWidth_deg / 2f;  // Used because we're interpolating between two buckets.

            var sun_left_deg = azimuth_deg - SunHalfAngle_deg - BucketHalfWidth_deg;   // 
            var sun_left_bucket_float = sun_left_deg * (HorizonSamplesF / 360f);
            var sun_left_bucket = (int)sun_left_bucket_float; // [0,1440)
            var frac = sun_left_bucket_float - sun_left_bucket;
            if (sun_left_bucket < 0) sun_left_bucket += HorizonSamples;  // [0,1440)

            var sun_right_bucket = Modulo(sun_left_bucket + 1, 0, 1440);
            var elevation_left = _buffer[sun_left_bucket + buffer_base];
            var elevation_right = _buffer[sun_right_bucket + buffer_base];
            var interpolated_horizon_deg = elevation_left + frac * (elevation_right - elevation_left);
            return target_elevation_deg - interpolated_horizon_deg;
        }

        protected static int Modulo(int i, int low, int high)
        {
            if (i < low)
                return i + (high - low);
            if (i >= high)
                return i - (high - low);
            return i;
        }

        public static unsafe byte[] GenerateShadowMap(
            float[] horizons, 
            int width, 
            int height, 
            ElevationMap dem, 
            int tileCol, 
            int tileRow, 
            Vector3d sunPos)
        {
            var result = new byte[width * height];
            
            fixed (float* horizonsPtr = horizons)
            {
                for (int y = 0; y < height; y++)
                {
                    var line = tileRow + y;
                    for (int x = 0; x < width; x++)
                    {
                        int pixelIdx = y * width + x;
                        var sample = tileCol + x;
                        
                        var mat = dem.GetMatrix(line, sample);
                        var (az_rad, el_rad) = dem.GetAzEl(sunPos, mat);

                        float az_deg = az_rad * 57.2957795f;
                        float el_deg = el_rad * 57.2957795f;

                        float* pixelHorizons = horizonsPtr + pixelIdx * HorizonSamples;
                        float frac = FastSunFraction(pixelHorizons, az_deg, el_deg);
                        
                        result[pixelIdx] = (byte)(255f * frac);
                    }
                }
            }
            return result;
        }
    }
}
