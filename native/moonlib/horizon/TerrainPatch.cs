using moonlib.math;
using System.Drawing;

namespace moonlib.horizon
{
    public class TerrainPatch
    {
        public const int PatchSizePixels = 128;
        public static Size PatchSize = new Size(PatchSizePixels, PatchSizePixels);
        public Rectangle Rect;
        public int Step = 1;
        public Point HighPixel;

        public List<Vector3> CasterPoints;

        /// <summary>
        /// An array of shadow caster points within this patch
        /// </summary>
        public Vector3 Casters;

        /// <summary>
        /// 
        /// </summary>
        public int CasterStep;

        public TerrainPatch(Point location)
        {
            Rect = new Rectangle(location, PatchSize);
            CasterPoints = new List<Vector3>();
        }

        /// <summary>
        /// Gets the grid location (in patch units) of this patch.
        /// </summary>
        public Point Location => Rect.Location;

        public float MaxElevation;

        public float MinElevation;
    }
}
