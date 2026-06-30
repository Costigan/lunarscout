using System.Drawing;

namespace travplan.utilities
{
    //TODO: Figure out whether this needs to go away or be merged with timeliner.map.views.DisplayTransform
    public struct OldDisplayTransform
    {
        public int OffsetX;
        public int OffsetY;
        public float Scale;

        public override string ToString() => $"<DisplayTransform Scale={Scale} OffsetX={OffsetX} OffsetY={OffsetY}";

        public PointF MouseToMap(Point m) => new PointF((m.X - OffsetX) * Scale, (m.Y - OffsetY) * Scale);
        public float MouseToMapX(int mx) => (mx - OffsetX) * Scale;
        public float MouseToMapY(int my) => (my - OffsetY) * Scale;

        /*
        public PointF MapToMouse(Point p)
        {
            float cellSize;
            int level, stride;
            TileTree.DrawCellSize(this, out cellSize, out stride, out level);
            var 
        }*/

        public float X(float mx) => mx / Scale + OffsetX;
        public float Y(float my) => my / Scale + OffsetY;
        public PointF MapToMouse(PointF p) => new PointF(X(p.X), Y(p.Y));

        public PointF this[PointF p] => new PointF(p.X / Scale + OffsetX, p.Y / Scale + OffsetY);
        public SizeF this[SizeF s] => new SizeF(s.Width / Scale, s.Height / Scale);

        public RectangleF this[Rectangle r] => new RectangleF(this[r.Location], this[r.Size]);
        public RectangleF this[RectangleF r] => new RectangleF(this[r.Location], this[r.Size]);

        //TODO: Change these?
        public Rectangle Transform(Rectangle r) => new Rectangle((int)(r.Left * Scale + OffsetX), (int)(r.Top * Scale + OffsetY), (int)(r.Width * Scale), (int)(r.Height * Scale));
        public Rectangle InvTransform(int Stride, Rectangle r) => new Rectangle(Stride * (int)((r.Left - OffsetX) / Scale), Stride * (int)((r.Y - OffsetY) / Scale), Stride * (int)(r.Width / Scale), Stride * (int)(r.Height / Scale));
        //public PointF this[System.Windows.Vector p] => new PointF((float)(p.X * Scale + OffsetX), (float)(p.Y * Scale + OffsetY));
    }
}
