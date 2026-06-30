namespace moonlib.math
{
    public struct RectangleD
    {
        public double Left { get; set; }
        public double Top { get; set; }
        public double Width { get; set; }
        public double Height { get; set; }

        public RectangleD(double l, double t, double w, double h) { Left = l; Top = t; Width = w; Height = h; }

        public RectangleD(PointD location, SizeD size) { Left = location.X; Top = location.Y; Width = size.Width; Height = size.Height; }
        public RectangleD(PointD ul, PointD lr) { Left = ul.X; Top = ul.Y; Width = lr.X - ul.X; Height = lr.Y - ul.Y; }

        public PointD Location
        {
            get => new PointD(Left, Top);
            set { var l = value; Left = l.X; Top = l.Y; }
        }
        public SizeD Size
        {
            get => new SizeD(Width, Height);
            set { var s = value; Width = s.Width; Height = s.Height; }
        }

        public double Right => Left + Width;
        public double Bottom => Top + Height;
    }
}
