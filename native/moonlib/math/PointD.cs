namespace moonlib.math
{
    public struct PointD
    {
        public double X { get; set; }
        public double Y { get; set; }

        public PointD(double x, double y)
        {
            X = x;
            Y = y;
        }

        public override string ToString() => $"PointD[{X},{Y}]";
    }
}
