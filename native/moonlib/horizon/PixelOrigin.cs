namespace moonlib.horizon
{
    public struct PixelOrigin
    {
        public PixelOrigin() { }
        public PixelOrigin(float x, float y, float z) => (X, Y, Z) = (x, y, z);
        public float X { get; set; }
        public float Y { get; set; }
        public float Z { get; set; }

        public override string ToString() => $"[{X}, {Y}, {Z}]";
    }

    public struct LatLonOrigin
    {
        public LatLonOrigin() { }
        public LatLonOrigin(double latitude, double longitude, float z) => (Latitude, Longitude, Z) = (latitude, longitude, z);

        public double Latitude { get; set; }
        public double Longitude { get; set; }
        public float Z { get; set; }
        public override string ToString() => $"[{Latitude}, {Longitude}, {Z}]";

    }
}
