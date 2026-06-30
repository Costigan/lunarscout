using moonlib.spice;

namespace moonlib
{
    public class BridgeSmoke
    {
        public static float AddOne(float x) => x + 1;

        public static int SpiceSmokeTest(int x)
        {
            var manager = SpiceManager.Singleton;
            return x + 1;
        }
    }
}
