using System.Collections.Generic;
using System.Linq;

namespace moonlib.util
{
    public class ListUtilities
    {
        public static List<List<T>> Split<T>(int n, IEnumerable<T> source)
        {
            var r = Enumerable.Range(0, n).Select(i => new List<T>()).ToList();
            var count = 0;
            foreach (var s in source)
                r[count++ % n].Add(s);
            return r;
        }
    }
}
