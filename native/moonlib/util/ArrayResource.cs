using System.Collections.Concurrent;
using System.Diagnostics;

namespace moonlib.util
{
    public class ArrayResource<T>
    {
        List<int> Sizes = new List<int>(8);
        List<ConcurrentStack<T[]>> Stacks = new List<ConcurrentStack<T[]>>(8);

        public T[] Allocate(int size)
        {
            lock (this)
            {
                for (var i = 0; i < Sizes.Count; i++)
                    if (Sizes[i] == size)
                        if (Stacks[i].TryPop(out var array))
                        {
                            Debug.Assert(array.Length == size);
                            return array;
                        }
                        else
                            return new T[size];

                Sizes.Add(size);
                Stacks.Add(new ConcurrentStack<T[]>());

                return new T[size];
            }
        }

        public void Deallocate(T[] array)
        {
            lock (this)
            {
                if (array == null)
                    throw new ArgumentNullException(nameof(array));
                for (var i = 0; i < Sizes.Count; i++)
                    if (Sizes[i] == array.Length)
                    {
                        Stacks[i].Push(array);
                        return;
                    }

                Sizes.Add(array.Length);
                var new_stack = new ConcurrentStack<T[]>();
                new_stack.Push(array);
                Stacks.Add(new_stack);
            }
        }
    }
}
