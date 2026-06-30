namespace moonlib
{
    internal sealed class SynchronousProgress<T> : IProgress<T>
    {
        private readonly Action<T> _handler;
        private readonly object _sync = new();

        public SynchronousProgress(Action<T> handler)
        {
            _handler = handler ?? throw new ArgumentNullException(nameof(handler));
        }

        public void Report(T value)
        {
            lock (_sync)
            {
                _handler(value);
            }
        }
    }
}
