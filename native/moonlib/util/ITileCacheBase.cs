namespace moonlib.util
{
    public interface ITileCacheBase : IDisposable
    {
        /// <summary>
        /// Prevent removal when OccupancyCacheManager.FlushCache() is called
        /// </summary>
        bool IsLockedInMemory { get; set; }

        /// <summary>
        /// Delete cached tiles or masks
        /// </summary>
        void Flush();
    }
}
