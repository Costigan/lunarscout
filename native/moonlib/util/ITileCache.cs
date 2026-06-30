using System.Drawing;

namespace moonlib.util
{
    /// <summary>
    /// This is a marker interface for TileCache, MaskCache and OccupancyMaskCache, 
    /// </summary>
    public interface ITileCache : ITileCacheBase
    {
        int ImageWidthInTiles { get; }
        int ImageHeightInTiles { get; }
        int TilesPerLayer { get; }
        int TileCacheLength { get; }
        long BytesPerLayer { get; }

        byte[] GetTile(int tile_layer_id);
        byte[] GetTile(int tile_id, int layer);
        byte[] GetTile(Point image_pt, int layer);
        byte GetByte(Point image_pt, int layer);

        (int dataset_id, int elevation_deci) Key { get; }

        int Count { get; }
    }
}
