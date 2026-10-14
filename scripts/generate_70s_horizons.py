import lunarscout as ls

print("version 1")

horizons = ls.generate_horizons(
    "/workspace/70south/horizons",
    [
        #"/d/viper/maps/lola/ldem_80s_20m.clipped.tif",
        #"/d/viper/maps/lola/ldem_70s_118m.tif",
        "/workspace/70south/ldem_80s_20m.clipped.tif",
        "/workspace/70south/ldem_70s_118m.tif",
    ],
    observer_height_m=0.0,
    compress=True,
    verbose=True,
)

print(horizons)
