import lunarscout as ls

horizons = ls.generate_horizons(
    "/e/lunar_analyst_scenarios/70south/lighting/horizons",
    [
        "/d/viper/maps/lola/ldem_80s_20m.clipped.tif",
        "/d/viper/maps/lola/ldem_70s_118m.tif",
    ],
    observer_height_m=0.0,
    compress=True,
    verbose=True,
)

print(horizons)
