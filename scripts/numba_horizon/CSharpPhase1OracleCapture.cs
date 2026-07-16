using System.Text.Json;
using System.Text.Json.Serialization;
using moonlib;
using moonlib.horizon;

if (args.Length != 1)
{
    Console.Error.WriteLine("Usage: CSharpPhase1OracleCapture <output-json>");
    return 2;
}

const string StereographicProj =
    "+proj=stere +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs";
string outputPath = Path.GetFullPath(args[0]);
Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
MoonlibBridge.EnsureGdalInitialized();
using var diagnosticGenerator = new QuadTreeHorizonGenerator(
    disableHierarchy: false,
    maxConcurrentGpuOps: 1,
    maxSegmentQueueSize: 1);

var cases = new List<object>();

var flat = CreateDem(121, 121, (_, _) => 0f);
cases.Add(CaptureSingle(
    diagnosticGenerator,
    "flat_east",
    "Flat spherical terrain sampled toward true east.",
    flat,
    azimuthDegrees: 90.0,
    maxDistanceMeters: 1000.0,
    expectations: new
    {
        elevation_degrees_min_exclusive = -1.0,
        elevation_degrees_max_exclusive = 0.0,
        all_slopes_finite = true,
    }));

const int obstacleSize = 121;
int obstacleCenter = obstacleSize / 2;

var remainingDirections = new[]
{
    (Name: "north", Azimuth: 0.0, RowOffset: -20, ColumnOffset: 0),
    (Name: "southeast", Azimuth: 135.0, RowOffset: 14, ColumnOffset: 14),
    (Name: "south", Azimuth: 180.0, RowOffset: 20, ColumnOffset: 0),
    (Name: "southwest", Azimuth: 225.0, RowOffset: 14, ColumnOffset: -14),
    (Name: "west", Azimuth: 270.0, RowOffset: 0, ColumnOffset: -20),
    (Name: "northwest", Azimuth: 315.0, RowOffset: -14, ColumnOffset: -14),
};
foreach (var directionCase in remainingDirections)
{
    var directionalDem = CreateDem(obstacleSize, obstacleSize, (row, col) =>
        row == obstacleCenter + directionCase.RowOffset &&
        col == obstacleCenter + directionCase.ColumnOffset ? 150f : 0f);
    cases.Add(CaptureSingle(
        diagnosticGenerator,
        $"single_obstacle_{directionCase.Name}",
        $"A 150 m obstacle toward true {directionCase.Name}.",
        directionalDem,
        directionCase.Azimuth,
        1200.0,
        new { elevation_degrees_min_exclusive = 5.0 }));
}

var flatTenMeter = CreateDem(121, 121, (_, _) => 0f, pixelMeters: 10.0);
cases.Add(CaptureSingle(
    diagnosticGenerator,
    "flat_east_10m",
    "Flat spherical terrain at 10 m/pixel toward true east.",
    flatTenMeter,
    90.0,
    1000.0,
    new
    {
        elevation_degrees_min_exclusive = -1.0,
        elevation_degrees_max_exclusive = 0.0,
    }));

var multiplePeaks = CreateDem(121, 121, (row, col) =>
    row == obstacleCenter && col == obstacleCenter + 10 ? 50f :
    row == obstacleCenter && col == obstacleCenter + 30 ? 300f : 0f);
cases.Add(CaptureSingle(
    diagnosticGenerator,
    "near_lower_far_higher_peaks",
    "A lower near peak and a horizon-setting higher far peak on one east ray.",
    multiplePeaks,
    90.0,
    1500.0,
    new { elevation_degrees_min_exclusive = 15.0 }));

var ridge = CreateDem(121, 121, (row, col) =>
    col == obstacleCenter + 20 && Math.Abs(row - obstacleCenter) <= 2 ? 150f : 0f);
foreach (double azimuth in new[] { 89.75, 90.0, 90.25 })
{
    string suffix = azimuth == 90.0 ? "center" : azimuth < 90.0 ? "north_edge" : "south_edge";
    cases.Add(CaptureSingle(
        diagnosticGenerator,
        $"ridge_adjacent_bin_{suffix}",
        "A north-south ridge crossing three adjacent quarter-degree azimuth bins.",
        ridge,
        azimuth,
        1200.0,
        new { elevation_degrees_min_exclusive = 5.0 }));
}

var negativeElevations = CreateDem(121, 121, (row, col) =>
    row == obstacleCenter && col == obstacleCenter + 20 ? -20f : -100f);
var elevatedOrigin = CenterOrigin(negativeElevations);
elevatedOrigin.Z = 20f;
cases.Add(CaptureAtOrigin(
    diagnosticGenerator,
    "negative_elevation_elevated_observer",
    "Negative terrain elevations with an observer elevated 20 m above the center.",
    negativeElevations,
    elevatedOrigin,
    90.0,
    1200.0,
    new { elevation_degrees_min_exclusive = 4.0 }));

var nodataHole = CreateDem(121, 121, (row, col) =>
    row == obstacleCenter && col == obstacleCenter + 10 ? -32000f : 0f);
cases.Add(CaptureSingle(
    diagnosticGenerator,
    "nodata_hole_east",
    "A nodata hole intersecting an otherwise valid east ray.",
    nodataHole,
    90.0,
    1200.0,
    new { all_slopes_finite = true }));
var nodataBorder = CreateDem(121, 121, (row, col) =>
    row == 0 || col == 0 || row == 120 || col == 120 ? -32000f : 0f);
cases.Add(CaptureSingle(
    diagnosticGenerator,
    "nodata_border_east",
    "A valid center surrounded by a nodata DEM border.",
    nodataBorder,
    90.0,
    1700.0,
    new { all_slopes_finite = true }));
var allNodataRay = CreateDem(121, 121, (row, col) =>
    row == obstacleCenter && col > obstacleCenter ? -32000f : 0f);
cases.Add(CaptureSingle(
    diagnosticGenerator,
    "entirely_nodata_east_ray",
    "Every terrain cell east of the observer is nodata.",
    allNodataRay,
    90.0,
    1200.0,
    new { all_slopes_finite = true }));

var boundaryDem = CreateDem(65, 65, (row, col) =>
    row == 32 && col == 24 ? 150f : 0f,
    anchorColumn: 4,
    anchorRow: 32);
var boundaryOrigin = new PixelOrigin { X = 4, Y = 32, Z = 0f };
cases.Add(CaptureAtOrigin(
    diagnosticGenerator,
    "observer_and_obstacle_near_dem_boundary",
    "Observer four pixels from the west boundary with an east obstacle.",
    boundaryDem,
    boundaryOrigin,
    90.0,
    1200.0,
    new { elevation_degrees_min_exclusive = 5.0 }));

var partialDimensions = CreateDem(63, 47, (row, col) =>
    row == 23 && col == 51 ? 150f : 0f);
cases.Add(CaptureSingle(
    diagnosticGenerator,
    "partial_non_power_of_four_dimensions",
    "A 63 by 47 partial DEM with a horizon-setting east obstacle.",
    partialDimensions,
    90.0,
    1000.0,
    new { elevation_degrees_min_exclusive = 5.0 }));

foreach (var thresholdCase in new[]
{
    (Id: "near_threshold_480m", Offset: 16),
    (Id: "far_threshold_510m", Offset: 17),
})
{
    var thresholdDem = CreateDem(121, 121, (row, col) =>
        row == obstacleCenter && col == obstacleCenter + thresholdCase.Offset ? 100f : 0f);
    cases.Add(CaptureSingle(
        diagnosticGenerator,
        thresholdCase.Id,
        "Obstacle immediately beside the 500 m near/far calculation threshold.",
        thresholdDem,
        90.0,
        900.0,
        new { elevation_degrees_min_exclusive = 5.0 }));
}

var obstacle = CreateDem(obstacleSize, obstacleSize, (row, col) =>
    row == obstacleCenter && col == obstacleCenter + 20 ? 150f : 0f);
cases.Add(CaptureSingle(
    diagnosticGenerator,
    "single_obstacle_east",
    "A 150 m obstacle 20 pixels east of the observer, sampled toward true east.",
    obstacle,
    azimuthDegrees: 90.0,
    maxDistanceMeters: 1200.0,
    expectations: new
    {
        elevation_degrees_min_exclusive = 5.0,
        comparison_case = "single_obstacle_west",
        minimum_elevation_advantage_degrees = 5.0,
    }));
cases.Add(CaptureSingle(
    diagnosticGenerator,
    "single_obstacle_west",
    "The same east-side obstacle, sampled in the opposite direction.",
    obstacle,
    azimuthDegrees: 270.0,
    maxDistanceMeters: 1200.0,
    expectations: new
    {
        comparison_case = "single_obstacle_east",
        maximum_elevation_disadvantage_degrees = -5.0,
    }));

var diagonalObstacle = CreateDem(obstacleSize, obstacleSize, (row, col) =>
    row == obstacleCenter - 14 && col == obstacleCenter + 14 ? 150f : 0f);
cases.Add(CaptureSingle(
    diagnosticGenerator,
    "single_obstacle_northeast",
    "A 150 m obstacle northeast of the observer, sampled at 45 degrees true azimuth.",
    diagonalObstacle,
    azimuthDegrees: 45.0,
    maxDistanceMeters: 1200.0,
    expectations: new
    {
        elevation_degrees_min_exclusive = 5.0,
        comparison_case = "single_obstacle_southwest",
        minimum_elevation_advantage_degrees = 5.0,
    }));
cases.Add(CaptureSingle(
    diagnosticGenerator,
    "single_obstacle_southwest",
    "The same northeast obstacle, sampled in the opposite diagonal direction.",
    diagonalObstacle,
    azimuthDegrees: 225.0,
    maxDistanceMeters: 1200.0,
    expectations: new
    {
        comparison_case = "single_obstacle_northeast",
        maximum_elevation_disadvantage_degrees = -5.0,
    }));

var inner = CreateDem(41, 41, (_, _) => 0f);
const int outerSize = 121;
int outerCenter = outerSize / 2;
var outer = CreateDem(outerSize, outerSize, (row, col) =>
    row == outerCenter && col == outerCenter + 45 ? 250f : 0f);
var multiOrigin = CenterOrigin(inner);
var innerOnly = ReferenceRayEmulator.Run(
    inner,
    multiOrigin,
    azimuthDeg: 90.0,
    outputPath: string.Empty,
    suppressCsv: true,
    maxDistanceMeters: 2500.0);
cases.Add(BuildCase(
    diagnosticGenerator,
    "multi_dem_inner_only_east",
    "The inner flat DEM alone, used as the nested-coverage comparison.",
    90.0,
    2500.0,
    multiOrigin,
    new[] { inner },
    new[] { innerOnly },
    new
    {
        comparison_case = "multi_dem_outer_obstacle_east",
        maximum_elevation_disadvantage_degrees = -5.0,
    }));

var multiResults = ReferenceRayEmulator.RunMultiDem(
    new List<ElevationMap> { inner, outer },
    multiOrigin,
    azimuthDeg: 90.0,
    maxDistanceMeters: 2500.0,
    suppressCsv: true);
cases.Add(BuildCase(
    diagnosticGenerator,
    "multi_dem_outer_obstacle_east",
    "Nested flat DEMs with a 250 m obstacle in the outer DEM toward true east.",
    90.0,
    2500.0,
    multiOrigin,
    new[] { inner, outer },
    multiResults,
    new
    {
        elevation_degrees_min_exclusive = 5.0,
        comparison_case = "multi_dem_inner_only_east",
        minimum_elevation_advantage_degrees = 5.0,
        minimum_pass_count = 2,
    }));

var coarseOuter = CreateDem(61, 61, (row, col) =>
    row == 30 && col == 52 ? 300f : 0f, pixelMeters: 60.0);
var differentResolutionResults = ReferenceRayEmulator.RunMultiDem(
    new List<ElevationMap> { inner, coarseOuter },
    multiOrigin,
    azimuthDeg: 90.0,
    maxDistanceMeters: 2500.0,
    suppressCsv: true);
cases.Add(BuildCase(
    diagnosticGenerator,
    "multi_dem_different_resolutions",
    "A 30 m inner DEM and 60 m outer DEM with the horizon-setting feature in the outer DEM.",
    90.0,
    2500.0,
    multiOrigin,
    new[] { inner, coarseOuter },
    differentResolutionResults,
    new
    {
        elevation_degrees_min_exclusive = 5.0,
        minimum_pass_count = 2,
    }));

var invalidPyramidDem = CreateDem(7, 5, (_, _) => -20000f);
invalidPyramidDem.Elevation![0, 0] = float.NaN;
invalidPyramidDem.Elevation[0, 1] = float.PositiveInfinity;
invalidPyramidDem.Elevation[0, 2] = float.NegativeInfinity;
invalidPyramidDem.Elevation[0, 3] = -19999f;
invalidPyramidDem.Elevation[0, 4] = 42f;
invalidPyramidDem.Elevation[2, 3] = -32000f;
invalidPyramidDem.Elevation[4, 6] = 100f;
var pyramidFixtures = new[]
{
    new
    {
        id = "invalid_values_odd_dimensions",
        description =
            "A 7x5 DEM containing NaN, infinities, the -20000 m cutoff, " +
            "a value immediately above the cutoff, and all-invalid factor-four blocks.",
        width = invalidPyramidDem.Width,
        height = invalidPyramidDem.Height,
        elevation_m = Flatten(invalidPyramidDem.Elevation),
        pyramid = CapturePyramid(diagnosticGenerator, invalidPyramidDem, 0),
        expectations = new
        {
            invalid_at_or_below_m = -20000f,
            valid_immediately_above_cutoff_m = -19999f,
            all_invalid_block_sentinel_m = -32000f,
            expected_nan_count_level0 = 1,
            expected_positive_infinity_count_level0 = 1,
            expected_negative_infinity_count_level0 = 1,
        },
    },
};

var subpatchSnapshot = diagnosticGenerator.CalculateSubpatchRaySegmentsForDiagnostics(
    new List<ElevationMap> { inner, outer },
    tileColumn: 0,
    tileRow: 0,
    tileWidth: 16,
    tileHeight: 16,
    numAzimuths: 16,
    maxDistanceMeters: 2500f,
    observerElevationMeters: 0f,
    subpatchSize: 8);
var convergenceDem = CreateDem(
    16,
    16,
    (_, _) => 0f,
    centerCrsX: 500000.0,
    centerCrsY: 500000.0);
var convergenceSnapshot = diagnosticGenerator.CalculateSubpatchRaySegmentsForDiagnostics(
    new List<ElevationMap> { convergenceDem },
    tileColumn: 0,
    tileRow: 0,
    tileWidth: 16,
    tileHeight: 16,
    numAzimuths: 16,
    maxDistanceMeters: 2500f,
    observerElevationMeters: 0f,
    subpatchSize: 8);
var subpatchFixtures = new[]
{
    new
    {
        id = "boundary_halo_multi_dem_16az",
        description =
            "A complete 4x4 halo-inclusive subpatch-center grid at the primary " +
            "DEM corner, with 16 azimuth bins and two DEM passes.",
        configuration = new
        {
            tile_column = 0,
            tile_row = 0,
            tile_width = 16,
            tile_height = 16,
            subpatch_size = 8,
            azimuth_count = 16,
            dem_count = 2,
            primary_dem_width = inner.Width,
            primary_dem_height = inner.Height,
            max_distance_m = 2500f,
            observer_elevation_m = 0f,
            layout = "[azimuth][subpatch_center][dem]",
        },
        grid_convergence = new
        {
            gamma_center_rad = subpatchSnapshot.GridConvergence.GammaCenter,
            d_gamma_dx_rad_per_pixel = subpatchSnapshot.GridConvergence.DGammaDx,
            d_gamma_dy_rad_per_pixel = subpatchSnapshot.GridConvergence.DGammaDy,
        },
        centers = subpatchSnapshot.Centers.Select(center => new
        {
            index = center.Index,
            grid_row = center.GridRow,
            grid_column = center.GridColumn,
            requested_center_column = center.RequestedCenterColumn,
            requested_center_row = center.RequestedCenterRow,
            segment_center_column = center.SegmentCenterColumn,
            segment_center_row = center.SegmentCenterRow,
        }),
        segments = subpatchSnapshot.Segments.Select(CaptureSegment),
        segment_count = subpatchSnapshot.Segments.Length,
    },
    new
    {
        id = "material_grid_convergence_16az",
        description =
            "A compact subpatch grid centered 500 km from both stereographic axes, " +
            "where grid convergence is materially nonzero.",
        configuration = new
        {
            tile_column = 0,
            tile_row = 0,
            tile_width = 16,
            tile_height = 16,
            subpatch_size = 8,
            azimuth_count = 16,
            dem_count = 1,
            primary_dem_width = convergenceDem.Width,
            primary_dem_height = convergenceDem.Height,
            max_distance_m = 2500f,
            observer_elevation_m = 0f,
            layout = "[azimuth][subpatch_center][dem]",
        },
        grid_convergence = new
        {
            gamma_center_rad = convergenceSnapshot.GridConvergence.GammaCenter,
            d_gamma_dx_rad_per_pixel = convergenceSnapshot.GridConvergence.DGammaDx,
            d_gamma_dy_rad_per_pixel = convergenceSnapshot.GridConvergence.DGammaDy,
        },
        centers = convergenceSnapshot.Centers.Select(center => new
        {
            index = center.Index,
            grid_row = center.GridRow,
            grid_column = center.GridColumn,
            requested_center_column = center.RequestedCenterColumn,
            requested_center_row = center.RequestedCenterRow,
            segment_center_column = center.SegmentCenterColumn,
            segment_center_row = center.SegmentCenterRow,
        }),
        segments = convergenceSnapshot.Segments.Select(CaptureSegment),
        segment_count = convergenceSnapshot.Segments.Length,
    },
};

var traversalOuter = CreateDem(
    1025,
    1025,
    (row, col) => row == inner.Height / 2 && col == 990 ? 5000f : 0f,
    anchorColumn: inner.Width / 2,
    anchorRow: inner.Height / 2);
var bufferDems = new List<ElevationMap> { inner, traversalOuter };
var bufferSnapshot = diagnosticGenerator.CaptureSubpatchBuffersForDiagnostics(
    bufferDems,
    tileColumn: inner.Width / 2,
    tileRow: inner.Height / 2,
    tileWidth: 1,
    tileHeight: 1,
    observerElevationMeters: 0f,
    subpatchSize: 8);
var horizonBufferFixtures = new[]
{
    new
    {
        id = "single_pixel_multi_dem_production",
        description =
            "Production hierarchy-enabled subpatch traversal for one primary-center " +
            "pixel, all 1,440 azimuths, and separate inner/outer DEM passes. The " +
            "outer DEM extends far enough east to force hierarchy descent toward " +
            "a 5,000 m synthetic obstacle.",
        configuration = new
        {
            tile_column = inner.Width / 2,
            tile_row = inner.Height / 2,
            tile_width = 1,
            tile_height = 1,
            subpatch_size = 8,
            azimuth_count = 1440,
            dem_count = 2,
            observer_elevation_m = 0f,
            hierarchy_enabled = true,
            output_layout = "[pixel][azimuth]",
        },
        dems = bufferDems.Select((dem, index) => new
        {
            index,
            width = dem.Width,
            height = dem.Height,
            projection = dem.Proj4,
            geo_transform = dem.GeoTransform,
            elevation_m = Flatten(dem.Elevation!),
            pyramid = CapturePyramid(diagnosticGenerator, dem, index),
        }),
        grid_convergence = new
        {
            gamma_center_rad = bufferSnapshot.GridConvergence.GammaCenter,
            d_gamma_dx_rad_per_pixel = bufferSnapshot.GridConvergence.DGammaDx,
            d_gamma_dy_rad_per_pixel = bufferSnapshot.GridConvergence.DGammaDy,
        },
        per_dem_slopes = bufferSnapshot.PerDemSlopes,
        final_slopes = bufferSnapshot.FinalSlopes,
        final_degrees = bufferSnapshot.FinalDegrees,
        traversal_trace = new
        {
            dem_pass = bufferSnapshot.TraversalTraceDemPass,
            azimuth_index = bufferSnapshot.TraversalTraceAzimuthIndex,
            azimuth_degrees = bufferSnapshot.TraversalTraceAzimuthIndex * 0.25,
            pixel_index = 0,
            action_codes = new
            {
                descend = 0,
                cull = 1,
                nodata_skip = 2,
                out_of_bounds = 3,
                level0_sample = 4,
            },
            steps = bufferSnapshot.TraversalTrace.Select(step => new
            {
                sequence = step.Sequence,
                parameter_distance_km = step.ParameterDistanceKm,
                true_distance_m = step.TrueDistanceMeters,
                level = step.Level,
                cell_x = step.CellX,
                cell_y = step.CellY,
                pixel_x = step.PixelX,
                pixel_y = step.PixelY,
                maximum_elevation_m = step.MaximumElevationMeters,
                sample_elevation_m = step.SampleElevationMeters,
                sample_slope = step.SampleSlope,
                advance_km = step.AdvanceKm,
                action = step.Action,
            }),
        },
    },
};

var report = new
{
    schema_version = 1,
    artifact_kind = "lunarscout_reference_ray_oracle_source",
    baseline_commit = Environment.GetEnvironmentVariable("LUNARSCOUT_BASELINE_COMMIT"),
    implementation = "moonlib.horizon.ReferenceRayEmulator",
    pyramid_implementation = "moonlib.horizon.QuadTreeHorizonGenerator.BuildOrLoadPyramid",
    accelerator = new
    {
        name = diagnosticGenerator.SelectedAcceleratorName,
        type = diagnosticGenerator.SelectedAcceleratorType.ToString(),
    },
    conventions = new
    {
        raster_axis_order = new[] { "y", "x" },
        azimuth = "Degrees clockwise from true north; reference emulator evaluates offsets 0, 1/12, and 1/6 degree and retains the ray with the maximum slope.",
        distance_unit = "m",
        elevation_unit = "m",
        slope_unit = "dimensionless rise/run",
        angle_unit = "degree",
        pixel_coordinates = "Floating-point pixel centers with X=column and Y=row.",
        moon_radius_m = 1737400.0,
        projection = StereographicProj,
        geo_transform = "[origin_x, pixel_width, row_rotation, origin_y, column_rotation, pixel_height]",
    },
    cases,
    pyramid_fixtures = pyramidFixtures,
    subpatch_fixtures = subpatchFixtures,
    horizon_buffer_fixtures = horizonBufferFixtures,
};

File.WriteAllText(
    outputPath,
    JsonSerializer.Serialize(
        report,
        new JsonSerializerOptions
        {
            WriteIndented = true,
            NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
        }) + "\n");
Console.WriteLine(outputPath);
return 0;

static object CaptureSingle(
    QuadTreeHorizonGenerator diagnosticGenerator,
    string id,
    string description,
    ElevationMap dem,
    double azimuthDegrees,
    double maxDistanceMeters,
    object expectations)
{
    var origin = CenterOrigin(dem);
    var result = ReferenceRayEmulator.Run(
        dem,
        origin,
        azimuthDegrees,
        outputPath: string.Empty,
        suppressCsv: true,
        maxDistanceMeters: maxDistanceMeters);
    return BuildCase(
        diagnosticGenerator,
        id,
        description,
        azimuthDegrees,
        maxDistanceMeters,
        origin,
        new[] { dem },
        new[] { result },
        expectations);
}

static object CaptureAtOrigin(
    QuadTreeHorizonGenerator diagnosticGenerator,
    string id,
    string description,
    ElevationMap dem,
    PixelOrigin origin,
    double azimuthDegrees,
    double maxDistanceMeters,
    object expectations)
{
    var result = ReferenceRayEmulator.Run(
        dem,
        origin,
        azimuthDegrees,
        outputPath: string.Empty,
        suppressCsv: true,
        maxDistanceMeters: maxDistanceMeters);
    return BuildCase(
        diagnosticGenerator,
        id,
        description,
        azimuthDegrees,
        maxDistanceMeters,
        origin,
        new[] { dem },
        new[] { result },
        expectations);
}

static object BuildCase(
    QuadTreeHorizonGenerator diagnosticGenerator,
    string id,
    string description,
    double azimuthDegrees,
    double maxDistanceMeters,
    PixelOrigin origin,
    IReadOnlyList<ElevationMap> dems,
    IReadOnlyList<EmulatorResult> passes,
    object expectations)
{
    double maximumSlope = passes.SelectMany(result => result.Slopes).Max();
    return new
    {
        id,
        description,
        azimuth_degrees = azimuthDegrees,
        max_distance_m = maxDistanceMeters,
        observer = new
        {
            pixel_x = origin.X,
            pixel_y = origin.Y,
            elevation_m = origin.Z,
        },
        dems = dems.Select((dem, index) => new
        {
            index,
            width = dem.Width,
            height = dem.Height,
            projection = dem.Proj4,
            geo_transform = dem.GeoTransform,
            elevation_m = Flatten(dem.Elevation!),
        }),
        pyramids = dems.Select((dem, index) =>
            CapturePyramid(diagnosticGenerator, dem, index)),
        passes = passes.Select((result, index) => new
        {
            dem_index = index,
            observer_latitude_rad = result.ObserverLatRad,
            observer_longitude_rad = result.ObserverLonRad,
            direction_me = new[] { result.DirectionX, result.DirectionY, result.DirectionZ },
            slopes = result.Slopes,
            trace = result.Trace.Select(sample => new
            {
                distance_m = sample.DistanceMeters,
                elevation_m = sample.ElevationMeters,
                slope = sample.Slope,
                pixel_x = sample.PixelX,
                pixel_y = sample.PixelY,
            }),
        }),
        ray_fit_passes = CaptureRayFits(dems, origin, azimuthDegrees, maxDistanceMeters),
        result = new
        {
            pass_count = passes.Count,
            sample_count = passes.Sum(result => result.Slopes.Length),
            maximum_slope = maximumSlope,
            elevation_degrees = Math.Atan(maximumSlope) * 180.0 / Math.PI,
        },
        expectations,
    };
}

static object CapturePyramid(
    QuadTreeHorizonGenerator diagnosticGenerator,
    ElevationMap dem,
    int demIndex)
{
    var snapshot = diagnosticGenerator.BuildPyramidForDiagnostics(dem);
    return new
    {
        dem_index = demIndex,
        downsample_factor = QuadTreeHorizonGenerator.PYR_DOWNSAMPLE_FACTOR,
        level0 = snapshot.Level0,
        mips = snapshot.Mips,
        levels = snapshot.Levels.Select((level, levelIndex) => new
        {
            level = levelIndex,
            offset = level.Offset,
            width = level.Width,
            height = level.Height,
            cell_size_x = level.CellSizeX,
            cell_size_y = level.CellSizeY,
        }),
        map_parameters = new
        {
            radius_m = snapshot.Map.R,
            scale = snapshot.Map.K0,
            false_easting_m = snapshot.Map.Fe,
            false_northing_m = snapshot.Map.Fn,
            inverse_transform_determinant = snapshot.Map.InvDet,
            transform_0 = snapshot.Map.T0,
            transform_1 = snapshot.Map.T1,
            transform_2 = snapshot.Map.T2,
            transform_3 = snapshot.Map.T3,
            transform_4 = snapshot.Map.T4,
            transform_5 = snapshot.Map.T5,
        },
        projection_parameters = new
        {
            radius_m = snapshot.Projection.R,
            latitude_origin_rad = snapshot.Projection.Lat0,
            longitude_origin_rad = snapshot.Projection.Lon0,
            scale = snapshot.Projection.K0,
            false_easting_m = snapshot.Projection.FalseEasting,
            false_northing_m = snapshot.Projection.FalseNorthing,
        },
    };
}

static object[] CaptureRayFits(
    IReadOnlyList<ElevationMap> dems,
    PixelOrigin origin,
    double azimuthDegrees,
    double maxDistanceMeters)
{
    var primaryDem = dems[0];
    var primaryProjection = QuadTreeHorizonGenerator.BuildProjectionParamsDouble(primaryDem);
    var centerCrs = primaryDem.PixelToCRS(new PixelPoint(origin.X, origin.Y));
    var (centerLat, centerLon) = QuadTreeHorizonGenerator.InverseProjectDouble(
        centerCrs.X,
        centerCrs.Y,
        primaryProjection);
    double centerTerrain = primaryDem.GetElevation(origin.X, origin.Y);
    double observerRadius = primaryProjection.R + centerTerrain + origin.Z;
    var observerVector = QuadTreeHorizonGenerator.LatLonToVectorMeters(
        centerLat,
        centerLon,
        observerRadius);
    var observerToMoon = QuadTreeHorizonGenerator.GetRotationMatrixd(centerLat, centerLon);
    var direction = QuadTreeHorizonGenerator.ComputeDirectionVector(
        observerToMoon,
        azimuthDegrees * Math.PI / 180.0);

    double startDistanceMeters = 1.0;
    var fits = new List<object>();
    for (int demIndex = 0; demIndex < dems.Count; demIndex++)
    {
        var dem = dems[demIndex];
        double pixelColumnMeters = Math.Sqrt(
            dem.GeoTransform[1] * dem.GeoTransform[1] +
            dem.GeoTransform[4] * dem.GeoTransform[4]);
        double pixelRowMeters = Math.Sqrt(
            dem.GeoTransform[2] * dem.GeoTransform[2] +
            dem.GeoTransform[5] * dem.GeoTransform[5]);
        double mapResolutionMeters = (pixelColumnMeters + pixelRowMeters) * 0.5;
        double demSizeMeters = Math.Min(
            dem.Width * mapResolutionMeters,
            dem.Height * mapResolutionMeters);
        double rayLimitMeters = Math.Min(maxDistanceMeters, demSizeMeters * 1.2);

        var sampleBuffer = new QuadTreeHorizonGenerator.RaySample[
            QuadTreeHorizonGenerator.MAX_RAY_SAMPLE_CAPACITY];
        int sampleCount = QuadTreeHorizonGenerator.BuildRaySamples(
            observerVector,
            direction,
            startDistanceMeters,
            rayLimitMeters,
            dem,
            mapResolutionMeters,
            sampleBuffer);
        var samples = sampleBuffer[..sampleCount];
        RaySegment segment = QuadTreeHorizonGenerator.FitRaySegmentForDiagnostics(
            samples,
            mapResolutionMeters,
            observerVector,
            primaryProjection.R + centerTerrain,
            dem,
            startDistanceMeters,
            demIndex);

        fits.Add(new
        {
            dem_index = demIndex,
            map_resolution_m = mapResolutionMeters,
            requested_start_distance_m = startDistanceMeters,
            ray_limit_m = rayLimitMeters,
            observer_vector_moon_centered_m = new[]
            {
                observerVector.X,
                observerVector.Y,
                observerVector.Z,
            },
            nominal_direction_moon_centered = new[]
            {
                direction.X,
                direction.Y,
                direction.Z,
            },
            samples = samples.Select(sample => new
            {
                distance_m = sample.DistanceMeters,
                pixel_x = sample.PixelX,
                pixel_y = sample.PixelY,
                latitude_rad = sample.LatRad,
                longitude_rad = sample.LonRad,
                row = sample.Row,
                column = sample.Col,
                terrain_height_m = sample.TerrainHeightMeters,
            }),
            segment = CaptureSegment(segment),
        });

        double lastDistanceMeters = sampleCount > 0
            ? sampleBuffer[sampleCount - 1].DistanceMeters
            : startDistanceMeters;
        startDistanceMeters = Math.Min(rayLimitMeters, lastDistanceMeters);
        if (startDistanceMeters >= maxDistanceMeters)
            break;
    }
    return fits.ToArray();
}

static object CaptureSegment(RaySegment segment) => new
{
    start_pixel_x = segment.StartPixel.X,
    start_pixel_y = segment.StartPixel.Y,
    dem_id = segment.DemId,
    x0 = segment.X0,
    y0 = segment.Y0,
    a1 = segment.A1,
    a2 = segment.A2,
    a3 = segment.A3,
    a4 = segment.A4,
    b1 = segment.B1,
    b2 = segment.B2,
    b3 = segment.B3,
    b4 = segment.B4,
    s_start_km = segment.SStart,
    s_end_km = segment.SEnd,
    s_start_chord_km = segment.SStartChord,
    planar_to_chord_c1 = segment.PlanarToChordC1,
    planar_to_chord_c2 = segment.PlanarToChordC2,
    planar_to_chord_c3 = segment.PlanarToChordC3,
};

static float[] Flatten(float[,] values)
{
    int height = values.GetLength(0);
    int width = values.GetLength(1);
    var flattened = new float[height * width];
    int index = 0;
    for (int row = 0; row < height; row++)
        for (int col = 0; col < width; col++)
            flattened[index++] = values[row, col];
    return flattened;
}

static ElevationMap CreateDem(
    int width,
    int height,
    Func<int, int, float> elevation,
    int? anchorColumn = null,
    int? anchorRow = null,
    double pixelMeters = 30.0,
    double centerCrsX = 0.0,
    double centerCrsY = 0.0)
{
    const string stereographicProj =
        "+proj=stere +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs";
    var data = new float[height, width];
    for (int row = 0; row < height; row++)
        for (int col = 0; col < width; col++)
            data[row, col] = elevation(row, col);

    var geoTransform = new[]
    {
        centerCrsX - (anchorColumn ?? width / 2) * pixelMeters,
        pixelMeters,
        0.0,
        centerCrsY + (anchorRow ?? height / 2) * pixelMeters,
        0.0,
        -pixelMeters,
    };
    return new ElevationMap(data, stereographicProj, geoTransform);
}

static PixelOrigin CenterOrigin(ElevationMap dem) => new()
{
    X = dem.Width / 2,
    Y = dem.Height / 2,
    Z = 0f,
};
