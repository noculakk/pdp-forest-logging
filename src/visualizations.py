import ee
import geemap

from src.histogram_matching import histogram_match


def visualize_forest_change(year):
    if year < 2001 or year > 2023:
        raise ValueError("Provide year between 2001-2023.")
    ee.Initialize()

    poland = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(
        ee.Filter.eq("ADM0_NAME", "Poland")
    )

    hansen_data = ee.Image("UMD/hansen/global_forest_change_2023_v1_11")

    loss = hansen_data.select("loss")
    loss_year = hansen_data.select("lossyear")
    loss_in_year = loss.updateMask(loss_year.eq(year - 2000))
    loss_in_year_poland = loss_in_year.clip(poland)

    satellite_data_following_year = (
        ee.ImageCollection("COPERNICUS/S2")
        .filterDate(f"{year + 1}-06-01", f"{year + 1}-08-31")
        .filterBounds(poland)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .median()
        .clip(poland)
    )

    satellite_data_previous_year = (
        ee.ImageCollection("COPERNICUS/S2")
        .filterDate(f"{year - 1}-06-01", f"{year - 1}-08-31")
        .filterBounds(poland)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .median()
        .clip(poland)
    )

    forest_cover = hansen_data.select("treecover2000").clip(poland).gte(25).selfMask()

    loss_viz = {"min": 0, "max": 1, "palette": ["red"]}
    satellite_viz = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000, "gamma": 1.2}

    map = geemap.Map()
    map.centerObject(poland, 6)
    map.addLayer(satellite_data_following_year, satellite_viz, f"SAT {year + 1}")
    map.addLayer(satellite_data_previous_year, satellite_viz, f"SAT {year - 1}")
    map.addLayer(poland, {"color": "blue"}, "Poland")
    map.addLayer(forest_cover, {"palette": ["green"]}, "Tree cover")
    map.addLayer(loss_in_year_poland, loss_viz, f"Year Loss {year}")

    return map


def create_labeled_image(
    region_geometry, min_deforestation_year, reference_year, forest_cover_threshold
):
    hansen_data = ee.Image("UMD/hansen/global_forest_change_2023_v1_11").clip(
        region_geometry
    )
    treecover = hansen_data.select("treecover2000").gte(forest_cover_threshold)
    lossyear = hansen_data.select("lossyear")
    year_of_loss = lossyear.add(2000)

    class_forest = hansen_data.select("treecover2000").gt(forest_cover_threshold)

    class_recent_deforestation = (
        (
            treecover.And(year_of_loss.gte(min_deforestation_year)).And(
                year_of_loss.lte(reference_year)
            )
        )
        .selfMask()
        .unmask(0)
    )

    labeled = (class_recent_deforestation.add(class_forest)).rename("land_class")

    return labeled


def show_training_samples(sat_image, train_points, test_points):
    all_points = ee.FeatureCollection(train_points).merge(test_points)
    color_mapped = all_points.map(
        lambda f: f.set(
            "style",
            {
                "color": ee.Algorithms.If(
                    f.getNumber("land_class").eq(0),
                    "grey",
                    ee.Algorithms.If(
                        f.getNumber("land_class").eq(1), "green", "orange"
                    ),
                ),
                "pointSize": 5,
            },
        )
    )
    m = geemap.Map()
    m.centerObject(all_points.geometry(), 8)
    rgb_viz = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000, "gamma": 1.2}
    m.addLayer(sat_image, rgb_viz, "Sentinel")
    m.addLayer(
        color_mapped.style(**{"styleProperty": "style"}), {}, "Train/Test Points"
    )
    return m


def classify_and_display_results(
    classifier, bands, bounding_box_buffer_m, reference_year, forest_cover_threshold
):
    print("Classifying entire Poland region...")
    poland = (
        ee.FeatureCollection("FAO/GAUL/2015/level0")
        .filter(ee.Filter.eq("ADM0_NAME", "Poland"))
        .first()
    )
    bbox = poland.geometry().bounds().buffer(bounding_box_buffer_m)
    poland_clipped = ee.Feature(poland).geometry().intersection(bbox)
    hansen_data = ee.Image("UMD/hansen/global_forest_change_2023_v1_11").clip(
        poland_clipped
    )
    sat_image = (
        ee.ImageCollection("COPERNICUS/S2")
        .filterDate(f"{reference_year + 1}-06-01", f"{reference_year + 1}-08-31")
        .filterBounds(poland_clipped)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .median()
        .clip(poland_clipped)
    )
    classified = sat_image.select(bands).classify(classifier).rename("classified")
    m = geemap.Map()
    m.centerObject(ee.FeatureCollection([poland]).geometry(), 6)
    m.add_basemap("OpenStreetMap")
    rgb_viz = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000, "gamma": 1.2}
    class_viz = {"min": 0, "max": 2, "palette": ["grey", "green", "orange"]}
    m.addLayer(sat_image, rgb_viz, "Obraz Sentinel")
    m.addLayer(classified, class_viz, "Klasyfikacja")
    m.addLayer(
        hansen_data.select("lossyear"),
        {"min": 0, "max": 23, "palette": ["black", "yellow", "red"]},
        "Hansen Loss Year",
    )
    forest_mask = (
        hansen_data.select("treecover2000").gte(forest_cover_threshold).selfMask()
    )
    m.addLayer(forest_mask, {"palette": ["green"]}, "Hansen Forest 2000")
    return m


def classify_and_subtract(
    classifier, bands, bounding_box_buffer_m, year_from, year_to, forest_cover_threshold
):
    print("Wczytuję dane i generuję klasyfikację dla całej Polski...")
    poland = (
        ee.FeatureCollection("FAO/GAUL/2015/level0")
        .filter(ee.Filter.eq("ADM0_NAME", "Poland"))
        .first()
    )
    bbox = poland.geometry().bounds().buffer(bounding_box_buffer_m)
    poland_clipped = ee.Feature(poland).geometry().intersection(bbox)
    hansen_data = ee.Image("UMD/hansen/global_forest_change_2023_v1_11").clip(
        poland_clipped
    )

    ref_sat_image = (
        ee.ImageCollection("COPERNICUS/S2")
        .filterDate(f"2020-06-01", f"2020-08-31")
        .filterBounds(poland_clipped)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .median()
        .clip(poland_clipped)
    )

    sat_images = []
    for yr in (year_from, year_to):
        sat_image = (
            ee.ImageCollection("COPERNICUS/S2")
            .filterDate(f"{yr}-06-01", f"{yr}-08-31")
            .filterBounds(poland_clipped)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .median()
            .clip(poland_clipped)
        )
        sat_images.append(sat_image)

    matched_sat_images = []
    for i in range(2):
        matched_sat_images.append(
            histogram_match(sat_images[i], ref_sat_image, poland_clipped, bands)
        )

    classified_from = (
        matched_sat_images[0]
        .select(bands)
        .classify(classifier)
        .rename("classified start")
    )
    classified_to = (
        matched_sat_images[1]
        .select(bands)
        .classify(classifier)
        .rename("classified start")
    )

    m = geemap.Map()
    m.centerObject(ee.FeatureCollection([poland]).geometry(), 6)
    rgb_viz = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000, "gamma": 1.2}
    class_viz = {"min": 0, "max": 2, "palette": ["grey", "green", "orange"]}
    m.addLayer(sat_images[0], rgb_viz, f"Sentinel {year_from}")
    m.addLayer(sat_images[1], rgb_viz, f"Sentinel {year_to}")
    m.addLayer(matched_sat_images[0], rgb_viz, f"Sentinel Matched {year_from}")
    m.addLayer(matched_sat_images[1], rgb_viz, f"Sentinel Matched {year_to}")
    m.addLayer(ref_sat_image, rgb_viz, "Sentinel 2020 (Reference)")

    m.addLayer(classified_from, class_viz, "Klasyfikacja from")
    m.addLayer(classified_to, class_viz, "Klasyfikacja to")
    # m.addLayer(hansen_data.select("lossyear"), {"min": 0, "max": 23, "palette": ["black", "yellow", "red"]}, "Hansen Loss Year")
    # forest_mask = hansen_data.select("treecover2000").gte(forest_cover_threshold).selfMask()
    # m.addLayer(forest_mask, {"palette": ["green"]}, "Hansen Forest 2000")
    return m
