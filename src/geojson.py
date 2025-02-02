import ee
import geemap


def export_forest_loss_to_geojson(year, output_file):
    if year < 2001 or year > 2023:
        raise ValueError("Provide year between 2001-2023.")

    ee.Initialize()

    poland = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(
        ee.Filter.eq("ADM0_NAME", "Poland")
    )
    poland_geometry = poland.geometry()

    hansen_data = ee.Image("UMD/hansen/global_forest_change_2023_v1_11")

    # Straty lasu
    loss = hansen_data.select("loss")
    loss_year = hansen_data.select("lossyear")

    loss_in_year = loss.updateMask(loss_year.eq(year - 2000))
    loss_in_year_poland = loss_in_year.clip(poland_geometry)

    loss_vector = loss_in_year_poland.reduceToVectors(
        geometry=poland_geometry,
        geometryType="polygon",
        reducer=ee.Reducer.countEvery(),
        scale=30,
        maxPixels=1e9,
        bestEffort=True,
    )

    geemap.ee_export_vector(ee_object=loss_vector, filename=output_file)
