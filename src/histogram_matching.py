import ee


def lookup(source_hist, target_hist):
    """Creates a lookup table to make a source histogram match a target histogram.

    Args:
        source_hist: The histogram to modify. Expects the Nx2 array format produced by ee.Reducer.autoHistogram.
        target_hist: The histogram to match to. Expects the Nx2 array format produced by ee.Reducer.autoHistogram.

    Returns:
        A dictionary with 'x' and 'y' properties that respectively represent the x and y
        array inputs to the ee.Image.interpolate function.
    """

    # Split the histograms by column and normalize the counts.
    source_values = source_hist.slice(1, 0, 1).project([0])
    source_counts = source_hist.slice(1, 1, 2).project([0])
    source_counts = source_counts.divide(source_counts.get([-1]))

    target_values = target_hist.slice(1, 0, 1).project([0])
    target_counts = target_hist.slice(1, 1, 2).project([0])
    target_counts = target_counts.divide(target_counts.get([-1]))

    # Find first position in target where targetCount >= srcCount[i], for each i.
    def make_lookup(n):
        return target_values.get(target_counts.gte(n).argmax())

    lookup = source_counts.toList().map(make_lookup)

    return {"x": source_values.toList(), "y": lookup}


def histogram_match(source_img, target_img, geometry, bands):

    args = {
        "reducer": ee.Reducer.autoHistogram(maxBuckets=256, cumulative=True),
        "geometry": geometry,
        "scale": 1,  # Need to specify a scale, but it doesn't matter what it is because bestEffort is true.
        "maxPixels": 65536 * 4 - 1,
        "bestEffort": True,
    }

    # Only use pixels in target that have a value in source (inside the footprint and unmasked).
    source = source_img.reduceRegion(**args)
    target = target_img.updateMask(source_img.mask()).reduceRegion(**args)

    interpolated_bands = []
    for band in bands:
        interpolated_band = source_img.select([band]).interpolate(
            **lookup(source.getArray(band), target.getArray(band))
        )
        interpolated_bands.append(interpolated_band)

    return ee.Image(
        ee.Image.cat(*interpolated_bands).copyProperties(
            source_img, ["system:time_start"]
        )
    )


def find_closest(target_image, image_col, days):
    """Filter images in a collection by date proximity and spatial intersection to a target image.

    Args:
        target_image: An ee.Image whose observation date is used to find near-date images in
          the provided image_col image collection. It must have a 'system:time_start' property.
        image_col: An ee.ImageCollection to filter by date proximity and spatial intersection
          to the target_image. Each image in the collection must have a 'system:time_start'
          property.
        days: A number that defines the maximum number of days difference allowed between
          the target_image and images in the image_col.

    Returns:
        An ee.ImageCollection that has been filtered to include those images that are within the
          given date proximity to target_image and intersect it spatially.
    """

    # Compute the timespan for N days (in milliseconds).
    range = ee.Number(days).multiply(1000 * 60 * 60 * 24)

    filter = ee.Filter.And(
        ee.Filter.maxDifference(range, "system:time_start", None, "system:time_start"),
        ee.Filter.intersects(".geo", None, ".geo"),
    )

    closest = ee.Join.saveAll("matches", "measure").apply(
        ee.ImageCollection([target_image]), image_col, filter
    )

    return ee.ImageCollection(ee.List(closest.first().get("matches")))
