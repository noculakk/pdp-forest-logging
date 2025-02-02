import pickle
import json

import ee

from src.visualizations import create_labeled_image


def extract_training_data(
    region_geometry,
    min_deforestation_year,
    reference_year,
    forest_cover_threshold,
    train_test_split,
    balanced=True,
    num_pixels=25000,
):
    labeled = create_labeled_image(
        region_geometry, min_deforestation_year, reference_year, forest_cover_threshold
    )
    sat_image = (
        ee.ImageCollection("COPERNICUS/S2")
        .filterDate(f"{reference_year + 1}-06-01", f"{reference_year + 1}-08-31")
        .filterBounds(region_geometry)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .median()
        .clip(region_geometry)
    )
    merged = sat_image.addBands(labeled)
    points_all = merged.sample(
        region=region_geometry,
        scale=30,
        numPixels=num_pixels,
        seed=42,
        tileScale=2,
        geometries=True,
    )

    none_points = points_all.filter(ee.Filter.eq("land_class", 0))
    forest_points = points_all.filter(ee.Filter.eq("land_class", 1))
    deforest_points = points_all.filter(ee.Filter.eq("land_class", 2))
    none_count = none_points.size().getInfo()
    forest_count = forest_points.size().getInfo()
    deforest_count = deforest_points.size().getInfo()

    if balanced:
        min_count = min(none_count, forest_count, deforest_count)

        none_points_limited = none_points.limit(min_count)
        forest_points_limited = forest_points.limit(min_count)
        deforest_points_limited = deforest_points.limit(min_count)

        balanced_points = none_points_limited.merge(forest_points_limited).merge(
            deforest_points_limited
        )

        print("Balanced points count:", min_count, "(per class)")
    else:
        balanced_points = points_all
        print(
            f"Classes counts: none={none_points.size().getInfo()}, forest={forest_points.size().getInfo()}, deforest={deforest_points.size().getInfo()}"
        )

    points_with_random = balanced_points.randomColumn("random_value", seed=42)
    train_points = points_with_random.filter(
        ee.Filter.lt("random_value", train_test_split)
    )
    test_points = points_with_random.filter(
        ee.Filter.gte("random_value", train_test_split)
    )

    return sat_image, labeled, train_points, test_points


def train_classifier(
    train_points,
    test_points,
    bands,
    classifier=None,
    disable_train_val=False,
    train_suite_name="final",
):
    print("Training classifier...")
    if classifier is None:
        classifier = ee.Classifier.smileRandomForest(numberOfTrees=50)

    classifier = classifier.train(
        features=train_points, classProperty="land_class", inputProperties=bands
    )

    classifier_str = "_".join(
        [f"{k}_{v}" for k, v in classifier.getInfo()["classifier"].items()]
    )
    classifier_str = classifier_str.replace(".", "_")

    print(f"Saving classifier as: {classifier_str}")

    print("Evaluating metrics...")
    test_pred = test_points.classify(classifier)

    conf_matrix_test_info = test_pred.errorMatrix(
        "land_class", "classification"
    ).getInfo()

    def compute_accuracy(conf_matrix):
        correct = sum(conf_matrix[i][i] for i in range(len(conf_matrix)))
        total = sum(sum(row) for row in conf_matrix)
        return correct / total if total > 0 else 0

    accuracy_test = compute_accuracy(conf_matrix_test_info)

    if not disable_train_val:
        train_pred = train_points.classify(classifier)
        conf_matrix_train_info = train_pred.errorMatrix(
            "land_class", "classification"
        ).getInfo()
        accuracy_train = compute_accuracy(conf_matrix_train_info)
        print("Confusion matrix (train):", conf_matrix_train_info)
        print("Accuracy (train):", accuracy_train)
    print("Confusion matrix (test):", conf_matrix_test_info)
    print("Accuracy (test):", accuracy_test)

    result = {
        "conf_matrix_train": conf_matrix_train_info if not disable_train_val else None,
        "accuracy_train": accuracy_train if not disable_train_val else None,
        "conf_matrix_test": conf_matrix_test_info,
        "accuracy_test": accuracy_test,
    }

    with open(f"results_{train_suite_name}/{classifier_str}.json", "w") as f:
        json.dump(result, f)

    with open(f"models_{train_suite_name}/{classifier_str}.pkl", "wb") as f:
        pickle.dump(classifier, f)

    return classifier
