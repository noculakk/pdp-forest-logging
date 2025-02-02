import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, classification_report


def prepare_results(results_path="results_final/"):
    df = pd.DataFrame()
    for file in os.listdir(results_path):
        if file.endswith(".json"):
            path = results_path + file

        with open(path) as f:
            data = json.load(f)

        df_temp = pd.DataFrame([data])
        df_temp["name"] = os.path.splitext(os.path.basename(path))[0]
        df = pd.concat([df, df_temp], ignore_index=True)

    df["name"] = df["name"].str.replace("type_Classifier_", "")

    return df


# Funkcja do obliczania metryk dla 3 klas
def calculate_metrics(conf_matrix):
    print(conf_matrix)
    conf_matrix = np.array(conf_matrix)

    # Konwersja macierzy konfuzji do list etykiet rzeczywistych i predykowanych
    true_labels = np.repeat(range(3), conf_matrix.sum(axis=1))
    pred_labels = np.concatenate(
        [np.full(conf_matrix[i, j], j) for i in range(3) for j in range(3)]
    )

    if len(true_labels) == 0 or len(pred_labels) == 0:
        return 0, 0, 0  # Obsługa pustych wartości

    # Obliczenie precision, recall i F1-macro przy użyciu sklearn
    precision, recall, f1_macro, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="macro", zero_division=0
    )
    report = classification_report(true_labels, pred_labels, zero_division=0)
    print(report)
    return precision, recall, f1_macro


def plot_confusion_matrices(df):
    fig, ax = plt.subplots(len(df), 2, figsize=(12, 40))
    axes = ax.flatten()
    for i in range(len(df)):
        sns.heatmap(
            df["conf_matrix_train"][i],
            annot=True,
            fmt="d",
            cmap=sns.color_palette("coolwarm", as_cmap=True),
            ax=axes[2 * i],
        )
        axes[2 * i].set_title(f"Train {df['name'][i]}")
        sns.heatmap(
            df["conf_matrix_test"][i],
            annot=True,
            fmt="d",
            cmap=sns.color_palette("coolwarm", as_cmap=True),
            ax=axes[2 * i + 1],
        )
        axes[2 * i + 1].set_title(f"Test {df['name'][i]}")
    plt.tight_layout()
