#---------------------------------------------------
# File: bradley_kmeans.py
# Authors: Bradley Mitchell, Alan Subedi, Alex Turner
# Course: CS430-01
# Date: 12/7/2023
# Python 3.8.10
#---------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import ConfusionMatrixDisplay


# kmeans_test
# Function to run kmeans clustering on the given dataset to predict binary and 5-level classification
# dataset: pandas DataFrame containing the preprocessed dataset
# testCase: list of strings to select the columns in the dataset that will be used as input variables for the kmeans algorithm
# case_name: string used as part of output image filenames
def kmeans_test(data: pd.DataFrame, testCase, case_name: str):

    # Run K-Means Clustering for binary classification
    kmeans_bin = KMeans(n_clusters=2, init='random', n_init='auto').fit(data[testCase])

    # Create result matrix
    result_matrix_bin = np.zeros((2,2), int)

    # Row = Actual Grade, Column = Predicted Cluster
    for i in range(0, data.shape[0]):
        row_pos = 0

        if (data.iloc[i].G3 >= 10): row_pos = 0 # Pass
        else: row_pos = 1                       # Fail

        result_matrix_bin[row_pos][kmeans_bin.labels_[i]] = result_matrix_bin[row_pos][kmeans_bin.labels_[i]] + 1

    result_dataframe_bin = pd.DataFrame(result_matrix_bin, columns=['C0', 'C1'], index=['Pass', 'Fail'])

    # Output results
    print("K-Means Clustering Results: Binary Classification")
    print(result_dataframe_bin)


    # Run K-Means Clustering for 5-level classification
    kmeans_5l = KMeans(n_clusters=5, init='random', n_init='auto').fit(data[testCase])

    # Create result matrix
    result_matrix_5l = np.zeros((5,5), int)

    # Row = Actual Grade, Column = Predicted Cluster
    for i in range(0, data.shape[0]):
        row_pos = 0

        if (data.iloc[i].G3 >= 16): row_pos = 0   # A
        elif (data.iloc[i].G3 >= 14): row_pos = 1 # B
        elif (data.iloc[i].G3 >= 12): row_pos = 2 # C
        elif (data.iloc[i].G3 >= 10): row_pos = 3 # D
        else: row_pos = 4                         # F

        result_matrix_5l[row_pos][kmeans_5l.labels_[i]] = result_matrix_5l[row_pos][kmeans_5l.labels_[i]] + 1

    result_dataframe_5l = pd.DataFrame(result_matrix_5l, columns=['C0', 'C1', 'C2', 'C3', 'C4'], index=['A', 'B', 'C', 'D', 'F'])

    # Output Results
    print("\nK-Means Clustering Results: 5 Level Classification")
    print(result_dataframe_5l)


    # Generate result images
    # 5 Level Classification
    # Assign Labels
    def get_letter(g: int):
        if (g >= 16): return 0
        elif (g >= 14): return 1
        elif (g >= 12): return 2
        elif (g >= 10): return 3
        else: return 4

    classification_5l = data['G3'].map(get_letter)

    # Generate and save result plot
    results = ConfusionMatrixDisplay.from_predictions(classification_5l, kmeans_5l.labels_)
    results_title = case_name + "-5-Level Results"
    results.ax_.set(
        title=results_title,
        xlabel="Cluster",
        ylabel="Final Grade",
        xticklabels=["0", "1", "2", "3", "4"],
        yticklabels=["A", "B", "C", "D", "F"]
    )
    plt.savefig(f'./visuals/kmeans/kmeans_results_{case_name}_5l.png')

    # Binary Classification
    # Assign Labels
    classification_bin = data['G3'].map(lambda g: 0 if g >= 10 else 1)

    # Generate and save result plot
    results = ConfusionMatrixDisplay.from_predictions(classification_bin, kmeans_bin.labels_)
    results_title = case_name + "-Binary Results"
    results.ax_.set(
        title=results_title,
        xlabel="Cluster",
        ylabel="Final Grade",
        xticklabels=["0", "1"],
        yticklabels=["Pass", "Fail"]
    )
    plt.savefig(f'./visuals/kmeans/kmeans_results_{case_name}_bin.png')


    return
