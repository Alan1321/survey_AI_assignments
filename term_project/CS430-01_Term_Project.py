#---------------------------------------------------
# File: CS430-01_Term_Project.py
# Authors: Bradley Mitchell, Alan Subedi, Alex Turner, Micah Mayers
# Course: CS430-01
# Date: 12/7/2023
# Python 3.8.10
#---------------------------------------------------

import pandas as pd
from os import path, makedirs
from bradley_kmeans import kmeans_test
from alex_nn import nn_test
from alan_svm import start_svm


# Constants
#FILE_PATH = "student-mat.csv" # 395 elements
FILE_PATH = "student-por.csv" # 649 elements
COL_NAMES = ['School', 'Sex', 'Age', 'Address', 'FamSize', 'Pstatus', 'Medu', 'Fedu',
             'Mjob', 'Fjob', 'Reason', 'Guardian', 'TravelTime', 'StudyTime', 'Failures',
             'SchoolSup', 'FamSup', 'Paid', 'Activities', 'Nursery', 'Higher', 'Internet',
             'Romantic', 'FamRel', 'FreeTime', 'GoOut', 'Dalc', 'Walc', 'Health', 'Absences',
             'G1', 'G2', 'G3']

# Input Variable Test Case A: Predict G3 using all input variables
INPUT_SETUP_A = ['School', 'Sex', 'Age', 'Address', 'FamSize', 'Pstatus', 'Medu', 'Fedu',
                 'Mjob', 'Fjob', 'Reason', 'Guardian', 'TravelTime', 'StudyTime', 'Failures',
                 'SchoolSup', 'FamSup', 'Paid', 'Activities', 'Nursery', 'Higher', 'Internet',
                 'Romantic', 'FamRel', 'FreeTime', 'GoOut', 'Dalc', 'Walc', 'Health', 'Absences',
                 'G1', 'G2']

# Input Variable Test Case B: Predict G3 without using G2
INPUT_SETUP_B = ['School', 'Sex', 'Age', 'Address', 'FamSize', 'Pstatus', 'Medu', 'Fedu',
                 'Mjob', 'Fjob', 'Reason', 'Guardian', 'TravelTime', 'StudyTime', 'Failures',
                 'SchoolSup', 'FamSup', 'Paid', 'Activities', 'Nursery', 'Higher', 'Internet',
                 'Romantic', 'FamRel', 'FreeTime', 'GoOut', 'Dalc', 'Walc', 'Health', 'Absences',
                 'G1']

# Input Variable Test Case C: Predict G3 without using G2 or G1
INPUT_SETUP_C = ['School', 'Sex', 'Age', 'Address', 'FamSize', 'Pstatus', 'Medu', 'Fedu',
                 'Mjob', 'Fjob', 'Reason', 'Guardian', 'TravelTime', 'StudyTime', 'Failures',
                 'SchoolSup', 'FamSup', 'Paid', 'Activities', 'Nursery', 'Higher', 'Internet',
                 'Romantic', 'FamRel', 'FreeTime', 'GoOut', 'Dalc', 'Walc', 'Health', 'Absences']

# Input Variable Test Case D: Only using G1 and G2
INPUT_SETUP_D = ['G1', 'G2']


def main():

    # Load dataset from file
    data = pd.read_csv(FILE_PATH, sep=';', header=0, names=COL_NAMES)

    # Preprocessing
    preprocessed_data = preprocess_data(data)

    # K-Means Clustering
    kmeansMakeVisuals = not path.isdir('./visuals/kmeans')
    if kmeansMakeVisuals:
        makedirs('./visuals/kmeans')

    print("\n\nRunning K-Means Clustering with input setup A:")
    kmeans_test(preprocessed_data, INPUT_SETUP_A, "SetupA")
    print("\n\nRunning K-Means Clustering with input setup B:")
    kmeans_test(preprocessed_data, INPUT_SETUP_B, "SetupB")
    print("\n\nRunning K-Means Clustering with input setup C:")
    kmeans_test(preprocessed_data, INPUT_SETUP_C, "SetupC")
    print("\n\nRunning K-Means Clustering with input setup D:")
    kmeans_test(preprocessed_data, INPUT_SETUP_D, "SetupD")

    # Neural Network Modeling
    finalGrades = preprocessed_data['G3']
    nnMakeVisuals = not path.isdir('./visuals/nn')
    if nnMakeVisuals:
        makedirs('./visuals/nn')

    print("\n\nRunning Neural Network Modeling with input setup A:")
    nn_test(preprocessed_data[INPUT_SETUP_A], finalGrades, 'SetupA', nnMakeVisuals)
    print("\nRunning Neural Network Modeling with input setup B:")
    nn_test(preprocessed_data[INPUT_SETUP_B], finalGrades, 'SetupB', nnMakeVisuals)
    print("\nRunning Neural Network Modeling with input setup C:")
    nn_test(preprocessed_data[INPUT_SETUP_C], finalGrades, 'SetupC', nnMakeVisuals)

    #everything SVM related is done by the function below
    start_svm(preprocessed_data, FILE_PATH, COL_NAMES, INPUT_SETUP_A, INPUT_SETUP_B, INPUT_SETUP_C, INPUT_SETUP_D, True)


# preprocess_data
# Function to perform preprocessing on the dataset
# dataset: pandas DataFrame containing the dataset to preprocess
#
# returns: pandas DataFrame containing the dataset produced by performing preprocessing on the input dataset
def preprocess_data(dataset: pd.DataFrame):

    # Make a deep copy of the data to preserve the unprocessed version of the dataset
    new_data = dataset.copy()

    # Preprocessing Step 1: Map all variables with non-numerical values to numerical values
    new_data['School'] = new_data['School'].replace({'GP': 0, 'MS': 1})                                               # School: GP = 0, MS = 1
    new_data['Sex'] = new_data['Sex'].replace({'F': 0, 'M': 1})                                                       # Sex: F = 0, M = 1
    new_data['Address'] = new_data['Address'].replace({'U': 0, 'R': 1})                                               # Address: U = 0, R = 1
    new_data['FamSize'] = new_data['FamSize'].replace({'LE3': 0, 'GT3': 1})                                           # FamSize: LE3 = 0, GT3 = 1
    new_data['Pstatus'] = new_data['Pstatus'].replace({'T': 0, 'A': 1})                                               # Pstatus: T = 0, A = 1
    new_data['Mjob'] = new_data['Mjob'].replace({'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4}) # Mjob: teacher = 0, health = 1, services = 2, at_home = 3, other = 4
    new_data['Fjob'] = new_data['Fjob'].replace({'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4}) # Fjob: teacher = 0, health = 1, services = 2, at_home = 3, other = 4
    new_data['Reason'] = new_data['Reason'].replace({'home': 0, 'reputation': 1, 'course': 2, 'other': 3})            # Reason: home = 0, reputation = 1, course = 2, other = 3
    new_data['Guardian'] = new_data['Guardian'].replace({'mother': 0, 'father': 1, 'other':2})                        # Guardian: mother = 0, father = 1, other = 2
    new_data['SchoolSup'] = new_data['SchoolSup'].replace({'no': 0, 'yes': 1})                                        # SchoolSup: no = 0, yes = 1
    new_data['FamSup'] = new_data['FamSup'].replace({'no': 0, 'yes': 1})                                              # FamSup: no = 0, yes = 1
    new_data['Paid'] = new_data['Paid'].replace({'no': 0, 'yes': 1})                                                  # Paid: no = 0, yes = 1
    new_data['Activities'] = new_data['Activities'].replace({'no': 0, 'yes': 1})                                      # Activities: no = 0, yes = 1
    new_data['Nursery'] = new_data['Nursery'].replace({'no': 0, 'yes': 1})                                            # Nursery: no = 0, yes = 1
    new_data['Higher'] = new_data['Higher'].replace({'no': 0, 'yes': 1})                                              # Higher: no = 0, yes = 1
    new_data['Internet'] = new_data['Internet'].replace({'no': 0, 'yes': 1})                                          # Internet: no = 0, yes = 1
    new_data['Romantic'] = new_data['Romantic'].replace({'no': 0, 'yes': 1})                                          # Romantic: no = 0, yes = 1

    return new_data


# Program start
if __name__ == "__main__":
    main()
