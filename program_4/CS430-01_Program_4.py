#---------------------------------------------------
# File: CS430-01_Program_4.py
# Authors: Bradley Mitchell, Alan Subedi, Alex Turner, Micah Mayers
# Course: CS430-01
# Date: 11/22/2023
# Python 3.8.10
#---------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.cluster import KMeans
from split_data import randomSplit

# Constants
FILE_PATH = "iris.data"
COL_NAMES = ['SL', 'SW', 'PL', 'PW', 'Class']
COLOR_MAP = colors.ListedColormap(['red', 'green', 'blue'])


def main():

    # Load dataset from file
    data = pd.read_csv(FILE_PATH, sep=',', header=None, names=COL_NAMES)

    # K-Means clustering code start
    print("Starting to run K-Means clustering algorithm on dataset")
    # Run K-Means clustering algorithm on the dataset
    kmeans = KMeans(n_clusters=3, init='random', n_init='auto').fit(data[['SL', 'SW', 'PL', 'PW']])

    # Initialize plot of the clusters
    fig, ax = plt.subplots()
    ax.set_title('Sepal Width vs. Sepal Length')
    ax.set_xlabel('Sepal Length (cm)')
    ax.set_ylabel('Sepal Width (cm)')
    ax.tick_params(direction='in', top=True, right=True)
    ax.set_autoscale_on(False)
    ax.set_xbound(4.0, 8.0)
    ax.set_ybound(1.0, 5.0)

    # Plot clustered data and generate legend
    scatter = plt.scatter(data.SL, data.SW, c=kmeans.labels_, cmap=COLOR_MAP)
    plt.legend(*scatter.legend_elements(), title='Clusters', loc='upper right')
    
    # Plot the final cluster centroids
    ax.scatter(kmeans.cluster_centers_[0,0], kmeans.cluster_centers_[0,1], c='black', marker='x', s=100, linewidths=2)
    ax.scatter(kmeans.cluster_centers_[1,0], kmeans.cluster_centers_[1,1], c='black', marker='x', s=100, linewidths=2)
    ax.scatter(kmeans.cluster_centers_[2,0], kmeans.cluster_centers_[2,1], c='black', marker='x', s=100, linewidths=2)

    # Save an image of the plot
    fig.savefig('kmeans_results.png')

    # K-Means clustering code end
    print("Finished running K-Means clustering algorithm on dataset \nA plot of the clusters has been saved as kmeans_results.png")


    # SVM preprocessing code start
    print("\nStarting to generate training and testing datasets using a random 80/20 split")

    # Split the dataset into a training set (80%) and a validation set (20%)
    trainSet, testSet = randomSplit(data, 30)

    # Create output files in libSVM dataset format using the random split
    # Training set
    trainSVM = open("iris_libsvm_train.data", 'w')
    for i in range(0,trainSet.shape[0]):
        if(trainSet.iat[i,4] == "Iris-setosa"): trainSVM.write("0 1:" + str(trainSet.iat[i,0]) + " 2:" + str(trainSet.iat[i,1]) + " 3:" + str(trainSet.iat[i,2]) + " 4:" + str(trainSet.iat[i,3]) + "\n")
        elif(trainSet.iat[i,4] == "Iris-versicolor"): trainSVM.write("1 1:" + str(trainSet.iat[i,0]) + " 2:" + str(trainSet.iat[i,1]) + " 3:" + str(trainSet.iat[i,2]) + " 4:" + str(trainSet.iat[i,3]) + "\n")
        elif(trainSet.iat[i,4] == "Iris-virginica"): trainSVM.write("2 1:" + str(trainSet.iat[i,0]) + " 2:" + str(trainSet.iat[i,1]) + " 3:" + str(trainSet.iat[i,2]) + " 4:" + str(trainSet.iat[i,3]) + "\n")
    trainSVM.close()
    print("Finished generating training set \nThe training set has been saved as iris_libsvm_train.data")

    # Testing set
    testSVM = open("iris_libsvm_test.data", 'w')
    for i in range(0,testSet.shape[0]):
        if(testSet.iat[i,4] == "Iris-setosa"): testSVM.write("0 1:" + str(testSet.iat[i,0]) + " 2:" + str(testSet.iat[i,1]) + " 3:" + str(testSet.iat[i,2]) + " 4:" + str(testSet.iat[i,3]) + "\n")
        elif(testSet.iat[i,4] == "Iris-versicolor"): testSVM.write("1 1:" + str(testSet.iat[i,0]) + " 2:" + str(testSet.iat[i,1]) + " 3:" + str(testSet.iat[i,2]) + " 4:" + str(testSet.iat[i,3]) + "\n")
        elif(testSet.iat[i,4] == "Iris-virginica"): testSVM.write("2 1:" + str(testSet.iat[i,0]) + " 2:" + str(testSet.iat[i,1]) + " 3:" + str(testSet.iat[i,2]) + " 4:" + str(testSet.iat[i,3]) + "\n")
    testSVM.close()
    print("Finished generating testing set \nThe testing set has been saved as iris_libsvm_test.data")


# Program start
if __name__ == "__main__":
    main()

    # Program end
    input("\nPress the Enter key to exit the program")
