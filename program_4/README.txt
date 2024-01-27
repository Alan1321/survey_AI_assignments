Team 16: Bradley Mitchell, Alan Subedi, Alex Turner, Micah Mayers
CS 430/530, Fall 2023
Homework 4: K-Means Clustering and Support Vector Machine

Dependencies:
- Python 3.8.10 or newer

- pandas 2.0.3 or newer
Information on installing pandas can be found at:
https://pandas.pydata.org/docs/getting_started/install.html

- matplotlib 3.7.2 or newer
Information on installing matplotlib can be found at:
https://matplotlib.org/stable/users/installing/index.html

- scikit-learn 1.3.2 or newer
Information on installing Scikit-learn can be found at:
https://scikit-learn.org/stable/install.html

- LibSVM 3.32 or newer
Information on installing LIBSVM can be found at:
https://www.csie.ntu.edu.tw/~cjlin/libsvm/

Instructions:
- Open terminal

- Navigate to folder

- Run "python CS430-01_Program_4.py"

- The program automaticly will run K-Means clustering on the dataset. Once K-Means clustering has finished, the program will generate a plot of the clusters. The plot will be saved as "kmeans_results.png" in the project directory.

- The program will also produce a random 80%/20% split of the dataset to use as a testing set and a training set. The testing and training sets will be saved as data files in the format used by LibSVM. The training set is saved as "iris_libsvm_train.data" and the testing set is saved as "iris_libsvm_test.data". Both files are saved to the project directory.

- The training set and testing set can be used as data files by LibSVM to classify the dataset. An example of the results produced by LibSVM, along with the version of the data files used to generate those results, can be found in the libsvm_results subdirectory of the project directory.

- If you want to run LibSVM using the data files, place both data files in the tools subdirectory of LibSVM and run the following command from the tools subdirectory:

easy.py iris_libsvm_train.data iris_libsvm_test.data

- The results produced by LibSVM will be printed to the terminal and saved to output files that can be found in the tools subdirectory.
