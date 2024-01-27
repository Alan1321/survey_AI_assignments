#---------------------------------------------------
# File: CS430-01_Program_3.py
# Authors: Bradley Mitchell, Alan Subedi, Alex Turner, Micah Mayers
# Course: CS430-01
# Date: 10/15/2023
# Python 3.8.10
#---------------------------------------------------

import numpy as np
import pandas as pd
from sigmoid import sigmoid
from scipy import optimize
from split_data import randomSplit

# Constants
file_path = "iris.data"
col_names = ['SL', 'SW', 'PL', 'PW', 'Class']
classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

def main():

    # Load dataset from file
    data = pd.read_csv(file_path, sep=',', header=None, names=col_names)

    # Create a binary integer representation for the classes
    for iris in classes:
        data[iris] = data.apply(lambda row: int(row['Class'] == iris), axis=1)

    # Split the dataset into a training set (80%) and a validation set (20%)
    trainSet, validSet = randomSplit(data, 30)

    # Classify each class against the other two classes
    for iris in classes:
        print(f'\n{iris}\n--------------------------------------------------')

        # Get theta using training set
        X = trainSet[['SL', 'SW', 'PL', 'PW']].astype(float)
        y = trainSet[iris].astype(float)
        optResult = get_theta(X, y)
        theta = optResult.x
        print(f'Theta = {theta}')

        # Predict classes for validation set
        valid_X = validSet[['SL', 'SW', 'PL', 'PW']].astype(float)
        valid_X.insert(0, 'x0', 1)
        valid_y = validSet[iris]
        valid_y_hat = predict_class(theta, valid_X)

        # Compare actual against predicted, then generate a correctness vector (1 = correct, 0 = incorrect)
        comparison = valid_y.compare(valid_y_hat, keep_shape=True, keep_equal=True, result_names=('Actual', 'Predicted'))
        correct = comparison.apply(lambda row: int(row['Actual'] == row['Predicted']), axis=1)

        # Build the confusion matrix
        truePos  = correct[(correct == 1) & (valid_y == 1)].count()
        trueNeg  = correct[(correct == 1) & (valid_y == 0)].count()
        falsePos = correct[(correct == 0) & (valid_y == 1)].count()
        falseNeg = correct[(correct == 0) & (valid_y == 0)].count()

        print(f"\nCONFUSION MATRIX:(T = true, F = false)")
        print(f"\t     Positive\t    Negative")
        print("\t---------------------------------")
        print(f"Pos\t|\tTP {truePos}\t|\tFP {falsePos}\t|")
        print("\t---------------------------------")
        print(f"Neg\t|\tFN {falseNeg}\t|\tTN {trueNeg}\t|")
        print("\t---------------------------------\n")
        
        # Calculate accuracy and precision
        m = validSet.shape[0]
        accuracy = (truePos + trueNeg) / m * 100
        precision = (truePos) / (truePos + falsePos) * 100
        print('Accuracy  = {:0.4f} %'.format(accuracy))
        print('Precision = {:0.4f} %'.format(precision))
    
    print('') # A pinch of whitespace at the end

# Hypothesis
def h(theta: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
    return sigmoid(X @ theta)


# Calculate parameters
def get_theta(X: pd.DataFrame, y: pd.DataFrame) -> optimize.OptimizeResult:
    epsilon = 1e-5 # Prevents divide by zero in log

    # Add column of ones
    X.insert(0, 'x0', 1)

    # Initialize theta
    m, n = X.shape
    theta = np.zeros(n)

    # Cost function
    def J(theta, X, y):
        hx = h(theta, X)
        return ((-y @ np.log(hx + epsilon)) - ((1 - y) @ np.log(1 - hx + epsilon))) / m

    # Gradient of cost function
    def J_grad(theta, X, y):
        return ((h(theta, X) - y) @ X) / m
    
    # Minimize via SciPy (Truncated Newton algorithm)
    return optimize.minimize(J, theta, args=(X, y), jac=J_grad, method='TNC')


def predict_class(theta, X):
    # Produce prediction vector given calculated parameters
    y_hat = h(theta, X)

    # 1 if y >= 0.5 (sample is given class)
    # 0 if y <  0.5 (sample is not given class)
    return y_hat.apply(lambda y: int(y >= 0.5))


# Program start
if __name__ == "__main__":
    main()
