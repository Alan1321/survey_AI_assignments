#---------------------------------------------------
# File: CS430-01_Program_2.py
# Authors: Bradley Mitchell, Alan Subedi, Alex Turner
# Course: CS430-01
# Date: 9/17/23
# Python 3.8.10
#---------------------------------------------------

import numpy as np
import pandas as pd
from utils import load_data
from gradient_descent import start
import sys

sys.setrecursionlimit(1000000)

# constants
file_path = "boston.txt"
column_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]


def main():
    #load data and get a panda dataframe back
    df = load_data(file_path, column_names)
    outputFile = open('output.txt', 'w')

    # print("\n--> Info: 2c and 2d (using gradient descent) takes really long time to calculate. Took my computer 7 minutes even when cost difference is only set to 0.001.\n")

    # #implemented a generalized gradient descent --> works for any number of thetas
    # outputFile.write("Part 1: Gradient Descent\n--------------------------------------------------------------------------------------------------\n")
    # thetas, squared_error = start(df, variables=['DIS', 'RAD', 'NOX'], alpha=0.01, cost_difference=0.001)
    # outputFile.write(f"2a. squared_error: {squared_error}\n")
    # outputFile.write(f"2a. thetas: {thetas} \n--------------------------------------------------------------------------------------------------\n")

    # thetas, squared_error = start(df, variables=["CRIM", "ZN", "INDUS", "CHAS", "RM", "AGE", "DIS","RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV", "NOX"], alpha=0.01, cost_difference=0.001)
    # outputFile.write(f"2b. squared_error: {squared_error}\n")
    # outputFile.write(f"2b. thetas: {thetas} \n--------------------------------------------------------------------------------------------------\n")

    # thetas, squared_error = start(df, variables=['AGE', 'TAX', 'MEDV'], alpha=0.01, cost_difference=0.001)
    # outputFile.write(f"2c. squared_error: {squared_error}\n")
    # outputFile.write(f"2c. thetas: {thetas} \n--------------------------------------------------------------------------------------------------\n")

    # thetas, squared_error = start(df, variables=["CRIM", "ZN", "INDUS", "CHAS", "RM", "AGE", "DIS","RAD", "TAX", "PTRATIO", "B", "LSTAT", "NOX", "MEDV"], alpha=0.01, cost_difference=0.001)
    # outputFile.write(f"2d. squared_error: {squared_error}\n")
    # outputFile.write(f"2d. thetas: {thetas}")

    ####################################################################################################################################################################

    data = load_data(file_path, column_names)

    # Add column of 1's to the dataset to use when calculating the value of theta 0
    data.insert(0, 'X_0', 1)

    # Split data
    trainSet = data.iloc[:456, :] # 456 rows for training
    validSet = data.iloc[456:, :] # 50 rows for validation
    m = trainSet.size

    # Calculate the values of theta for cases 2a and 2c using normal equations
    theta_2a_normal = normalEquation(trainSet.loc[:, ["X_0", "DIS", "RAD"]], trainSet.loc[:, ["NOX"]])
    theta_2c_normal = normalEquation(trainSet.loc[:, ["X_0", "AGE", "TAX"]], trainSet.loc[:, ["MEDV"]])

    outputFile.write("\n\nPart 2: Normal Equations\n--------------------------------------------------------------------------------------------------\n")
    outputFile.write(f"2a. thetas:\n{theta_2a_normal}\n--------------------------------------------------------------------------------------------------\n")
    outputFile.write(f"2c. thetas:\n{theta_2c_normal}\n")
    #theta_2b_normal = normalEquation(trainSet.loc[:, ["X_0", "CRIM", "ZN", "INDUS", "CHAS", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]], trainSet.loc[:, ["NOX"]])
    #theta_2d_normal = normalEquation(trainSet.loc[:, ["X_0", "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]], trainSet.loc[:, ["MEDV"]])
    #print(theta_2b_normal)
    #print(theta_2d_normal)

    # Feature normalization/scaling
    mean, std = trainSet.mean(), trainSet.std()
    trainSet = (trainSet - mean) / std
    validSet = (validSet - mean) / std

# Hypothesis
def h(theta, X):
	return theta.T * X

# normalEquation
# Calculates the values of theta using normal equations
# x: pandas dataframe containing the M x (N + 1) matrix of x parameters.
# y: pandas dataframe containing the M x 1 matrix of the given y values.
# M = number of training samples. N = number of features.
# returns theta_normal: pandas dataframe containing the theta values calculated using normal equations.
def normalEquation(x: pd.DataFrame, y: pd.DataFrame):
    theta_normal: pd.DataFrame
    theta_normal = np.linalg.inv(x.T @ x) @ x.T @ y
    return theta_normal

# Program start
if __name__ == "__main__":
    main()