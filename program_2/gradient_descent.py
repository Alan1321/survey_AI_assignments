import numpy as np
import pandas as pd
import sys
sys.setrecursionlimit(1000000)

def start(df: pd.DataFrame, variables=[], alpha=0.01, cost_difference=0.001):
    #initialize all thetas to zeroes
    init_thetas = [0 for i in range(len(variables))]
    #only select the required column names
    partial_data = df[variables]
    #split datasets into train/test
    train_partial_data = partial_data[0:456]
    test_partial_data = partial_data[456:]
    #normalize train/test 
    train_partial_data = normalize(train_partial_data, variables)
    test_partial_data = normalize(test_partial_data, variables)
    #start gradient descent here
    thetas = general_gradient_descent(train_partial_data, init_thetas, variables, alpha, cost_difference)
    #calculate squared sum error
    prediction = predict(test_partial_data, thetas, variables)

    return thetas, prediction

def general_cost_function(df: pd.DataFrame, thetas: list, variables: list):
    total_rows = len(df)
    total_columns = df.shape[1]
    sum_0 = 0
    for i in range(total_rows):
        x, y = get_xy_values(df, i, variables)
        sum_0 += pow(sum(multiply_arrays(thetas,x)) - y, 2)
    sum_0 = sum_0 / (2*total_rows)
    return sum_0

def general_differentiation(df: pd.DataFrame, thetas: list, variables: list):
    total_rows = len(df)
    total_columns = df.shape[1]
    #for theta0
    sum_0 = 0
    for row in range(total_rows):
        x, y = get_xy_values(df, row, variables)
        sum_0 += sum(multiply_arrays(thetas,x)) - y
    sum_0 = sum_0 / total_rows
    
    #for other thetas
    theta_sums = [0 for i in range(total_columns - 1)]
    index = 0
    for k in range(len(thetas)-1):
        for row in range(total_rows):
            x, y = get_xy_values(df, row, variables)
            theta_sums[index] = theta_sums[index] + (sum(multiply_arrays(thetas,x)) - y) * x[index+1]
        theta_sums[index] = theta_sums[index] / total_rows
        index += 1
    theta_sums = [sum_0] + theta_sums
    return theta_sums
    
    
#TODO: pass thetas as np array
def general_gradient_descent(df: pd.DataFrame, thetas: list, variables: list, alpha: float, cost_difference):
    prev_cost = general_cost_function(df, thetas, variables)
    thetas = subtract_list(thetas, multiply_num_and_list(alpha, general_differentiation(df, thetas, variables)))
    cost = general_cost_function(df, thetas, variables)
    if(abs(prev_cost - cost) > cost_difference):
        return general_gradient_descent(df, thetas, variables, alpha, cost_difference)
    return thetas


def normalize(df, variables):
    length = len(variables) - 1
    for i in range(length):
        mean = df[variables[i]].mean()
        std = df[variables[i]].std()
        if mean != 0:
            df.loc[:, variables[i]] = (df[variables[i]] - mean)/std
    return df


def predict(df: pd.DataFrame, thetas: list, variables: list):
    total_rows = len(df)
    total_columns = df.shape[1]
    sum_0 = 0
    for row in range(total_rows):
        x, y = get_xy_values(df, df.index[0]+row, variables)
        y_hat = sum(multiply_arrays(thetas, x))
        sum_0 += pow(y_hat - y,2)
    return sum_0

#utils methods
def multiply_arrays(a, b):
    result = [x * y for x, y in zip(a, b)]
    return result

def get_xy_values(df, row, variables):
    col_size = df.shape[1]
    x = [1]
    for i in range(col_size - 1):
        x.append(df[variables[i]][row])
    y = df[variables[col_size-1]][row]
    return x, y

def multiply_num_and_list(num, arr):
    result = [element * num for element in arr]
    return result

def subtract_list(list1, list2):
    result = [a - b for a, b in zip(list1, list2)]
    return result

#reads file and returns a pandas dataFrame
def read_data(file_path):
    data = []
    #read all data
    with open(file_path, 'r') as file:
        lines = file.readlines()
    #only start after line 22 --> where actual values start
    for line in lines[22:]:
        line = line.strip()
        line = line.split()
        array_of_floats = [float(val) for val in line]
        data.append(array_of_floats)
    #data is read with \n as delimiter, so merge such that we have a row of 14 columns
    merged_data = []
    for i in range(0, len(data), 2):
        merged_array = data[i] + data[i + 1] if i + 1 < len(data) else data[i]
        merged_data.append(merged_array)
    #once merged convert the 2D array into pandas dataframe
    df = pd.DataFrame(merged_data, columns=column_names)
    return df