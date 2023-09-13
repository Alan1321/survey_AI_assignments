import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.setrecursionlimit(1000000)

def read_data(file_path, x_label, y_label):
    return pd.read_csv(file_path,header=None,names=[x_label,y_label])

def plot_data(df, y_predictions=[], prediction=False):
    # Create the plot
    plt.plot(df['Population'],df['Profit'],'rx', label='Training data')
    if prediction:
        plt.plot(df["Population"],y_predictions,'b',label='Linear Regression')
        plt.legend(loc='lower right')
    plt.xticks([4,6,8,10,12,14,16,18,20,22,24])
    plt.yticks([-5,0,5,10,15,20,25])
    plt.xlabel("Population of the city in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.show()
    
def cost_function(df, theta0, theta1):
    m = len(df)
    sum = 0
    for i in range(m):
        x = df['Population'][i]
        y = df['Profit'][i]
        sum += pow((theta0 + (theta1 * x) - y), 2)
    sum = sum / (2*m)
    return sum    

def differentiate_theta0(df, theta0, theta1):
    m = len(df)
    sum = 0
    for i in range(m):
        x = df['Population'][i]
        y = df['Profit'][i]
        sum += (theta0 + (theta1 * x) - y)
    sum = sum / m
    return sum

def differentiate_theta1(df, theta0, theta1):
    m = len(df)
    sum = 0
    for i in range(m):
        x = df['Population'][i]
        y = df['Profit'][i]
        sum += (theta0 + (theta1 * x) - y) * x
    sum = sum / m
    return sum

def gradient_descent(df, theta0, theta1, alpha, cost_difference):
    prev_cost = cost_function(df, theta0, theta1)
    tmp_theta0 = theta0 - (alpha*differentiate_theta0(df, theta0, theta1))
    tmp_theta1 = theta1 - (alpha*differentiate_theta1(df, theta0, theta1))
    theta0 = tmp_theta0
    theta1 = tmp_theta1
    cost = cost_function(df, theta0, theta1)
    if(abs(prev_cost - cost) > cost_difference):
        return gradient_descent(df, theta0, theta1, alpha, cost_difference)
    return theta0, theta1

def predict(theta0, theta1, x):
    return (theta0 + (theta1*x))