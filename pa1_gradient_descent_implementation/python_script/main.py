#everything has been implemented from scrtach in utils.py file
from utils import read_data, gradient_descent, predict, plot_data

#constants
file_path = "data.txt"
x_label = "Population"
y_label = "Profit"
theta0 = 0
theta1 = 0
alpha = 0.01
cost_difference = 0.00001

#read data as a panda dataframe object
df = read_data(file_path, x_label, y_label)
plot_data(df)

#performing gradient descent for the optimal theta0 and theta1 values
theta0, theta1 = gradient_descent(df, theta0, theta1, alpha, cost_difference)

print("Printing the theta values")
print(f"Theta0: {theta0}, Theta1: {theta1}")
print("\nPredicting results for inputs 35k and 70k")
print(f"x: {35000}, y: {predict(theta0, theta1, 3.5)}")
print(f"x: {70000}, y: {predict(theta0, theta1, 7)}")

y_predictions = [predict(theta0, theta1, df['Population'][i]) for i in range(len(df))]

plot_data(df, y_predictions, True)