import numpy as np
import matplotlib.pyplot as plt

# sample data
x  = np.array([1,2,3,4,5])
y = np.array([2,4,6,8,10])

q0=0
q1=0
alpha = 0.1
 # if the learning rate is too small eg: 0.001, it will take a large number of epoches to converge

# Hypothesis function
def predict(x, q0, q1):
    return q0+q1*x

# Error function
def error(predicted_value, actual_value):
    return predicted_value - actual_value

# Gradient descent function
def gradient_descent(alpha, x, y, q0, q1,epoches):
    m = len(x)
    for i in range(epoches):
        q0 = q0 - alpha * (1/m) * np.sum(error(predict(x, q0, q1),y)* 1)
        q1 = q1 - alpha * (1/m) * np.sum(error(predict(x, q0, q1),y)* x)
    return q0, q1

q0 , q1 = gradient_descent(alpha, x, y , q0, q1,1000)

# Prepare data for plotting
x_data = x
y_data = y

# Create a figure and axes
plt.figure(figsize=(8, 6))

# Plot the original data points
plt.scatter(x_data, y_data, label='Data Points', color='blue')

# Create x values for the regression line
x_line = np.linspace(min(x_data) - 1, max(x_data) + 1, 100)

# Calculate y values for the regression line
y_line = q0 + q1 * x_line

# Plot the regression line
plt.plot(x_line, y_line, color='red', linestyle='--',   )

# Add labels and a title
plt.xlabel('X Data')
plt.ylabel('Y Data')
plt.title('Plot of Data with Learned Linear Regression Line')
plt.legend()
plt.show()