# The code below implements a simple Logistic Regression model from scratch
# using gradient descent to find the optimal parameters.
# It is designed to be a learning tool, similar to your linear regression example.

import numpy as np
import matplotlib.pyplot as plt

# --- 1. Generate Sample Data for Binary Classification ---
# Unlike linear regression, where y is a continuous value, logistic regression
# is used for classification, so the target variable 'y' is binary (0 or 1).
# We'll create a 2D dataset with two features (x1, x2) to make the
# decision boundary visually clear.
np.random.seed(0)

# Class 0 data points
x0_x = np.random.normal(2, 1, 50)  # Center around x=2
x0_y = np.random.normal(2, 1, 50)  # Center around y=2
y0 = np.zeros(50)                  # Label them as class 0

# Class 1 data points
x1_x = np.random.normal(5, 1, 50)  # Center around x=5
x1_y = np.random.normal(5, 1, 50)  # Center around y=5
y1 = np.ones(50)                   # Label them as class 1

# Combine all data
X = np.vstack((np.concatenate((x0_x, x1_x)), np.concatenate((x0_y, x1_y)))).T
y = np.concatenate((y0, y1))

# --- 2. Initialize Model Parameters and Hyperparameters ---
# We now have three parameters: q0 (bias), q1 (weight for x1), and q2 (weight for x2).
q0 = 0
q1 = 0
q2 = 0

# Set a learning rate and number of epochs for gradient descent.
# A higher learning rate might converge faster but could overshoot the minimum.
# A lower rate is more stable but requires more epochs.
alpha = 0.1
epochs = 5000

# --- 3. Define the Sigmoid (Hypothesis) Function ---
# This is the core difference from linear regression. Instead of a straight line,
# logistic regression uses a sigmoid function to "squash" the output of the
# linear combination into a probability between 0 and 1.
# The formula is h(x) = 1 / (1 + e^(-z)), where z = q0 + q1*x1 + q2*x2
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# The predict function now returns a probability.
def predict(X, q0, q1, q2):
    linear_combination = q0 + q1 * X[:, 0] + q2 * X[:, 1]
    return sigmoid(linear_combination)

# --- 4. Gradient Descent Function ---
# The update rule for the parameters looks very similar to linear regression,
# but the 'error' term (predict - y) is now based on the sigmoid hypothesis.
def gradient_descent(alpha, X, y, q0, q1, q2, epochs):
    m = len(y)
    for i in range(epochs):
        predictions = predict(X, q0, q1, q2)
        error = predictions - y

        # The gradients are the average of the error multiplied by the respective feature.
        # This is essentially the same as the linear regression update rule, but
        # 'predict' is now the sigmoid function's output.
        q0 = q0 - alpha * (1/m) * np.sum(error)
        q1 = q1 - alpha * (1/m) * np.sum(error * X[:, 0])
        q2 = q2 - alpha * (1/m) * np.sum(error * X[:, 1])

        # Optional: Print cost every 1000 epochs to monitor convergence
        if i % 1000 == 0:
            # The cost function for logistic regression is different from
            # the squared error cost of linear regression.
            # J(q) = -1/m * sum(y*log(h(x)) + (1-y)*log(1-h(x)))
            cost = (-1/m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
            print(f"Epoch {i}: Cost = {cost}")
            
    return q0, q1, q2

# Run gradient descent to train the model
q0 , q1, q2 = gradient_descent(alpha, X, y, q0, q1, q2, epochs)

# --- 5. Plot the Results ---
# We'll plot the data points and the learned decision boundary.
plt.figure(figsize=(8, 6))

# Plot the data points, colored by their class
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='Class 0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Class 1')

# The decision boundary is where the probability of being class 1 is 0.5.
# This occurs when q0 + q1*x1 + q2*x2 = 0.
# We can rearrange this to solve for x2 to get the equation of a line:
# x2 = (-q1/q2) * x1 - (q0/q2)
if q2 != 0:  # Avoid division by zero
    x_boundary = np.array([X[:, 0].min(), X[:, 0].max()])
    y_boundary = (-q1 / q2) * x_boundary - (q0 / q2)
    plt.plot(x_boundary, y_boundary, color='green', linestyle='--', label='Decision Boundary')

# Add labels and a title
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.grid(True)
plt.show()

# You can also check the final parameters
print(f"\nFinal parameters: q0={q0}, q1={q1}, q2={q2}")
