from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import fmin_bfgs
from ex2_reg_functions import sigmoid, h, predict, plot_data, map_feature, cost_function_reg, gradient_reg


# Loading data and dividing it into X and y
data = np.loadtxt('ex2data2.txt', delimiter=',')
m = len(data)

y = data[:, -1]
X = data[:, 0:-1]

# Plotting data
print("Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.")
plot_data(X, y)

input('\nProgram paused. Press enter to continue.')

# Create polynomial features
X = map_feature(X[:, 0], X[:, 1])
m, n = X.shape
n = n - 1 # Taking the extra column of ones into consideration

# Set up initial parameters and regularization parameter
initial_theta = np.zeros(n + 1)
reg_lambda = 1


# Compute and display initial cost and gradient.
cost = cost_function_reg(initial_theta, X, y, reg_lambda)
grad = gradient_reg(initial_theta, X, y, reg_lambda)
print("Cost at initial theta (zeros): {}".format(cost))
print("Expected cost (approx): 0.693")
print("Gradient at initial theta (zeros) - first five values only:")
print(grad[:5])
print("Expected gradients (approx) - first five values only:")
print("[0.0085, 0.0188, 0.0001, 0.0503, 0.0115]")

# Compute and display cost and gradient for parameters as all ones and lambda = 10
test_theta = np.ones(n + 1)
reg_lambda = 10
cost = cost_function_reg(test_theta, X, y, reg_lambda)
grad = gradient_reg(test_theta, X, y, reg_lambda)
print("Cost at test theta (with lambda = 10): {}".format(cost))
print("Expected cost (approx): 3.16")
print("Gradient at test theta - first five values only:")
print(grad[:5])
print("Expected gradients (approx) - first five values only:")
print("[0.3460, 0.1614, 0.1948, 0.2269, 0.0922]")

input('\nProgram paused. Press enter to continue.')

# Using an advanced optimization algorithm - fmin_bfgs - to find the optimal value of theta
reg_lambda = 1
theta = fmin_bfgs(f=cost_function_reg, x0=initial_theta, fprime=gradient_reg, args=(X, y, reg_lambda), maxiter=400)
cost = cost_function_reg(theta, X, y, reg_lambda)

input('\nProgram paused. Press enter to continue.')

# Calculate accuracy of the algorithm on the training set.
p = predict(X, theta)

print('Train Accuracy: {}'.format(np.mean((p == y)) * 100))
print('Expected accuracy (with lambda = 1) (approx): 83.1')