from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import fmin_bfgs
from ex2_functions import plot_data, plot_decision_boundary, sigmoid, h, cost_function, gradient, predict


# Loading data and dividing it into X and y
data = np.loadtxt('ex2data1.txt', delimiter=',')
m = len(data)

y = data[:, -1]
X = data[:, 0:-1]

# Plotting data
print("Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.")
plot_data(X, y)

input('\nProgram paused. Press enter to continue.')

# Append a column of 1s to the start of X
m, n = X.shape
X = np.hstack((np.ones((m, 1)), X))

# This is a faster way to add a column of 1s to X
# X = np.ones((m, data.shape[1]))
# X[:, 1:] = data[:, 0:-1]

# Set up initial parameters
initial_theta = np.zeros(n + 1)

# Compute and display initial cost and gradient.
cost = cost_function(initial_theta, X, y)
grad = gradient(initial_theta, X, y)
print("Cost at initial theta (zeros): {}".format(cost))
print("Expected cost (approx): 0.693")
print("Gradient at initial theta (zeros):")
print(grad)
print("Expected gradient (approx):\n[-0.1000, -12.0092, -11.2628]")

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([-24, 0.2, 0.2]).T
cost = cost_function(test_theta, X, y)
grad = gradient(test_theta, X, y)
print("Cost at test theta: {}".format(cost))
print("Expected cost (approx): 0.218")
print("Gradient at test theta:")
print(grad)
print("Expected gradient (approx):\n[0.043, 2.566, 2.647]")

input('\nProgram paused. Press enter to continue.')

# Using an advanced optimization algorithm - fmin_bfgs - to find the optimal value of theta
theta = fmin_bfgs(f=cost_function, x0=initial_theta, fprime=gradient, args=(X, y), maxiter=400)
cost = cost_function(theta, X, y)

# Print the results
print('Cost at theta found by fmin_bfgs: {}'.format(cost))
print('Expected cost (approx): 0.203')
print('theta found by fmin_bfgs:')
print(theta)
print('Expected theta (approx):\n[-25.161, 0.206, 0.201]')

input('\nProgram paused. Press enter to continue.')

prob = sigmoid(np.array([1, 45, 85]).dot(theta))
print("For a student with scores 45 and 85, we predict an admission probability of {}".format(prob))
print('Expected value: 0.775 +/- 0.002')

# Calculate accuracy of the algorithm on the training set.
p = predict(X, theta)

print('Train Accuracy: {}'.format(np.mean((p == y)) * 100))
print('Expected accuracy (approx): 89.0')

# Plot the decision boundary.
plot_decision_boundary(X[:, 1:], y, theta)