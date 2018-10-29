from matplotlib import pyplot as plt
import numpy as np

def plot_data(X, y):
    plt.title("Microchip Tests")
    plt.xlabel("Microchip Test 1")
    plt.ylabel("Microchip Test 2")
    failed = X[y == 0]
    passed = X[y == 1]
    plt.plot(failed[:, 0], failed[:, 1], "yo", label="y = 0")
    plt.plot(passed[:, 0], passed[:, 1], "r+", label="y = 1")
    plt.legend()
    plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def h(X, theta):
    return sigmoid(X.dot(theta))


def predict(X, theta):
    return np.round(h(X, theta))


def cost_function_reg(theta, X, y, reg_lambda):
    m = len(y)
    y_zero = (1 - y).dot(np.log(1 - h(X, theta)))
    y_one = y.dot(np.log(h(X, theta)))
    reg = (reg_lambda / (2 * m)) * sum(theta[1:] ** 2)
    J = (-1 / m) * (y_zero + y_one) + reg
    return J


def gradient_reg(theta, X, y, reg_lambda):
    m = len(y)
    reg = (reg_lambda / m) * theta
    reg[0] = 0
    return ((h(X, theta) - y).dot(X) / m) + reg


def map_feature(X1, X2):
    degree = 6
    out = np.ones((X1.shape[0], 1))
    for i in range(1, degree + 1):
        for j in range(i + 1):
            col = ((X1 ** (i - j)) * (X2 ** j)).reshape(X1.shape[0], 1)
            out = np.hstack((out, col))

    return out

