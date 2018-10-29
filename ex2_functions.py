from matplotlib import pyplot as plt
import numpy as np

def plot_data(X, y):
    plt.title("Admitted?")
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    not_admitted = X[y == 0]
    admitted = X[y == 1]
    plt.plot(not_admitted[:, 0], not_admitted[:, 1], "yo", label="Not admitted")
    plt.plot(admitted[:, 0], admitted[:, 1], "r+", label="Admitted")
    plt.legend()
    plt.show()

def plot_decision_boundary(X, y, theta):
    slope = - theta[1] / theta[2]
    intercept = - (theta[0] - 0.5) / theta[2]
    x_lim = [X[:, 0].min(), X[:, 0].max()] 
    y_lim = [slope * i + intercept for i in x_lim] 

    plt.title("Decision Boundary")
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    not_admitted = X[y == 0]
    admitted = X[y == 1]
    plt.plot(not_admitted[:, 0], not_admitted[:, 1], "yo", label="Not admitted")
    plt.plot(admitted[:, 0], admitted[:, 1], "r+", label="Admitted")
    plt.plot(x_lim, y_lim)
    plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def h(X, theta):
    return sigmoid(X.dot(theta))


def cost_function(theta, X, y):
    m = len(y)
    y_zero = (1 - y).dot(np.log(1 - h(X, theta)))
    y_one = y.dot(np.log(h(X, theta)))
    J = (-1 / m) * (y_zero + y_one)
    return J


def gradient(theta, X, y):
    m = len(y)
    return (h(X, theta) - y).dot(X) / m


def predict(X, theta):
    return np.round(h(X, theta))