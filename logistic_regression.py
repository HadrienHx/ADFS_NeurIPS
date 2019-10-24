import numpy as np
import scipy


def logistic(x, y, theta):
    return scipy.special.expit(np.multiply(y, np.dot(theta, x)))


# Global model with the full dataset
class LogisticRegression(object):
    def __init__(self, dataset, c):
        self.X = dataset.X
        self.y = dataset.y
        self.c = c
        self.d = dataset.d
        w, _ = np.linalg.eig(np.dot(self.X, np.transpose(self.X)))

        self.smoothness = (0.25 * max(w)) + c
        self.strong_convexity = c
        self.kappa = self.smoothness / self.strong_convexity
        self.smoothnesses = [0.25 * np.dot(x, x) for x in self.X.T]

    def compute_error(self, theta):
        return np.sum(-np.log(logistic(self.X, self.y, theta))
                      ) + 0.5 * self.c * np.dot(theta, theta)

    # Allows to retrieve single point models that correspond to this node
    def get_local_models_and_probas(self, node_id, nb_nodes):
        # Compute which part of the dataset this node has to process
        nb_samples_per_node = len(self.y) / nb_nodes
        start = int(node_id * nb_samples_per_node)
        end = int(start + nb_samples_per_node)

        local_c = self.c / nb_nodes
        # Construct models for communication (L2Penalty) and virtual (SinglePointLogisticRegression) nodes
        models = [L2Penalty(local_c)] + [
            SinglePointLogisticRegression(x, y)
            for x, y in zip(self.X.T[start:end], self.y[start:end])
        ]
        # Get sampling probabilities for virtual nodes
        probas = np.array(
            [np.sqrt(1 + L / local_c) for L in self.smoothnesses])
        # Function to check whether a given sample is in this node
        is_local_range = lambda x: (x > start - 1) and (x < end)
        return models, probas, is_local_range

    # Retrieves the minimum error using accelerated gradient descent
    def get_min_error(self, nb_iters):
        y_prev = np.zeros((self.d, ))
        g = np.zeros((self.d, ))
        sq_kappa = self.smoothness / self.strong_convexity
        gamma = (sq_kappa - 1.) / (sq_kappa + 1.)
        for i in range(nb_iters):
            y = g - self.get_gradient(g) / self.smoothness
            g = (1 + gamma) * y - gamma * y_prev
            y_prev = y
        return self.compute_error(g)

    def get_gradient(self, theta):
        s = np.dot(self.X,
                   np.multiply(self.y, logistic(self.X, -self.y, theta)))
        return self.c * theta - s


# Logistic Regression for one data point only
class SinglePointLogisticRegression(object):
    inner_newton_steps = 5

    def __init__(self, Xi, yi):
        self.X = Xi
        self.Li = np.dot(Xi, Xi)
        self.yLi = yi * self.Li
        self.warm_start = 0.
        self.smoothness = 0.25 * self.Li

    def get_smoothness(self):
        return self.smoothness

    def get_rescaled_parameter(self, theta):
        return theta / self.smoothness

    # Local subproblem solved with Newton Method
    def get_primal_prox(self, theta, eta):
        x = self.warm_start
        Xtheta = theta * self.Li  # Xtheta = np.dot(self.X, theta) if theta is not 1D
        for i in range(self.inner_newton_steps):
            e = np.exp(self.yLi * x)
            hess = self.Li / eta + e * np.power(self.yLi / (1. + e), 2)
            g = (self.Li * x - Xtheta) / eta - self.yLi * scipy.special.expit(
                -self.yLi * x)
            x = x - g / hess

        self.warm_start = x
        return x  #  return x * self.X if theta is not 1D

    def project_on_Xi(self, z):
        return np.dot(self.X, z) / self.Li


# Class for the communication node, to simply consider it as any other model
class L2Penalty(object):
    def __init__(self, c):
        self.c = c

    def get_smoothness(self):
        return self.c

    def get_strong_convexity(self):
        return self.c

    def get_rescaled_parameter(self, theta):
        return theta / self.c
