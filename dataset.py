import numpy as np


class ClassificationDataset(object):
    def __init__(self, seed=None, nb_points=10, d=5, sigma=1.):
        self.d = d
        self.N = nb_points
        self.rs = np.random.RandomState(seed)
        self.generate_classification_data(sigma)

    # Generates a binary classification dataset with two Gaussians
    def generate_classification_data(self, sigma):
        n1 = int(round(0.5 * self.N))  # Number of examples in class 1

        # Generate positive and negative samples
        X1 = self.rs.normal(1., sigma, (n1, self.d))
        X2 = self.rs.normal(-1., sigma, (self.N - n1, self.d))

        X = np.concatenate([X1, X2])
        y = np.concatenate([np.ones((n1, )), -np.ones((self.N - n1, ))])

        # Shuffle positive and negative samples
        new_args = list(range(len(X)))
        self.rs.shuffle(new_args)

        self.X = X[new_args].T
        self.y = y[new_args]
