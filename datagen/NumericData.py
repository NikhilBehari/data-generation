import numpy as np
from scipy.stats import multivariate_normal


def generate_data_linear(low_bound=1, high_bound=10, num_points=50, b0=0, b1=3.55, error_var=6):
    """Generate data with a linear relationship using Gaussian error

    :param low_bound: lower bound of x data (default 1)
    :param high_bound: upper bound of x data (default 10)
    :param num_points: number of data points (default 50)
    :param b0: intercept coefficient (default 0)
    :param b1: slope coefficient (default 3.55)
    :param error_var: Gaussian error variance (default 6)
    :return: [x_data, y_data]
    """
    x_vals = np.random.uniform(low_bound, high_bound, num_points)
    error = np.random.normal(0, error_var, num_points)
    y_vals = (x_vals*b1 + b0 + error)
    return np.array([x_vals, y_vals])


def generate_data_cluster(num_points=50, sample_balance=10, k=2, dims=3, random_factor=10):
    """Generate categorical cluster data

    :param num_points: number of data points (default 50)
    :param sample_balance: balance of sampling frequency per class (default 10)
    :param k: number of data classes (default 2)
    :param dims: number of dimensions (default 3)
    :param random_factor: difference in cluster means (default 10)
    :return: [x_data, y_data]
    """
    sample_prob = np.random.dirichlet(np.ones(k) * sample_balance, size=1)[0]
    means = np.multiply(np.random.rand(k, dims), np.random.randint(1, random_factor, [k, dims]))
    covariances = np.random.rand(k, dims, dims)

    for mat_ind in range(len(covariances)):
        covariances[mat_ind] = np.dot(covariances[mat_ind], covariances[mat_ind].T)

    x_data = []
    y_data = []
    for n in range(num_points):
        index = np.random.choice(k, 1, p=sample_prob)[0]
        x_data.append(multivariate_normal.rvs(means[index], covariances[index]))
        y_data.append(index)

    return [np.array(x_data), np.array(y_data)]
