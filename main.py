import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def generate_data_linear(low_bound=1, high_bound=10, num_points=50, b1=3.55, b0=0, error_var=6):
    # generate data with linear relationship using Gaussian error
    x_vals = np.random.uniform(low_bound, high_bound, num_points)
    error = np.random.normal(0, error_var, num_points)
    y_vals = (x_vals*b1 + b0 + error)
    return np.array([x_vals, y_vals])


def linear_data_example():
    # sample usage
    data = generate_data_linear(0, 1, 150, 3, 0, 1)
    plt.scatter(data[0], data[1])
    plt.show()


def generate_data_cluster(num_points=100, sample_balance=10, k=2, dims=3, random_factor=10):
    # generate n-D data clusters
    sample_prob = np.random.dirichlet(np.ones(k)*sample_balance, size=1)[0]
    means = np.multiply(np.random.rand(k, dims), np.random.randint(1, random_factor, dims))
    covariances = np.random.rand(k, dims, dims)
    x_data = []
    y_data = []
    for n in range(num_points):
        index = np.random.choice(k, 1, p=sample_prob)[0]
        x_data.append(multivariate_normal.rvs(means[index], covariances[index]))
        y_data.append(index)

    return [np.array(x_data), np.array(y_data)]


def cluster_data_example():
    # visualize 2D data clusters
    data = generate_data_cluster(num_points=500, sample_balance=10, k=5, random_factor=10)
    x_data = data[0].T
    plt.scatter(x_data[0], x_data[1], c=data[1], alpha=0.8, edgecolors='none', cmap='Spectral')
    plt.show()

cluster_data_example()