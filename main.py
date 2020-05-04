import numpy as np
import matplotlib.pyplot as plt


def generate_data_linear(low_bound=1, high_bound=10, num_points=50, b1=3.55, b0=0, error_var=6):
    # generate data with linear relationship using Gaussian error
    x_vals = np.random.uniform(low_bound, high_bound, num_points)
    error = np.random.normal(0, error_var, num_points)
    y_vals = (x_vals*b1 + b0 + error)
    return np.array([x_vals, y_vals])


# sample usage
data = generate_data_linear(0, 1, 50, 3, 0, 1)
plt.scatter(data[0], data[1])
plt.show()
