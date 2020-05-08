from DataGen import NumericData
import matplotlib.pyplot as plt

gen_obj = NumericData


def linear_data_example():
    # sample usage
    data = gen_obj.generate_data_linear(0, 1, 150, 2, 5, 1)
    plt.scatter(data[0], data[1], alpha=0.8,
                edgecolors='none', c='blue')
    plt.show()


def cluster_data_example():
    # visualize 2D data clusters
    data = gen_obj.generate_data_cluster(num_points=500, sample_balance=100, k=5, random_factor=15)
    x_data = data[0].T
    plt.scatter(x_data[0], x_data[1], c=data[1],
                alpha=0.8, edgecolors='none',
                cmap=plt.cm.get_cmap('Spectral', 5))
    plt.colorbar()
    plt.show()


linear_data_example()
cluster_data_example()
