from datagen import NumericData
from datagen import ClassroomData
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def linear_data_example():
    # sample usage
    data = NumericData.generate_data_linear(0, 1, 150, 2, 5, 1)
    plt.scatter(data[0], data[1], alpha=0.8,
                edgecolors='none', c='blue')
    plt.show()


def cluster_data_example():
    # visualize 2D data clusters
    data = NumericData.generate_data_cluster(num_points=500, sample_balance=100, k=5, random_factor=15)
    x_data = data[0].T
    plt.scatter(x_data[0], x_data[1], c=data[1],
                alpha=0.8, edgecolors='none',
                cmap=plt.cm.get_cmap('Spectral', 5))
    plt.colorbar()
    plt.show()


def exam_data_example():
    # visualize exam scores
    data = ClassroomData.generate_data_exams(num_students=5, num_exams=6, trend_up=20, exam_var=6)
    names = data[0]
    data = np.delete(data, 0, 0)
    data = data.T

    df = pd.DataFrame({'Exam': range(1, 7),
                       names[0]: data[0].astype(np.float),
                       names[1]: data[1].astype(np.float),
                       names[2]: data[2].astype(np.float),
                       names[3]: data[3].astype(np.float),
                       names[4]: data[4].astype(np.float)})

    for column in df.drop('Exam', axis=1):
        plt.plot(df['Exam'], df[column], label=column, alpha=0.8)
    plt.gca().set_ylim([0, 100])
    plt.xlabel("Exam Number")
    plt.ylabel("Exam Score")
    plt.legend(loc=4)
    plt.show()


exam_data_example()
linear_data_example()
cluster_data_example()
