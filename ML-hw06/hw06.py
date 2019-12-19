from PIL import Image
from scipy.spatial.distance import pdist, squareform
import numpy as np


def read_data(file_name):
    image = Image.open(file_name).load()
    data = []
    for i in range(100):
        for j in range(100):
            data.append([i, j, image[i, j]])

    return data


def kernel(x, y, lambda_s, lambda_c):
    spatial = -lambda_s * (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2
    color = -lambda_c * (x[2] - y[2]) ** 2 + (x[3] - y[3]) ** 2 + (x[4] - y[4]) ** 2
    return np.exp(spatial) * np.exp(color)


def visualize(data, label):
    pass


def k_mean(data, num_cluster):
    label = np.zeros((len(data), 1))
    term3 = np.zeros(len(data), 1)
    mu = []

    kernel_pair = squareform(pdist(data, lambda u, v: kernel(u, v, 1, 1)))

    while (True):

        for cluster_id in range(num_cluster):
            res = 0
            num = 0
            for p in range(len(data)):
                for q in range(len(data)):
                    if label[p] == cluster_id and label[q] == cluster_id:
                        res += kernel_pair[p][q]
                        num += 1

            term3[cluster_id] = (1.0 / (num ** 2)) * res

        for cluster_id in range(num_cluster):
            res = 0
            num = 0
            for j in range(len(data)):
                for n in range(len(data)):
                    if label[n] == cluster_id:
                        res += kernel_pair[j][n]
                        num += 1
                term2[j][cluster_id] = (2 / num) * res

        for j in range(len(data)):
            all = []
            for cluster_id in range(num_cluster):
                all[cluster_id] = kernel_pair[j][j] - term2[j][cluster_id] + term3[cluster_id]





if __name__ == '__main__':
    print("HW06 - 0860832")

    img_data = read_data(file_name="image1.png")
    # label = k_mean(data=img_data, num_cluster=2)
    # visualize(img_data, label)
