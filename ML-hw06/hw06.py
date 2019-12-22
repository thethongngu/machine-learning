from PIL import Image
from scipy.spatial.distance import pdist, squareform
import numpy as np


def read_data(file_name):
    image = Image.open(file_name).load()
    data = []
    for i in range(100):
        for j in range(100):
            data.append([i, j, image[i, j][0], image[i, j][1], image[i, j][2]])

    return data


def kernel(x, y, lambda_s, lambda_c):
    spatial = -lambda_s * ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)
    color = -lambda_c * ((x[2] - y[2]) ** 2 + (x[3] - y[3]) ** 2 + (x[4] - y[4]) ** 2)
    return np.exp(spatial) * np.exp(color)


def visualize(data, label):
    pass


def k_mean(data, num_cluster):
    res_label = np.zeros((len(data), 1))
    term2 = np.zeros((len(data), num_cluster))
    term3 = np.zeros((len(data), 1))

    # random assign to cluster
    for i in range(num_cluster):
        for j in range(int(len(data) / num_cluster)):
            res_label[i * int(len(data) / num_cluster) + j] = i

    kernel_pair = squareform(pdist(data, lambda u, v: kernel(u, v, 0.01, 0.01)))
    count = 0

    while True:

        print(res_label)

        count += 1
        for cluster_id in range(num_cluster):
            res = 0
            num = 0
            for p in range(len(data)):
                for q in range(len(data)):
                    if res_label[p] == cluster_id and res_label[q] == cluster_id:
                        res += kernel_pair[p][q]
                        num += 1

            if num == 0:
                term3[cluster_id] = 0
            else:
                term3[cluster_id] = (1.0 / (num ** 2)) * res

        for cluster_id in range(num_cluster):
            res = 0
            num = 0
            for j in range(len(data)):
                for n in range(len(data)):
                    if res_label[n] == cluster_id:
                        res += kernel_pair[j][n]
                        num += 1

                if num == 0:
                    term2[j][cluster_id] = 0
                else:
                    term2[j][cluster_id] = (2 / num) * res

        old_label = np.copy(res_label)
        for j in range(len(data)):
            dist = np.zeros((len(data), num_cluster))
            for cluster_id in range(num_cluster):
                dist[j][cluster_id] = kernel_pair[j][j] - term2[j][cluster_id] + term3[cluster_id]
            res_label[j] = np.argmin(dist[j])

        stop = True
        for j in range(len(data)):
            if res_label[j] != old_label[j]:
                stop = False
                break

        if stop:
            break

    return res_label


def spectral(data, num_cluster, is_normalized=False):
    n = len(data)
    k = num_cluster

    W = squareform(pdist(data, lambda u, v: kernel(u, v, 0.01, 0.01)))
    D = np.diag(np.sum(W, axis=1))

    L = D - W
    if is_normalized:
        L = np.matmul(np.linalg.inv(D), L)

    val, vec = np.linalg.eig(L)
    sorted_id = val.argsort()

    U = np.zeros((n, k))
    for i in range(k):
        U[:, i] = vec[sorted_id[k]]

    if is_normalized:
        norm_value = np.sum(U, axis=1)
        U = U / norm_value[:, None]

    res_label = k_mean(U, k)
    return res_label


if __name__ == '__main__':
    print("HW06 - 0860832")

    img_data = read_data(file_name="image1.png")
    label = k_mean(data=img_data, num_cluster=2)
    # visualize(img_data, label)
