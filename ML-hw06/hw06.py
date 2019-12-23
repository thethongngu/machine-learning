from PIL import Image
from scipy.spatial.distance import pdist, squareform

import numpy as np
import matplotlib.pyplot as plt

HEIGHT = 20
WIDTH = 20


def read_data(file_name):
    image = Image.open(file_name)
    image.thumbnail((WIDTH, HEIGHT), Image.NEAREST)

    pixels = image.load()

    data = []
    for i in range(0, WIDTH):
        for j in range(0, HEIGHT):
            data.append([i, j, pixels[i, j][0], pixels[i, j][1], pixels[i, j][2]])

    draw_input(data)

    return np.array(data)


def RBF(x, y, lambda_s, lambda_c):
    spatial = -lambda_s * ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)
    color = -lambda_c * ((x[2] - y[2]) ** 2 + (x[3] - y[3]) ** 2 + (x[4] - y[4]) ** 2)
    res = np.exp(spatial) * np.exp(color)
    return res


def visualize(data, label, num_cluster, iteration):
    colors = ['red', 'blue', 'green']
    title = "Kernel K-Means Iteration " + str(iteration)

    plt.xlim([0, WIDTH])
    plt.ylim([0, HEIGHT])
    plt.ylim(max(plt.ylim()), min(plt.ylim()))

    for i in range(len(data)):
        plt.scatter(x=data[i][0], y=data[i][1], color=colors[int(label[i])])

    # for i in range(0, k):
    #     col = next(color)
    #     plt.scatter(means[i][0], means[i][1], s=32, c=col)

    plt.title(title)
    plt.show()


def draw_input(data):
    plt.xlim([0, WIDTH])
    plt.ylim([0, HEIGHT])
    plt.ylim(max(plt.ylim()), min(plt.ylim()))

    for i in range(len(data)):
        plt.scatter(x=data[i][0], y=data[i][1], color=(data[i][2] / 255.0, data[i][3] / 255.0, data[i][4] / 255.0))

    plt.show()


def build_gram_matrix(data):
    kernel_pair = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        kernel_pair[i][i] = 1
        for j in range(i + 1, len(data)):
            kernel_pair[i][j] = RBF(data[i], data[j], 0.001, 0.001)
            kernel_pair[j][i] = kernel_pair[i][j]

    return kernel_pair


def k_mean(data, num_cluster, kernel_pair=None):
    res_label = np.zeros((len(data), 1), dtype=int)
    term2 = np.zeros((len(data), num_cluster))
    term3 = np.zeros((len(data), 1))

    # assign clusters
    # for i in range(0, len(data) - 1):
    #     res_label[i] = 1
    # for i in range(len(data) - 1, len(data)):
    #     res_label[i] = 0

    for i in range(num_cluster):
        length = int(len(data) / num_cluster)
        for j in range(length):
            res_label[i * length + j] = i

    if kernel_pair is None:
        kernel_pair = build_gram_matrix(data)
    count = 0

    while True:

        # print(res_label)
        visualize(data, res_label, num_cluster, count)
        count += 1

        num_element = np.zeros((len(data), 1))
        for i in range(len(res_label)):
            num_element[res_label[i]] += 1

        for cluster_id in range(num_cluster):
            res = 0.0
            for p in range(len(data)):
                for q in range(len(data)):
                    if res_label[p] == cluster_id and res_label[q] == cluster_id:
                        res += kernel_pair[p][q]

            if num_element[cluster_id] == 0:
                term3[cluster_id] = 0
            else:
                term3[cluster_id] = (res / (num_element[cluster_id] ** 2))

        for cluster_id in range(num_cluster):
            for j in range(len(data)):
                res = 0.0
                for n in range(len(data)):
                    if res_label[n] == cluster_id:
                        res += kernel_pair[j][n]

                if num_element[cluster_id] == 0:
                    term2[j][cluster_id] = 0
                else:
                    term2[j][cluster_id] = 2.0 * (res / num_element[cluster_id])

        old_label = np.copy(res_label)
        for j in range(len(data)):
            dist = np.zeros((num_cluster, 1))

            print()
            print(j)
            print("Pos: (%s, %s)" % (data[j][0], data[j][1]))

            for cluster_id in range(num_cluster):
                dist[cluster_id] = kernel_pair[j][j] - term2[j][cluster_id] + term3[cluster_id]

                print("1: %s" % kernel_pair[j][j])
                print("2: %s" % term2[j][cluster_id])
                print("3: %s" % term3[cluster_id])

            res_label[j] = np.argmin(dist)

            print(dist)
            print(res_label[j])

        diff = 0
        for j in range(len(data)):
            if res_label[j] != old_label[j]:
                diff += 1

        if diff < 5:
            break

    visualize(data, res_label, num_cluster, count)

    return res_label


def spectral(data, num_cluster, is_normalized=False):
    n = len(data)
    k = num_cluster

    W = build_gram_matrix(data)
    D = np.diag(np.sum(W, axis=1))

    L = D - W
    if is_normalized:
        L = np.matmul(np.linalg.inv(D), L)

    val, vec = np.linalg.eig(L)
    sorted_id = np.argsort(val)

    U = (vec[:, sorted_id])[:, 1: k + 1]

    if is_normalized:
        norm_value = np.sum(U, axis=1)
        U = U / norm_value[:, None]

    res_label = k_mean(data=U, num_cluster=k, kernel_pair=W)
    return res_label


if __name__ == '__main__':
    print("HW06 - 0860832")

    img_data = read_data(file_name="image1.png")
    # label = k_mean(data=img_data, num_cluster=2)

    label = spectral(data=img_data, num_cluster=2)
