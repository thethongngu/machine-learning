import random

from PIL import Image
from scipy.spatial.distance import pdist, squareform, cdist

import sys
import numpy as np
import matplotlib.pyplot as plt

HEIGHT = 20
WIDTH = 20
LAMBDA_S = 0.001
LAMBDA_C = 0.001


def read_data(file_name):
    image = Image.open(file_name)
    image.thumbnail((WIDTH, HEIGHT), Image.ANTIALIAS)

    pixels = image.load()

    data = []
    for i in range(0, WIDTH):
        for j in range(0, HEIGHT):
            data.append([i, j, pixels[i, j][0], pixels[i, j][1], pixels[i, j][2]])

    draw_image(data)

    return np.array(data)


def draw_image(data):
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.ylim(max(plt.ylim()), min(plt.ylim()))

    for i in range(len(data)):
        plt.scatter(x=data[i][0], y=data[i][1], color=(data[i][2] / 255.0, data[i][3] / 255.0, data[i][4] / 255.0))

    plt.show()


def draw_clusters(data, data_label, iteration, name):
    colors = ['red', 'blue', 'green', 'yellow', 'orange']
    title = name + " - Iteration " + str(iteration)

    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.ylim(max(plt.ylim()), min(plt.ylim()))

    for i in range(len(data)):
        plt.scatter(x=data[i][0], y=data[i][1], color=colors[data_label[i]])

    plt.title(title)
    plt.show()


def draw_eigenspace(data, data_label, name):
    colors = ['red', 'blue', 'green', 'yellow', 'orange']

    for i in range(len(data)):
        plt.scatter(x=data[i][0], y=data[i][1], color=colors[data_label[i]])

    plt.title(name)
    plt.show()


def build_gram_matrix(data, gamma_s, gamma_c):
    coordinates = data[:, :3]
    colors = data[:, 3:]

    spatial_RBF = -gamma_s * squareform(pdist(coordinates, 'sqeuclidean'))
    color_RBF = -gamma_c * squareform(pdist(colors, 'sqeuclidean'))
    kernel_pair = np.exp(spatial_RBF) * np.exp(color_RBF)

    return kernel_pair


def init_centers(data, num_cluster, method=""):
    centers = np.zeros((num_cluster, data.shape[1]))

    if method == "random":
        ids = np.random.randint(0, len(data), size=num_cluster)
        for i in range(len(ids)):
            centers[i] = data[ids[i]]

    if method == "k-mean++":

        # create first center randomly
        centers[0] = data[np.random.randint(0, len(data))]

        # find remained centers for each cluster_id
        for cluster_id in range(1, num_cluster):
            dist = cdist(data, centers, 'sqeuclidean')
            centers[cluster_id] = np.argmax(np.min(dist, axis=1), axis=0)

    return centers


def assign_label_by_mean(data, mean, num_cluster):
    res_label = np.zeros((len(data), 1), dtype=int)
    for i in range(len(data)):
        dist = np.zeros((num_cluster, 1))
        for cluster_id in range(num_cluster):
            mean_id = mean[cluster_id]
            dist[cluster_id] = np.linalg.norm(data[i] - data[mean_id], 2)
        res_label[i] = np.argmin(dist)
    return res_label


def kernel_k_mean(data, num_cluster, original_data, init_mean, kernel_matrix=None):
    term2 = np.zeros((len(data), num_cluster))
    term3 = np.zeros((len(data), 1))

    # initialization
    centers = init_centers(data, num_cluster, init_mean)
    dist = cdist(data, centers, 'sqeuclidean')
    data_label = np.argmin(dist, axis=1)

    if kernel_matrix is None:
        kernel_matrix = build_gram_matrix(data, 0.001, 0.001)

    iteration = 0

    # k-mean
    while True:

        title = "Kernel K-mean - Iteration: %s - Init: %s" % (iteration, "K-mean++" if init_mean else "")
        iteration += 1

        # count number of elements of each cluster
        num_element = np.zeros((len(data), 1))
        for i in range(len(res_label)):
            num_element[res_label[i]] += 1

        # calculate the third term of applied kernel distance
        for cluster_id in range(num_cluster):
            res = 0.0
            for p in range(len(data)):
                for q in range(len(data)):
                    if res_label[p] == cluster_id and res_label[q] == cluster_id:
                        res += kernel_matrix[p][q]

            if num_element[cluster_id] == 0:
                term3[cluster_id] = 0
            else:
                term3[cluster_id] = (res / (num_element[cluster_id] ** 2))

        # calculate the second term of applied kernel distance
        for cluster_id in range(num_cluster):
            for j in range(len(data)):
                res = 0.0
                for n in range(len(data)):
                    if res_label[n] == cluster_id:
                        res += kernel_matrix[j][n]

                if num_element[cluster_id] == 0:
                    term2[j][cluster_id] = 0
                else:
                    term2[j][cluster_id] = 2.0 * (res / num_element[cluster_id])

        # assign point to nearest mean
        old_label = np.copy(res_label)
        for j in range(len(data)):
            dist = np.zeros((num_cluster, 1))

            # print()
            # print(j)
            # print("Pos: (%s, %s)" % (data[j][0], data[j][1]))

            for cluster_id in range(num_cluster):
                dist[cluster_id] = kernel_matrix[j][j] - term2[j][cluster_id] + term3[cluster_id]

                # print("1: %s" % kernel_pair[j][j])
                # print("2: %s" % term2[j][cluster_id])
                # print("3: %s" % term3[cluster_id])

            res_label[j] = np.argmin(dist)

            # print(dist)
            # print(res_label[j])

        # check difference
        diff = 0
        for j in range(len(data)):
            if res_label[j] != old_label[j]:
                diff += 1

        if diff < 5:
            break

    title = "Kernel K-mean - Iteration: %s - Init: %s" % (iteration, "K-mean++" if init_mean else "")
    draw_clusters(original_data, res_label, iteration, title)

    return res_label


def k_mean(data, num_cluster, original_data, init_center, title=""):
    # initialization
    centers = init_centers(data, num_cluster, init_center)
    data_label = np.zeros((len(data), 1), dtype=int)
    iteration = 0

    # k-mean
    while True:

        iteration += 1

        # calculate cluster of each point
        old_label = data_label
        dist = cdist(data, centers, 'sqeuclidean')
        data_label = np.argmin(dist, axis=1)

        draw_clusters(original_data, data_label, iteration, title)
        draw_eigenspace(data, data_label, "aaa")

        # calculate new center
        total = np.zeros((num_cluster, data.shape[1]), dtype=float)
        count = np.zeros((num_cluster, 1), dtype=int)

        for p in range(len(data)):
            total[data_label[p]] += data[p]
            count[data_label[p]] += 1

        for cluster_id in range(num_cluster):
            if count[cluster_id] == 0:
                count[cluster_id] = 1

            centers[cluster_id] = total[cluster_id] / count[cluster_id]

        diff = np.count_nonzero(old_label != data_label)
        if diff < 5:
            break

    return data_label


def spectral(data, num_cluster, is_normalized=False, init_center="random", W=None):
    if W is None:
        W = build_gram_matrix(data, 0.001, 0.001)

    if is_normalized:
        D_power = np.diag(np.power(np.sum(W, axis=1), -0.5))
        L = np.eye(len(data)) - np.matmul(np.matmul(D_power, W), D_power)
    else:
        D = np.diag(np.sum(W, axis=1))
        L = D - W

    val, vec = np.linalg.eig(L)
    sorted_id = np.argsort(val)[1: num_cluster + 1]

    U = vec[:, sorted_id]

    if is_normalized:
        norm_value = np.linalg.norm(U, 2, axis=1)
        for i in range(norm_value.shape[0]):
            if norm_value[i] == 0:
                norm_value[i] = 1

        U /= norm_value[:, None]

    title = "%s - Init: %s" % (
        "Normalized Cut" if is_normalized else "Ratio Cut",
        "K-mean++" if init_center else "Random"
    )

    data_label = k_mean(data=U, num_cluster=num_cluster, original_data=data, init_center=init_center, title=title)

    draw_eigenspace(U, data_label, title)

    return data_label


if __name__ == '__main__':
    print("HW06 - 0860832")

    # Image 1
    img_data = read_data(file_name="image1.png")
    gram_matrix = build_gram_matrix(img_data, 0.001, 0.001)

    # kernel_k_mean(data=img_data, num_cluster=2, original_data=img_data, init_mean=False, kernel_pair=gram_matrix)
    spectral(data=img_data, num_cluster=2, W=gram_matrix)
    spectral(data=img_data, num_cluster=2, is_normalized=True, W=gram_matrix)

    # kernel_k_mean(data=img_data, num_cluster=3, original_data=img_data, init_mean=False, kernel_pair=gram_matrix)
    # spectral(data=img_data, num_cluster=3, W=gram_matrix)
    # spectral(data=img_data, num_cluster=3, is_normalized=True, W=gram_matrix)
    #
    # kernel_k_mean(data=img_data, num_cluster=4, original_data=img_data, kernel_pair=gram_matrix, init_mean=False)
    # spectral(data=img_data, num_cluster=4, W=gram_matrix)
    # spectral(data=img_data, num_cluster=4, is_normalized=True, W=gram_matrix)
    #
    # kernel_k_mean(data=img_data, num_cluster=3, original_data=img_data, kernel_pair=gram_matrix, init_mean=True)
    # spectral(data=img_data, num_cluster=3, init_mean=True, W=gram_matrix)
    # spectral(data=img_data, num_cluster=3, init_mean=True, is_normalized=True, W=gram_matrix)

    # Image 2
    # img_data = read_data(file_name="image2.png")
    # gram_matrix = build_gram_matrix(img_data, 0.001, 0.001)

    # kernel_k_mean(data=img_data, num_cluster=2, original_data=img_data, init_mean=False, kernel_pair=gram_matrix)
    # spectral(data=img_data, num_cluster=2, kernel_pair=gram_matrix)
    # spectral(data=img_data, num_cluster=2, is_normalized=True, W=gram_matrix)
    #
    # kernel_k_mean(data=img_data, num_cluster=3, original_data=img_data, init_mean=False, kernel_pair=gram_matrix)
    # spectral(data=img_data, num_cluster=3, kernel_pair=gram_matrix)
    # spectral(data=img_data, num_cluster=3, is_normalized=True, kernel_pair=gram_matrix)
    #
    # kernel_k_mean(data=img_data, num_cluster=4, original_data=img_data, kernel_pair=gram_matrix, init_mean=False)
    # spectral(data=img_data, num_cluster=4, kernel_pair=gram_matrix)
    # spectral(data=img_data, num_cluster=4, is_normalized=True, kernel_pair=gram_matrix)
    #
    # kernel_k_mean(data=img_data, num_cluster=3, original_data=img_data, kernel_pair=gram_matrix, init_mean=True)
    # spectral(data=img_data, num_cluster=3, kernel_pair=gram_matrix)
    # spectral(data=img_data, num_cluster=3, is_normalized=True, kernel_pair=gram_matrix)
