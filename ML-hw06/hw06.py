import random

from PIL import Image
from scipy.spatial.distance import pdist, squareform, cdist

import sys
import numpy as np
import matplotlib.pyplot as plt

HEIGHT = 100
WIDTH = 100
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

    plt.savefig("image%s/%s" % (active_image, title))
    plt.show()


def draw_eigenspace(data, data_label, name):
    colors = ['red', 'blue', 'green', 'yellow', 'orange']

    for i in range(len(data)):
        plt.scatter(x=data[i][0], y=data[i][1], color=colors[data_label[i]])

    plt.title(name)
    plt.savefig("image%s/%s" % (active_image, name))
    plt.show()


def build_kernel_distance(data, gamma_s, gamma_c, is_matrix=True):
    coordinates = data[:, :3]
    colors = data[:, 3:]

    if is_matrix:
        spatial_RBF = -gamma_s * squareform(pdist(coordinates, 'sqeuclidean'))
        color_RBF = -gamma_c * squareform(pdist(colors, 'sqeuclidean'))
        kernel_distance = np.exp(spatial_RBF) * np.exp(color_RBF)

    else:
        spatial_RBF = -gamma_s * pdist(coordinates, 'sqeuclidean')
        color_RBF = -gamma_c * pdist(colors, 'sqeuclidean')
        kernel_distance = np.exp(spatial_RBF) * np.exp(color_RBF)

    return kernel_distance


def init_kmean(data, num_cluster, method=""):
    if method == "random":
        centers = np.zeros((num_cluster, data.shape[1]))
        ids = np.random.randint(0, len(data), size=num_cluster)
        for i in range(len(ids)):
            centers[i] = data[ids[i]]

    else:

        # create first center randomly
        centers_array = [data[np.random.randint(0, len(data))]]
        centers = np.array(centers_array)

        # find remained centers for each cluster_id
        for cluster_id in range(1, num_cluster):
            dist = cdist(data, centers, 'sqeuclidean')
            centers_array.append(data[np.argmax(np.min(dist, axis=1), axis=0)])
            centers = np.array(centers_array)

    return centers


def init_kernel_kmean(data, num_cluster, method="", kernel_matrix=None):
    A = np.zeros((data.shape[0], num_cluster))
    data_label = np.zeros((len(data),), dtype=int)
    A[:, 0] = 1

    if method == "random":
        for i in range(len(data)):
            cluster_id = np.random.randint(0, num_cluster)
            A[i, cluster_id] = 1
            data_label[i] = cluster_id

    if method == "k-mean++":
        center_ids = np.zeros((1,), dtype=int)
        center_ids[0] = np.random.randint(0, len(data))

        for i in range(1, num_cluster):

            dist = np.min(kernel_matrix[:, center_ids], axis=1)
            next_id = np.argmin(dist, axis=0)
            center_ids = np.append(center_ids, next_id)

            old_label = data_label
            data_label = np.argmin(kernel_matrix[:, center_ids], axis=1)
            for j in range(len(data)):
                A[j, old_label[j]] = 0
                A[j, data_label[j]] = 1

    return A, data_label


def kernel_kmean(data, num_cluster, init_mean, kernel_matrix=None):
    # calculate gram matrix
    if kernel_matrix is None:
        kernel_matrix = build_kernel_distance(data, 0.001, 0.001)

    A, data_label = init_kernel_kmean(data, num_cluster, init_mean, kernel_matrix)

    iteration = 0

    # k-mean
    while True:

        title = "Kernel k-mean - Cluster %s - Initialize %s" % (num_cluster, init_mean)
        draw_clusters(data, data_label, iteration, title)
        iteration += 1

        # count number of elements of each cluster
        count_point = np.sum(A, axis=0)
        for i in range(count_point.shape[0]):
            if count_point[i] == 0:
                count_point[i] = 1

        second_term = (-2.0 * (kernel_matrix @ A)) / count_point[None, :]

        third_term = np.zeros((num_cluster,))
        for cluster_id in range(num_cluster):
            point_ids = np.where(A[:, cluster_id] == 1)[0]
            points = data[point_ids]
            pair_distance = build_kernel_distance(points, 0.001, 0.001, is_matrix=False)
            third_term[cluster_id] = np.sum(pair_distance) / (count_point[cluster_id] ** 2)

        dist = second_term + third_term[None, :]
        old_label = data_label
        data_label = np.argmin(dist, axis=1)

        for i in range(len(data)):
            A[i, old_label[i]] = 0
            A[i, data_label[i]] = 1

        diff = np.count_nonzero(old_label != data_label)
        if diff < 5:
            break

    draw_clusters(data, data_label, iteration, title)

    return data_label


def kmean(data, num_cluster, original_data, init_center, title=""):
    # initialization
    centers = init_kmean(data, num_cluster, init_center)
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
        # draw_eigenspace(data, data_label, "aaa")

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


def spectral(data, num_cluster, is_normalized, init_center, W=None):
    if W is None:
        W = build_kernel_distance(data, 0.001, 0.001)

    if is_normalized:
        D_power = np.diag(np.power(np.sum(W, axis=1), -0.5))
        L = np.eye(len(data)) - np.matmul(np.matmul(D_power, W), D_power)
    else:
        D = np.diag(np.sum(W, axis=1))
        L = D - W

    val, vec = np.linalg.eig(L)
    sorted_id = np.argsort(val)[1: num_cluster + 1]

    U = vec[:, sorted_id].astype(float)

    if is_normalized:
        norm_value = np.linalg.norm(U, 2, axis=1)
        for i in range(norm_value.shape[0]):
            if norm_value[i] == 0:
                norm_value[i] = 1

        U /= norm_value[:, None]

    title = "%s - Cluster %s - Init %s" % (
        "Normalized Cut" if is_normalized else "Ratio Cut",
        num_cluster,
        init_center
    )

    data_label = kmean(data=U, num_cluster=num_cluster, original_data=data, init_center=init_center, title=title)

    draw_eigenspace(U, data_label, "Eigen - " + title)

    return data_label


active_image = 1

if __name__ == '__main__':
    print("HW06 - 0860832")

    # Image 1
    img_data = read_data(file_name="image1.png")
    gram_matrix = build_kernel_distance(img_data, 0.001, 0.001)

    # kernel_kmean(data=img_data, num_cluster=2, init_mean="random", kernel_matrix=gram_matrix)
    # spectral(data=img_data, num_cluster=2, is_normalized=False, init_center="random", W=gram_matrix)
    spectral(data=img_data, num_cluster=2, is_normalized=True, init_center="random", W=gram_matrix)

    kernel_kmean(data=img_data, num_cluster=3, init_mean="random", kernel_matrix=gram_matrix)
    spectral(data=img_data, num_cluster=3, is_normalized=False, init_center="random", W=gram_matrix)
    spectral(data=img_data, num_cluster=3, is_normalized=True, init_center="random", W=gram_matrix)

    kernel_kmean(data=img_data, num_cluster=4, init_mean="random", kernel_matrix=gram_matrix)
    spectral(data=img_data, num_cluster=4, is_normalized=False, init_center="random", W=gram_matrix)
    spectral(data=img_data, num_cluster=4, is_normalized=True, init_center="random", W=gram_matrix)

    kernel_kmean(data=img_data, num_cluster=3, init_mean="k-mean++", kernel_matrix=gram_matrix)
    spectral(data=img_data, num_cluster=3, is_normalized=False, init_center="k-mean++", W=gram_matrix)
    spectral(data=img_data, num_cluster=3, is_normalized=True, init_center="k-mean++", W=gram_matrix)

    active_image = 2

    # Image 2
    img_data = read_data(file_name="image2.png")
    gram_matrix = build_kernel_distance(img_data, 0.001, 0.001)

    kernel_kmean(data=img_data, num_cluster=2, init_mean="random", kernel_matrix=gram_matrix)
    spectral(data=img_data, num_cluster=2, is_normalized=False, init_center="random", W=gram_matrix)
    spectral(data=img_data, num_cluster=2, is_normalized=True, init_center="random", W=gram_matrix)

    kernel_kmean(data=img_data, num_cluster=3, init_mean="random", kernel_matrix=gram_matrix)
    spectral(data=img_data, num_cluster=3, is_normalized=False, init_center="random", W=gram_matrix)
    spectral(data=img_data, num_cluster=3, is_normalized=True, init_center="random", W=gram_matrix)

    kernel_kmean(data=img_data, num_cluster=4, init_mean="random", kernel_matrix=gram_matrix)
    spectral(data=img_data, num_cluster=4, is_normalized=False, init_center="random", W=gram_matrix)
    spectral(data=img_data, num_cluster=4, is_normalized=True, init_center="random", W=gram_matrix)

    kernel_kmean(data=img_data, num_cluster=3, init_mean="k-mean++", kernel_matrix=gram_matrix)
    spectral(data=img_data, num_cluster=3, is_normalized=False, init_center="k-mean++", W=gram_matrix)
    spectral(data=img_data, num_cluster=3, is_normalized=True, init_center="k-mean++", W=gram_matrix)
