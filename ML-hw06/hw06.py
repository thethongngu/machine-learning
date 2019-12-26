from PIL import Image

import sys
import numpy as np
import matplotlib.pyplot as plt

HEIGHT = 50
WIDTH = 50
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

    draw_input(data)

    return np.array(data)


def RBF(x, y, gamma_s, gamma_c):
    spatial = -gamma_s * ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)
    color = -gamma_c * ((x[2] - y[2]) ** 2 + (x[3] - y[3]) ** 2 + (x[4] - y[4]) ** 2)
    res = np.exp(spatial) * np.exp(color)
    return res


def draw_clusters(data, res_label, iteration, name):
    colors = ['red', 'blue', 'green', 'yellow']
    title = name + " - Iteration " + str(iteration)

    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.ylim(max(plt.ylim()), min(plt.ylim()))

    for i in range(len(data)):
        plt.scatter(x=data[i][0], y=data[i][1], color=colors[res_label[i][0]])

    plt.title(title)
    plt.show()


def draw_input(data):
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.ylim(max(plt.ylim()), min(plt.ylim()))

    for i in range(len(data)):
        plt.scatter(x=data[i][0], y=data[i][1], color=(data[i][2] / 255.0, data[i][3] / 255.0, data[i][4] / 255.0))

    plt.show()


def draw_eigenspace(data, res_label, num_cluster, name):
    colors = ['red', 'blue', 'green', 'yellow']

    for cluster_id in range(num_cluster):
        plt.clf()
        for i in range(len(data)):
            if res_label[i] == cluster_id:
                point = data[i]
                plt.scatter(x=point[0], y=point[1], color=colors[res_label[i][0]])
        plt.title(name)
        plt.show()


def build_gram_matrix(data, ls, lc):
    kernel_pair = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        kernel_pair[i][i] = 1
        for j in range(i + 1, len(data)):
            kernel_pair[i][j] = RBF(data[i], data[j], ls, lc)
            kernel_pair[j][i] = kernel_pair[i][j]

    return kernel_pair


def init_k_mean(data, num_cluster):
    mean = [np.random.randint(0, len(data))]
    point_to_mean = np.zeros((len(data), 1))

    for cluster_id in range(1, num_cluster):

        for i in range(len(data)):
            d = sys.maxsize
            for mean_id in mean:
                d = min(d, np.linalg.norm(data[i] - data[mean_id], 2))

            point_to_mean[i] = d

        mean.append(np.argmax(point_to_mean))

    return mean


def assign_label_by_mean(data, mean, num_cluster):
    res_label = np.zeros((len(data), 1), dtype=int)
    for i in range(len(data)):
        dist = np.zeros((num_cluster, 1))
        for cluster_id in range(num_cluster):
            mean_id = mean[cluster_id]
            dist[cluster_id] = np.linalg.norm(data[i] - data[mean_id], 2)
        res_label[i] = np.argmin(dist)
    return res_label


def k_mean(data, num_cluster, original_data, init_mean=False, kernel_pair=None, title=""):
    res_label = np.zeros((len(data), 1), dtype=int)
    term2 = np.zeros((len(data), num_cluster))
    term3 = np.zeros((len(data), 1))

    # initialization
    if not init_mean:
        mean = []
        for cluster_id in range(num_cluster):
            mean.append(np.random.randint(0, len(data)))
        res_label = assign_label_by_mean(data, mean, num_cluster)
    else:
        mean = init_k_mean(data, num_cluster)
        res_label = assign_label_by_mean(data, mean, num_cluster)

    # calculate gram matrix
    if kernel_pair is None:
        kernel_pair = build_gram_matrix(data, 0.001, 0.001)
    count = 0

    # k-mean
    while True:

        draw_clusters(original_data, res_label, count, title)
        count += 1

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
                        res += kernel_pair[p][q]

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
                        res += kernel_pair[j][n]

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
                dist[cluster_id] = kernel_pair[j][j] - term2[j][cluster_id] + term3[cluster_id]

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

    draw_clusters(original_data, res_label, count, title)

    return res_label


def spectral(data, num_cluster, is_normalized=False, init_mean=False, W=None, title=""):
    n = len(data)
    k = num_cluster

    if W is None:
        W = build_gram_matrix(data, 0.001, 0.001)

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

    res_label = k_mean(data=U, num_cluster=k, original_data=data, init_mean=init_mean, kernel_pair=W, title=title)

    draw_eigenspace(U, res_label, num_cluster, title)

    return res_label


if __name__ == '__main__':
    print("HW06 - 0860832")

    # Image 1
    # img_data = read_data(file_name="image1.png")
    # gram_matrix = build_gram_matrix(img_data, 0.001, 0.001)

    # k_mean(data=img_data, num_cluster=2, original_data=img_data, init_mean=False, kernel_pair=gram_matrix, title="kmean - 2 cluster")
    # spectral(data=img_data, num_cluster=2, W=gram_matrix, title="Ratio Cut - 2 cluster")
    # spectral(data=img_data, num_cluster=2, is_normalized=True, W=gram_matrix, title="Normalized Cut - 2 cluster")
    #
    # k_mean(data=img_data, num_cluster=3, original_data=img_data, init_mean=False, kernel_pair=gram_matrix, title="kmean - 3 cluster")
    # spectral(data=img_data, num_cluster=3, W=gram_matrix, title="Ratio Cut - 3 cluster")
    # spectral(data=img_data, num_cluster=3, is_normalized=True, W=gram_matrix, title="Normalized Cut - 3 cluster")
    #
    # k_mean(data=img_data, num_cluster=4, original_data=img_data, kernel_pair=gram_matrix, init_mean=False, title="kmean - 4 cluster")
    # spectral(data=img_data, num_cluster=4, W=gram_matrix, title="Ratio Cut - 4 cluster")
    # spectral(data=img_data, num_cluster=4, is_normalized=True, W=gram_matrix, title="Normalized Cut - 4 cluster")

    # k_mean(data=img_data, num_cluster=3, original_data=img_data, kernel_pair=gram_matrix, init_mean=True, title="kmean++ - 3 cluster")
    # spectral(data=img_data, num_cluster=3, W=gram_matrix, title="Ratio Cut kmean++ - 3 cluster")
    # spectral(data=img_data, num_cluster=3, is_normalized=True, W=gram_matrix, title="Normalized Cut kmean++ - 3 cluster")

    # Image 2
    img_data = read_data(file_name="image2.png")
    gram_matrix = build_gram_matrix(img_data, 0.001, 0.001)

    k_mean(data=img_data, num_cluster=2, original_data=img_data, init_mean=False, kernel_pair=gram_matrix, title="kmean - 2 cluster")
    spectral(data=img_data, num_cluster=2, W=gram_matrix, title="Ratio Cut - 2 cluster")
    spectral(data=img_data, num_cluster=2, is_normalized=True, W=gram_matrix, title="Normalized Cut - 2 cluster")

    k_mean(data=img_data, num_cluster=3, original_data=img_data, init_mean=False, kernel_pair=gram_matrix, title="kmean - 3 cluster")
    spectral(data=img_data, num_cluster=3, W=gram_matrix, title="Ratio Cut - 3 cluster")
    spectral(data=img_data, num_cluster=3, is_normalized=True, W=gram_matrix, title="Normalized Cut - 3 cluster")

    k_mean(data=img_data, num_cluster=4, original_data=img_data, kernel_pair=gram_matrix, init_mean=False, title="kmean - 4 cluster")
    spectral(data=img_data, num_cluster=4, W=gram_matrix, title="Ratio Cut - 4 cluster")
    spectral(data=img_data, num_cluster=4, is_normalized=True, W=gram_matrix, title="Normalized Cut - 4 cluster")

    k_mean(data=img_data, num_cluster=3, original_data=img_data, kernel_pair=gram_matrix, init_mean=True, title="kmean++ - 3 cluster")
    spectral(data=img_data, num_cluster=3, W=gram_matrix, title="Ratio Cut kmean++ - 3 cluster")
    spectral(data=img_data, num_cluster=3, is_normalized=True, W=gram_matrix, title="Normalized Cut kmean++ - 3 cluster")
