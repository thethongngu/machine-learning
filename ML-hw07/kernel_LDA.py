from PIL import Image
from scipy.spatial.distance import pdist, squareform, cdist

import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import random

num_subject = 15
num_property = 11
num_train = 135
num_test = 30

height = 195
width = 231
num_pixel = height * width

np.set_printoptions(threshold=np.inf)


def read_data(path):
    count_train = 0
    count_test = 0
    training = np.zeros((num_pixel, num_train))
    testing = np.zeros((num_pixel, num_test))
    train_label = np.zeros((num_train,), dtype=int)
    test_label = np.zeros((num_test,), dtype=int)

    train_path = path + '/Training/*'
    for file in glob.glob(train_path):
        img = Image.open(file)
        training[:, count_train] = np.array(img).reshape(-1, )
        pos = file.find('subject') + 7
        train_label[count_train] = int(file[pos: pos + 2]) - 1
        count_train += 1

    test_path = path + '/Testing/*'
    for file in glob.glob(test_path):
        img = Image.open(file)
        testing[:, count_test] = np.array(img).reshape(-1, )
        pos = file.find('subject') + 7
        test_label[count_test] = int(file[pos: pos + 2]) - 1
        count_test += 1

    return training, testing, train_label, test_label


def show_image(data):
    emin = np.min(data)
    emax = np.max(data)
    data = (data - emin) * 255 / (emax - emin)
    Image.fromarray(data.reshape(width, height)).show()


def kernel_PCA(training, low_dim, kernel_type):
    if kernel_type == 'RBF':
        gamma = 1e-3
        K_train = np.exp(-gamma * squareform(pdist(training.T, 'sqeuclidean')))  # (135, 135)
    if kernel_type == 'linear':
        K_train = training.T @ training

    one = np.ones((num_train, num_train)) / num_train
    K_center = K_train - one @ K_train - K_train @ one + one @ K_train @ one
    eigen_value, eigen_vector = np.linalg.eigh(K_center)

    sorted_id = np.argsort(eigen_value)
    eigen_vector = eigen_vector[sorted_id].T[::-1][: low_dim]
    eigen_value = eigen_value.T[::-1][: low_dim]
    eigen_vector = np.true_divide(eigen_vector, np.linalg.norm(eigen_vector, ord=2, axis=1).reshape(-1, 1))
    W = eigen_vector[:low_dim]

    lowd_train = W @ K_train
    return lowd_train, W


def kernel_LDA(training, testing, train_label, test_label, kernel_type):
    low_dim = num_train - num_subject
    lowd_data, eigens_pca = kernel_PCA(training, low_dim, kernel_type)  # (120, 135), (low_dim, f)
    print(lowd_data.shape)
    print(eigens_pca.shape)

    mean_all = (np.sum(lowd_data, axis=1) / num_train).reshape((low_dim, 1))  # (f, 1)
    mean_class = np.zeros((num_subject, low_dim))  # (15, f)
    count_item = np.zeros((num_subject, 1))

    for image_id in range(num_train):
        class_id = train_label[image_id]
        mean_class[class_id] += lowd_data[:, image_id].T  # (15, f) + (1, f)
        count_item[class_id] += 1

    for class_id in range(num_subject):
        count_item[class_id] = max(1, count_item[class_id])

    mean_class = np.true_divide(mean_class, count_item)

    SW = np.zeros((low_dim, low_dim))
    for image_id in range(num_train):
        class_id = train_label[image_id]
        sw_diff = lowd_data[:, image_id] - mean_class[class_id].reshape(low_dim, 1)  # (f, 9) - (f, 1) = (f, 9)
        SW += sw_diff @ sw_diff.T  # (f, f)

    SB = np.zeros((low_dim, low_dim))
    for class_id in range(num_subject):
        sb_diff = mean_class[class_id].T - mean_all  # (f, 1) - (f, 1) = (f, 1)
        SB += sb_diff @ sb_diff.T * count_item[class_id]

    eigen_value, eigen_vector = np.linalg.eigh(np.linalg.inv(SW) @ SB)
    sorted_id = np.argsort(eigen_value)
    eigen_vector = eigen_vector[sorted_id][::-1]
    W = eigen_vector[:25]  # (25, 120)

    W = W @ eigens_pca  # (25, 120) x (120, f) = (25, f)

    print(W.shape)
    print(eigens_pca.shape)

    # ------------ face recognition --------------------
    lowd_train = W @ training
    lowd_test = W @ testing
    dist = cdist(lowd_test.T, lowd_train.T, 'euclidean')

    k = 15
    smallest_ids = train_label[np.argsort(dist, axis=1)[:, :k]]
    prediction = np.zeros((num_test,), dtype=int)
    for i in range(num_test):
        prediction[i] = np.argmax(np.bincount(smallest_ids[i]))

    error_rate = np.count_nonzero(prediction != test_label) / num_test
    print("Error rate: ", str(error_rate))


if __name__ == '__main__':
    train, test, train_label, test_label = read_data(os.getcwd() + '/Yale_Face_Database/')
    kernel_LDA(train, test, train_label, test_label, 'linear')
