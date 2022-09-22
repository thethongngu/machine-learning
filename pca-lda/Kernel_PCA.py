from PIL import Image
from scipy.spatial.distance import pdist, squareform, cdist

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
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
        train_label[count_train] = int(file[pos: pos + 2])
        count_train += 1

    test_path = path + '/Testing/*'
    for file in glob.glob(test_path):
        img = Image.open(file)
        testing[:, count_test] = np.array(img).reshape(-1, )
        pos = file.find('subject') + 7
        test_label[count_test] = int(file[pos: pos + 2])
        count_test += 1

    return training, testing, train_label, test_label


def show_image(data):
    emin = np.min(data)
    emax = np.max(data)
    data = (data - emin) * 255 / (emax - emin)
    Image.fromarray(data.reshape(width, height)).show()


def kernel_PCA(training, testing, train_label, test_label, kernel_type):

    if kernel_type == 'RBF':
        gamma = 1e-3
        K_train = np.exp(-gamma * squareform(pdist(training.T, 'sqeuclidean')))  # (135, 135)
        K_test = np.exp(-gamma * cdist(training.T, testing.T, 'sqeuclidean'))  # (135, 135)
    if kernel_type == 'linear':
        K_train = training.T @ training
        K_test = training.T @ testing

    one = np.ones((num_train, num_train)) / num_train
    K_center = K_train - one @ K_train - K_train @ one + one @ K_train @ one
    eigen_value, eigen_vector = np.linalg.eigh(K_center)

    low_dim = 25
    sorted_id = np.argsort(eigen_value)
    eigen_vector = eigen_vector[sorted_id].T[::-1][: low_dim]
    eigen_value = eigen_value.T[::-1][: low_dim]
    eigen_vector = np.true_divide(eigen_vector, np.linalg.norm(eigen_vector, ord=2, axis=1).reshape(-1, 1))
    W = eigen_vector[:low_dim]

    lowd_train = W @ K_train  # (25, 135) x (135, 135) = (25, 135)
    lowd_test = W @ K_test  # (25, 135) x (135, 30) = (25, 30)

    k = 15
    low_dist = cdist(lowd_test.T, lowd_train.T, 'euclidean')  # (30 x 135)
    smallest_ids = train_label[np.argsort(low_dist, axis=1)[:, :k]]
    prediction = np.zeros((num_test,), dtype=int)
    for i in range(num_test):
        prediction[i] = np.argmax(np.bincount(smallest_ids[i]))

    error_rate = np.count_nonzero(prediction != test_label) / num_test
    print("Error rate: ", str(error_rate))


if __name__ == '__main__':
    train, test, train_label, test_label = read_data(os.getcwd() + '/Yale_Face_Database/')
    kernel_PCA(train, test, train_label, test_label, 'RBF')
