from PIL import Image
from scipy.spatial.distance import cdist

import numpy as np
import glob
import matplotlib.pyplot as plt
import random

num_subject = 15
num_property = 11
num_train = 135
num_test = 30

height = 50
width = 50
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
        img = img.resize((height, width))

        training[:, count_train] = np.array(img).reshape(-1, )
        pos = file.find('subject') + 7
        train_label[count_train] = int(file[pos: pos + 2])
        count_train += 1

    train_path = path + '/Testing/*'
    for file in glob.glob(train_path):
        img = Image.open(file)
        img = img.resize((height, width))

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


def LDA(training, testing, train_label, test_label):
    mean_all = (np.sum(training, axis=1) / num_train).reshape((num_pixel, 1))  # (f, 1)
    mean_class = np.zeros((num_subject, num_pixel))  # (15, f)
    num_train_per_class = 9

    for class_id in range(num_subject):
        start_id = class_id * num_train_per_class
        end_id = start_id + num_train_per_class - 1
        mean_class[class_id] = np.sum(training[:, start_id: end_id], axis=1) / num_train_per_class  # (15, f)
        show_image(mean_class[class_id])

    SW = np.zeros((num_pixel, num_pixel))
    SB = np.zeros((num_pixel, num_pixel))
    for class_id in range(num_subject):
        start_id = class_id * num_train_per_class
        end_id = start_id + num_train_per_class - 1
        sw_diff = training[:, start_id: end_id] - mean_class[class_id].reshape(num_pixel, 1)  # (f, 9) - (f, 1) = (f, 9)
        SW += sw_diff @ sw_diff.T  # (f, f)

        sb_diff = mean_class[class_id].T - mean_all  # (f, 1) - (f, 1) = (f, 1)
        SB += sb_diff @ sb_diff.T

    low_dim = 25
    eigen_value, eigen_vector = np.linalg.eigh(np.linalg.inv(SW) @ SB)
    sorted_id = np.argsort(eigen_value)
    eigen_vector = eigen_vector[sorted_id][::-1]
    W = eigen_vector[:low_dim]  # (low_dim, f)

    # ------------ show first 25 fisherfaces --------------------
    fig = plt.figure(figsize=(50, 50))

    for i in range(low_dim):
        fig.add_subplot(5, 5, i + 1)
        plt.imshow(W[i].reshape(height, width), cmap='gray')
        plt.axis('off')
        plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off',
                        labelright='off', labelbottom='off')

    plt.show()


if __name__ == '__main__':
    train, test, train_label, test_label = read_data(
        '/home/thethongngu/Documents/code/machine-learning/ML-hw07/Yale_Face_Database/'
    )
    LDA(train, test, train_label, test_label)
