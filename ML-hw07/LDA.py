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


def PCA(training, low_dim):
    mean = (np.sum(training, axis=1) / num_train).reshape((num_pixel, 1))  # (f, 1)
    mean0_data = training - mean  # (f, 135)

    K = (mean0_data.T @ mean0_data) / num_train  # 135 x 135

    eigen_value, eigen_vector = np.linalg.eigh(K)  # (135, ), (135, 135)
    eigen_vector = (mean0_data @ eigen_vector).T  # ((f x 135) @ (135, 135)).T = (135 x f)

    sorted_id = np.argsort(eigen_value)
    eigen_vector = eigen_vector[sorted_id][::-1]
    eigen_vector = np.true_divide(eigen_vector, np.linalg.norm(eigen_vector, ord=2, axis=1).reshape(-1, 1))
    W = eigen_vector[:low_dim]  # (low_diw x f)

    return W @ training, W


def LDA(training, testing, train_label, test_label):

    low_dim = num_train - num_subject
    lowd_data, eigens_pca = PCA(training, low_dim)  # (120, 135), (low_dim, f)

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

    # ------------ show first 25 fisherfaces --------------------
    fig = plt.figure(figsize=(50, 50))

    for i in range(25):
        fig.add_subplot(5, 5, i + 1)
        plt.imshow(W[i].reshape(width, height), cmap='gray')
        plt.axis('off')
        plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off',
                        labelright='off', labelbottom='off')

    plt.show()

    # ------------ reconstruct 10 random images --------------------
    fig = plt.figure(figsize=(50, 20))
    random_ids = random.sample(range(num_train), 10)
    random_imgs = training[:, random_ids]  # (f, 10)
    reconstructed_img = random_imgs.T @ W.T @ W  # (10, f) * (f, 25) * (25, f) = (10, f)

    for i in range(10):
        fig.add_subplot(2, 5, i + 1)
        plt.imshow(reconstructed_img[i].reshape(width, height), cmap='gray')
        plt.axis('off')
        plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off',
                        labelright='off', labelbottom='off')

    plt.show()

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
    train, test, train_label, test_label = read_data(
        '/Users/thethongngu/Documents/machine-learning/ML-hw07/Yale_Face_Database/'
    )
    LDA(train, test, train_label, test_label)
