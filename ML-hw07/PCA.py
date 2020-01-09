from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

num_subject = 15
num_property = 11
num_image = num_subject * num_property

height = 195
width = 231
num_pixel = height * width

np.set_printoptions(threshold=np.inf)


def read_data(path):
    prefix = 'subject'
    suffix = ['centerlight', 'glasses', 'happy', 'leftlight', 'noglasses',
              'normal', 'rightlight', 'sad', 'sleepy', 'surprised', 'wink']

    num_train = 0
    num_test = 0
    training = np.zeros((height * width, 135))
    testing = np.zeros((height * width, 30))

    for subj_id in range(num_subject):
        test_id = np.random.randint(0, num_property, 2)
        if test_id[0] == test_id[1]:
            test_id[1] = (test_id[1] + 1) % num_property

        for prop_id in range(num_property):
            image_path = path + prefix + ('%02d' % (subj_id + 1)) + '.' + suffix[prop_id] + '.pgm'
            img = Image.open(image_path)
            img = img.resize((width, height), Image.NEAREST)

            if prop_id in test_id:
                testing[:, num_test] = np.array(img).reshape(-1, )
                num_test += 1
            else:
                training[:, num_train] = np.array(img).reshape(-1, )
                num_train += 1

    return training, testing


def show_image(data):
    Image.fromarray(data.reshape(height, width)).show()


def PCA(data):
    mean = (np.sum(data, axis=1) / num_image).reshape((num_pixel, 1))
    mean0_data = data - mean  # 2500 x 135

    K = (mean0_data.T @ mean0_data) / num_image  # 135 x 135

    eigen_value, eigen_vector = np.linalg.eig(K)  # (135, ), (135, 135)
    eigen_vector = (mean0_data @ eigen_vector).T  # ((2500 x 135) @ (135, 135)).T = (135 x 2500)

    low_dim = 25
    sorted_id = np.argsort(eigen_value)
    eigen_vector = eigen_vector[sorted_id][::-1].astype(float)
    eigen_vector = np.true_divide(eigen_vector, np.linalg.norm(eigen_vector, ord=2, axis=1).reshape(-1, 1))

    W = eigen_vector[:-low_dim - 1:-1]
    emin = np.min(W, axis=1).reshape(-1, 1)
    emax = np.max(W, axis=1).reshape(-1, 1)
    W = (W - emin) * 255 / (emax - emin)

    # ------------ 1 --------------------
    fig = plt.figure(figsize=(50, 50))

    for i in range(25):
        fig.add_subplot(5, 5, i + 1)
        plt.imshow(W[i].reshape(height, width), cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off',
                        labelright='off', labelbottom='off')

    plt.show()

    # ------------------------------------

    projected_data = W @ mean0_data
    return projected_data


if __name__ == '__main__':
    train, test = read_data('YALE/centered/')
    projected = PCA(train)
