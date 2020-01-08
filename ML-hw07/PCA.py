from PIL import Image
import numpy as np

num_subject = 1
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

    show_image(mean0_data[:, 0])

    K = (mean0_data.T @ mean0_data) / num_image  # 135 x 135

    eigen_value, eigen_vector = np.linalg.eig(K)  # (135, 135), (135, )
    eigen_vector = (mean0_data @ eigen_vector).T  # (2500 x 135) @ (135, 135) = (135 x 2500)

    eigen_value = eigen_value[::-1].astype(float)
    eigen_vector = eigen_vector[::-1].astype(float)
    print((np.sqrt(eigen_value).reshape(-1, 1)).shape)
    eigen_vector /= (np.sqrt(eigen_value).reshape(-1, 1))

    low_dim = 4
    W = eigen_vector[:, :4]  # (2500 x 4)

    for i in range(low_dim):
        show_image(W[:, i])

    projected_data = W.T @ mean0_data
    return projected_data


if __name__ == '__main__':
    train, test = read_data('YALE/centered/')
    projected = PCA(train)
    # Image.fromarray(projected[0], 'L').show()
