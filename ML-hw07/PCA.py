from PIL import Image
import numpy as np

num_subject = 15
num_property = 11
height = 243
width = 320


def read_data(path):
    prefix = 'subject'
    suffix = ['centerlight', 'glasses', 'happy', 'leftlight', 'noglasses',
              'normal', 'rightlight', 'sad', 'sleepy', 'surprised', 'wink']

    num_train = 0
    num_test = 0
    training = np.zeros((135, height * width), dtype=int)
    testing = np.zeros((30, height * width), dtype=int)

    for subj_id in range(num_subject):
        test_id = np.random.randint(0, 11, 2)

        for prop_id in range(num_property):
            image_path = path + '/' + prefix + ('%02d' % (subj_id + 1)) + '.' + suffix[prop_id]
            img = Image.open(image_path)

            if prop_id in test_id:
                img.show()
                testing[num_test] = np.asarray(img, dtype=int).reshape(-1, )
                num_test += 1
            else:
                training[num_train] = np.asarray(img, dtype=int).reshape(-1, )
                num_train += 1

    return training, testing


if __name__ == '__main__':

    read_data('YALE/yalefaces')
