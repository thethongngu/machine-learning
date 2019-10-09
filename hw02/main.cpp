#include <iostream>
#include <fstream>
#include <vector>
#include "image.h"

#define debug(a) std::cout << #a << ": " << a << std::endl

const std::string kMnistTrainImagePath = "../data/train-images-idx3-ubyte";
const std::string kMnistTrainLabelPath = "../data/train-labels-idx1-ubyte";

std::vector<Image> images;

auto big_to_small(unsigned int data) {
    unsigned int byte00 = (data & 0x000000ffu) << 24u;
    unsigned int byte01 = (data & 0x0000ff00u) << 8u;
    unsigned int byte02 = (data & 0x00ff0000u) >> 8u;
    unsigned int byte03 = (data & 0xff000000u) >> 24u;

    return byte00 | byte01 | byte02 | byte03;
}

void read_mnist_data() {

    std::ifstream image_file, label_file;
    int magic_number, num_images;
    int num_row, num_col;

    image_file.open(kMnistTrainImagePath, std::ios::binary);
    label_file.open(kMnistTrainLabelPath, std::ios::binary);

    if (image_file.is_open() && label_file.is_open()) {

        label_file.read((char *)&magic_number, sizeof(magic_number));
        label_file.read((char *)&num_images, sizeof(num_images));

        image_file.read((char *)&magic_number, sizeof(magic_number));
        image_file.read((char *)&num_images, sizeof(num_images));
        image_file.read((char *)&num_row, sizeof(num_row));
        image_file.read((char *)&num_col, sizeof(num_col));

        num_images = big_to_small(num_images);
        num_row = big_to_small(num_row);
        num_col = big_to_small(num_col);

        unsigned char image_data[784];
        char label_data[1];
        for(int i = 0; i < 1; i++) {

            image_file.read((char *)image_data, sizeof(image_data));
            for(int j = 0; j < 28 * 28; j++) std::cout << (int)image_data[j] << std::endl;



            label_file.read(label_data, sizeof(label_data));


            debug(label_data);
//            Image image(image_data, (unsigned int)label_data[0]);
//            images.push_back(image);
        }
    }
}

int main() {
    read_mnist_data();
    return 0;
}

