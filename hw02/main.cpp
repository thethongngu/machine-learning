#include <iostream>
#include <fstream>
#include <vector>
#include "image.h"

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

    std::ifstream image_file;
    int magic_number, num_images;
    int num_row, num_col;

    image_file.open(kMnistTrainImagePath, std::ios::binary);

    if (image_file.is_open()) {
        image_file.read((char *)&magic_number, sizeof(magic_number));
        image_file.read((char *)&num_images, sizeof(num_images));
        image_file.read((char *)&num_row, sizeof(num_row));
        image_file.read((char *)&num_col, sizeof(num_col));

        num_images = big_to_small(num_images);
        num_row = big_to_small(num_row);
        num_col = big_to_small(num_col);

        char *image_data = new char[Image::kImageSize];
        for(int i = 0; i < num_images; i++) {

            image_file.read(image_data, sizeof(image_data));
            for(char *pixel = image_data; pixel < image_data + Image::kImageSize; pixel++) {
                *pixel = big_to_small(*pixel);
            }

            Image image(image_data);
            images.push_back(image);
        }
    }
}

int main() {
    read_mnist_data();
    return 0;
}

