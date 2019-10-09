#include <iostream>
#include <fstream>
#include <vector>

const std::string kMnistTrainImagePath = "../data/train-images-idx3-ubyte";
const std::string kMnistTrainLabelPath = "../data/train-labels-idx1-ubyte";

auto big_to_small(unsigned int data) {
    unsigned int byte00 = (data & 0x000000ffu) << 24u;
    unsigned int byte01 = (data & 0x0000ff00u) << 8u;
    unsigned int byte02 = (data & 0x00ff0000u) >> 8u;
    unsigned int byte03 = (data & 0xff000000) >> 24u;

    return byte00 | byte01 | byte02 | byte03;
}

void read_mnist_data() {

    std::ifstream image_file;
    int num_images;

    image_file.open(kMnistTrainImagePath, std::ios::binary);

    if (image_file.is_open()) {
        image_file.read((char *)&num_images, 4);
        image_file.read((char *)&num_images, 4);
        std::cout << "Number: " << big_to_small(num_images);
    }
}

int main() {
    read_mnist_data();
    return 0;
}

