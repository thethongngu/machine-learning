//
// Created by Thong Nguyen on 10/10/19.
//

#include <fstream>
#include <array>
#include "input_data.h"

InputData::InputData(const std::string& image_file, const std::string& label_file) {
    read_image_file(image_file);
    read_label_file(label_file);
}

auto InputData::big_to_small_endian(unsigned int data) {
    unsigned int byte00 = (data & 0x000000ffu) << 24u;
    unsigned int byte01 = (data & 0x0000ff00u) << 8u;
    unsigned int byte02 = (data & 0x00ff0000u) >> 8u;
    unsigned int byte03 = (data & 0xff000000u) >> 24u;

    return byte00 | byte01 | byte02 | byte03;
}

void InputData::read_image_file(const std::string& image_file) {

    std::ifstream f;
    f.open(image_file, std::ios::binary);
    if (f.is_open()) {
        unsigned int magic_number;
        f.read((char *) &magic_number, sizeof(magic_number));
        f.read((char *) &num_image, sizeof(num_image));
        f.read((char *) &num_row, sizeof(num_col));
        f.read((char *) &num_col, sizeof(num_col));

        num_image = big_to_small_endian(num_image);
        num_row = big_to_small_endian(num_row);
        num_col = big_to_small_endian(num_col);

        unsigned int image_size = num_row * num_col;
        unsigned char pixel_data[image_size];

        for(unsigned int i = 0; i < num_image; i++) {
            f.read((char *)pixel_data, sizeof(pixel_data));
            std::array<unsigned int, 28 * 28> image{};
            for(unsigned int j = 0; j < image_size; j++) image[j] = (unsigned int)pixel_data[j];
            image_data.push_back(image);
        }
    }
}

void InputData::read_label_file(const std::string& label_file) {
    std::ifstream f;
    f.open(label_file, std::ios::binary);

    if (f.is_open()) {
        unsigned int _;
        f.read((char *) &_, sizeof(_));
        f.read((char *) &_, sizeof(_));
        f.read((char *) &_, sizeof(_));
        f.read((char *) &_, sizeof(_));

        char label = 0;
        for(unsigned int i = 0; i < num_image; i++) {
            f.read(&label, sizeof(label));
            label_data.push_back((unsigned int)label);
        }
    }
}

std::array<unsigned int, 784> InputData::get_image(unsigned int i) {
    return image_data[i];
}

unsigned int InputData::get_label(unsigned int i) {
    return label_data[i];
}
