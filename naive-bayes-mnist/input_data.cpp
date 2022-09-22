//
// Created by Thong Nguyen on 10/10/19.
//

#include <fstream>
#include <array>
#include <iostream>
#include "input_data.h"

#define debug(a) std::cout << #a << " = " << a << std::endl

InputData::InputData(const std::string& image_file, const std::string& label_file) {
    ReadImageFile(image_file);
    ReadLabelFile(label_file);
}

auto InputData::ConvertBigToSmallEndian(unsigned int data) {
    unsigned int byte00 = (data & 0x000000ffu) << 24u;
    unsigned int byte01 = (data & 0x0000ff00u) << 8u;
    unsigned int byte02 = (data & 0x00ff0000u) >> 8u;
    unsigned int byte03 = (data & 0xff000000u) >> 24u;

    return byte00 | byte01 | byte02 | byte03;
}

void InputData::ReadImageFile(const std::string& image_file) {

    std::ifstream f;
    f.open(image_file, std::ios::binary);
    if (f.is_open()) {
        unsigned int magic_number;
        f.read((char *) &magic_number, sizeof(magic_number));
        f.read((char *) &num_image, sizeof(num_image));
        f.read((char *) &num_row, sizeof(num_col));
        f.read((char *) &num_col, sizeof(num_col));

        num_image = ConvertBigToSmallEndian(num_image);
        num_row = ConvertBigToSmallEndian(num_row);
        num_col = ConvertBigToSmallEndian(num_col);

        unsigned int image_size = num_row * num_col;
        unsigned char pixel_data[image_size];

        for(unsigned int i = 0; i < num_image; i++) {
            f.read((char *)&pixel_data, sizeof(pixel_data));
            std::array<unsigned int, 28 * 28> image{};
            for(unsigned int j = 0; j < image_size; j++) image[j] = (unsigned int)pixel_data[j];
            image_data.push_back(image);
        }
    }
}

void InputData::ReadLabelFile(const std::string& label_file) {
    std::ifstream f;
    f.open(label_file, std::ios::binary);

    if (f.is_open()) {
        unsigned int a;
        f.read((char *) &a, sizeof(a));
        f.read((char *) &a, sizeof(a));

        char label = 0;
        for(unsigned int i = 0; i < num_image; i++) {
            f.read(&label, sizeof(label));
            label_data.push_back((unsigned int)label);
            label_image_id[label_data.back()].push_back(i);
        }
    }
}

const std::array<unsigned int, 784> & InputData::GetImage(unsigned int i) const {
    return image_data[i];
}

unsigned int InputData::GetLabel(unsigned int i) const {
    return label_data[i];
}

const std::vector<unsigned int> & InputData::GetAllImagesIDByLabel(unsigned int label_id) const {
    return label_image_id[label_id];
}

int InputData::GetNumImagesByLabel(unsigned int label_id) const {
    return label_image_id[label_id].size();
}

unsigned int InputData::GetNumImages() const {
    return num_image;
}

void InputData::printImage(unsigned int x) const {
    for(int i = 0; i < 28; i++) {
        for(int j = 0; j < 28; j++) std::cout << std::setw(4) << image_data[x][(28 * i) + j];
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
