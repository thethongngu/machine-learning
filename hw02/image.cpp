//
// Created by thethongngu on 10/9/19.
//

#include "image.h"

Image::Image(const char *data, uint8_t l) {
    for(int i = 0; i < kImageSize; i++) {
        image_data[i] = data[i];
    }
    label = l;
}

Image::PixelType &Image::operator[](unsigned int i) {
    return image_data[i];
}

Image::LabelType Image::getLabel() {
    return label;
};

