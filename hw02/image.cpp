//
// Created by thethongngu on 10/9/19.
//

#include "image.h"

Image::Image(const char *data) {
    for(int i = 0; i < kImageSize; i++) {
        image_data[i] = data[i];
    }
}

Image::PixelType &Image::operator[](unsigned int i) {

};

