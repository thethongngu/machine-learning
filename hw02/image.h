//
// Created by thethongngu on 10/9/19.
//

#ifndef HW02_IMAGE_H
#define HW02_IMAGE_H


#include <array>

class Image {
public:
    typedef int8_t PixelType;
    const static unsigned int kImageSize = 28 * 28;

public:
    Image(const char *data);

    PixelType& operator[](unsigned int i);

private:
    std::array<PixelType, 28 * 28> image_data{};

};


#endif //HW02_IMAGE_H
