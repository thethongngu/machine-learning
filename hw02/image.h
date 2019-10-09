//
// Created by thethongngu on 10/9/19.
//

#ifndef HW02_IMAGE_H
#define HW02_IMAGE_H


#include <array>

class Image {
public:
    typedef uint8_t PixelType;
    typedef uint8_t LabelType;
    const static unsigned int kImageSize = 28 * 28;

public:
    Image(const char *data, uint8_t l);

    PixelType& operator[](unsigned int i);
    LabelType getLabel();

private:
    std::array<PixelType, 28 * 28> image_data{};
    LabelType label{};
};


#endif //HW02_IMAGE_H
