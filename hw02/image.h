//
// Created by thethongngu on 10/9/19.
//

#ifndef HW02_IMAGE_H
#define HW02_IMAGE_H


#include <array>

class Image {
public:
    typedef unsigned int PixelType;

public:
    void getPixel(unsigned int i, unsigned int j);
    void setPixel(unsigned int i, unsigned int j);
    std::istream& operator >>(Image& image);

private:
    std::array<PixelType, 28 * 28> image_data;

};


#endif //HW02_IMAGE_H
