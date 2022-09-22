//
// Created by Thong Nguyen on 10/10/19.
//

#ifndef HW02_INPUT_DATA_H
#define HW02_INPUT_DATA_H


#include <string>
#include <vector>
#include <array>
#include <iomanip>

class InputData {
public:
    InputData(const std::string& image_file, const std::string& label_file);

    static auto ConvertBigToSmallEndian(unsigned int data);
    const std::array<unsigned int, 784> & GetImage(unsigned int i) const;
    unsigned int GetNumImages() const;
    unsigned int GetLabel(unsigned int i) const;
    const std::vector<unsigned int> & GetAllImagesIDByLabel(unsigned int label_id) const;
    int GetNumImagesByLabel(unsigned int label_id) const;
    void printImage(unsigned int i) const;

private:
    unsigned int num_image{};
    unsigned int num_row{};
    unsigned int num_col{};
    std::vector<std::array<unsigned int, 784> > image_data;
    std::vector<unsigned int> label_data;
    std::vector<unsigned int> label_image_id[10];

    void ReadImageFile(const std::string& image_file);
    void ReadLabelFile(const std::string& label_file);
};


#endif //HW02_INPUT_DATA_H
