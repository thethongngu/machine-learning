//
// Created by Thong Nguyen on 10/10/19.
//

#ifndef HW02_INPUT_DATA_H
#define HW02_INPUT_DATA_H


#include <string>
#include <vector>
#include <array>

class InputData {
public:
    InputData(const std::string& image_file, const std::string& label_file);

    static auto big_to_small_endian(unsigned int data);
    const std::array<unsigned int, 784> & get_image(unsigned int i) const;
    unsigned int get_num_images() const;
    unsigned int get_label(unsigned int i);
    const std::vector<unsigned int> & get_image_id_by_label(unsigned int label_id) const;
    int get_num_images_by_class(unsigned int label_id) const;

private:
    unsigned int num_image{};
    unsigned int num_row{};
    unsigned int num_col{};
    std::vector<std::array<unsigned int, 784> > image_data;
    std::vector<unsigned int> label_data;
    std::vector<unsigned int> label_image_id[10];

    void read_image_file(const std::string& image_file);
    void read_label_file(const std::string& label_file);
};


#endif //HW02_INPUT_DATA_H
