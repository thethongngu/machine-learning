#include <iostream>
#include <fstream>
#include <vector>
#include "input_data.h"

#define debug(a) std::cout << #a << ": " << a << std::endl

const std::string kMnistTrainImagePath = "../data/train-images-idx3-ubyte";
const std::string kMnistTrainLabelPath = "../data/train-labels-idx1-ubyte";

int main() {
    InputData input_data(kMnistTrainImagePath, kMnistTrainLabelPath);


    return 0;
}

