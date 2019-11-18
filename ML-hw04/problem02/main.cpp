#include <iostream>
#include "input_data.h"

const std::string kMnistTrainImagePath = "../data/train-images-idx3-ubyte";
const std::string kMnistTrainLabelPath = "../data/train-labels-idx1-ubyte";
const std::string kMnistTestImagePath = "../data/t10k-images-idx3-ubyte";
const std::string kMnistTestLabelPath = "../data/t10k-labels-idx1-ubyte";

int main() {
    InputData input_data(kMnistTrainImagePath, kMnistTrainLabelPath);
    InputData test_data(kMnistTestImagePath, kMnistTestLabelPath);
}