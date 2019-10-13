#include <iostream>
#include <fstream>
#include <vector>
#include "input_data.h"
#include "bayes_classifier.h"
#include "beta_distribution.h"

#define debug(a) std::cout << #a << ": " << a << std::endl

const std::string kMnistTrainImagePath = "../data/train-images-idx3-ubyte";
const std::string kMnistTrainLabelPath = "../data/train-labels-idx1-ubyte";
const std::string kMnistTestImagePath = "../data/t10k-images-idx3-ubyte";
const std::string kMnistTestLabelPath = "../data/t10k-labels-idx1-ubyte";
const std::string betaDataFile = "../data/testfile.txt";

int main() {
    InputData input_data(kMnistTrainImagePath, kMnistTrainLabelPath);
    InputData test_data(kMnistTestImagePath, kMnistTestLabelPath);

//    NaiveBayesClassifier::discrete_classify(input_data, test_data);
    BetaDistribution::fit(betaDataFile);
    return 0;
}

