//
// Created by Thong Nguyen on 10/10/19.
//

#ifndef HW02_BAYES_CLASSIFIER_H
#define HW02_BAYES_CLASSIFIER_H


#include "input_data.h"

class NaiveBayesClassifier {
public:
    explicit NaiveBayesClassifier();

    static void discrete_classify(const InputData& input_data, const InputData& test_data);
    static void continuous_classify(const InputData& input_data, const InputData& test_data);
};


#endif //HW02_BAYES_CLASSIFIER_H
