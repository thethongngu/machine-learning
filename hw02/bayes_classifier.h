//
// Created by Thong Nguyen on 10/10/19.
//

#ifndef HW02_BAYES_CLASSIFIER_H
#define HW02_BAYES_CLASSIFIER_H


#include "input_data.h"

class NaiveBayesClassifier {
public:
    explicit NaiveBayesClassifier();

    void discrete_classify(const InputData& input_data, const InputData& test_data);
    double continuous_classify();

private:
    double label_bin[10][28 * 28][32] = {0};
    double total_bin[28 * 28][32] = {0};

    double prob_c[10] = {0};
    double prob_x_c[10][28 * 28][32] = {0};
    double prob_x[28 * 28][32] = {0};
};


#endif //HW02_BAYES_CLASSIFIER_H
