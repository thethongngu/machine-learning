//
// Created by Thong Nguyen on 10/10/19.
//

#ifndef HW02_BAYES_CLASSIFIER_H
#define HW02_BAYES_CLASSIFIER_H


#include "input_data.h"

class NaiveBayesClassifier {
public:
    explicit NaiveBayesClassifier();

    double discrete_classify(const InputData& input_data);
    double continuous_classify();

private:
    double prob_label[10]{};
    double label_bin[10][28 * 28][32]{};
    double total_bin[28 * 28][32]{};

    double discrete_classify(const InputData &input_data);
};


#endif //HW02_BAYES_CLASSIFIER_H
