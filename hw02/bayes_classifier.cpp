//
// Created by Thong Nguyen on 10/10/19.
//

#include <cmath>
#include <iostream>
#include <iomanip>
#include "bayes_classifier.h"

#define debug(a) std::cout << #a << " = " << a << std::endl

void NaiveBayesClassifier::discrete_classify(const InputData &input_data, InputData test_data) {

    for (int label = 0; label < 10; label++) {
        const std::vector<unsigned int> &image_ids = input_data.GetAllImagesIDByLabel(label);

        // for calculating p(x|c)
        for (unsigned int image_id : image_ids) {
            std::array<unsigned int, 784> image = input_data.GetImage(image_id);

            for (int pixel = 0; pixel < image.size(); pixel++) {
                label_bin[label][pixel][image[pixel] / 8]++;
            }
        }
    }

    // Avoid empty bin (log-scale return -inf)
    auto min_bin = 1000000000;
    for (int label = 0; label < 10; label++) {
        for (int pixel = 0; pixel < 28 * 28; pixel++) {
            for (int bin = 0; bin < 32; bin++) {
                if (std::fabs(label_bin[label][pixel][bin]) > 2.22045e-016) {
                    min_bin = fmin(min_bin, label_bin[label][pixel][bin]);
                }
            }
        }
    }

    // for calculating p(x)
    for (int label = 0; label < 10; label++) {
        for (int pixel = 0; pixel < 28 * 28; pixel++) {
            for (int bin = 0; bin < 32; bin++) {
                label_bin[label][pixel][bin] = fmax(min_bin, label_bin[label][pixel][bin]);
            }
        }
    }

    unsigned int num_train_images = input_data.GetNumImages();
    for (int label = 0; label < 10; label++) {
        unsigned int num_class_images = input_data.GetNumImagesByLabel(label);

        prob_c[label] = (double) num_class_images / num_train_images;   // p(c)

        for (int pixel = 0; pixel < 28 * 28; pixel++) {
            for (int bin = 0; bin < 32; bin++) {
                prob_x_c[label][pixel][bin] = (double) label_bin[label][pixel][bin] / num_class_images;  // p(x|c)
            }
        }
    }

    int num_wrong = 0;
    int num_test_images = test_data.GetNumImages();
    for (int i = 0; i < num_test_images; i++) {
        std::array<unsigned int, 784> test_image = test_data.GetImage(i);
        int prediction = 0;

        double posterior = 0.0, max_posterior = -1000000000;
        double sum = 0.0;
        double posteriors[10];
        for (int label = 0; label < 10; label++) {

            double prior = log(prob_c[label]);
            double likelihood = 0.0;
            for (int pixel = 0; pixel < test_image.size(); pixel++) {
                likelihood += log(prob_x_c[label][pixel][test_image[pixel] / 8]);
            }
            posterior = (likelihood + prior);

            if (posterior > max_posterior) {
                max_posterior = posterior;
                prediction = label;
            }
            sum += posterior;
            posteriors[label] = posterior;
        }

        std::cout << "Posterior (in log scale):" << std::endl;
        for(int k = 0; k < 10; k++) {
            std::cout << k << ": " << std::setprecision(17) << posteriors[k] / sum << std::endl;
        }
        num_wrong += (prediction != test_data.GetLabel(i)) ? 1 : 0;
        std::cout << "Prediction: " << prediction << ", Ans: " << test_data.GetLabel(i) << std::endl;
    }

    for(int label = 0; label < 10; label++) {
        std::cout << label << ": " << std::endl;
        for(int i = 0; i < 28; i++) {
            for(int j = 0; j < 28; j++) {
                double white = 0.0;
                double black = 0.0;
                for (int bin = 0; bin < 16; bin++) white += prob_x_c[label][(28 * i) + j][bin];
                for (int bin = 16; bin < 32; bin++) black += prob_x_c[label][(28 * i) + j][bin];

                if (black >= white) {
                    std::cout << " 1";
                } else {
                    std::cout << " 0";
                }
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << "Error rate: " << std::setprecision(4) << (double) num_wrong / test_data.GetNumImages() << std::endl;
}

double NaiveBayesClassifier::continuous_classify() {
    return 0;
}

NaiveBayesClassifier::NaiveBayesClassifier() = default;
