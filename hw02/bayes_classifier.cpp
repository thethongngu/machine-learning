//
// Created by Thong Nguyen on 10/10/19.
//

#include <cmath>
#include <iostream>
#include <iomanip>
#include "bayes_classifier.h"

#define debug(a) std::cout << #a << " = " << a << std::endl

void NaiveBayesClassifier::discrete_classify(const InputData &input_data, const InputData &test_data) {

    for (int label = 0; label < 10; label++) {
        const std::vector<unsigned int> &image_ids = input_data.GetAllImagesIDByLabel(label);

        // for calculating p(x|c)
        for (unsigned int image_id : image_ids) {
            std::array<unsigned int, 784> image = input_data.GetImage(image_id);
//            for(int i = 0; i < 28; i++) {
//                for(int j = 0; j < 28; j++) std::cout << std::setw(4) << image[(28 * i) + j];
//                std::cout << std::endl;
//            }
//            std::cout << std::endl;

            for (int pixel = 0; pixel < image.size(); pixel++) {
                label_bin[label][pixel][image[pixel] / 8]++;
            }
        }
    }

    // Avoid empty bin (log-scale return -inf)
    auto min_bin = 1000000000;
    for(int label = 0; label < 10; label++) {
        for(int pixel = 0; pixel < 28 * 28; pixel++) {
            for(int bin = 0; bin < 32; bin++) {
                if (std::fabs(label_bin[label][pixel][bin] - 0.0) > 2.22045e-016) {
                    min_bin = fmin(min_bin, label_bin[label][pixel][bin]);
                }
            }
        }
    }

    // for calculating p(x)
    for(int label = 0; label < 10; label++) {
        for (int pixel = 0; pixel < 28 * 28; pixel++) {
            for (int bin = 0; bin < 32; bin++) {
                label_bin[label][pixel][bin] = fmax(min_bin, label_bin[label][pixel][bin]);
                total_bin[pixel][bin] += label_bin[label][pixel][bin];
            }
        }
    }

    // Calculate probability
    unsigned int num_images = input_data.GetNumImages();
    for (int label = 0; label < 10; label++) {

        unsigned int num_class_images = input_data.GetNumImagesByLabel(label);

        // p(c)
        prob_c[label] = (double) num_class_images / num_images;

        // p(x|c)
        for (int pixel = 0; pixel < 28 * 28; pixel++) {
            for (int bin = 0; bin < 32; bin++) {
                prob_x_c[label][pixel][bin] = (double) label_bin[label][pixel][bin] / num_class_images;
            }
        }

        // p(x)
        for (int pixel = 0; pixel < 28 * 28; pixel++) {
            for (int bin = 0; bin < 32; bin++) {
                prob_x[pixel][bin] = (double) total_bin[pixel][bin] / num_images;
            }
        }

    }

    int num_wrong = 0;
    for (int i = 0; i < test_data.GetNumImages(); i++) {
        std::cout << "Posterior (in log scale):" << std::endl;
        std::array<unsigned int, 784> test_image = test_data.GetImage(i);
        int prediction = 0;

        double marginal = 1.0;
        for (int pixel = 0; pixel < test_image.size(); pixel++) {
            marginal += log(prob_x[pixel][test_image[pixel] / 8]);
        }

        double posterior = 0.0, max_posterior = -1000000000;
        for (int label = 0; label < 10; label++) {
            double likelihood = 1.0;
            for (int pixel = 0; pixel < test_image.size(); pixel++) {
                likelihood += log(prob_x_c[label][pixel][test_image[pixel] / 8]);
            }

            double prior = log(prob_c[label]);
            posterior = (likelihood * prior) / marginal;
            if (posterior > max_posterior) {
                max_posterior = posterior;
                prediction = label;
            }

            std::cout << label << ": " << std::setprecision(17) << posterior << std::endl;
        }
        num_wrong += (prediction != test_data.GetLabel(i)) ? 1 : 0;
        std::cout << "Prediction: " << prediction << ", Ans: " << test_data.GetLabel(i) << std::endl;
    }

    std::cout << "Error rate: " << (double)num_wrong / test_data.GetNumImages() << std::endl;
}

double NaiveBayesClassifier::continuous_classify() {
    return 0;
}

NaiveBayesClassifier::NaiveBayesClassifier() = default;
