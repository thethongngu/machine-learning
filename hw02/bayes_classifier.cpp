//
// Created by Thong Nguyen on 10/10/19.
//

#include <cmath>
#include <iostream>
#include <iomanip>
#include "bayes_classifier.h"

void NaiveBayesClassifier::discrete_classify(const InputData &input_data, const InputData &test_data) {

    for (int label = 0; label < 10; label++) {
        const std::vector<unsigned int> &image_ids = input_data.get_image_id_by_label(label);

        // for calculating p(x|c)
        for (unsigned int image_id : image_ids) {
            std::array<unsigned int, 784> image = input_data.get_image(image_id);
            for (size_t pixel_pos = 0; pixel_pos < image.size(); pixel_pos++) {
                label_bin[label][pixel_pos][image[pixel_pos] / 8]++;
            }
        }

        // for updating p(x)
        for (size_t pixel_pos = 0; pixel_pos < 28 * 28; pixel_pos++) {
            for (size_t bin_pos = 0; bin_pos < 32; bin_pos++) {
                total_bin[pixel_pos][bin_pos] += label_bin[label][pixel_pos][bin_pos];
            }
        }
    }

    // Calculate probability
    unsigned int num_images = input_data.get_num_images();
    for (int label = 0; label < 10; label++) {

        unsigned int num_class_images = input_data.get_num_images_by_class(label);

        // p(c)
        prob_c[label] = (double) num_class_images / num_images;

        // p(x|c)
        for (int pixel_pos = 0; pixel_pos < 28 * 28; pixel_pos++) {
            for (int bin_id = 0; bin_id < 32; bin_id++) {
                prob_x_c[label][pixel_pos][bin_id] = (double) label_bin[label][pixel_pos][bin_id] / num_class_images;
            }
        }

        // p(x)
        for (int pixel_pos = 0; pixel_pos < 28 * 28; pixel_pos++) {
            for (int bin_id = 0; bin_id < 32; bin_id++) {
                prob_x[pixel_pos][bin_id] = (double) total_bin[pixel_pos][bin_id] / num_images;
            }
        }

    }

    for (int i = 0; i < 1; i++) {
        std::array<unsigned int, 784> test_image = test_data.get_image(i);

        double marginal = 0.0;
        for (size_t pixel_pos = 0; pixel_pos < test_image.size(); pixel_pos++) {
            marginal += log10(prob_x[pixel_pos][test_image[pixel_pos] / 8]);
        }

        std::cout << "Posterior (in log scale):" << std::endl;

        double posterior = 0.0;
        for (int label = 0; label < 10; label++) {
            double likelihood = 0.0;
            for (size_t pixel_pos = 0; pixel_pos < test_image.size(); pixel_pos++) {
                likelihood += log10(prob_x_c[label][pixel_pos][test_image[pixel_pos] / 8]);
            }
            posterior = (likelihood + log10(prob_c[label])) - marginal;

            std::cout << label << ": " << std::setprecision(17) << posterior << std::endl;
        }

    }
}

double NaiveBayesClassifier::continuous_classify() {
    return 0;
}

NaiveBayesClassifier::NaiveBayesClassifier() = default;
