//
// Created by Thong Nguyen on 10/10/19.
//

#include <cmath>
#include <iostream>
#include <iomanip>
#include "bayes_classifier.h"

#define debug(a) std::cout << #a << " = " << a << std::endl

void NaiveBayesClassifier::discrete_classify(const InputData &input_data, const InputData& test_data) {

    double label_bin[10][28 * 28][32];
    double prob_c[10];
    double prob_x_c[10][28 * 28][32];

    for(int label = 0; label < 10; label++) {
        for(int pixel = 0; pixel < 28 * 28; pixel++) {
            for(int bin = 0; bin < 32; bin++) {
                label_bin[label][pixel][bin] = 0;
                prob_x_c[label][pixel][bin] = 0;
            }
        }
    }

    for(int i = 0; i < 10; i++) prob_c[i] = 0;

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
        for (int k = 0; k < 10; k++) {
            std::cout << k << ": " << std::setprecision(17) << posteriors[k] / sum << std::endl;
        }
        num_wrong += (prediction != test_data.GetLabel(i)) ? 1 : 0;
        std::cout << "Prediction: " << prediction << ", Ans: " << test_data.GetLabel(i) << std::endl;
    }

    // Print imagination number
    for (int label = 0; label < 10; label++) {
        std::cout << label << ": " << std::endl;
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
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

void NaiveBayesClassifier::continuous_classify(const InputData& input_data, const InputData& test_data) {

    double mean[10][28 * 28];
    double pixel_squared[10][28 * 28];
    double variance[10][28 * 28];

    for(int label = 0; label < 10; label++) {
        for(int pixel = 0; pixel < 28 * 28; pixel++) {
            mean[label][pixel] = 0.0;
            variance[label][pixel] = 0.0;
        }
    }

    for (int label = 0; label < 10; label++) {
        const std::vector<unsigned int> &image_ids = input_data.GetAllImagesIDByLabel(label);
        for (unsigned int image_id : image_ids) {
            std::array<unsigned int, 784> image = input_data.GetImage(image_id);
            for (int pixel = 0; pixel < image.size(); pixel++) {
                mean[label][pixel] += image[pixel];
                pixel_squared[label][pixel] += image[pixel] * image[pixel];
            }
        }
    }

    for(int label = 0; label < 10; label++) {
        int num_class_images = input_data.GetNumImagesByLabel(label);
        for(int pixel = 0; pixel < 28 * 28; pixel++) {
            mean[label][pixel] /= num_class_images;
        }
    }

    for(int label = 0; label < 10; label++) {
        const std::vector<unsigned int> &image_ids = input_data.GetAllImagesIDByLabel(label);
        for (unsigned int image_id : image_ids) {
            std::array<unsigned int, 784> image = input_data.GetImage(image_id);
            for (int pixel = 0; pixel < image.size(); pixel++) {
                variance[label][pixel] += (image[pixel] - mean[label][pixel]) * (image[pixel] - mean[label][pixel]);
            }
        }
    }

    for(int label = 0; label < 10; label++) {
        int num_class_images = input_data.GetNumImagesByLabel(label);
        for(int pixel = 0; pixel < 28 * 28; pixel++) {
            variance[label][pixel] /= (num_class_images);
        }
    }

    double eps = 1;
    for(int label = 0; label < 10; label++) {
        for(int pixel = 0; pixel < 28 * 28; pixel++) {
            if (variance[label][pixel] == 0) {
                variance[label][pixel] = eps;
            }
        }
    }

    int num_wrong = 0;
    int num_test_images = test_data.GetNumImages();
    const double PI = 3.14159265358979323846;
    for (int i = 0; i < num_test_images; i++) {

        std::array<unsigned int, 784> test_image = test_data.GetImage(i);
        int prediction = 0;

        double posterior = 0.0, max_posterior = -1000000000;
        double sum = 0.0;
        double posteriors[10];
        for(int label = 0; label < 10; label++) {
            double prior = log((double)input_data.GetNumImagesByLabel(label) / input_data.GetNumImages());
            double likelihood = 0.0;

            for (int pixel = 0; pixel < 28 * 28; pixel++) {
                double a = log((1.0) / (std::sqrt(2 * PI * variance[label][pixel])));
                double b = ((test_image[pixel] - mean[label][pixel]) * (test_image[pixel] - mean[label][pixel])) / (2.0 * variance[label][pixel]);
                likelihood += (a - b);
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
        for (int k = 0; k < 10; k++) {
            std::cout << k << ": " << std::setprecision(17) << posteriors[k] / sum << std::endl;
        }
        num_wrong += (prediction != test_data.GetLabel(i)) ? 1 : 0;
        std::cout << "Prediction: " << prediction << ", Ans: " << test_data.GetLabel(i) << std::endl << std::endl;
    }

    // Print imagination number
    for (int label = 0; label < 10; label++) {
        std::cout << label << ": " << std::endl;
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                if (mean[label][(28 * i) + j] >= 128) {
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

NaiveBayesClassifier::NaiveBayesClassifier() = default;
