#include <iostream>
#include <numeric>
#include <random>
#include <chrono>
#include "input_data.h"

const std::string kMnistTrainImagePath = "../data/train-images-idx3-ubyte";
const std::string kMnistTrainLabelPath = "../data/train-labels-idx1-ubyte";
const int IMAGE_SIZE = 28 * 28;
const int NUM_IMAGE = 60000;

double lambda[10];
double mu[IMAGE_SIZE][10], last_mu[IMAGE_SIZE][10];
double w[NUM_IMAGE][10];
int num_step = 0;
double diff;

void init_params() {
    srand((unsigned) time(nullptr));
    for (double &x : lambda) x = 0.1;
    for (int pixel_id = 0; pixel_id < IMAGE_SIZE; pixel_id++) {
        for (int label = 0; label < 10; label++) {
            mu[pixel_id][label] = (double) rand() / RAND_MAX;
        }
    }
    for (int image_id = 0; image_id < NUM_IMAGE; image_id++) {
        for (int label = 0; label < 10; label++) {
            w[image_id][label] = 0.1;
        }
    }
}

bool is_converge() {
    num_step++;
    diff = 0.0;
    for (int image_id = 0; image_id < IMAGE_SIZE; image_id++) {
        for (int label = 0; label < 10; label++) {
            diff += std::abs(mu[image_id][label] - last_mu[image_id][label]);
        }
    }
    printf("\nNo. of Iteration: %d, Difference: %f", num_step, diff);
    std::cout << "\n-----------------------------------------------------------" << std::endl;

//    if (num_step < 10) return false; else true;

    return diff < 10;
}

void do_E(const InputData &data) {
    double p_label[10];

    for (double &label: p_label) {
        label = 0;
    }

    for (int image_id = 0; image_id < NUM_IMAGE; image_id++) {
        auto image = data.GetImage(image_id);

        for (int label = 0; label < 10; label++) {
            double p_image = 1.0;
            for (auto pixel_id = 0; pixel_id < IMAGE_SIZE; pixel_id++) {
                p_image *= (image[pixel_id] == 1) ? mu[pixel_id][label] : (1 - mu[pixel_id][label]);
            }
            p_label[label] = lambda[label] * p_image;
        }
        double marginal = std::accumulate(p_label, p_label + 10, 0.0);
        if (marginal == 0) marginal = 1;

        for (int label = 0; label < 10; label++) w[image_id][label] = p_label[label] / marginal;
    }
}

void do_M(const InputData &data) {
    double sum_w[10];

    for (double &label : sum_w) {
        label = 0;
    }

    // store last mu for calculating difference
    for (int pixel_id = 0; pixel_id < IMAGE_SIZE; pixel_id++) {
        for (int label = 0; label < 10; label++) {
            last_mu[pixel_id][label] = mu[pixel_id][label];
        }
    }

    // maximize parameters
    for (int label = 0; label < 10; label++) {
        for (int image_id = 0; image_id < NUM_IMAGE; image_id++) {
            sum_w[label] += w[image_id][label];
        }
    }
    for (int pixel_id = 0; pixel_id < IMAGE_SIZE; pixel_id++) {
        for (int label = 0; label < 10; label++) {
            double mu_pixel_label = 0.0;
            for (int image_id = 0; image_id < NUM_IMAGE; image_id++) {
                mu_pixel_label += data.GetImage(image_id)[pixel_id] * w[image_id][label];
            }

            mu[pixel_id][label] = mu_pixel_label / ((sum_w[label] == 0) ? 1 : sum_w[label]);
        }
    }
    for (int label = 0; label < 10; label++) {
        lambda[label] = sum_w[label] / NUM_IMAGE;
    }

    // print imagination number
    for (int label = 0; label < 10; label++) {
        printf("\nClass %d:", label);
        for (int pixel_id = 0; pixel_id < IMAGE_SIZE; pixel_id++) {
            if (pixel_id % 28 == 0) printf("\n");
            if (mu[pixel_id][label] >= 0.5) printf("1 "); else printf("0 ");
        }
    }
}

void print_data(const InputData &data) {
    double p_label[10];
    int matrix[10][2][2];

    for (auto &confusion : matrix) {
        confusion[0][0] = confusion[0][1] = confusion[1][0] = confusion[1][1] = 0;
    }
    for (double &label : p_label) {
        label = 0;
    }

    std::vector<int> image_of_class[10];
    int predict_result[60000];

    for (int image_id = 0; image_id < NUM_IMAGE; image_id++) {
        auto image = data.GetImage(image_id);
        double max_prob = 0.0;
        int max_label = 0;

        // calculate label probability for each class
        for (int label = 0; label < 10; label++) {
            double p_image_label = 1.0;
            for (int pixel_id = 0; pixel_id < IMAGE_SIZE; pixel_id++) {
                p_image_label *= (image[pixel_id] == 1) ? mu[pixel_id][label] : (1 - mu[pixel_id][label]);
            }
            p_label[label] = lambda[label] * p_image_label;
            if (p_label[label] > max_prob) {
                max_prob = p_label[label];
                max_label = label;
            }
        }

        image_of_class[max_label].push_back(image_id);
        predict_result[image_id] = max_label;
    }

    int real_label[10];
    for (int class_id = 0; class_id < 10; class_id++) {
        int count_label[10];
        for (int i = 0; i < 10; i++) count_label[i] = 0;
        for (int i = 0; i < image_of_class[class_id].size(); i++) {
            int image_id = image_of_class[class_id][i];
            count_label[data.GetLabel(image_id)]++;
        }

        int max_label = 0, max_value = 0;
        for (int label = 0; label < 10; label++) {
            if (count_label[label] > max_value) {
                max_value = count_label[label];
                max_label = label;
            }
        }
        real_label[class_id] = max_label;
    }

    // store value to confusion matrix
    int guess_true = 0;
    for (int image_id = 0; image_id < NUM_IMAGE; image_id++) {
        auto real_predict = real_label[predict_result[image_id]];
        auto ground_truth = data.GetLabel(image_id);

        if (real_predict == ground_truth) {
            guess_true++;
            matrix[ground_truth][0][0]++;
            for (int other_label = 0; other_label < 10; other_label++) {
                if (other_label != ground_truth) matrix[other_label][1][1]++;
            }
        } else {
            matrix[ground_truth][0][1]++;
            matrix[real_predict][1][0]++;
            for(int other_label = 0; other_label < 10; other_label++) {
                if (other_label != ground_truth && other_label != real_predict) matrix[other_label][1][1]++;
            }
        }
    }

    // print imagination number
    for (int label = 0; label < 10; label++) {
        printf("\nLabeled class %d:", real_label[label]);
        for (int pixel_id = 0; pixel_id < IMAGE_SIZE; pixel_id++) {
            if (pixel_id % 28 == 0) printf("\n");
            if (mu[pixel_id][label] >= 0.5) printf("1 "); else printf("0 ");
        }
    }

    for (int label = 0; label < 10; label++) {
        std::cout << "\n------------------------------------------------------------" << std::endl;
        printf("Confusion matrix %d:\n", label);
        printf("                         Predict number %d      |      Predict not number %d\n", label, label);
        printf("Is    number %d                 %d                             %d           \n", label,
               matrix[label][0][0], matrix[label][0][1]);
        printf("Isn't number %d                 %d                             %d           \n", label,
               matrix[label][1][0], matrix[label][1][1]);

        printf("\n");
        printf("Sensitivity (Successfully predict number %d): %f\n", label,
               (double) matrix[label][0][0] / (matrix[label][0][0] + matrix[label][1][0]));
        printf("Specificity (Successfully predict not number %d): %f\n", label,
               (double) matrix[label][1][1] / (matrix[label][0][1] + matrix[label][1][1]));
    }

    printf("Total iteration to converge: %d\n", num_step);
    printf("Total error rate: %f\n", (double) (NUM_IMAGE - guess_true) / NUM_IMAGE);
}

int main() {
    InputData input_data(kMnistTrainImagePath, kMnistTrainLabelPath);

    init_params();
    do {
        do_E(input_data);
        do_M(input_data);
    } while (!is_converge());

    print_data(input_data);
}