#include <iostream>
#include <numeric>
#include "input_data.h"

const std::string kMnistTrainImagePath = "../data/train-images-idx3-ubyte";
const std::string kMnistTrainLabelPath = "../data/train-labels-idx1-ubyte";
const int IMAGE_SIZE = 28 * 28;

double lambda[10];
double mu[IMAGE_SIZE][10], last_mu[IMAGE_SIZE][10];
double w[IMAGE_SIZE][10];
int num_step = 0;
double diff;

void init_params() {
    for (double &x : lambda) x = 0.1;
    for (int image_id = 0; image_id < 60000; image_id++) {
        for (int label = 0; label < 10; label++) {
            mu[image_id][label] = 0.0;
            w[image_id][label] = 0.1;
        }
    }
}

bool is_converge() {
    num_step++;
    diff = 0.0;
    for (int image_id = 0; image_id < 60000; image_id++) {
        for (int label = 0; label < 10; label++) {
            diff += std::abs(mu[image_id][label] - last_mu[image_id][label]);
        }
    }
    printf("No. of Iteration: %d, Difference: %ff", num_step, diff);
    std::cout << "-----------------------------------------------------------" << std::endl;
    return diff < 0.0001;
}

void do_E(const InputData &data) {
    double p_label[10];

    for (int image_id = 0; image_id < data.GetNumImages(); image_id++) {
        auto image = data.GetImage(image_id);
        double p_image = 1.0;
        for (int label = 0; label < 10; label++) {
            for (auto pixel_id = 0; pixel_id < IMAGE_SIZE; pixel_id++) {
                p_image *= (image[pixel_id] == 0) ? mu[pixel_id][label] : (1 - mu[pixel_id][label]);
            }
            p_label[label] = lambda[label] * p_image;
        }
        double marginal = std::accumulate(p_label, p_label + 10, 0.0);
        if (marginal == 0) marginal = 1;

        for (int label = 0; label < 10; label++) w[image_id][label] = p_image / marginal;
    }
}

void do_M(const InputData &data) {
    double sum_w[10];

    // store last mu for calculating difference
    for (int image_id = 0; image_id < 60000; image_id++) {
        for (int label = 0; label < 10; label++) {
            last_mu[image_id][label] = mu[image_id][label];
        }
    }

    // maximize parameters
    for (int pixel_id = 0; pixel_id < IMAGE_SIZE; pixel_id++) {
        for (int label = 0; label < 10; label++) {
            double mu_pixel_label = 0.0;
            sum_w[label] = 0.0;
            for (int image_id = 0; image_id < data.GetNumImages(); image_id++) {
                mu_pixel_label += data.GetImage(image_id)[pixel_id] * w[image_id][label];
                sum_w[label] += w[image_id][label];
            }
            if (sum_w[label] == 0) sum_w[label] = 1;

            mu[pixel_id][label] = mu_pixel_label / sum_w[label];
        }

        for (int label = 0; label < 10; label++) {
            lambda[label] = sum_w[label] / data.GetNumImages();
        }
    }

    // print imagination number
    for (int label = 0; label < 10; label++) {
        printf("Class %d:", label);
        for (int pixel_id = 0; pixel_id < IMAGE_SIZE; pixel_id++) {
            if (pixel_id % 28 == 0) printf("\n");
            if (mu[pixel_id][label] >= 0.5) printf("1 "); else printf("0 ");
        }
    }
}

void print_data(const InputData &data) {
    double p_label[10];
    int matrix[10][2][2];

    for (int image_id = 0; image_id < 60000; image_id++) {
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

        // store value to confusion matrix
        auto ground_truth = data.GetLabel(image_id);
        if (max_label == ground_truth) {
            matrix[ground_truth][0][0]++;
            for (int other_label = 0; other_label < 10; other_label++) {
                if (other_label != ground_truth) matrix[other_label][1][1]++;
            }
        } else {
            matrix[ground_truth][0][1]++;
            matrix[max_label][1][0]++;
        }
    }

    for (int label = 0; label < 10; label++) {
        std::cout << "------------------------------------------------------------" << std::endl;
        printf("Confusion matrix %d:\n", label);
        printf("                         Predict number %d      |      Predict not number %d\n", label, label);
        printf("Is    number %d                 %d              |              %d           \n", label,
               matrix[label][0][0], matrix[label][0][1]);
        printf("Isn't number %d                 %d              |              %d           \n", label,
               matrix[label][1][0], matrix[label][1][1]);

        printf("Sensitivity (Successfully predict number %d): %ff\n", label,
               (double) matrix[label][0][0] / (matrix[label][0][0] + matrix[label][1][0]));
        printf("Specificity (Successfully predict number %d): %ff\n", label,
               (double) matrix[label][0][1] / (matrix[label][0][1] + matrix[label][1][1]));
    }

    int guess_true = 0;
    for (int label = 0; label < 10; label++) guess_true += matrix[label][0][0];

    printf("Total iteration to converge: %d", num_step);
    printf("Total error rate: %ff", (double) (60000 - guess_true) / 60000);
}

int main() {
    InputData input_data(kMnistTrainImagePath, kMnistTrainLabelPath);

    init_params();
    do {
        do_E(input_data);
        do_M(input_data);
    } while (is_converge());

    print_data(input_data);
}