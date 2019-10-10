//
// Created by Thong Nguyen on 10/10/19.
//

#include "bayes_classifier.h"

double NaiveBayesClassifier::discrete_classify(const InputData& input_data) {

    for(int i = 0; i < 10; i++) {
        const std::vector<unsigned int>& image_ids = input_data.get_image_id_by_label(i);
        prob_label[i] = (double)image_ids.size() / input_data.get_num_images();

        // Calculate p(x|c), which c = i
        for(unsigned int image_id : image_ids) {
            std::array<unsigned int, 784> image = input_data.get_image(image_id);
            for(size_t k = 0; k < image.size(); k++) {
                label_bin[i][k][image[k] / 8]++;
            }
        }

        // Update p(x)
        for(size_t j = 0; j < 28 * 28; j++) {
            for(size_t k = 0; k < 32; k++) {
                total_bin[j][k] += label_bin[i][j][k];
            }
        }
    }


}

double NaiveBayesClassifier::continuous_classify() {
    return 0;
}

NaiveBayesClassifier::NaiveBayesClassifier() = default;
