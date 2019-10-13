//
// Created by thethongngu on 10/13/19.
//

#include <fstream>
#include <cstring>
#include <iostream>
#include <cmath>
#include "beta_distribution.h"

void BetaDistribution::fit(const std::string& data_file) {
    std::ifstream f;
    f.open(data_file);

    std::string data;
    if (!f.is_open()) {
        std::cout << "Error reading file";
    }

    long long fac[15];
    fac[0] = 1;  fac[1] = 1;
    for(int i = 2; i < 15; i++) fac[i] = fac[i - 1] * i;

    while (!f.eof()) {
        f >> data;
        std::cout << data << std::endl;

        double count1 = 0;
        for(int i = 0; i < data.length(); i++) {
            if (data[i] == '1') count1++;
        }
        double N = data.length();
        double count0 = N - count1;
    }
}
