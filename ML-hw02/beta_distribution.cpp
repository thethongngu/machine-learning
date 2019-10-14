//
// Created by thethongngu on 10/13/19.
//

#include <fstream>
#include <cstring>
#include <iostream>
#include <cmath>
#include <iomanip>
#include "beta_distribution.h"

void BetaDistribution::fit(const std::string& data_file) {
    std::ifstream f;
    f.open(data_file);

    std::string data;
    if (!f.is_open()) {
        std::cout << "Error reading file";
    }

    int case_id = 0;
    int a, b;
    std::cout << "Input a, b: ";
    std::cin >> a >> b;
    while (!f.eof()) {
        case_id++;
        f >> data;

        double count1 = 0;
        double N = data.length();
        for(int i = 0; i < N; i++) {
            if (data[i] == '1') count1++;
        }
        double count0 = N - count1;
        double likelihood = fac(N) / (fac(count1) * fac(count0)) * (pow(count1 / N, count1) * pow(count0 / N, count0));

        std::cout << "case " << case_id << ": " << data << std::endl;
        std::cout << "Likelihood: " << std::setprecision(17) << likelihood << std::endl;
        std::cout << "Beta prior:     " << "a = " << a << " b = " << b << std::endl;
        a += count1;  b += count0;
        std::cout << "Beta posterior: " << "a = " << a << " b = " << b << std::endl << std::endl;
    }
}

double BetaDistribution::pow(double x, double y) {
    double res = 1.0;
    for(int i = 0; i < y; i++) res *= x;
    return res;
}

double BetaDistribution::fac(double x) {
    if (x == 0 || x == 1) return 1LL;

    double res = 1.0;
    for(int i = 2; i <= x; i++) res *= i;
    return res;
}
