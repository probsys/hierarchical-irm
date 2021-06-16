// Copyright 2021 MIT Probabilistic Computing Project
// Apache License, Version 2.0, refer to LICENSE.txt

#include <algorithm>

#include "util_math.hh"

using std::vector;

const double INF = std::numeric_limits<double>::infinity();

// http://matlab.izmiran.ru/help/techdoc/ref/betaln.html
double lbeta(int z, int w) {
    return lgamma(z) + lgamma(w) - lgamma(z + w);
}

vector<double> linspace(double start, double stop, int num, bool endpoint) {
    double step = (stop - start) / (num - endpoint);
    vector<double> v;
    for (int i = 0; i < num; i++) {
        v.push_back(start + step * i);
    }
    return v;
}

vector<double> log_linspace(double start, double stop, int num, bool endpoint) {
    auto v = linspace(log(start), log(stop), num, endpoint);
    for (int i = 0; i < v.size(); i++) {
        v[i] = exp(v[i]);
    }
    return v;
}

vector<double> log_normalize(const std::vector<double> &weights){
    double Z = logsumexp(weights);
    vector<double> result(weights.size());
    for (int i = 0; i < weights.size(); i++) {
        result[i] = weights[i] - Z;
    }
    return result;
}

double logsumexp(const vector<double> &weights) {
    double m = *std::max_element(weights.begin(), weights.end());
    double s = 0;
    for (auto w : weights) {
        s += exp(w - m);
    }
    return log(s) + m;
}

int choice(const std::vector<double> &weights, PRNG *prng) {
    std::discrete_distribution<int> dist(weights.begin(), weights.end());
    int idx = dist(*prng);
    return idx;
}

int log_choice(const std::vector<double> &weights, PRNG *prng) {
    vector<double> log_weights_norm = log_normalize(weights);
    vector<double> weights_norm;
    for (double w : log_weights_norm) {
        weights_norm.push_back(exp(w));
    }
    return choice(weights_norm, prng);
}

vector<vector<int>> product(const vector<vector<int>> &lists) {
    // https://rosettacode.org/wiki/Cartesian_product_of_two_or_more_lists#C.2B.2B
    vector<vector<int>> result;
    for (const auto &l : lists) {
        if (l.size() == 0) {
            return result;
        }
    }
    for (const auto &e : lists[0]) {
        result.push_back({e});
    }
    for (size_t i = 1; i < lists.size(); ++i) {
        vector<vector<int>> temp;
        for (auto &e : result) {
            for (auto f : lists[i]) {
                auto e_tmp = e;
                e_tmp.push_back(f);
                temp.push_back(e_tmp);
          }
        }
        result = temp;
      }
      return result;
    }
