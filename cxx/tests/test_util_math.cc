// Copyright 2021 MIT Probabilistic Computing Project
// Apache License, Version 2.0, refer to LICENSE.txt

#include <cassert>
#include <cstdio>
#include <vector>

#include "util_math.hh"

int main(int argc, char **argv) {
    vector<vector<int>> x {{1}, {2, 3}, {1, 10, 11}};

    auto cartesian = product(x);
    assert(cartesian.size() == 6);
    assert((cartesian.at(0) == vector<int>{1, 2, 1}));
    assert((cartesian.at(1) == vector<int>{1, 2, 10}));
    assert((cartesian.at(2) == vector<int>{1, 2, 11}));
    assert((cartesian.at(3) == vector<int>{1, 3, 1}));
    assert((cartesian.at(4) == vector<int>{1, 3, 10}));
    assert((cartesian.at(5) == vector<int>{1, 3, 11}));

    x.push_back({});
    cartesian = product(x);
    assert(cartesian.size() == 0);
}
