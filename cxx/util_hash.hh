// Copyright 2021 MIT Probabilistic Computing Project
// Apache License, Version 2.0, refer to LICENSE.txt

// Hash functions for std::<vector>
// https://stackoverflow.com/a/27216842

#pragma once

#include <string>
#include <vector>

using std::vector;
using std::string;

struct VectorIntHash {
    int operator()(const vector<int> &V) const {
        int hash = V.size();
        for(auto &i : V) {
            hash ^= i + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

struct VectorStringHash {
    int operator()(const vector<string> &V) const {
        int hash = V.size();
        for(auto &s : V) {
            hash ^= std::hash<string>{}(s) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};
