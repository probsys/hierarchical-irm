// Copyright 2021 MIT Probabilistic Computing Project
// Apache License, Version 2.0, refer to LICENSE.txt

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <map>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using std::map;
using std::string;
using std::tuple;
using std::vector;

#define uset std::unordered_set
#define umap std::unordered_map

// https://stackoverflow.com/q/2241327/
typedef std::mt19937 PRNG;

typedef map<string, vector<string>> T_schema;

extern const double INF;
