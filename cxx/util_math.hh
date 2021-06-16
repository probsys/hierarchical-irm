// Copyright 2021 MIT Probabilistic Computing Project
// Apache License, Version 2.0, refer to LICENSE.txt

#pragma once

#include "globals.hh"

double lbeta(int z, int w);

vector<double> linspace(double start, double stop, int num, bool endpoint);
vector<double> log_linspace(double start, double stop, int num, bool endpoint);
vector<double> log_normalize(const vector<double> &weights);
double logsumexp(const vector<double> &weights);

int choice(const vector<double> &weights, PRNG *prng);
int log_choice(const vector<double> &weights, PRNG *prng);

vector<vector<int>> product(const vector<vector<int>> &lists);
