// Copyright 2021 MIT Probabilistic Computing Project
// Apache License, Version 2.0, refer to LICENSE.txt

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "globals.hh"
#include "hirm.hh"
#include "util_io.hh"
#include "util_math.hh"

int main(int argc, char **argv) {
    string path_base = "assets/two_relations";
    int seed = 1;
    int iters = 2;

    PRNG prng (seed);

    string path_schema = path_base + ".schema";
    std::cout << "loading schema from " << path_schema << std::endl;
    auto schema = load_schema(path_schema);
    for (auto const &[relation, domains] : schema) {
        printf("relation: %s, ", relation.c_str());
        printf("domains: ");
        for (auto const &domain : domains) {
            printf("%s ", domain.c_str());
        }
        printf("\n");
    }

    string path_obs = path_base + ".obs";
    std::cout << "loading observations from " << path_obs << std::endl;
    auto observations = load_observations(path_obs);
    T_encoding encoding = encode_observations(schema, observations);

    IRM irm (schema, &prng);
    incorporate_observations(irm, encoding, observations);
    printf("running for %d iterations\n", iters);
    for (int i = 0; i < iters; i++) {
        irm.transition_cluster_assignments_all();
        for (auto const &[d, domain] : irm.domains) {
            domain->crp.transition_alpha();
        }
        double x = irm.logp_score();
        printf("iter %d, score %f\n", i, x);
    }

    string path_clusters = path_base + ".irm";
    std::cout << "writing clusters to " << path_clusters << std::endl;
    to_txt(path_clusters, irm, encoding);

    map<int, map<int, double>> expected_p0 {
        {0,     { {0, 1},   {10, 1},    {100, .5} } },
        {10,    { {0, 0},   {10, 0},    {100, .5} } },
        {100,   { {0, .66}, {10, .66},  {100, .5} } },
    };

    vector<vector<int>> indexes {{0, 10, 100}, {0, 10, 100}};
    for (const auto &l : product(indexes)) {
        assert(l.size() == 2);
        auto x1 = l.at(0);
        auto x2 = l.at(1);
        auto p0 = irm.relations.at("R1")->logp({x1, x2}, 0);
        auto p0_irm = irm.logp({{"R1", {x1, x2}, 0}});
        assert(abs(p0 - p0_irm) < 1e-10);
        auto p1 = irm.relations.at("R1")->logp({x1, x2}, 1);
        auto Z = logsumexp({p0, p1});
        assert(abs(Z) < 1e-10);
        assert(abs(exp(p0) - expected_p0[x1].at(x2)) < .1);
    }

    for (const auto &l : vector<vector<int>> {{0, 10, 100}, {110, 10, 100}}) {
        auto x1 = l.at(0);
        auto x2 = l.at(1);
        auto x3 = l.at(2);
        auto p00 = irm.logp({{"R1", {x1, x2}, 0}, {"R1", {x1, x3}, 0}});
        auto p01 = irm.logp({{"R1", {x1, x2}, 0}, {"R1", {x1, x3}, 1}});
        auto p10 = irm.logp({{"R1", {x1, x2}, 1}, {"R1", {x1, x3}, 0}});
        auto p11 = irm.logp({{"R1", {x1, x2}, 1}, {"R1", {x1, x3}, 1}});
        auto Z = logsumexp({p00, p01, p10, p11});
        assert(abs(Z) < 1e-10);
    }

    IRM irx ({}, &prng);
    from_txt(&irx, path_schema, path_obs, path_clusters);
    // Check log scores agree.
    for (const auto &d : {"D1", "D2"}) {
        auto dm = irm.domains.at(d);
        auto dx = irx.domains.at(d);
        dx->crp.alpha = dm->crp.alpha;
    }
    assert(abs(irx.logp_score() - irm.logp_score()) < 1e-8);
    // Check domains agree.
    for (const auto &d : {"D1", "D2"}) {
        auto dm                    =  irm.domains.at(d);
        auto dx                    =  irx.domains.at(d);
        assert(dm->items           == dx->items);
        assert(dm->crp.assignments == dx->crp.assignments);
        assert(dm->crp.tables      == dx->crp.tables);
        assert(dm->crp.N           == dx->crp.N);
        assert(dm->crp.alpha       == dx->crp.alpha);
    }
    // Check relations agree.
    for (const auto &r : {"R1", "R2"}) {
        auto rm = irm.relations.at(r);
        auto rx = irx.relations.at(r);
        assert(rm->data     == rx->data);
        assert(rm->data_r   == rx->data_r);
        assert(rm->clusters.size() == rx->clusters.size());
        for (const auto &[z, clusterm] : rm->clusters) {
            auto clusterx = rx->clusters.at(z);
            assert(clusterm->N == clusterx->N);
        }
    }
}
