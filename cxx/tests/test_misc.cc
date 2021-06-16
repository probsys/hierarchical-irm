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
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "hirm.hh"
#include "util_hash.hh"
#include "util_io.hh"
#include "util_math.hh"

int main(int argc, char **argv) {

    srand(1);
    PRNG prng (1);

    BetaBernoulli bb (&prng);
    bb.incorporate(1);
    bb.incorporate(1);
    printf("%f\n", exp(bb.logp(1)));
    for (int i = 0; i < 100; i++) {
        printf("%1.f ", bb.sample());
    }
    printf("\n");

    CRP crp (&prng);
    crp.alpha = 1.5;
    printf("starting crp\n");
    T_item foo = 1, food = 2, sultan = 3, ali = 4;
    printf("%f\n", crp.logp_score());
    crp.incorporate(foo, 1);
    printf("%f\n", crp.logp_score());
    crp.incorporate(food, 1);
    printf("%f\n", crp.logp_score());
    crp.incorporate(sultan, 12);
    printf("%f\n", crp.logp_score());
    crp.incorporate(ali, 0);
    std::cout << "tables count 10 " << crp.tables.count(10) << std::endl;
    for (auto const &i : crp.tables[0]) {
        std::cout << i << " ";
    }
    for (auto const &i : crp.tables[1]) {
        std::cout << i << " ";
    }
    printf("\n");
    std::cout << "assignments ali? " << crp.assignments.count(ali) << std::endl;
    crp.unincorporate(ali);
    std::cout << "assignments ali? " << crp.assignments.count(ali) << std::endl;
    printf("%f %d\n", crp.logp_score(), crp.assignments[-1]);

    printf("=== tables_weights\n");
    auto tables_weights = crp.tables_weights();
    for (auto &tw : tables_weights) {
        printf("table %d weight %f\n", tw.first, tw.second);
    }

    printf("=== tables_weights_gibbs\n");
    auto tables_weights_gibbs = crp.tables_weights_gibbs(1);
    for (auto &tw : tables_weights_gibbs) {
        printf("table %d weight %f\n", tw.first, tw.second);
    }
    printf("==== tables_weights_gibbs_singleton\n");
    auto tables_weights_gibbs_singleton = crp.tables_weights_gibbs(12);
    for (auto &tw : tables_weights_gibbs_singleton) {
        printf("table %d weight %f\n", tw.first, tw.second);
    }
    printf("==== log probability\n");
    printf("%f\n", crp.logp(0));

    printf("=== DOMAIN === \n");
    Domain d ("foo", &prng);
    string relation1 = "ali";
    string relation2 = "mubarak";
    T_item salman = 1;
    T_item mansour = 2;
    d.incorporate(salman);
    for (auto &item : d.items) {
        printf("item %d: ", item);
    }
    d.set_cluster_assignment_gibbs(salman, 12);
    d.incorporate(salman);
    d.incorporate(mansour, 5);
    for (auto &item : d.items) {
        printf("item %d: ", item);
    }
    // d.unincorporate(salman);
    for (auto &item : d.items) {
        printf("item %d: ", item);
    }
    // d.unincorporate(relation2, salman);
    // assert (d.items.size() == 0);
    // d.items[01].insert("foo");

    umap<int, uset<int>> m;
    m[1].insert(10);
    m[1] = uset<int>();
    for (auto &ir: m) {
        printf("%d\n", ir.first);
        for (auto &x : ir.second) {
            printf("%d\n", x);
        }
    }

    printf("== RELATION == \n");
    Domain D1 ("D1", &prng);
    Domain D2 ("D2", &prng);
    Domain D3 ("D3", &prng);
    D1.incorporate(0);
    D2.incorporate(1);
    D3.incorporate(3);
    Relation R1 ("R1", {&D1, &D2, &D3}, &prng);
    printf("arity %ld\n", R1.domains.size());
    R1.incorporate({0, 1, 3}, 1);
    R1.incorporate({1, 1, 3}, 1);
    R1.incorporate({3, 1, 3}, 1);
    R1.incorporate({4, 1, 3}, 1);
    R1.incorporate({5, 1, 3}, 1);
    R1.incorporate({0, 1, 4}, 0);
    R1.incorporate({0, 1, 6}, 1);
    auto z1 = R1.get_cluster_assignment({0, 1, 3});
    for (int x : z1) {
        printf("%d,", x);
    }
    auto z2 = R1.get_cluster_assignment_gibbs({0, 1, 3}, D2, 1, 191);
    printf("\n");
    for (int x : z2) {
        printf("%d,", x);
    }
    printf("\n");

    double lpg = R1.logp_gibbs_approx(D1, 0, 1);
    printf("logp gibbs %f\n", lpg);
    lpg = R1.logp_gibbs_approx(D1, 0, 0);
    printf("logp gibbs %f\n", lpg);
    lpg = R1.logp_gibbs_approx(D1, 0, 10);
    printf("logp gibbs %f\n", lpg);

    printf("calling set_cluster_assignment_gibbs\n");
    R1.set_cluster_assignment_gibbs(D1, 0, 1);
    printf("new cluster %d\n", D1.get_cluster_assignment(0));
    D1.set_cluster_assignment_gibbs(0, 1);

    printf("%lu\n", R1.data.size());
    // R1.unincorporate({0, 1, 3});
    printf("%lu\n", R1.data.size());

    printf("== HASHING UTIL == \n");
    std::unordered_map<const vector<int>, int, VectorIntHash> map_int;
    map_int[{1, 2}] = 7;
    printf("%d\n", map_int.at({1,2}));
    std::unordered_map<const vector<string>, int, VectorStringHash> map_str;
    map_str[{"1", "2", "3"}] = 7;
    printf("%d\n", map_str.at({"1","2", "3"}));


    printf("===== IRM ====\n");
    map<string, vector<string>> schema1 {
        {"R1", {"D1", "D1"}},
        {"R2", {"D1", "D2"}},
        {"R3", {"D3", "D1"}},
    };
    IRM irm(schema1, &prng);

    for (auto const &kv : irm.domains) {
        printf("%s %s; ", kv.first.c_str(), kv.second->name.c_str());
        for (auto const r : irm.domain_to_relations.at(kv.first)) {
            printf("%s ", r.c_str());
        }
        printf("\n");
    }
    for (auto const &kv : irm.relations) {
        printf("%s ", kv.first.c_str());
        for (auto const d : kv.second->domains) {
            printf("%s ", d->name.c_str());
        }
        printf("\n");
    }

    printf("==== READING IO ===== \n");
    auto schema = load_schema("assets/animals.binary.schema");
    for (auto const &i : schema) {
        printf("relation: %s\n", i.first.c_str());
        printf("domains: ");
        for (auto const &j : i.second) {
            printf("%s ", j.c_str());
        }
        printf("\n");
    }

    IRM irm3 (schema, &prng);
    auto observations = load_observations("assets/animals.binary.obs");
    auto encoding = encode_observations(schema, observations);
    auto item_to_code = std::get<0>(encoding);
    for (auto const &i : observations) {
        auto relation = std::get<0>(i);
        auto value = std::get<2>(i);
        auto item = std::get<1>(i);
        printf("incorporating %s ", relation.c_str());
        printf("%1.f ", value);
        int counter = 0;
        T_items items_code;
        for (auto const &item : std::get<1>(i)) {
            auto domain = schema.at(relation)[counter];
            counter += 1;
            auto code = item_to_code.at(domain).at(item);
            printf("%s(%d) ", item.c_str(), code);
            items_code.push_back(code);
        }
        printf("\n");
        irm3.incorporate(relation, items_code, value);
    }

    for (int i = 0; i < 4; i++) {
        irm3.transition_cluster_assignments({"animal", "feature"});
        irm3.transition_cluster_assignments_all();
        for (auto const &[d, domain]: irm3.domains) {
            domain->crp.transition_alpha();
        }
        double x = irm3.logp_score();
        printf("iter %d, score %f\n", i, x);
    }

    string path_clusters = "assets/animals.binary.irm";
    to_txt(path_clusters, irm3, encoding);

    auto rel = irm3.relations.at("has");
    auto &enc = std::get<0>(encoding);
    auto lp0 = rel->logp({enc["animal"]["tail"], enc["animal"]["bat"]}, 0);
    auto lp1 = rel->logp({enc["animal"]["tail"], enc["animal"]["bat"]}, 1);
    auto lp_01 = logsumexp({lp0, lp1});
    assert(abs(lp_01) < 1e-5);
    printf("log prob of has(tail, bat)=0 is %1.2f\n", lp0);
    printf("log prob of has(tail, bat)=1 is %1.2f\n", lp1);
    printf("logsumexp is %1.2f\n", lp_01);

    IRM irm4 ({}, &prng);
    from_txt(&irm4,
        "assets/animals.binary.schema",
        "assets/animals.binary.obs",
        path_clusters);
    irm4.domains.at("animal")->crp.alpha = irm3.domains.at("animal")->crp.alpha;
    irm4.domains.at("feature")->crp.alpha = irm3.domains.at("feature")->crp.alpha;
    assert(abs(irm3.logp_score() - irm4.logp_score()) < 1e-8);
    for (const auto &d : {"animal", "feature"}) {
        auto d3 = irm3.domains.at(d);
        auto d4 = irm4.domains.at(d);
        assert(d3->items == d4->items);
        assert(d3->crp.assignments == d4->crp.assignments);
        assert(d3->crp.tables == d4->crp.tables);
        assert(d3->crp.N == d4->crp.N);
        assert(d3->crp.alpha == d4->crp.alpha);
    }
    for (const auto &r : {"has"}) {
        auto r3 = irm3.relations.at(r);
        auto r4 = irm4.relations.at(r);
        assert(r3->data == r4->data);
        assert(r3->data_r == r4->data_r);
        assert(r3->clusters.size() == r4->clusters.size());
        for (const auto &[z, cluster3] : r3->clusters) {
            auto cluster4 = r4->clusters.at(z);
            assert(cluster3->N == cluster4->N);
        }
    }
}
