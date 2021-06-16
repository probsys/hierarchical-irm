// Copyright 2021 MIT Probabilistic Computing Project
// Apache License, Version 2.0, refer to LICENSE.txt

#include <cassert>
#include <random>

#include "hirm.hh"
#include "util_hash.hh"
#include "util_io.hh"
#include "util_math.hh"

int main(int argc, char **argv) {

    srand(1);
    PRNG prng (1);

    string path_base    = "assets/animals.unary";
    auto path_schema    = path_base + ".schema";
    auto path_obs       = path_base + ".obs";

    printf("== HIRM == \n");
    auto schema_unary = load_schema(path_schema);
    auto observations_unary = load_observations(path_obs);
    auto encoding_unary = encode_observations(schema_unary, observations_unary);

    HIRM hirm (schema_unary, &prng);
    incorporate_observations(hirm, encoding_unary, observations_unary);
    int n_obs_unary = 0;
    for (const auto &[z, irm] : hirm.irms) {
        for (const auto &[r, relation] : irm->relations) {
            n_obs_unary += relation->data.size();
        }
    }
    assert(n_obs_unary == observations_unary.size());

    hirm.transition_cluster_assignments_all();
    hirm.transition_cluster_assignments_all();
    hirm.set_cluster_assignment_gibbs("solitary", 120);
    hirm.set_cluster_assignment_gibbs("water", 741);
    for (int i = 0; i < 20; i++) {
        hirm.transition_cluster_assignments_all();
        for (const auto &[t, irm] : hirm.irms) {
            irm->transition_cluster_assignments_all();
            for (const auto &[d, domain] : irm->domains) {
                domain->crp.transition_alpha();
            }
        }
        hirm.crp.transition_alpha();
        printf("%d %f [", i, hirm.logp_score());
        for (const auto &[t, customers] : hirm.crp.tables) {
            printf("%ld ", customers.size());
        }
        printf("]\n");
    }

    // TODO: Removing the relation causes solitary to have no observations,
    // which causes the serialization test.  Instead, we need a
    // to_txt_dataset([relation | irm | hirm]) which writes the latest
    // dataset to disk and is used upon reloading the data.
    // hirm.remove_relation("solitary");
    // hirm.transition_cluster_assignments_all();
    // hirm.add_relation("solitary", {"animal"});
    // hirm.transition_cluster_assignments_all();

    string path_clusters = path_base + ".hirm";
    to_txt(path_clusters, hirm, encoding_unary);

    auto &enc = std::get<0>(encoding_unary);

    // Marginally normalized.
    int persiancat = enc["animal"]["persiancat"];
    auto p0_black_persiancat = hirm.logp({{"black", {persiancat}, 0.}});
    auto p1_black_persiancat = hirm.logp({{"black", {persiancat}, 1.}});
    assert(abs(logsumexp({p0_black_persiancat, p1_black_persiancat})) < 1e-10);

    // Marginally normalized.
    int sheep = enc["animal"]["sheep"];
    auto p0_solitary_sheep = hirm.logp({{"solitary", {sheep}, 0.}});
    auto p1_solitary_sheep = hirm.logp({{"solitary", {sheep}, 1.}});
    assert(abs(logsumexp({p0_solitary_sheep, p1_solitary_sheep})) < 1e-10);

    // Jointly normalized.
    auto p00_black_persiancat_solitary_sheep = hirm.logp(
        {{"black", {persiancat}, 0.}, {"solitary", {sheep}, 0.}});
    auto p01_black_persiancat_solitary_sheep = hirm.logp(
        {{"black", {persiancat}, 0.}, {"solitary", {sheep}, 1.}});
    auto p10_black_persiancat_solitary_sheep = hirm.logp(
        {{"black", {persiancat}, 1.}, {"solitary", {sheep}, 0.}});
    auto p11_black_persiancat_solitary_sheep = hirm.logp(
        {{"black", {persiancat}, 1.}, {"solitary", {sheep}, 1.}});
    auto Z = logsumexp({
        p00_black_persiancat_solitary_sheep,
        p01_black_persiancat_solitary_sheep,
        p10_black_persiancat_solitary_sheep,
        p11_black_persiancat_solitary_sheep,
    });
    assert(abs(Z) < 1e-10);

    // Independence
    assert(abs(p00_black_persiancat_solitary_sheep - (p0_black_persiancat + p0_solitary_sheep)) < 1e-8);
    assert(abs(p01_black_persiancat_solitary_sheep - (p0_black_persiancat + p1_solitary_sheep)) < 1e-8);
    assert(abs(p10_black_persiancat_solitary_sheep - (p1_black_persiancat + p0_solitary_sheep)) < 1e-8);
    assert(abs(p11_black_persiancat_solitary_sheep - (p1_black_persiancat + p1_solitary_sheep)) < 1e-8);

    // Load the clusters.
    HIRM hirx ({}, &prng);
    from_txt(&hirx, path_schema, path_obs, path_clusters);

    assert(hirm.irms.size() == hirx.irms.size());
    // Check IRMs agree.
    for (const auto &[table, irm] : hirm.irms) {
        auto irx = hirx.irms.at(table);
        // Check log scores agree.
        for (const auto &[d, dm] : irm->domains) {
            auto dx = irx->domains.at(d);
            dx->crp.alpha = dm->crp.alpha;
        }
        assert(abs(irx->logp_score() - irm->logp_score()) < 1e-8);
        // Check domains agree.
        for (const auto &[d, dm] : irm->domains) {
            auto dx                    =  irx->domains.at(d);
            assert(dm->items           == dx->items);
            assert(dm->crp.assignments == dx->crp.assignments);
            assert(dm->crp.tables      == dx->crp.tables);
            assert(dm->crp.N           == dx->crp.N);
            assert(dm->crp.alpha       == dx->crp.alpha);
        }
        // Check relations agree.
        for (const auto &[r, rm] : irm->relations) {
            auto rx = irx->relations.at(r);
            assert(rm->data     == rx->data);
            assert(rm->data_r   == rx->data_r);
            assert(rm->clusters.size() == rx->clusters.size());
            for (const auto &[z, clusterm] : rm->clusters) {
                auto clusterx = rx->clusters.at(z);
                assert(clusterm->N == clusterx->N);
            }
        }
    }
    hirx.crp.alpha = hirm.crp.alpha;
    assert(abs(hirx.logp_score() - hirm.logp_score()) < 1e-8);
}
