// Copyright 2020
// See LICENSE.txt

#pragma once
#include "globals.hh"
#include "util_hash.hh"
#include "util_math.hh"

typedef int T_item;
typedef vector<T_item> T_items;
typedef VectorIntHash H_items;

class Distribution {
public:
    int N = 0;
    virtual void incorporate(double x) = 0;
    virtual void unincorporate(double x) = 0;
    virtual double logp(double x) const = 0;
    virtual double logp_score() const = 0;
    virtual double sample() = 0;
    ~Distribution(){};
};

class BetaBernoulli : public Distribution {
public:
    double  alpha = 1;       // hyperparameter
    double  beta = 1;        // hyperparameter
    int     s = 0;           // sum of observed values
    PRNG    *prng;

    BetaBernoulli(PRNG *prng) {
        this->prng = prng;
    }
    void incorporate(double x){
        assert(x == 0 || x == 1);
        N += 1;
        s += x;
    }
    void unincorporate(double x) {
        assert(x == 0 || x ==1);
        N -= 1;
        s -= x;
        assert(0 <= s);
        assert(0 <= N);
    }
    double logp(double x) const {
        double log_denom = log(N + alpha + beta);
        if (x == 1) { return log(s + alpha) - log_denom; }
        if (x == 0) { return log(N - s + beta) - log_denom; }
        assert(false);
    }
    double logp_score() const {
        double v1 = lbeta(s + alpha, N - s + beta);
        double v2 = lbeta(alpha, beta);
        return v1 - v2;
    }
    double sample() {
        double p = exp(logp(1));
        vector<int> items {0, 1};
        vector<double> weights {1-p, p};
        auto idx = choice(weights, prng);
        return items[idx];
    }

    // Disable copying.
    BetaBernoulli & operator=(const BetaBernoulli&) = delete;
    BetaBernoulli(const BetaBernoulli&) = delete;
};

class CRP {
public:
    double                  alpha = 1;   // concentration parameter
    int                     N = 0;       // number of customers
    umap<int, uset<T_item>> tables;      // map from table id to set of customers
    umap<T_item, int>       assignments; // map from customer to table id
    PRNG                    *prng;

    CRP(PRNG *prng) {
        this->prng = prng;
    }

    void incorporate(const T_item &item, int table) {
        assert(assignments.count(item) == 0);
        if (tables.count(table) == 0) {
            tables[table] = uset<T_item>();
        }
        tables.at(table).insert(item);
        assignments[item] = table;
        N += 1;
    }
    void unincorporate(const T_item &item) {
        assert(assignments.count(item) == 1);
        auto table = assignments.at(item);
        tables.at(table).erase(item);
        if (tables.at(table).size() == 0) {
            tables.erase(table);
        }
        assignments.erase(item);
        N -= 1;
    }
    int sample() {
        auto crp_dist = tables_weights();
        vector<int> items(crp_dist.size());
        vector<double> weights(crp_dist.size());
        int i = 0;
        for(auto &[table, weight] : crp_dist) {
            items[i] = table;
            weights[i] = weight;
            i += 1;
        }
        auto idx = choice(weights, prng);
        return items[idx];
    }
    double logp(int table) const {
        auto dist = tables_weights();
        if (dist.count(table) == 0) {
            return -INF;
        }
        double numer = dist[table];
        double denom = N + alpha;
        return log(numer) - log(denom);
    }
    double logp_score() const {
        double term1 = tables.size() * log(alpha);
        double term2 = 0;
        for (const auto &[table, customers] : tables) {
            term2 += lgamma(customers.size());
        }
        double term3 = lgamma(alpha);
        double term4 = lgamma(N + alpha);
        return term1 + term2 + term3 - term4;
    }
    umap<int, double> tables_weights() const {
        umap<int, double> dist;
        if (N == 0) {
            dist[0] = 1;
            return dist;
        }
        int t_max = 0;
        for (const auto &[table, customers] : tables) {
            dist[table] = customers.size();
            t_max = std::max(table, t_max);
        }
        dist[t_max + 1] = alpha;
        return dist;
    }
    umap<int, double> tables_weights_gibbs(int table) const {
        assert(N > 0);
        assert(tables.count(table) > 0);
        auto dist = tables_weights();
        dist.at(table) -= 1;
        if (dist.at(table) == 0) {
            dist.at(table) = alpha;
            int t_max = 0;
            for (const auto &[table, weight] : dist) {
                t_max = std::max(table, t_max);
            }
            dist.erase(t_max);
        }
        return dist;
    }
    void transition_alpha() {
        if (N == 0) { return; }
        auto grid = log_linspace(1. / N, N + 1, 20, true);
        vector<double> logps;
        for (const auto &g : grid) {
            this->alpha = g;
            auto logp_g = logp_score();
            logps.push_back(logp_g);
        }
        auto idx = log_choice(logps, prng);
        this->alpha = grid[idx];
    }

    // Disable copying.
    CRP & operator=(const CRP&) = delete;
    CRP(const CRP&) = delete;
};


class Domain {
public:
    const string                name;   // human-readable name
    uset<T_item>                items;  // set of items
    CRP                         crp;    // clustering model for items
    PRNG                        *prng;

    Domain(const string &name, PRNG *prng)
            :   name(name),
                crp(prng) {
        assert(name.size() > 0);
        this->prng = prng;
    }
    void incorporate(const T_item &item, int table=-1) {
        if (items.count(item) > 0) {
            assert(table == -1);
        }
        if (items.count(item) == 0) {
            items.insert(item);
            auto t = 0 <= table ? table : crp.sample();
            crp.incorporate(item, t);
        }
    }
    void unincorporate(const T_item &item) {
        printf("Not implemented\n");
        exit(EXIT_FAILURE);
        // assert(items.count(item) == 1);
        // assert(items.at(item).count(relation) == 1);
        // items.at(item).erase(relation);
        // if (items.at(item).size() == 0) {
        //     crp.unincorporate(item);
        //     items.erase(item);
        // }
    }
    int get_cluster_assignment(const T_item &item) const {
        assert(items.count(item) == 1);
        return crp.assignments.at(item);
    }
    void set_cluster_assignment_gibbs(const T_item &item, int table) {
        assert(items.count(item) == 1);
        assert(crp.assignments.at(item) != table);
        crp.unincorporate(item);
        crp.incorporate(item, table);
    }
    umap<int, double> tables_weights() const {
        return crp.tables_weights();
    }
    umap<int, double> tables_weights_gibbs(const T_item &item) const {
        auto table = get_cluster_assignment(item);
        return crp.tables_weights_gibbs(table);
    }

    // Disable copying.
    Domain & operator=(const Domain&) = delete;
    Domain(const Domain&) = delete;
};


class Relation {
public:
    // human-readable name
    const string                                            name;
    // list of domain pointers
    const vector<Domain*>                                   domains;
    // map from cluster multi-index to Distribution pointer
    umap<const vector<int>, Distribution*, VectorIntHash>   clusters;
    // map from item to observed data
    umap<const T_items, double, H_items>                    data;
    // map from domain name to reverse map from item to
    // set of items that include that item
    umap<string, umap<T_item, uset<T_items, H_items>>>      data_r;
    PRNG                                                    *prng;

    Relation(const string &name, const vector<Domain*> &domains, PRNG *prng)
            :   name(name),
                domains(domains) {
        assert(domains.size() > 0);
        assert(name.size() > 0);
        this->prng = prng;
        for (const auto &d : domains) {
            this->data_r[d->name] = umap<T_item, uset<T_items, H_items>>();
        }
    }

    ~Relation() {
        for (auto [z, cluster] : clusters) {
            delete cluster;
        }
    }

    void incorporate(const T_items &items, double value) {
        assert(data.count(items) == 0);
        data[items] = value;
        for (int i = 0; i < domains.size(); i++) {
            domains[i]->incorporate(items[i]);
            if (data_r.at(domains[i]->name).count(items[i]) == 0){
                data_r.at(domains[i]->name)[items[i]] = uset<T_items, H_items>();
            }
            data_r.at(domains[i]->name).at(items[i]).insert(items);
        }
        auto z = get_cluster_assignment(items);
        if (clusters.count(z) == 0) {
            // Invalid discussion as using pointers now;
            //      Cannot use clusters[z] because BetaBernoulli
            //      does not have a default constructor, whereas operator[]
            //      calls default constructor when the key does not exist.
            clusters[z] = new BetaBernoulli(prng);
        }
        clusters.at(z)->incorporate(value);
    }

    void unincorporate(const T_items &items) {
        printf("Not implemented\n");
        exit(EXIT_FAILURE);
        // auto x = data.at(items);
        // auto z = get_cluster_assignment(items);
        // clusters.at(z)->unincorporate(x);
        // if (clusters.at(z)->N == 0) {
        //     delete clusters.at(z);
        //     clusters.erase(z);
        // }
        // for (int i = 0; i < domains.size(); i++) {
        //     const string &n = domains[i]->name;
        //     if (data_r.at(n).count(items[i]) > 0) {
        //         data_r.at(n).at(items[i]).erase(items);
        //         if (data_r.at(n).at(items[i]).size() == 0) {
        //             data_r.at(n).erase(items[i]);
        //             domains[i]->unincorporate(name, items[i]);
        //         }
        //     }
        // }
        // data.erase(items);
    }

    vector<int> get_cluster_assignment(const T_items &items) const {
        assert(items.size() == domains.size());
        vector<int> z(domains.size());
        for (int i = 0; i < domains.size(); i++) {
            z[i] = domains[i]->get_cluster_assignment(items[i]);
        }
        return z;
    }

    vector<int> get_cluster_assignment_gibbs(const T_items &items,
            const Domain &domain, const T_item &item, int table) const {
        assert(items.size() == domains.size());
        vector<int> z(domains.size());
        int hits = 0;
        for (int i = 0; i < domains.size(); i++) {
            if ((domains[i]->name == domain.name) && (items[i] == item)) {
                z[i] = table;
                hits++;
            } else {
                z[i] = domains[i]->get_cluster_assignment(items[i]);
            }
        }
        assert(hits > 0);
        return z;
    }

    // Implementation of approximate Gibbs data probabilities (faster).

    double logp_gibbs_approx_current(const Domain &domain, const T_item &item) {
        double logp = 0.;
        for (const auto &items : data_r.at(domain.name).at(item)) {
            auto x = data.at(items);
            auto z = get_cluster_assignment(items);
            auto cluster = clusters.at(z);
            cluster->unincorporate(x);
            double lp = cluster->logp(x);
            cluster->incorporate(x);
            logp += lp;
        }
        return logp;
    }

    double logp_gibbs_approx_variant(const Domain &domain, const T_item &item, int table) {
        double logp = 0.;
        for (const auto &items : data_r.at(domain.name).at(item)) {
            auto x = data.at(items);
            auto z = get_cluster_assignment_gibbs(items, domain, item, table);
            double lp;
            if (clusters.count(z) == 0){
                BetaBernoulli cluster (prng);
                lp = cluster.logp(x);
            } else {
                lp = clusters.at(z)->logp(x);
            }
            logp += lp;
        }
        return logp;
    }

    double logp_gibbs_approx(const Domain &domain, const T_item &item, int table) {
        auto table_current = domain.get_cluster_assignment(item);
        double logp;
        if (table_current == table) {
            logp = logp_gibbs_approx_current(domain, item);
        } else {
            logp = logp_gibbs_approx_variant(domain, item, table);
        }
        return logp;
    }


    // Implementation of exact Gibbs data probabilities.

    umap<vector<int> const, vector<T_items>, VectorIntHash>
    get_cluster_to_items_list(Domain const &domain, const T_item &item) {
        umap<const vector<int>, vector<T_items>, VectorIntHash> m;
        for (const auto &items : data_r.at(domain.name).at(item)) {
            auto x = data.at(items);
            auto z = get_cluster_assignment(items);
            m[z].push_back(items);
        }
        return m;
    }

    double logp_gibbs_exact_current(const vector<T_items> &items_list) {
        assert(items_list.size() > 0);
        auto z = get_cluster_assignment(items_list[0]);
        auto cluster = clusters.at(z);
        auto logp0 = cluster->logp_score();
        for (const auto &items : items_list) {
            auto x = data.at(items);
            // assert(z == get_cluster_assignment(items));
            cluster->unincorporate(x);
        }
        auto logp1 = cluster->logp_score();
        for (const auto &items : items_list) {
            auto x = data.at(items);
            cluster->incorporate(x);
        }
        assert(cluster->logp_score() == logp0);
        return logp0 - logp1;
    }

    double logp_gibbs_exact_variant(const Domain &domain, const T_item &item,
            int table, const vector<T_items> &items_list) {
        assert(items_list.size() > 0);
        auto z = get_cluster_assignment_gibbs(items_list[0], domain, item, table);

        BetaBernoulli aux (prng);
        Distribution * cluster = clusters.count(z) > 0 ? clusters.at(z) : &aux;
        // auto cluster = self.clusters.get(z, self.aux())
        auto logp0 = cluster->logp_score();
        for (const auto &items : items_list) {
            // assert(z == get_cluster_assignment_gibbs(items, domain, item, table));
            auto x = data.at(items);
            cluster->incorporate(x);
        }
        auto logp1 = cluster->logp_score();
        for (const auto &items : items_list) {
            auto x = data.at(items);
            cluster->unincorporate(x);
        }
        assert(cluster->logp_score() == logp0);
        return logp1 - logp0;
    }

    vector<double> logp_gibbs_exact(const Domain &domain,
            const T_item &item, vector<int> tables) {
        auto cluster_to_items_list = get_cluster_to_items_list(domain, item);
        auto table_current = domain.get_cluster_assignment(item);
        vector<double> logps;
        for (const auto &table : tables) {
            double lp_table = 0;
            for (const auto &[z, items_list] : cluster_to_items_list) {
                double lp_cluster;
                if (table == table_current) {
                    lp_cluster = logp_gibbs_exact_current(items_list);
                } else {
                    lp_cluster = logp_gibbs_exact_variant(
                        domain, item, table, items_list);
                }
                lp_table += lp_cluster;
            }
            logps.push_back(lp_table);
        }
        return logps;
    }

    double logp(const T_items &items, double value) {
        // TODO: Falsely assumes cluster assignments of items
        // from same domain are identical, see note in hirm.py
        assert(items.size() == domains.size());
        vector<vector<T_item>> tabl_list;
        vector<vector<double>> wght_list;
        vector<vector<int>> indx_list;
        for (int i = 0; i < domains.size(); i++) {
            auto domain = domains.at(i);
            auto item = items.at(i);
            vector<T_item> t_list;
            vector<double> w_list;
            vector<int> i_list;
            if (domain->items.count(item) > 0) {
                auto z = domain->get_cluster_assignment(item);
                t_list = {z};
                w_list = {0};
                i_list = {0};
            } else {
                auto tables_weights = domain->tables_weights();
                auto Z = log(1 + domain->crp.N);
                int idx = 0;
                for (const auto &[t, w] : tables_weights) {
                    t_list.push_back(t);
                    w_list.push_back(log(w) - Z);
                    i_list.push_back(idx++);
                }
                assert(idx == t_list.size());
            }
            tabl_list.push_back(t_list);
            wght_list.push_back(w_list);
            indx_list.push_back(i_list);
        }
        vector<double> logps;
        for (const auto &indexes : product(indx_list)) {
            assert(indexes.size() == domains.size());
            vector<int> z;
            double logp_w = 0;
            for (auto i = 0; i < domains.size(); i++) {
                auto zi = tabl_list.at(i).at(indexes[i]);
                auto wi = wght_list.at(i).at(indexes[i]);
                z.push_back(zi);
                logp_w += wi;
            }
            BetaBernoulli aux (prng);
            Distribution * cluster = clusters.count(z) > 0 ? clusters.at(z) : &aux;
            auto logp_z = cluster->logp(value);
            auto logp_zw = logp_z + logp_w;
            logps.push_back(logp_zw);
        }
        return logsumexp(logps);
    }

    double logp_score() const {
        double logp = 0.;
        for (const auto &[z, cluster] : clusters) {
            logp += cluster->logp_score();
        }
        return logp;
    }

    void set_cluster_assignment_gibbs(const Domain &domain,
            const T_item &item, int table) {
        auto table_current = domain.get_cluster_assignment(item);
        assert(table != table_current);
        for (const auto &items : data_r.at(domain.name).at(item)) {
            auto x = data.at(items);
            // Remove from current cluster.
            auto z_prev = get_cluster_assignment(items);
            auto cluster_prev = clusters.at(z_prev);
            cluster_prev->unincorporate(x);
            if (cluster_prev->N == 0) {
                delete clusters.at(z_prev);
                clusters.erase(z_prev);
            }
            // Move to desired cluster.
            auto z_new = get_cluster_assignment_gibbs(items, domain, item, table);
            if (clusters.count(z_new) == 0) {
                // Move to fresh cluster.
                clusters[z_new] = new BetaBernoulli(prng);
                clusters.at(z_new)->incorporate(x);
            } else {
                // Move to existing cluster.
                assert((clusters.at(z_new)->N > 0));
                clusters.at(z_new)->incorporate(x);
            }
        }
        // Caller should invoke domain.set_cluster_gibbs
    }

    bool has_observation(const Domain &domain, const T_item &item) {
        return data_r.at(domain.name).count(item) > 0;
    }

    // Disable copying.
    Relation & operator=(const Relation&) = delete;
    Relation(const Relation&) = delete;
};

class IRM {
public:
    T_schema                    schema;              // schema of relations
    umap<string, Domain*>       domains;             // map from name to Domain
    umap<string, Relation*>     relations;           // map from name to Relation
    umap<string, uset<string>>  domain_to_relations; // reverse map
    PRNG                        *prng;

    IRM(const T_schema &schema, PRNG *prng) {
        this->prng = prng;
        for (const auto &[r, ds] : schema) {
            this->add_relation(r, ds);
        }
    }

    ~IRM() {
        for (auto [d, domain] : domains)     { delete domain; }
        for (auto [r, relation] : relations) { delete relation; }
    }

    void incorporate(const string &r, const T_items &items, double value) {
        relations.at(r)->incorporate(items, value);
    }

    void unincorporate(const string &r, const T_items &items) {
        relations.at(r)->unincorporate(items);
    }

    void transition_cluster_assignments_all() {
        for (const auto &[d, domain] : domains) {
            for (auto item : domain->items) {
                transition_cluster_assignment_item(d, item);
            }
        }
    }

    void transition_cluster_assignments(vector<string> ds) {
        for (const auto &d : ds) {
            for (auto item : domains.at(d)->items) {
                transition_cluster_assignment_item(d, item);
            }
        }
    }

    void transition_cluster_assignment_item(const string &d, const T_item &item) {
        Domain *domain = domains.at(d);
        auto crp_dist = domain->tables_weights_gibbs(item);
        // Compute probability of each table.
        vector<int> tables;
        vector<double> logps;
        for (const auto &[table, n_customers] : crp_dist) {
            tables.push_back(table);
            logps.push_back(log(n_customers));
        }
        for (const auto &r : domain_to_relations.at(d)) {
            auto relation = relations.at(r);
            if (relation->has_observation(*domain, item)) {
                auto lp_relation = relation->logp_gibbs_exact(*domain, item, tables);
                assert(lp_relation.size() == tables.size());
                assert(lp_relation.size() == logps.size());
                for (int i = 0; i < logps.size(); i++) {
                    logps[i] += lp_relation[i];
                }
            }
        }
        // Sample new table.
        assert(tables.size() == logps.size());
        auto idx = log_choice(logps, prng);
        auto choice = tables[idx];
        // Move to new table (if necessary).
        if (choice != domain->get_cluster_assignment(item)) {
            for (const auto &r : domain_to_relations.at(d)) {
                auto relation = relations.at(r);
                if (relation->has_observation(*domain, item)) {
                    relation->set_cluster_assignment_gibbs(*domain, item, choice);
                }
            }
            domain->set_cluster_assignment_gibbs(item, choice);
        }
    }

    double logp(const vector<tuple<string, T_items, double>> &observations) {
        umap<string, uset<T_items, H_items>>    relation_items_seen;
        umap<string, uset<T_item>>              domain_item_seen;
        vector<tuple<string, T_item>>           item_universe;
        vector<vector<int>>                     index_universe;
        vector<vector<double>>                  weight_universe;
        umap<string, umap<T_item, tuple<int, vector<int>>>> cluster_universe;
        // Compute all cluster combinations.
        for (const auto &[r, items, value] : observations) {
            // Assert observation is unique.
            assert(relation_items_seen[r].count(items) == 0);
            relation_items_seen[r].insert(items);
            // Process each (domain, item) in the observations.
            auto relation = relations.at(r);
            auto arity = relation->domains.size();
            assert(items.size() == arity);
            for (int i = 0; i < arity; i++) {
                // Skip if (domain, item) processed.
                auto domain = relation->domains.at(i);
                auto item = items.at(i);
                if (domain_item_seen[domain->name].count(item) > 0) {
                    assert(cluster_universe[domain->name].count(item) > 0);
                    continue;
                }
                domain_item_seen[domain->name].insert(item);
                // Obtain tables, weights, indexes for this item.
                vector<int>     t_list;
                vector<double>  w_list;
                vector<int>     i_list;
                if (domain->items.count(item) > 0) {
                    auto z = domain->get_cluster_assignment(item);
                    t_list = {z};
                    w_list = {0};
                    i_list = {0};
                } else {
                    auto tables_weights = domain->tables_weights();
                    auto Z = log(1 + domain->crp.N);
                    int idx = 0;
                    for (const auto &[t, w] : tables_weights) {
                        t_list.push_back(t);
                        w_list.push_back(log(w) - Z);
                        i_list.push_back(idx++);
                    }
                    assert(idx == t_list.size());
                }
                // Add to universe.
                item_universe.push_back({domain->name, item});
                index_universe.push_back(i_list);
                weight_universe.push_back(w_list);
                auto loc = index_universe.size() - 1;
                cluster_universe[domain->name][item] = {loc, t_list};
            }
        }
        assert(item_universe.size() == index_universe.size());
        assert(item_universe.size() == weight_universe.size());
        // Compute data probability given cluster combinations.
        vector<double> logps;
        for (const auto &indexes : product(index_universe)) {
            double logp_indexes = 0;
            // Compute weight of cluster assignments.
            double weight = 0;
            for (int i = 0; i < indexes.size(); i++) {
                weight += weight_universe.at(i).at(indexes[i]);
            }
            logp_indexes += weight;
            // Compute weight of data given cluster assignments.
            for (const auto &[r, items, value] : observations) {
                auto relation = relations.at(r);
                vector<int> z;
                for (int i = 0; i < relation->domains.size(); i++) {
                    auto domain = relation->domains.at(i);
                    auto item = items.at(i);
                    auto &[loc, t_list] = cluster_universe.at(domain->name).at(item);
                    auto t = t_list.at(indexes.at(loc));
                    z.push_back(t);
                }
                BetaBernoulli aux (prng);
                Distribution * cluster = relation->clusters.count(z) > 0 \
                    ? relation->clusters.at(z)
                    : &aux;
                logp_indexes += cluster->logp(value);
            }
            logps.push_back(logp_indexes);
        }
        return logsumexp(logps);
    }

    double logp_score() const {
        double logp_score_crp = 0.;
        for (const auto &[d, domain] : domains) {
            logp_score_crp += domain->crp.logp_score();
        }
        double logp_score_relation = 0.;
        for (const auto &[r, relation] : relations) {
            logp_score_relation += relation->logp_score();
        }
        return logp_score_crp + logp_score_relation;
    }

    void add_relation(const string &r, const vector<string> &ds) {
        assert(schema.count(r) == 0);
        assert(relations.count(r) == 0);
        vector<Domain*> doms;
        for (const auto &d : ds) {
            if (domains.count(d) == 0) {
                assert(domain_to_relations.count(d) == 0);
                domains[d] = new Domain(d, prng);
                domain_to_relations[d] = uset<string>();
            }
            domain_to_relations.at(d).insert(r);
            doms.push_back(domains.at(d));
        }
        relations[r] = new Relation(r, doms, prng);
        schema[r] = ds;
    }

    void remove_relation(const string &r) {
        uset<string> ds;
        for (const auto &domain : relations.at(r)->domains) {
            ds.insert(domain->name);
        }
        for (const auto &d : ds) {
            domain_to_relations.at(d).erase(r);
            // TODO: Remove r from domains.at(d)->items
            if (domain_to_relations.at(d).size() == 0) {
                domain_to_relations.erase(d);
                delete domains.at(d);
                domains.erase(d);
            }
        }
        delete relations.at(r);
        relations.erase(r);
        schema.erase(r);
    }

    // Disable copying.
    IRM & operator=(const IRM&) = delete;
    IRM(const IRM&) = delete;
};


class HIRM {
public:
    T_schema            schema;             // schema of relations
    umap<int, IRM*>     irms;               // map from cluster id to IRM
    umap<string, int>   relation_to_code;   // map from relation name to code
    umap<int, string>   code_to_relation;   // map from code to relation
    CRP                 crp;                // clustering model for relations
    PRNG                *prng;

    HIRM(const T_schema &schema, PRNG *prng) : crp(prng) {
        this->prng = prng;
        for (const auto &[r, ds] : schema) {
            this->add_relation(r, ds);
        }
    }

    void incorporate(const string &r, const T_items &items, double value) {
        auto irm = relation_to_irm(r);
        irm->incorporate(r, items, value);
    }
    void unincorporate(const string &r, const T_items &items) {
        auto irm = relation_to_irm(r);
        irm->unincorporate(r, items);
    }

    int relation_to_table(const string &r) {
        auto rc = relation_to_code.at(r);
        return crp.assignments.at(rc);
    }
    IRM * relation_to_irm(const string &r) {
        auto rc = relation_to_code.at(r);
        auto table = crp.assignments.at(rc);
        return irms.at(table);
    }
    Relation * get_relation(const string &r) {
        auto irm = relation_to_irm(r);
        return irm->relations.at(r);
    }

    void transition_cluster_assignments_all() {
        for (const auto &[r, rc] : relation_to_code) {
            transition_cluster_assignment_relation(r);
        }
    }
    void transition_cluster_assignments(vector<string> rs) {
        for (const auto &r : rs) {
            transition_cluster_assignment_relation(r);
        }
    }
    void transition_cluster_assignment_relation(const string &r) {
        auto rc = relation_to_code.at(r);
        auto table_current = crp.assignments.at(rc);
        auto relation = get_relation(r);
        auto crp_dist = crp.tables_weights_gibbs(table_current);
        vector<string> domains;
        for (const auto &d : relation->domains) {
            domains.push_back(d->name);
        }
        vector<int> tables;
        vector<double> logps;
        int * table_aux = NULL;
        IRM * irm_aux = NULL;
        // Compute probabilities of each table.
        for (const auto &[table, n_customers] : crp_dist) {
            IRM * irm;
            if (irms.count(table) == 0) {
                irm = new IRM({}, prng);
                assert(table_aux == NULL);
                assert(irm_aux == NULL);
                table_aux = (int *) malloc(sizeof(*table_aux));
                *table_aux = table;
                irm_aux = irm;
            } else {
                irm = irms.at(table);
            }
            if (table != table_current) {
                irm->add_relation(r, domains);
                for (const auto &[items, value] : relation->data) {
                    irm->incorporate(r, items, value);
                }
            }
            auto lp_data = irm->relations.at(r)->logp_score();
            auto lp_crp = log(n_customers);
            logps.push_back(lp_crp + lp_data);
            tables.push_back(table);
        }
        // Sample new table.
        auto idx = log_choice(logps, prng);
        auto choice = tables[idx];
        int new_size = 0;
        if (crp.tables.count(choice) > 0) {
            new_size = crp.tables.at(choice).size();
        }
        // Remove relation from all other tables.
        for (const auto &[table, customers] : crp.tables) {
            auto irm = irms.at(table);
            if (table != choice) {
                assert(irm->relations.count(r) == 1);
                irm->remove_relation(r);
            }
            if (irm->relations.size() == 0) {
                assert(crp.tables[table].size() == 1);
                assert(table == table_current);
                irms.erase(table);
                delete irm;
            }
        }
        // Add auxiliary table if necessary.
        if ((table_aux != NULL) && (choice == *table_aux)) {
            assert(irm_aux != NULL);
            irms[choice] = irm_aux;
        } else {
            delete irm_aux;
        }
        free(table_aux);
        // Update the CRP.
        crp.unincorporate(rc);
        crp.incorporate(rc, choice);
        assert(irms.size() == crp.tables.size());
        for (const auto &[table, irm] : irms) {
            assert(crp.tables.count(table) == 1);
        }
    }

    void set_cluster_assignment_gibbs(const string &r, int table) {
        assert(irms.size() == crp.tables.size());
        auto rc = relation_to_code.at(r);
        auto table_current = crp.assignments.at(rc);
        auto relation = get_relation(r);
        auto irm = relation_to_irm(r);
        auto observations = relation->data;
        vector<string> domains;
        for (const auto &d : relation->domains) {
            domains.push_back(d->name);
        }
        // Remove from current IRM.
        irm->remove_relation(r);
        if (irm->relations.size() == 0) {
            irms.erase(table_current);
            delete irm;
        }
        // Add to target IRM.
        if (irms.count(table) == 0) {
            irm = new IRM({}, prng);
            irms[table] = irm;
        }
        irm = irms.at(table);
        irm->add_relation(r, domains);
        for (const auto &[items, value] : observations) {
            irm->incorporate(r, items, value);
        }
        // Update CRP.
        crp.unincorporate(rc);
        crp.incorporate(rc, table);
        assert(irms.size() == crp.tables.size());
        for (const auto &[table, irm] : irms) {
            assert(crp.tables.count(table) == 1);
        }
    }

    void add_relation(string const &r, const vector<string> &ds) {
        assert(schema.count(r) == 0);
        schema[r] = ds;
        auto offset = (code_to_relation.size() == 0) ? 0
            : std::max_element(
                code_to_relation.begin(),
                code_to_relation.end())->first;
        auto rc = 1 + offset;
        auto table = crp.sample();
        crp.incorporate(rc, table);
        if (irms.count(table) == 1) {
           irms.at(table)->add_relation(r, ds);
        } else {
           irms[table] = new IRM({{r, ds}}, prng);
        }
        assert(relation_to_code.count(r) == 0);
        assert(code_to_relation.count(rc) == 0);
        relation_to_code[r] = rc;
        code_to_relation[rc] = r;
    }
    void remove_relation(string const &r) {
        schema.erase(r);
        auto rc = relation_to_code.at(r);
        auto table = crp.assignments.at(rc);
        auto singleton = crp.tables.at(table).size() == 1;
        crp.unincorporate(rc);
        irms.at(table)->remove_relation(r);
        if (singleton) {
            auto irm = irms.at(table);
            assert(irm->relations.size() == 0);
            irms.erase(table);
            delete irm;
        }
        relation_to_code.erase(r);
        code_to_relation.erase(rc);
    }

    double logp(const vector<tuple<string, T_items, double>> &observations) {
        umap<int, vector<tuple<string, T_items, double>>> obs_dict;
        for (const auto &[r, items, value] : observations) {
            auto rc = relation_to_code.at(r);
            auto table = crp.assignments.at(rc);
            if (obs_dict.count(table) == 0) {
                obs_dict[table] = {};
            }
            obs_dict.at(table).push_back({r, items, value});
        }
        double logp = 0;
        for (const auto &[t, o] : obs_dict) {
            logp += irms.at(t)->logp(o);
        }
        return logp;
    }

    double logp_score() {
        auto logp_score_crp = crp.logp_score();
        double logp_score_irms = 0;
        for (const auto &[table, irm] : irms) {
            logp_score_irms += irm->logp_score();
        }
        return logp_score_crp + logp_score_irms;
    }

    ~HIRM() {
        for (const auto &[table, irm] : irms) { delete irm; }
    }

    // Disable copying.
    HIRM & operator=(const HIRM&) = delete;
    HIRM(const HIRM&) = delete;
};
