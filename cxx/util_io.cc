// Copyright 2021 MIT Probabilistic Computing Project
// Apache License, Version 2.0, refer to LICENSE.txt

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>

#include "util_io.hh"

T_schema load_schema(const string &path) {
    std::ifstream fp (path, std::ifstream::in);
    assert(fp.good());

    map<string, vector<string>> schema;
    string line;
    while (std::getline(fp, line)) {
        std::istringstream stream (line);

        string dist;
        string relname;
        vector<string> domains;

        stream >> dist;
        stream >> relname;
        for (string w; stream >> w; ) {
            domains.push_back(w);
        }
        assert(domains.size() > 0);
        schema[relname] = domains;
    }
    fp.close();
    return schema;
}

T_observations load_observations(const string &path) {
    std::ifstream fp (path, std::ifstream::in);
    assert(fp.good());

    vector<tuple<string, vector<string>, double>> observations;
    string line;
    while (std::getline(fp, line)) {
        std::istringstream stream (line);

        double value;
        string relname;
        vector<string> items;

        stream >> value;
        stream >> relname;
        for (string w; stream >> w; ) {
            items.push_back(w);
        }
        assert(items.size() > 0);
        auto entry= std::make_tuple(relname, items, value);
        observations.push_back(entry);
    }
    fp.close();
    return observations;
}

// Assumes that T_item is integer.
T_encoding encode_observations(const T_schema &schema,
        const T_observations &observations) {
    // Counter and encoding maps.
    map<string, int> domain_item_counter;
    T_encoding_f item_to_code;
    T_encoding_r code_to_item;
    // Create a counter of items for each domain.
    for (const auto &[r, domains]: schema) {
        for (const auto &domain : domains) {
            domain_item_counter[domain] = 0;
            item_to_code[domain] = map<string, T_item>();
            code_to_item[domain] = map<T_item, string>();
        }
    }
    // Create the codes for each item.
    for (const auto &i : observations) {
        auto relation = std::get<0>(i);
        auto items = std::get<1>(i);
        int counter = 0;
        for (const auto &item : items) {
            // Obtain domain that item belongs to.
            auto domain = schema.at(relation).at(counter);
            counter += 1;
            // Compute its code, if necessary.
            if (item_to_code.at(domain).count(item) == 0) {
                int code = domain_item_counter[domain];
                item_to_code[domain][item] = code;
                code_to_item[domain][code] = item;
                domain_item_counter[domain]++;
            }
        }
    }
    return std::make_pair(item_to_code, code_to_item);
}

void incorporate_observations(IRM &irm, const T_encoding &encoding,
        const T_observations &observations) {
    auto item_to_code = std::get<0>(encoding);
    for (const auto &[relation, items, value] : observations) {
        int counter = 0;
        T_items items_e;
        for (const auto &item : items) {
            auto domain = irm.schema.at(relation)[counter];
            counter += 1;
            int code = item_to_code.at(domain).at(item);
            items_e.push_back(code);
        }
        irm.incorporate(relation, items_e, value);
    }
}

void incorporate_observations(HIRM &hirm, const T_encoding &encoding,
        const T_observations &observations) {
    int j = 0;
    auto item_to_code = std::get<0>(encoding);
    for (const auto &[relation, items, value] : observations) {
        int counter = 0;
        T_items items_e;
        for (const auto &item : items) {
            auto domain = hirm.schema.at(relation)[counter];
            counter += 1;
            int code = item_to_code.at(domain).at(item);
            items_e.push_back(code);
        }
        hirm.incorporate(relation, items_e, value);
    }
}

void to_txt(std::ostream &fp, const IRM &irm, const T_encoding &encoding) {
    auto code_to_item = std::get<1>(encoding);
    for (const auto &[d, domain]: irm.domains) {
        auto i0 = domain->crp.tables.begin();
        auto i1 = domain->crp.tables.end();
        map<int, uset<T_item>> tables (i0, i1);
        for (const auto &[table, items] : tables) {
            fp << domain->name << " ";
            fp << table << " ";
            int i = 1;
            for (const auto &item : items) {
                fp << code_to_item.at(domain->name).at(item);
                if (i++ < items.size()) {
                    fp << " ";
                }
            }
            fp << "\n";
        }
    }
}

void to_txt(std::ostream &fp, const HIRM &hirm, const T_encoding &encoding){
    // Write the relation clusters.
    auto i0 = hirm.crp.tables.begin();
    auto i1 = hirm.crp.tables.end();
    map<int, uset<T_item>> tables (i0, i1);
    for (const auto &[table, rcs] : tables) {
        fp << table << " ";
        int i = 1;
        for (const auto rc : rcs) {
            fp << hirm.code_to_relation.at(rc);
            if (i ++ < rcs.size()) {
                fp << " ";
            }
        }
        fp << "\n";
    }
    fp << "\n";
    // Write the IRMs.
    int j = 0;
    for (const auto &[table, rcs] : tables) {
        const auto &irm = hirm.irms.at(table);
        fp << "irm=" << table << "\n";
        to_txt(fp, *irm, encoding);
        if (j < tables.size() - 1) {
            fp << "\n";
            j += 1;
        }
    }
}

void to_txt(const string &path, const IRM &irm, const T_encoding &encoding) {
    std::ofstream fp (path);
    assert(fp.good());
    to_txt(fp, irm, encoding);
    fp.close();
}

void to_txt(const string &path, const HIRM &hirm, const T_encoding &encoding) {
    std::ofstream fp (path);
    assert(fp.good());
    to_txt(fp, hirm, encoding);
    fp.close();
}

map<string, map<int, vector<string>>>
load_clusters_irm(const string &path) {
    std::ifstream fp (path, std::ifstream::in);
    assert(fp.good());

    map<string, map<int, vector<string>>> clusters;
    string line;
    while (std::getline(fp, line)) {
        std::istringstream stream (line);

        string domain;
        int table;
        vector<string> items;

        stream >> domain;
        stream >> table;
        for (string w; stream >> w; ) {
            items.push_back(w);
        }
        assert(items.size() > 0);
        assert(clusters[domain].count(table) == 0);
        clusters[domain][table] = items;
    }
    fp.close();
    return clusters;
}


int isnumeric(const std::string & s) {
    for (char c : s) { if (!isdigit(c)) { return false; } }
    return !s.empty() && true;
}


tuple<
    map<int, vector<string>>,                       // x[table] = {relation list}
    map<int, map<string, map<int, vector<string>>>> // x[table][domain][table] = {item list}
    >
load_clusters_hirm(const string &path) {
    std::ifstream fp (path, std::ifstream::in);
    assert(fp.good());

    map<int, vector<string>> relations;
    map<int, map<string, map<int, vector<string>>>> irms;

    string line;
    int irmc = 0;

    while (std::getline(fp, line)) {
        std::istringstream stream (line);

        string first;
        stream >> first;

        // Parse a relation cluster.
        if (isnumeric(first)) {
            int table = std::stoi(first);
            vector<string> items;
            for (string item; stream >> item; ) {
                items.push_back(item);
            }
            assert(items.size() > 0);
            assert(relations.count(table) == 0);
            relations[table] = items;
            continue;
        }

        // Skip a new line.
        if (first.size() == 0) {
            irmc = -1;
            continue;
        }

        // Parse an irm= line.
        if (first.rfind("irm=", 0) == 0) {
            assert(irmc = -1);
            assert(first.size() > 4);
            auto x = first.substr(4);
            irmc = std::stoi(x);
            assert(irms.count(irmc) == 0);
            irms[irmc] = {};
            continue;
        }

        // Parse a domain cluster.
        assert(irmc > -1);
        assert(irms.count(irmc) == 1);
        string second;
        stream >> second;
        assert(second.size() > 0);
        assert(isnumeric(second));
        auto &domain = first;
        auto table = std::stoi(second);
        vector<string> items;
        for (string item; stream >> item; ) {
            items.push_back(item);
        }
        assert(items.size() > 0);
        if (irms.at(irmc).count(domain) == 0) {
            irms.at(irmc)[domain] = {};
        }
        assert(irms.at(irmc).at(domain).count(table) == 0);
        irms.at(irmc).at(domain)[table] = items;
    }

    assert(relations.size() == irms.size());
    for (const auto &[t, rs] : relations) {
        assert(irms.count(t) == 1);
    }
    fp.close();
    return std::make_pair(relations, irms);
}

void from_txt(IRM * const irm,
        const string &path_schema,
        const string &path_obs,
        const string &path_clusters) {
    // Load the data.
    auto schema = load_schema(path_schema);
    auto observations = load_observations(path_obs);
    auto encoding = encode_observations(schema, observations);
    auto clusters = load_clusters_irm(path_clusters);
    // Add the relations.
    assert(irm->schema.size() == 0);
    assert(irm->domains.size() == 0);
    assert(irm->relations.size() == 0);
    assert(irm->domain_to_relations.size() == 0);
    for (const auto &[r, ds] : schema) {
        irm->add_relation(r, ds);
    }
    // Add the domain entities with fixed clustering.
    T_encoding_f item_to_code = std::get<0>(encoding);
    for (const auto &[domain, tables] : clusters) {
        assert(irm->domains.at(domain)->items.size() == 0);
        for (const auto &[table, items] : tables) {
            assert(0 <= table);
            for (const auto &item : items) {
                auto code = item_to_code.at(domain).at(item);
                irm->domains.at(domain)->incorporate(code, table);
            }
        }
    }
    // Add the observations.
    incorporate_observations(*irm, encoding, observations);
}

void from_txt(HIRM * const hirm,
        const string &path_schema,
        const string &path_obs,
        const string &path_clusters) {
    auto schema = load_schema(path_schema);
    auto observations = load_observations(path_obs);
    auto encoding = encode_observations(schema, observations);
    auto [relations, irms] = load_clusters_hirm(path_clusters);
    // Add the relations.
    assert(hirm->schema.size() == 0);
    assert(hirm->irms.size() == 0);
    assert(hirm->relation_to_code.size() == 0);
    assert(hirm->code_to_relation.size() == 0);
    for (const auto &[r, ds] : schema) {
        hirm->add_relation(r, ds);
        assert(hirm->irms.size() == hirm->crp.tables.size());
        hirm->set_cluster_assignment_gibbs(r, -1);
    }
    // Add each IRM.
    for (const auto &[table, rs] : relations) {
        assert(hirm->irms.size() == hirm->crp.tables.size());
        // Add relations to the IRM.
        for (const auto &r : rs) {
            assert(hirm->irms.size() == hirm->crp.tables.size());
            auto table_current = hirm->relation_to_table(r);
            if (table_current != table) {
                assert(hirm->irms.size() == hirm->crp.tables.size());
                hirm->set_cluster_assignment_gibbs(r, table);
            }
        }
        // Add the domain entities with fixed clustering to this IRM.
        // TODO: Duplicated code with from_txt(IRM)
        auto irm = hirm->irms.at(table);
        auto clusters = irms.at(table);
        assert(irm->relations.size() == rs.size());
        T_encoding_f item_to_code = std::get<0>(encoding);
        for (const auto &[domain, tables] : clusters) {
            assert(irm->domains.at(domain)->items.size() == 0);
            for (const auto &[t, items] : tables) {
                assert(0 <= t);
                for (const auto &item : items) {
                    auto code = item_to_code.at(domain).at(item);
                    irm->domains.at(domain)->incorporate(code, t);
                }
            }
        }
    }
    assert(hirm->irms.count(-1) == 0);
    // Add the observations.
    incorporate_observations(*hirm, encoding, observations);
}
