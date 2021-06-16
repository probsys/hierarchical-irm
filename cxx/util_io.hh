// Copyright 2020
// See LICENSE.txt

#pragma once

#include "globals.hh"
#include "hirm.hh"

typedef map<string, map<string, T_item>> T_encoding_f;
typedef map<string, map<T_item, string>> T_encoding_r;
typedef tuple<T_encoding_f, T_encoding_r> T_encoding;

typedef tuple<string, vector<string>, double> T_observation;
typedef vector<T_observation> T_observations;

typedef umap<string, T_item> T_assignment;
typedef umap<string, T_assignment> T_assignments;

// disk IO
T_schema load_schema(const string &path);
T_observations load_observations(const string &path);
T_encoding encode_observations(const T_schema &schema, const T_observations &observations);

void incorporate_observations(IRM &irm, const T_encoding &encoding,
    const T_observations &observations);
void incorporate_observations(HIRM &hirm, const T_encoding &encoding,
    const T_observations &observations);

void to_txt(const string &path, const IRM &irm, const T_encoding &encoding);
void to_txt(const string &path, const HIRM &irm, const T_encoding &encoding);
void to_txt(std::ostream &fp, const IRM &irm, const T_encoding &encoding);
void to_txt(std::ostream &fp, const HIRM &irm, const T_encoding &encoding);

map<string, map<int, vector<string>>> load_clusters_irm(const string &path);
tuple<
    map<int, vector<string>>,                       // x[table] = {relation list}
    map<int, map<string, map<int, vector<string>>>> // x[table][domain][table] = {item list}
    >
load_clusters_hirm(const string &path);

void from_txt(IRM * const irm, const string &path_schema, const string &path_obs, const string &path_clusters);
void from_txt(HIRM * const irm, const string &path_schema, const string &path_obs, const string &path_clusters);
