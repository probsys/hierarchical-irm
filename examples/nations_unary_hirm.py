# Copyright 2021 MIT Probabilistic Computing Project
# Apache License, Version 2.0, refer to LICENSE.txt

import json
import os
import random

import matplotlib.pyplot as plt

from pprint import pprint

from hirm import HIRM
from hirm.util_io import load_observations
from hirm.util_io import load_schema
from hirm.util_io import to_dict_HIRM
from hirm.util_io import to_txt_hirm
from hirm.util_plot import plot_binary_relation
from hirm.util_plot import plot_hirm_crosscat

dirname = os.path.dirname(os.path.abspath(__file__))
path_schema = os.path.join(dirname, 'datasets', 'nations.unary.schema')
path_obs = os.path.join(dirname, 'datasets', 'nations.unary.obs')
schema = load_schema(path_schema)
data = load_observations(path_obs)
features = [r for r in schema if len(schema[r]) == 1]
predicates = [r for r in schema if len(schema[r]) == 2]

prng = random.Random(12)
hirm = HIRM(schema, prng=prng)
for relation, items, value in data:
    hirm.incorporate(relation, items, value)

print(hirm.logp_score())
for i in range(10):
    hirm.transition_cluster_assignments()
    for irm in hirm.irms.values():
        irm.transition_cluster_assignments()
    print(i, hirm.logp_score(), [len(c) for c in hirm.crp.tables.values()])

pprint(hirm.crp.tables)

fig, ax = plot_hirm_crosscat(hirm, features)
fig.set_size_inches((30, 10))
fig.set_tight_layout(True)
path_features = os.path.join('assets', 'nations.unary.hirm.features.png')
fig.savefig(path_features)
print(path_features)
for r in predicates:
    irm = hirm.relation_to_irm(r)
    fig, ax = plot_binary_relation(irm.relations[r])
    fname = os.path.join('assets', 'nations.unary.hirm.%s.png' % (r,))
    fig.set_tight_layout(True)
    fig.savefig(fname)
    print(fname)
    plt.close(fig)

d = to_dict_HIRM(hirm)
path_json = os.path.join('assets', 'nations.unary.hirm.json')
with open(path_json, 'w') as f:
    json.dump(d, f, indent=4)
print(path_json)

path_clusters = os.path.join('assets', 'nations.unary.hirm')
to_txt_hirm(path_clusters, hirm)
print(path_clusters)
