# Copyright 2021 MIT Probabilistic Computing Project
# Apache License, Version 2.0, refer to LICENSE.txt

import json
import os
import random

import matplotlib.pyplot as plt

from pprint import pprint

from hirm import IRM
from hirm.util_io import load_observations
from hirm.util_io import load_schema
from hirm.util_io import to_dict_IRM
from hirm.util_io import to_txt_irm
from hirm.util_plot import plot_binary_relation
from hirm.util_plot import plot_ternary_relation

dirname = os.path.dirname(os.path.abspath(__file__))
path_schema = os.path.join(dirname, 'datasets', 'nations.binary.schema')
path_obs = os.path.join(dirname, 'datasets', 'nations.binary.obs')
schema = load_schema(path_schema)
data = load_observations(path_obs)

prng = random.Random(12)
irm = IRM(schema, prng=prng)
for relation, items, value in data:
    irm.incorporate(relation, items, value)

for i in range(10):
    irm.transition_cluster_assignments()
    print(i, irm.logp_score())

pprint(irm.domains['country'].crp.tables)
pprint(irm.domains['feature'].crp.tables)
pprint(irm.domains['predicate'].crp.tables)

fig, ax = plot_binary_relation(irm.relations['has'], transpose=True)
fig.set_tight_layout(True)
fig.set_size_inches((20, 10))
path_features = os.path.join('assets', 'nations.binary.irm.features.png')
fig.savefig(path_features)
print(path_features)
for predicate in irm.domains['predicate'].items:
    fig, ax = plot_ternary_relation(irm.relations['applies'], predicate)
    fname = os.path.join('assets', 'nations.binary.irm.%s.png' % (predicate,))
    fig.set_tight_layout(True)
    fig.savefig(fname)
    print(fname)
    plt.close(fig)

d = to_dict_IRM(irm)
path_json = os.path.join('assets', 'nations.binary.irm.json')
with open(path_json, 'w') as f:
    json.dump(d, f, indent=4)
print(path_json)

path_clusters = os.path.join('assets', 'nations.binary.irm')
to_txt_irm(path_clusters, irm)
print(path_clusters)
