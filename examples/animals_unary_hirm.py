# Copyright 2021 MIT Probabilistic Computing Project
# Apache License, Version 2.0, refer to LICENSE.txt

import os
import random

import matplotlib.pyplot as plt

from hirm import HIRM
from hirm.util_io import load_schema
from hirm.util_io import load_observations
from hirm.util_io import to_txt_hirm

from hirm.util_plot import plot_hirm_crosscat

dirname = os.path.dirname(os.path.abspath(__file__))
path_schema = os.path.join(dirname, 'datasets', 'animals.unary.schema')
path_obs = os.path.join(dirname, 'datasets', 'animals.unary.obs')
schema = load_schema(path_schema)
data = load_observations(path_obs)
features = [r for r in schema if len(schema[r]) == 1]

prng = random.Random(12)
hirm = HIRM(schema, prng=prng)
for relation, items, value in data:
    print(relation, items, value)
    hirm.incorporate(relation, items, value)

fig, ax = plot_hirm_crosscat(hirm, features)
fig.set_tight_layout(True)

print(hirm.logp_score())
for i in range(10):
    hirm.transition_cluster_assignments()
    for irm in hirm.irms.values():
        irm.transition_cluster_assignments()
    print(i, hirm.logp_score(), [len(c) for c in hirm.crp.tables.values()])

fig, ax = plot_hirm_crosscat(hirm, features)
plt.show()
fig.set_tight_layout(True)

path_fig = os.path.join('assets', 'animals.unary.hirm.png')
fig.savefig(path_fig)
print(path_fig)

path_clusters = os.path.join('assets', 'animals.unary.hirm')
to_txt_hirm(path_clusters, hirm)
print(path_clusters)
