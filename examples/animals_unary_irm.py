# Copyright 2021 MIT Probabilistic Computing Project
# Apache License, Version 2.0, refer to LICENSE.txt

import os
import random

from pprint import pprint

import matplotlib.pyplot as plt

from hirm import IRM
from hirm.util_io import load_schema
from hirm.util_io import load_observations
from hirm.util_io import to_txt_irm
from hirm.util_plot import plot_unary_relations

dirname = os.path.dirname(os.path.abspath(__file__))
path_schema = os.path.join(dirname, 'datasets', 'animals.unary.schema')
path_obs = os.path.join(dirname, 'datasets', 'animals.unary.obs')
schema = load_schema(path_schema)
data = load_observations(path_obs)

prng = random.Random(13412)
irm = IRM(schema, prng=prng)
for relation, items, value in data:
    irm.incorporate(relation, items, value)

for i in range(20):
    irm.transition_cluster_assignments()
    print(i, irm.logp_score())
pprint(irm.domains['animal'].crp.tables)

fig, ax = plot_unary_relations(list(irm.relations.values()))
plt.show()
fig.set_tight_layout(True)

path_fig = os.path.join('assets', 'animals.unary.irm.png')
fig.savefig(path_fig)
print(path_fig)

path_clusters = os.path.join('assets', 'animals.unary.irm')
to_txt_irm(path_clusters, irm)
print(path_clusters)
