# Copyright 2021 MIT Probabilistic Computing Project
# Apache License, Version 2.0, refer to LICENSE.txt

import os
import random

from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

from hirm import IRM
from hirm.tests.util_test import make_two_clusters
from hirm.util_io import to_txt_irm
from hirm.util_plot import plot_unary_relations

schema, items_D1, items_D2, data = make_two_clusters()

# Update schema to unary encoding.
schema = {'Feature-%02d' % (j,): ('D1',) for j in range(40)}

# Write schema to disk.
path_schema = os.path.join('assets', 'two_clusters.unary.schema')
with open(path_schema, 'w') as f:
    for j in range(40):
        f.write('bernoulli Feature-%02d D1\n' % (j,))
print(path_schema)
# Write observations to disk.
path_obs = os.path.join('assets', 'two_clusters.unary.obs')
with open(path_obs, 'w') as f:
    for ((i, j), value) in data:
        f.write('%d Feature-%02d %d\n' % (value, j, i))
print(path_obs)

# Plot the synthetic data.
X = np.zeros((30, 40))
for (i, j), v in data:
    X[i,j] = v
fig, ax = plt.subplots()
ax.imshow(X, cmap='Greys')
ax.xaxis.tick_top()
ax.set_xticks(np.arange(X.shape[1]))
ax.set_yticks(np.arange(X.shape[0]))
ax.set_title('Raw Data')

# Make an IRM.
irm = IRM(schema, prng=random.Random(1))
for (i, j), v in data:
    irm.incorporate('Feature-%02d' % (j,), (i,), v)

# Plot the prior.
fig, ax = plot_unary_relations(list(irm.relations.values()))
ax.set_title('Prior Sample')

# Run inference.
for i in range(20):
    irm.transition_cluster_assignments()
pprint(irm.domains['D1'].crp.tables)

# Write the results.
path_clusters = os.path.join('assets', 'two_clusters.unary.irm')
to_txt_irm(path_clusters, irm)
print(path_clusters)

# Plot the posterior.
fig, ax = plot_unary_relations(list(irm.relations.values()))
ax.set_title('Posterior Sample')
plt.show()

path_figure = os.path.join('assets', 'two_clusters.unary.irm.png')
fig.set_tight_layout(True)
fig.savefig(path_figure)
print(path_figure)
