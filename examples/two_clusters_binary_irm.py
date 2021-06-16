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
from hirm.util_plot import plot_binary_relation

schema, items_D1, items_D2, data = make_two_clusters()

# Write schema to disk.
path_schema = os.path.join('assets', 'two_clusters.binary.schema')
with open(path_schema, 'w') as f:
    f.write('bernoulli R1 D1 D2\n')
print(path_schema)
# Write observations to disk.
path_obs = os.path.join('assets', 'two_clusters.binary.obs')
with open(path_obs, 'w') as f:
    for ((i, j), value) in data:
        f.write('%d R1 %d %d\n' % (value, i, j))
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
    irm.incorporate('R1', (i, j), v)

# Plot the prior.
fig, ax = plot_binary_relation(irm.relations['R1'])
ax.set_title('Prior Sample')

# Run inference.
for i in range(20):
    irm.transition_cluster_assignments()
pprint(irm.domains['D1'].crp.tables)
pprint(irm.domains['D2'].crp.tables)

# Write the results.
path_clusters = os.path.join('assets', 'two_clusters.binary.irm')
to_txt_irm(path_clusters, irm)
print(path_clusters)

# Plot the posterior.
fig, ax = plot_binary_relation(irm.relations['R1'])
ax.set_title('Posterior Sample')
plt.show()

path_figure = os.path.join('assets', 'two_clusters.binary.irm.png')
fig.set_tight_layout(True)
fig.savefig(path_figure)
print(path_figure)
