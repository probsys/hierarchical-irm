# Copyright 2021 MIT Probabilistic Computing Project
# Apache License, Version 2.0, refer to LICENSE.txt

import os
import random

import matplotlib.pyplot as plt
import numpy as np

NOISY=1
prng = random.Random(1)
def flip(p):
    return prng.random() < p if NOISY else p > 0.5

P_LO = .1
P_HI = .65

# ===== Synthetic data generation.
items_D1 = [
    list(range(0, 300)),
    list(range(300, 400)),
]
data_r1 \
    = [((i, j), flip(P_LO)) for i in items_D1[0] for j in items_D1[0]] \
    + [((i, j), flip(P_HI)) for i in items_D1[0] for j in items_D1[1]] \
    + [((i, j), flip(P_HI)) for i in items_D1[1] for j in items_D1[0]] \
    + [((i, j), flip(P_LO)) for i in items_D1[1] for j in items_D1[1]] \

data_r2 \
    = [((i, j), flip(P_HI)) for i in items_D1[0] for j in items_D1[0]] \
    + [((i, j), flip(P_LO)) for i in items_D1[0] for j in items_D1[1]] \
    + [((i, j), flip(P_LO)) for i in items_D1[1] for j in items_D1[0]] \
    + [((i, j), flip(P_HI)) for i in items_D1[1] for j in items_D1[1]] \

xlabels = {'R1': '$D_1$', 'R2': '$D_1$'}
ylabels = {'R1': '$D_1$', 'R2': ''}

# Plot the synthetic data.
fig, axes = plt.subplots(ncols=2)
for relation, data, ax in [('R1', data_r1, axes[0]), ('R2', data_r2, axes[1])]:
    n = max(max(z) for z in items_D1)
    X = np.zeros((n+1, n+1))
    for (i, j), v in data:
        X[i,j] = v
    ax.imshow(X, cmap='Greys')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel(xlabels[relation])
    ax.set_ylabel(ylabels[relation], rotation=0, labelpad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(.05, .95, '$%s_%s$' % (relation[0], relation[1]),
        ha='left', va='top',
        transform=ax.transAxes,
        bbox={'facecolor': 'red', 'alpha': 1, 'edgecolor':'k'})

figname = os.path.join('assets', 'two_relations_anti.data.png')
fig.set_size_inches((3,1.5))
fig.set_tight_layout(True)
fig.savefig(figname)
print(figname)

# TODO: Compare output for clustering R1 and R2 using:
#   - IRM, with a higher-order encoding R': D1 x D1 X R -> {0, 1}
#   - HIRM, with a direct encoding of R1 and R2.
