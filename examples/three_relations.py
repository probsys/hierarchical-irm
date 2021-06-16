# Copyright 2021 MIT Probabilistic Computing Project
# Apache License, Version 2.0, refer to LICENSE.txt

import os
import random

import matplotlib.pyplot as plt
import numpy as np

from hirm import HIRM
from hirm.util_io import to_txt_hirm
from three_relations_plot import xlabels
from three_relations_plot import ylabels
from three_relations_plot import plot_hirm_clusters

ITERS = int(os.environ.get('ITERS', '100'))
NOISY = os.environ.get('NOISY', True)
SHUFFLE = os.environ.get('SHUFFLE', True)
noisy_str = 'noisy' if NOISY else 'clean'
prng = random.Random(1)
def flip(p):
    return prng.random() < p if NOISY else p > 0.5

P_LO = .1
P_HI = .9

# ===== Synthetic data generation.
schema = {
    'R1': ('D1', 'D1'),
    'R2': ('D1', 'D2'),
    'R3': ('D1', 'D3'),
}
n_items = {'D1': 300, 'D2': 300, 'D3': 300}

# R1
items_D1_R1 = [
    list(range(0, 200)),
    list(range(200, 300)),
]
data_r1_d10_d10 = [((i, j), flip(P_LO)) for i in items_D1_R1[0] for j in items_D1_R1[0]]
data_r1_d10_d11 = [((i, j), flip(P_HI)) for i in items_D1_R1[0] for j in items_D1_R1[1]]
data_r1_d11_d10 = [((i, j), flip(P_HI)) for i in items_D1_R1[1] for j in items_D1_R1[0]]
data_r1_d11_d11 = [((i, j), flip(P_LO)) for i in items_D1_R1[1] for j in items_D1_R1[1]]
data_r1 =  data_r1_d10_d10 + data_r1_d10_d11 + data_r1_d11_d10 + data_r1_d11_d11

# R2
items_D1_R2 = [
    list(range(0, 300))[::2],
    list(range(0, 300))[1::2]
]
items_D2_R2 = [
    list(range(0, 150))[::2],
    list(range(0, 150))[1::2],
    list(range(150, 300))[::2],
    list(range(150, 300))[1::2]
]
data_r2_d10_d20 = [((i, j), flip(P_LO)) for i in items_D1_R2[0] for j in items_D2_R2[0]]
data_r2_d10_d21 = [((i, j), flip(P_HI)) for i in items_D1_R2[0] for j in items_D2_R2[1]]
data_r2_d10_d22 = [((i, j), flip(P_LO)) for i in items_D1_R2[0] for j in items_D2_R2[2]]
data_r2_d10_d23 = [((i, j), flip(P_HI)) for i in items_D1_R2[0] for j in items_D2_R2[3]]
data_r2_d11_d20 = [((i, j), flip(P_HI)) for i in items_D1_R2[1] for j in items_D2_R2[0]]
data_r2_d11_d21 = [((i, j), flip(P_LO)) for i in items_D1_R2[1] for j in items_D2_R2[1]]
data_r2_d11_d22 = [((i, j), flip(P_HI)) for i in items_D1_R2[1] for j in items_D2_R2[3]]
data_r2_d11_d23 = [((i, j), flip(P_LO)) for i in items_D1_R2[1] for j in items_D2_R2[2]]
data_r2 \
    = data_r2_d10_d20 + data_r2_d10_d21 + data_r2_d10_d22 + data_r2_d10_d23 \
    + data_r2_d11_d20 + data_r2_d11_d21 + data_r2_d11_d22 + data_r2_d11_d23

# R3
items_D1_R3 = items_D1_R1
items_D3_R3 = [
    list(range(0, 100)),
    list(range(100, 200)),
    list(range(200, 300)),
]

data_r3_d10_d30 = [((i, j), flip(P_HI)) for i in items_D1_R3[0] for j in items_D3_R3[0]]
data_r3_d10_d31 = [((i, j), flip(P_LO)) for i in items_D1_R3[0] for j in items_D3_R3[1]]
data_r3_d10_d32 = [((i, j), flip(P_HI)) for i in items_D1_R3[0] for j in items_D3_R3[2]]
data_r3_d11_d30 = [((i, j), flip(P_LO)) for i in items_D1_R3[1] for j in items_D3_R3[0]]
data_r3_d11_d31 = [((i, j), flip(P_HI)) for i in items_D1_R3[1] for j in items_D3_R3[1]]
data_r3_d11_d32 = [((i, j), flip(P_HI)) for i in items_D1_R3[1] for j in items_D3_R3[2]]
data_r3 \
    = data_r3_d10_d30 + data_r3_d10_d31 + data_r3_d10_d32 \
    + data_r3_d11_d30 + data_r3_d11_d31 + data_r3_d11_d32

# Write schema to disk.
path_schema = os.path.join('assets', 'three_relations.schema')
with open(path_schema, 'w') as f:
    f.write('bernoulli R1 D1 D1\n')
    f.write('bernoulli R2 D1 D2\n')
    f.write('bernoulli R3 D1 D3\n')
print(path_schema)
# Write observations to disk.
path_obs = os.path.join('assets', 'three_relations.obs')
with open(path_obs, 'w') as f:
    for ((i, j), value) in data_r1:
        f.write('%d R1 %d %d\n' % (value, i, j))
    for ((i, j), value) in data_r2:
        f.write('%d R2 %d %d\n' % (value, i, j))
    for ((i, j), value) in data_r3:
        f.write('%d R3 %d %d\n' % (value, i, j))
print(path_obs)

# Plot the synthetic data.
fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0,0].set_axis_off()
for relation, data, ax in [
        ('R1', data_r1, axes[1,1]),
        ('R2', data_r2, axes[1,0]),
        ('R3', data_r3, axes[0,1]),
    ]:
    nr = n_items[schema[relation][0]]
    nc = n_items[schema[relation][1]]
    X = np.zeros((nr, nc))
    for (i, j), v in data:
        X[i,j] = v
    if relation == 'R3':
        X = X.T
    if SHUFFLE:
        if relation == 'R1':
            n = n_items['D1']
            pi = prng.sample(list(range(n)), k=n)
            X = np.asarray([
                [X[pi[r], pi[c]] for c in range(n)]
                for r in range(n)
            ])
        elif relation == 'R2':
            nr = n_items['D1']
            nc = n_items['D2']
            pir = prng.sample(list(range(nr)), k=nr)
            pic = prng.sample(list(range(nc)), k=nc)
            X = np.asarray([
                [X[pir[r], pic[c]] for c in range(nc)]
                for r in range(nr)
            ])
        if relation == 'R3':
            nr = n_items['D1']
            nc = n_items['D3']
            pir = prng.sample(list(range(nr)), k=nr)
            pic = prng.sample(list(range(nc)), k=nc)
            X = np.asarray([
                [X[pir[r], pic[c]] for c in range(nc)]
                for r in range(nr)
            ])
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
figname = os.path.join('assets', 'three_relations.%s.data.png' % (noisy_str,))
fig.set_size_inches((3.5, 3.5))
fig.subplots_adjust(wspace=.1, hspace=.1)
fig.savefig(figname)
print(figname)

# ===== Make an HIRM for three relations and learn partition.
def learn_hirm(seed, steps):
    hirm = HIRM(schema, prng=random.Random(seed))
    hirm.seed = seed
    for relation, data in [
            ('R1', data_r1),
            ('R2', data_r2),
            ('R3', data_r3),
        ]:
        for (i, j), v in data:
            hirm.incorporate(relation, (i, j), v)
    print(hirm.logp_score())
    for i in range(steps):
        hirm.transition_cluster_assignments()
        for irm in hirm.irms.values():
            irm.transition_cluster_assignments()
        print(i, [len(c) for c in hirm.crp.tables.values()], hirm.logp_score())
    return hirm

if __name__ == '__main__':
    seed = int(os.environ.get('SEED', '0'))
    iters = int(os.environ.get('ITERS', '20'))
    print('running with seed %d for %d iters' % (seed, iters))
    hirm = learn_hirm(seed, iters)
    path_clusters = os.path.join('assets', 'three_relations.%s.%d.hirm' % (noisy_str, seed,))
    to_txt_hirm(path_clusters, hirm)
    print(path_clusters)
    figname = '%s.png' % (path_clusters)
    plot_hirm_clusters(hirm, figname)
