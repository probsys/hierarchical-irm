# Copyright 2021 MIT Probabilistic Computing Project
# Apache License, Version 2.0, refer to LICENSE.txt

import os
import random

from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

from hirm import IRM
from hirm.util_plot import plot_binary_relation

prng = random.Random(1)

# ===== Synthetic data generation.
items_D1_R1 = [
    list(range(0, 10)) + list(range(20,30)),
    list(range(10, 20)),
]
items_D2_R1 = [
    list(range(0, 20)),
    list(range(20, 40)),
]
data_r1_d10_d20 = [((i, j), 0) for i in items_D1_R1[0] for j in items_D2_R1[0]]
data_r1_d10_d21 = [((i, j), 1) for i in items_D1_R1[0] for j in items_D2_R1[1]]
data_r1_d11_d20 = [((i, j), 1) for i in items_D1_R1[1] for j in items_D2_R1[0]]
data_r1_d11_d21 = [((i, j), 0) for i in items_D1_R1[1] for j in items_D2_R1[1]]
data_r1 =  data_r1_d10_d20 + data_r1_d10_d21 + data_r1_d11_d20 + data_r1_d11_d21

items_D1_R2 = [
    list(range(0, 30))[::2],
    list(range(0, 30))[1::2]
]
items_D2_R2 = [
    list(range(0, 20))[::2],
    list(range(0, 20))[1::2],
    list(range(20, 40))[::2],
    list(range(20, 40))[1::2]
]
data_r2_d10_d20 = [((i, j), 0) for i in items_D1_R2[0] for j in items_D2_R2[0]]
data_r2_d10_d21 = [((i, j), 1) for i in items_D1_R2[0] for j in items_D2_R2[1]]
data_r2_d10_d22 = [((i, j), 0) for i in items_D1_R2[0] for j in items_D2_R2[2]]
data_r2_d10_d23 = [((i, j), 1) for i in items_D1_R2[0] for j in items_D2_R2[3]]
data_r2_d11_d20 = [((i, j), 1) for i in items_D1_R2[1] for j in items_D2_R2[0]]
data_r2_d11_d21 = [((i, j), 0) for i in items_D1_R2[1] for j in items_D2_R2[1]]
data_r2_d11_d22 = [((i, j), 1) for i in items_D1_R2[1] for j in items_D2_R2[3]]
data_r2_d11_d23 = [((i, j), 0) for i in items_D1_R2[1] for j in items_D2_R2[2]]
data_r2 \
    = data_r2_d10_d20 + data_r2_d10_d21 + data_r2_d10_d22 + data_r2_d10_d23 \
    + data_r2_d11_d20 + data_r2_d11_d21 + data_r2_d11_d22 + data_r2_d11_d23

xlabels = {'R1': 'D2', 'R2': 'D2'}
ylabels = {'R1': 'D1', 'R2': ''}

# Write schema to disk.
path_schema = os.path.join('assets', 'two_relations.schema')
with open(path_schema, 'w') as f:
    f.write('bernoulli R1 D1 D2\n')
    f.write('bernoulli R2 D1 D2\n')
print(path_schema)
# Write observations to disk.
path_obs = os.path.join('assets', 'two_relations.obs')
with open(path_obs, 'w') as f:
    for ((i, j), value) in data_r1:
        f.write('%d R1 %d %d\n' % (value, i, j))
    for ((i, j), value) in data_r2:
        f.write('%d R2 %d %d\n' % (value, i, j))
print(path_obs)

# Plot the synthetic data.
fig, axes = plt.subplots(ncols=2)
for relation, data, ax in [('R1', data_r1, axes[0]), ('R2', data_r2, axes[1])]:
    X = np.zeros((30, 40))
    for (i, j), v in data:
        X[i,j] = v
    nr = 30
    nc = 40
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
    ax.set_ylabel(ylabels[relation])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(.05, .95, relation,
        ha='left', va='top',
        transform=ax.transAxes,
        bbox={'facecolor': 'red', 'alpha': 1, 'edgecolor':'k'})
figname = os.path.join('assets', 'two_relations.data.png')
fig.set_size_inches((4,2))
fig.set_tight_layout(True)
fig.savefig(figname)
print(figname)

# ===== Make an IRM for both relations (using seed that underfits).
schema = {'R1': ('D1', 'D2'), 'R2': ('D1', 'D2')}
irm = IRM(schema, prng=random.Random(1))
for relation, data in [
        ('R1', data_r1),
        ('R2', data_r2)
    ]:
    for (i, j), v in data:
        irm.incorporate(relation, (i, j), v)

# Run inference.
for i in range(100):
    irm.transition_cluster_assignments()
pprint(irm.domains['D1'].crp.tables)
pprint(irm.domains['D2'].crp.tables)
pprint(irm.logp_score())

# Plot the posterior.
fig, axes = plt.subplots(ncols=2)
for relation, data, ax in [('R1', data_r1, axes[0]), ('R2', data_r2, axes[1])]:
    plot_binary_relation(irm.relations[relation], ax=ax)
    score = irm.relations[relation].logp_score()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel(xlabels[relation])
    ax.set_ylabel(ylabels[relation])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(.5, -.1, 'log score = %1.2f' % (score,),
        ha='center', va='top', color='k',
        transform=ax.transAxes)
figname = os.path.join('assets', 'two_relations.underfit.png')
fig.set_size_inches((4,2))
fig.set_tight_layout(True)
fig.savefig(figname)
print(figname)


# ===== Make an IRM for both relations (using seed that overfits).
schema = {'R1': ('D1', 'D2'), 'R2': ('D1', 'D2')}
irm = IRM(schema, prng=random.Random(10))
for relation, data in [
        ('R1', data_r1),
        ('R2', data_r2)
    ]:
    for (i, j), v in data:
        irm.incorporate(relation, (i, j), v)

# Run inference.
for i in range(200):
    irm.transition_cluster_assignments()
pprint(irm.domains['D1'].crp.tables)
pprint(irm.domains['D2'].crp.tables)
pprint(irm.logp_score())

# Plot the posterior.
fig, axes = plt.subplots(ncols=2)
for relation, data, ax in [('R1', data_r1, axes[0]), ('R2', data_r2, axes[1])]:
    plot_binary_relation(irm.relations[relation], ax=ax)
    score = irm.relations[relation].logp_score()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel(xlabels[relation])
    ax.set_ylabel(ylabels[relation])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(.5, -.1, 'log score = %1.2f' % (score,),
        ha='center', va='top', color='k',
        transform=ax.transAxes)
figname = os.path.join('assets', 'two_relations.overfit.png')
fig.set_size_inches((4,2))
fig.set_tight_layout(True)
fig.savefig(figname)
print(figname)

# ===== Make IRM for each relation separately.
irm1 = IRM({'R1': ('D1', 'D2')}, prng=random.Random(1))
irm2 = IRM({'R2': ('D1', 'D2')}, prng=random.Random(10))
for (i, j), v in data_r1:
    irm1.incorporate('R1', (i, j), v)
for (i, j), v in data_r2:
    irm2.incorporate('R2', (i, j), v)

# Run inference.
for i in range(100):
    irm1.transition_cluster_assignments()
    irm2.transition_cluster_assignments()
pprint(irm1.domains['D1'].crp.tables)
pprint(irm2.domains['D2'].crp.tables)
pprint(irm1.logp_score())
pprint(irm2.logp_score())

# Plot the posterior.
fig, axes = plt.subplots(ncols=2)
plot_binary_relation(irm1.relations['R1'], ax=axes[0])
score1 = irm1.relations['R1'].logp_score()
axes[0].xaxis.set_label_position('top')
axes[0].set_xlabel(xlabels['R1'])
axes[0].set_ylabel(ylabels['R1'])
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].text(.5, -.1, 'log score = %1.2f' % (score1,),
    ha='center', va='top', color='k', transform=axes[0].transAxes)

plot_binary_relation(irm2.relations['R2'], ax=axes[1])
score2 = irm2.relations['R2'].logp_score()
axes[1].xaxis.set_label_position('top')
axes[1].set_xlabel(xlabels['R2'])
axes[1].set_ylabel(ylabels['R2'])
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].text(.5, -.1, 'log score = %1.2f' % (score2,),
    ha='center', va='top', color='k', transform=axes[1].transAxes)

figname = os.path.join('assets', 'two_relations.separate.png')
fig.set_size_inches((4,2))
fig.set_tight_layout(True)
fig.savefig(figname)
print(figname)
plt.show()

# from hirm.util_math import log_normalize
# p1, p2 = np.exp(log_normalize([score, score1 + score2]))
# print((p1, p2))

