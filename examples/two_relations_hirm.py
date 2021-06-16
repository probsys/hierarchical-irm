# Copyright 2021 MIT Probabilistic Computing Project
# Apache License, Version 2.0, refer to LICENSE.txt

import os
import random

import matplotlib.pyplot as plt
import numpy as np

from hirm import HIRM
from hirm.util_plot import plot_binary_relation

prng = random.Random(1)
NOISY = os.environ.get('NOISY', None)
noisy_str = 'noisy' if NOISY else 'clean'
def flip(p):
    return prng.random() < p if NOISY else p > 0.5

items_D1_R1 = [
    list(range(0, 10)) + list(range(20,30)),
    list(range(10, 20)),
]
items_D2_R1 = [
    list(range(0, 20)),
    list(range(20, 40)),
]
data_r1_d10_d20 = [((i, j), flip(.2)) for i in items_D1_R1[0] for j in items_D2_R1[0]]
data_r1_d10_d21 = [((i, j), flip(.7)) for i in items_D1_R1[0] for j in items_D2_R1[1]]
data_r1_d11_d20 = [((i, j), flip(.8)) for i in items_D1_R1[1] for j in items_D2_R1[0]]
data_r1_d11_d21 = [((i, j), flip(.15)) for i in items_D1_R1[1] for j in items_D2_R1[1]]
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
data_r2_d10_d20 = [((i, j), flip(.1)) for i in items_D1_R2[0] for j in items_D2_R2[0]]
data_r2_d10_d21 = [((i, j), flip(.2)) for i in items_D1_R2[0] for j in items_D2_R2[1]]
data_r2_d10_d22 = [((i, j), flip(.15)) for i in items_D1_R2[0] for j in items_D2_R2[2]]
data_r2_d10_d23 = [((i, j), flip(.8)) for i in items_D1_R2[0] for j in items_D2_R2[3]]
data_r2_d11_d20 = [((i, j), flip(.8)) for i in items_D1_R2[1] for j in items_D2_R2[0]]
data_r2_d11_d21 = [((i, j), flip(.3)) for i in items_D1_R2[1] for j in items_D2_R2[1]]
data_r2_d11_d22 = [((i, j), flip(.9)) for i in items_D1_R2[1] for j in items_D2_R2[3]]
data_r2_d11_d23 = [((i, j), flip(.1)) for i in items_D1_R2[1] for j in items_D2_R2[2]]
data_r2 \
    = data_r2_d10_d20 + data_r2_d10_d21 + data_r2_d10_d22 + data_r2_d10_d23 \
    + data_r2_d11_d20 + data_r2_d11_d21 + data_r2_d11_d22 + data_r2_d11_d23

# Plot the synthetic data.
fig, axes = plt.subplots(ncols=2)
for relation, data, ax in [('R1', data_r1, axes[0]), ('R2', data_r2, axes[1])]:
    X = np.zeros((30, 40))
    for (i, j), v in data:
        X[i,j] = v
    ax.imshow(X, cmap='Greys')
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(X.shape[1]))
    ax.set_yticks(np.arange(X.shape[0]))
    ax.set_title('Raw Data %s' % (relation,))
figname = os.path.join('assets', 'two_relations_hirm.%s.data.png' % (noisy_str))
fig.set_size_inches((4,2))
fig.set_tight_layout(True)
fig.savefig(figname)
print(figname)

# ===== Make an HIRM for both relations.
# Using NOISY=0; seed=108; iters=100 learns cross product.
schema = {'R1': ('D1', 'D2'), 'R2': ('D1', 'D2')}
hirm = HIRM(schema, prng=random.Random(108))
for relation, data in [
        ('R1', data_r1),
        ('R2', data_r2)
    ]:
    for (i, j), v in data:
        hirm.incorporate(relation, (i, j), v)
print(hirm.crp.assignments)

# Run inference.
iters = 100
hirm.set_cluster_assignment_gibbs('R1', 100)
for i in range(iters):
    hirm.transition_cluster_assignments()
    hirm.transition_crp_alpha()
    for irm in hirm.irms.values():
        irm.transition_cluster_assignments()
        irm.transition_crp_alphas()
    print(hirm.crp.assignments)
    print(hirm.logp_score())

# Plot the posterior.
fig, axes = plt.subplots(ncols=2)
for relation, data, ax in [('R1', data_r1, axes[0]), ('R2', data_r2, axes[1])]:
    irm = hirm.relation_to_irm(relation)
    plot_binary_relation(irm.relations[relation], ax=ax)
    score = irm.relations[relation].logp_score()
    ax.set_title('Posterior Sample %s, score %1.2f' % (relation, score,))

figname = os.path.join('assets', 'two_relations_hirm.%s.png' % (noisy_str))
fig.set_size_inches((4,2))
fig.set_tight_layout(True)
fig.savefig(figname)
print(figname)

plt.show()
