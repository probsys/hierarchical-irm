# Copyright 2021 MIT Probabilistic Computing Project
# Apache License, Version 2.0, refer to LICENSE.txt

import os
import sys

import matplotlib.pyplot as plt

from hirm.util_io import from_txt_hirm
from hirm.util_plot import plot_binary_relation

xlabels = {'R1': '', 'R2': '$D_2$', 'R3': '$D_1$'}
ylabels = {'R1': '', 'R2': '$D_1$', 'R3': '$D_3$'}

def plot_hirm_clusters(hirm, figname):
    fig, axes = plt.subplots(nrows=2, ncols=2)
    axes[0,0].set_axis_off()

    bbox = {'facecolor': 'red', 'alpha': 1, 'edgecolor':'k'}

    irm_R1 = hirm.relation_to_irm('R1')
    plot_binary_relation(irm_R1.relations['R1'], ax=axes[1,1])
    score1 = irm_R1.relations['R1'].logp_score()
    axes[1,1].xaxis.set_label_position('top')
    axes[1,1].set_xlabel(xlabels['R1'])
    axes[1,1].set_ylabel(ylabels['R1'], rotation=0, labelpad=10)
    axes[1,1].set_xticks([])
    axes[1,1].set_yticks([])
    axes[1,1].text(.05, .95, '$R_1$', ha='left', va='top',
        transform=axes[1,1].transAxes, bbox=bbox)

    irm_R2 = hirm.relation_to_irm('R2')
    plot_binary_relation(irm_R2.relations['R2'], ax=axes[1,0])
    score2 = irm_R2.relations['R2'].logp_score()
    axes[1,0].xaxis.set_label_position('top')
    axes[1,0].set_xlabel(xlabels['R2'])
    axes[1,0].set_ylabel(ylabels['R2'], rotation=0, labelpad=10)
    axes[1,0].set_xticks([])
    axes[1,0].set_yticks([])
    axes[1,0].text(.05, .95, '$R_2$', ha='left', va='top',
        transform=axes[1,0].transAxes, bbox=bbox)

    irm_R3 = hirm.relation_to_irm('R3')
    plot_binary_relation(irm_R3.relations['R3'], ax=axes[0,1], transpose=1)
    score3 = irm_R3.relations['R3'].logp_score()
    axes[0,1].xaxis.set_label_position('top')
    axes[0,1].set_xlabel(xlabels['R3'])
    axes[0,1].set_ylabel(ylabels['R3'], rotation=0, labelpad=10)
    axes[0,1].set_xticks([])
    axes[0,1].set_yticks([])
    axes[0,1].text(.05, .95, '$R_3$', ha='left', va='top',
        transform=axes[0,1].transAxes, bbox=bbox)

    print(score1, score2, score3)

    fig.set_size_inches((3.5, 3.5))
    fig.subplots_adjust(wspace=.1, hspace=.1)
    fig.savefig(figname)
    print(figname)

if __name__ == '__main__':
    path_clusters = sys.argv[1]
    path_schema = os.path.join('assets', 'three_relations.schema')
    path_obs = os.path.join('assets', 'three_relations.obs')
    hirm = from_txt_hirm(path_schema, path_obs, path_clusters)
    basename = os.path.basename(path_clusters)
    figname = os.path.join('assets', '%s.png' % (basename,))
    plot_hirm_clusters(hirm, figname)
