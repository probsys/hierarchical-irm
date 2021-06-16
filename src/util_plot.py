# Copyright 2021 MIT Probabilistic Computing Project
# Apache License, Version 2.0, refer to LICENSE.txt

import copy

import matplotlib.pyplot as plt
import numpy as np

### Generic plotting functions
### ==========================

def get_fig_ax(ax=None):
    """Create new fig, ax for unspecified axes."""
    if ax is None:
        fig, ax = plt.subplots()
    return ax.get_figure(), ax

def sort_crp_tables(tables):
    """Sort cluster assignments by number"""
    keys = sorted(tables,
        key=lambda t: (len(tables[t]), min(tables[t])),
        reverse=True)
    items = [item for table in keys for item in tables[table]]
    dividers = [len(tables[table]) for table in keys]
    return (items, np.cumsum(dividers))

def sort_data_binary_relation(data, tables_rows, tables_cols):
    """Sort rows and columns of a binary relation by cluster size."""
    (items_rows, dividers_rows) = sort_crp_tables(tables_rows)
    (items_cols, dividers_cols) = sort_crp_tables(tables_cols)
    X = np.asarray([
        [data.get((i,j), np.nan) for j in items_cols]
        for i in items_rows
    ])
    return (X, (items_rows, items_cols), (dividers_rows, dividers_cols))

def sort_data_ternary_relation(data, predicate, tables_rows, tables_cols):
    """Sort rows and columns of a ternary relation by cluster size."""
    (items_rows, dividers_rows) = sort_crp_tables(tables_rows)
    (items_cols, dividers_cols) = sort_crp_tables(tables_cols)
    X = np.asarray([
        [data.get((predicate, i,j), np.nan) for j in items_cols]
        for i in items_rows
    ])
    return (X, (items_rows, items_cols), (dividers_rows, dividers_cols))

def plot_data_matrix_sorted(X, items, dividers, transpose=None, ax=None):
    """Plot clustered 2D matrix."""
    # Adapted from https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/image_annotated_heatmap.html
    fig, ax = get_fig_ax(ax)
    X = X.T if transpose else X
    items_rows, items_cols = items[::-1] if transpose else items
    dividers_rows, dividers_cols = dividers[::-1] if transpose else dividers
    imshow(X, items_rows, items_cols, dividers_rows, dividers_cols, ax)
    return fig, ax

def imshow(X, items_rows, items_cols, dividers_rows, dividers_cols, ax):
    """Main function for rendering an IRM matrix."""
    cmap = copy.copy(plt.get_cmap('Greys'))
    cmap.set_bad(color='gray')
    # Use aspect='auto'
    # https://stackoverflow.com/q/44654421/1405543
    ax.imshow(X, cmap=cmap)
    # Set ticks.
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(X.shape[1]))
    ax.set_yticks(np.arange(X.shape[0]))
    ax.set_yticklabels(items_rows)
    ax.set_xticklabels(items_cols, rotation=90)
    if len(dividers_rows) > 1:
        for i in dividers_rows[:-1]:
            ax.axhline(i-.5, color='r', linewidth=2)
    if len(dividers_cols) > 1:
        for i in dividers_cols[:-1]:
            ax.axvline(i-.5, color='r', linewidth=2)
    # Make a thin grid on the minor axis.
    # ax.set_xticks(np.arange(X.shape[1]+1)-.5, minor=True)
    # ax.set_yticks(np.arange(X.shape[0]+1)-.5, minor=True)
    # ax.grid(which='minor', color='green', linestyle='-', linewidth=1)
    # ax.tick_params(which='minor', top=False, left=False)

### Relation plotting functions
### ===========================

def sort_binary_relation(relation):
    """Sort rows and columns of a binary relation by cluster size."""
    assert len(relation.domains) == 2
    tables_rows = relation.domains[0].crp.tables
    tables_cols = relation.domains[1].crp.tables
    data = relation.data
    return sort_data_binary_relation(data, tables_rows, tables_cols)

def sort_ternary_relation(relation, predicate):
    """Sort rows and columns of a ternary relation by cluster size."""
    assert len(relation.domains) == 3
    tables_rows = relation.domains[1].crp.tables
    tables_cols = relation.domains[2].crp.tables
    data = relation.data
    return sort_data_ternary_relation(data, predicate, tables_rows, tables_cols)

def sort_unary_relations(relations):
    domain = relations[0].domains[0]
    assert all(len(relation.domains) == 1 for relation in relations)
    assert all(relation.domains[0] is domain for relation in relations)
    items_rows, dividers_rows = sort_crp_tables(domain.crp.tables)
    items_cols = [relation.name for relation in relations]
    X = np.asarray([
        [relation.data.get((i,), np.nan) for relation in relations]
        for i in items_rows
    ])
    return X, (items_rows, dividers_rows), (items_cols, [])

def plot_binary_relation(relation, transpose=None, ax=None):
    """Plot matrix for a ternary relation, curried at first value = predicate."""
    X, items, dividers = sort_binary_relation(relation)
    return plot_data_matrix_sorted(X, items, dividers, transpose=transpose, ax=ax)

def plot_ternary_relation(relation, predicate, transpose=None, ax=None):
    """Plot matrix for a ternary relation, curried at first value = predicate."""
    X, items, dividers = sort_ternary_relation(relation, predicate)
    return plot_data_matrix_sorted(X, items, dividers, transpose=transpose, ax=ax)

def plot_unary_relations(relations, ax=None):
    """Plot partition of unary 'relations' learned by IRM, ala DPMM."""
    fig, ax = get_fig_ax(ax)
    X, (items_rows, dividers_rows), (items_cols, dividers_cols) \
        = sort_unary_relations(relations)
    imshow(X, items_rows, items_cols, dividers_rows, dividers_cols, ax)
    return fig, ax

def plot_hirm_crosscat(hirm, relations):
    """Plot partition of unary 'relations' learned by HIRM, ala CrossCat."""
    domain = hirm.relation(relations[0]).domains[0]
    for r in relations:
        assert len(hirm.relation(r).domains) == 1
        assert hirm.relation(r).domains[0].name == domain.name
    tables = set([hirm.crp.assignments[relation] for relation in relations])
    fig, axes = plt.subplots(ncols=len(tables))
    axes = np.atleast_1d(axes)
    for table, ax in zip(tables, axes):
        relations_table = [
            hirm.relation(r) for r in relations
            if hirm.crp.assignments[r] == table
        ]
        X, (items_rows, dividers_rows), (items_cols, dividers_cols) \
            = sort_unary_relations(relations_table)
        imshow(X, items_rows, items_cols, dividers_rows, dividers_cols, ax)
    for ax in axes:
        ax.set_aspect('auto')
    return fig, axes
