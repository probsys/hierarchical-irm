# Copyright 2021 MIT Probabilistic Computing Project
# Apache License, Version 2.0, refer to LICENSE.txt

import itertools
import json
import math
import os
import random
import tempfile

import numpy as np

from hirm import IRM
from hirm.util_io import from_dict_IRM
from hirm.util_io import from_txt_irm
from hirm.util_io import to_dict_IRM
from hirm.util_io import to_txt_irm
from hirm.util_math import logsumexp
from hirm.tests.util_test import make_two_clusters

def test_irm_add_relation():
    schema = {
        'Flippers'      : ['Animal'],           # Feature
        'Strain Teeth'  : ['Animal'],           # Feature
        'Swims'         : ['Animal'],           # Feature
        'Arctic'        : ['Animal'],           # Feature
        'Hunts'         : ['Animal', 'Animal'], # Likes relation
    }
    random.seed(1)
    model = IRM({})
    for relation, domains in schema.items():
        model.add_relation(relation, domains)
    model.incorporate('Arctic', ('Bear',), 1)
    model.incorporate('Hunts', ('Bear', 'Bear'), 0)
    model.incorporate('Hunts', ('Bear', 'Fish'), 1)
    model.transition_cluster_assignments()
    # model.unincorporate('Hunts', ('Bear', 'Bear'))
    # model.unincorporate('Hunts', ('Bear', 'Fish'))
    for relation in schema:
        model.remove_relation(relation)
    assert not model.relations
    assert not model.domains
    assert not model.domain_to_relations

def test_irm_two_clusters():
    schema, items_D1, items_D2, data = make_two_clusters()
    irm = IRM(schema, prng=random.Random(1))
    for (i, j), v in data:
        irm.incorporate('R1', (i, j), v)
    # Run inference.
    for i in range(20):
        irm.transition_cluster_assignments()
    assert len(irm.domains['D1'].crp.tables) == 2
    assert set(items_D1[0]) in irm.domains['D1'].crp.tables.values()
    assert set(items_D1[1]) in irm.domains['D1'].crp.tables.values()
    assert set(items_D2[0]) in irm.domains['D2'].crp.tables.values()
    assert set(items_D2[1]) in irm.domains['D2'].crp.tables.values()
    # Check probabilities.
    expected_p0 = {
        (0, 0)     : 1.,
        (0, 10)    : 1.,
        (0, 100)   : .5,
        (10, 0)    : 0.,
        (10, 10)   : 0.,
        (10, 100)  : .5,
        (100, 0)   : .66,
        (100, 10)  : .66,
        (100, 100) : .5
    }
    for x1, x2 in itertools.product([0, 10, 100], [0, 10, 100]):
        p0 = irm.relations['R1'].logp((x1, x2), 0)
        p0_irm = irm.logp((('R1', (x1, x2), 0),))
        assert np.allclose(p0, p0_irm)
        p1 = irm.relations['R1'].logp((x1, x2), 1)
        assert np.allclose(logsumexp([p0, p1]), 0)
        assert abs(math.exp(p0) - expected_p0[(x1, x2)]) < 0.1
    for (x1, x2, x3) in [(0, 10, 100), (110, 10, 100)]:
        p00 = irm.logp([
            ('R1', (x1, x2), 0),
            ('R1', (x1, x3), 0)
        ])
        p01 = irm.logp([
            ('R1', (x1, x2), 0),
            ('R1', (x1, x3), 1)
        ])
        p10 = irm.logp([
            ('R1', (x1, x2), 1),
            ('R1', (x1, x3), 0)
        ])
        p11 = irm.logp([
            ('R1', (x1, x2), 1),
            ('R1', (x1, x3), 1)
        ])
        assert np.allclose(logsumexp([p00, p01, p10, p11]), 0)

def check_irms_agree(irm, x):
    schema, items_D1, items_D2, data = make_two_clusters()
    for d in ['D1', 'D2']:
        assert x.domains[d].crp.assignments == irm.domains[d].crp.assignments
        assert x.domains[d].crp.tables == irm.domains[d].crp.tables
        assert x.domains[d].items == irm.domains[d].items
    assert x.relations['R1'].data == irm.relations['R1'].data
    assert x.relations['R1'].data_r == irm.relations['R1'].data_r
    # Run inference.
    for i in range(20):
        x.transition_cluster_assignments()
    assert len(x.domains['D1'].crp.tables) == 2
    assert set(items_D1[0]) in x.domains['D1'].crp.tables.values()
    assert set(items_D1[1]) in x.domains['D1'].crp.tables.values()
    assert set(items_D2[0]) in x.domains['D2'].crp.tables.values()
    assert set(items_D2[1]) in x.domains['D2'].crp.tables.values()

def test_irm_two_clusters_serialize_json_dict():
    schema, items_D1, items_D2, data = make_two_clusters()
    prng = random.Random(1)
    irm = IRM(schema, prng=prng)
    for (i, j), v in data:
        irm.incorporate('R1', (i, j), v)
    # Serialize the prior IRM to dict and JSON.
    d1 = to_dict_IRM(irm)
    d2 = json.loads(json.dumps(d1))
    irm1 = from_dict_IRM(d1, prng=prng)
    irm2 = from_dict_IRM(d2, prng=prng)
    for x in [irm1, irm2]:
        check_irms_agree(irm, x)

def test_irm_two_clusters_serliaze_txt():
    schema, items_D1, items_D2, data = make_two_clusters()
    prng = random.Random(1)
    irm = IRM(schema, prng=prng)
    for (i, j), v in data:
        irm.incorporate('R1', (i, j), v)
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        path_schema = f.name
        f.write('bernoulli R1 D1 D2\n')
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        path_obs = f.name
        for (i, j), v in data:
            f.write('%d R1 %d %d\n' % (v, i, j))
    with tempfile.NamedTemporaryFile(delete=False) as f:
        path_irm = f.name
        to_txt_irm(path_irm, irm)
    irm1 = from_txt_irm(path_schema, path_obs, path_irm)
    check_irms_agree(irm, irm1)
    os.remove(path_schema)
    os.remove(path_obs)
    os.remove(path_irm)
