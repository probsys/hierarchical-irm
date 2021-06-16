# Copyright 2021 MIT Probabilistic Computing Project
# Apache License, Version 2.0, refer to LICENSE.txt

import ast

from . import hirm

def intify(x):
    if x.isnumeric():
        assert int(x) == float(x)
        return int(x)
    return x

def load_schema(path):
    """Load a schema from path."""
    signatures = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            assert 3 <= len(parts)
            dist = parts[0]
            assert dist == 'bernoulli'
            feature = parts[1]
            domains = tuple(parts[2:])
            signatures[feature] = domains
    return signatures

def load_observations(path):
    """Load a dataset from path."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            assert 3 <= len(parts)
            x = float(parts[0])
            relation = parts[1]
            items = tuple(intify(x) for x in parts[2:])
            data.append((relation, items, x))
    return data

def load_clusters_irm(path):
    """Load clusters from path."""
    clusters = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            assert 3 <= len(parts)
            domain = parts[0]
            table = int(parts[1])
            items = tuple(intify(x) for x in parts[2:])
            if domain not in clusters:
                clusters[domain] = {}
            clusters[domain][table] = items
    return clusters

def load_clusters_hirm(path):
    """Load clusters from path."""
    irms = {}
    relations = {}
    current_irm = 0
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if parts[0].isnumeric():
                assert 2 <= len(parts)
                table = int(parts[0])
                items = tuple(parts[1:])
                assert table not in relations
                relations[table] = items
                continue
            if len(parts) == 1 and not parts[0]:
                current_irm = None
                continue
            if len(parts) == 1 and parts[0].startswith('irm='):
                assert current_irm is None
                current_irm = int(parts[0].split('=')[1])
                assert current_irm not in irms
                irms[current_irm] = {}
                continue
            if 2 <= len(parts):
                assert current_irm is not None
                assert current_irm in irms
                domain = parts[0]
                table = int(parts[1])
                items = tuple(intify(x) for x in parts[2:])
                if domain not in irms[current_irm]:
                    irms[current_irm][domain] = {}
                assert table not in irms[current_irm][domain]
                irms[current_irm][domain][table] = items
                continue
            assert False, 'Failed to process line'
    assert set(relations) == set(irms)
    return relations, irms

# Serialization to/from JSON compatible dictionaries
# NB: Caveats of json.dumps
#   - dict keys must be string (no tuples)
#   - dict keys that are integers automatically stringified.
#   - tuples automatically converted to listified.
#   - sets are not JSON serializable.

def to_dict_BetaBernoulli(x):
    return {'alpha': x.alpha, 'beta': x.beta, 'N': x.N, 's': x.s}
def from_dict_BetaBernoulli(d, prng=None):
    x = hirm.BetaBernoulli(alpha=d['alpha'], beta=d['beta'], prng=prng)
    x.N = d['N']
    x.s = d['s']
    return x

def to_dict_CRP(x):
    return {
        'alpha': x.alpha,
        'N': x.N,
        'tables': {repr(t): list(v) for t,v in x.tables.items()},
        'assignments': {repr(t): v for t,v in x.assignments.items()}
    }
def from_dict_CRP(d, prng=None):
    x = hirm.CRP(d['alpha'], prng=prng)
    x.N = d['N']
    x.tables = {ast.literal_eval(t): set(v) for t,v in d['tables'].items()}
    x.assignments = {ast.literal_eval(t): v for t,v in d['assignments'].items()}
    return x

def to_dict_Domain(x):
    return {
      'name': x.name,
      'items': list(x.items),
      'crp': to_dict_CRP(x.crp)
    }
def from_dict_Domain(d, prng=None):
    x = hirm.Domain(d['name'], prng=prng)
    x.items = set(d['items'])
    x.crp = from_dict_CRP(d['crp'])
    return x

def to_dict_Relation(x):
    return {
        'name' : x.name,
        'domains' : [d.name for d in x.domains], # Serialize names only.
        'clusters' : {repr(c): to_dict_BetaBernoulli(v) for c,v in x.clusters.items()},
        'data' : {repr(c): v for c,v in x.data.items()},
        'data_r' : {
            repr(k) : {repr(k1): list(v1) for k1, v1 in v.items()}
            for k, v in x.data_r.items()
        }
    }
def from_dict_Relation(d, prng=None):
    x = hirm.Relation(d['name'], [], prng=prng)
    x.domains = d['domains']
    x.clusters = {
        ast.literal_eval(c): from_dict_BetaBernoulli(v, prng=prng)
        for c,v in d['clusters'].items()
    }
    x.data = {ast.literal_eval(c): v for c,v in d['data'].items()}
    x.data_r = {
        ast.literal_eval(k): {
            ast.literal_eval(k1): set(tuple(y) for y in v1)
            for k1,v1 in v.items()
        }
        for k, v in d['data_r'].items()
    }
    return x

def to_dict_IRM(x):
    return {
        'schema': x.schema,
        'domains': {k: to_dict_Domain(v) for k,v in x.domains.items()},
        'relations':  {k: to_dict_Relation(v) for k,v in x.relations.items()},
        'domain_to_relations': {k: list(v) for k,v in x.domain_to_relations.items()}
    }
def from_dict_IRM(d, prng=None):
    x = hirm.IRM({}, prng=prng)
    x.schema = d['schema']
    x.domains = {k: from_dict_Domain(v, prng=prng) for k,v in d['domains'].items()}
    x.relations = {k: from_dict_Relation(v, prng=prng) for k,v in d['relations'].items()}
    x.domain_to_relations = {k: set(v) for k,v in d['domain_to_relations'].items()}
    # Resolve Domain names into Domain objects.
    for relation in x.relations.values():
        relation.domains = tuple([x.domains[d] for d in relation.domains])
    return x

def to_dict_HIRM(x):
    return {
        'schema': x.schema,
        'crp': to_dict_CRP(x.crp),
        'irms' : {k: to_dict_IRM(v) for k, v in x.irms.items()}
    }
def from_dict_HIRM(d, prng=None):
    x = hirm.HIRM({}, prng=prng)
    x.schema = d['schema']
    x.crp = from_dict_CRP(d['crp'], prng=prng)
    x.irms = {int(k): from_dict_IRM(v, prng=prng) for k,v in d['irms'].items()}
    return x

def to_txt_irm(path, irm):
    with open(path, 'w') as f:
        for domain in irm.domains.values():
            tables = sorted(domain.crp.tables)
            for table in tables:
                customers = domain.crp.tables[table]
                customers_str = ' '.join(str(c) for c in customers)
                f.write('%s %d %s' % (domain.name, table, customers_str))
                f.write('\n')

def to_txt_hirm(path, hirm):
    with open(path, 'w') as f:
        tables = sorted(hirm.crp.tables)
        for table in tables:
            customers = hirm.crp.tables[table]
            customers_str = ' '.join(str(c) for c in customers)
            f.write('%d %s' % (table, customers_str))
            f.write('\n')
        f.write('\n')
        j = 0
        for table in tables:
            f.write('irm=%d\n' % (table,))
            irm = hirm.irms[table]
            for domain in irm.domains.values():
                for table, customers in domain.crp.tables.items():
                    customers_str = ' '.join(str(c) for c in customers)
                    f.write('%s %d %s' % (domain.name, table, customers_str))
                    f.write('\n')
            if j != len(hirm.irms) - 1:
                f.write('\n')
                j += 1

def from_txt_irm(path_schema, path_obs, path_clusters):
    schema = load_schema(path_schema)
    observations = load_observations(path_obs)
    clusters = load_clusters_irm(path_clusters)
    irm = hirm.IRM(schema)
    for domain, tables in clusters.items():
        for table, items in tables.items():
            for item in items:
                irm.domains[domain].incorporate(item, table=table)
    for (relation, items, x) in observations:
        irm.incorporate(relation, items, x)
    return irm

def from_txt_hirm(path_schema, path_obs, path_clusters):
    schema = load_schema(path_schema)
    observations = load_observations(path_obs)
    relations, irms = load_clusters_hirm(path_clusters)
    hirmm = hirm.HIRM(schema)
    for table in relations:
        for relation in relations[table]:
            if hirmm.crp.assignments[relation] != table:
                hirmm.set_cluster_assignment_gibbs(relation, table)
        irm = hirmm.irms[table]
        for domain, tables in irms[table].items():
            for t, items in tables.items():
                for item in items:
                    irm.domains[domain].incorporate(item, table=t)
    for (relation, items, x) in observations:
        hirmm.incorporate(relation, items, x)
    return hirmm
