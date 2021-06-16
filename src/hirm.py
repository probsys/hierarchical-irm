# Copyright 2021 MIT Probabilistic Computing Project
# Apache License, Version 2.0, refer to LICENSE.txt

import itertools
import math
import random

from scipy.special import betaln
from scipy.special import gammaln

from .util_math import log_choices
from .util_math import log_linspace
from .util_math import logsumexp

INF = float('inf')

class BetaBernoulli:
    def __init__(self, alpha=1, beta=1, prng=None):
        self.alpha = alpha      # Hyperparameter
        self.beta = beta        # Hyperparameter
        self.N = 0              # Number of incorporated observations
        self.s = 0              # Sum of incorporated observations
        self.prng = prng or random
    def incorporate(self, x):
        assert x in [0, 1]
        self.N += 1
        self.s += x
    def unincorporate(self, x):
        assert x in [0, 1]
        self.N -= 1
        self.s -= x
        assert 0 <= self.N
        assert 0 <= self.s
    def logp(self, x):
        log_denom = math.log(self.N + self.alpha + self.beta)
        if x == 1:
            return math.log(self.s + self.alpha) - log_denom
        if x == 0:
            return math.log(self.N - self.s + self.beta) - log_denom
        return -INF
        # assert False, 'Bad value %s' % (repr(x),)
    def logp_score(self):
        n = betaln(self.s + self.alpha, self.N - self.s + self.beta)
        d = betaln(self.alpha, self.beta)
        return n - d
    def sample(self):
        p = math.exp(self.logp(1))
        return self.prng.choices([0, 1], [p, 1-p])[0]
    def __repr__(self):
        return 'BetaBernoulli(alpha=%f, beta=%f, N=%d, s=%d)' \
            % (self.alpha, self.beta, self.N, self.s)
    def __str__(self):
        return repr(self)

class CRP:
    def __init__(self, alpha=1, prng=None):
        self.alpha = alpha          # Concentration parameter.
        self.N = 0                  # Number of customers
        self.tables = {}            # Map from table to set of items
        self.assignments = {}       # Map from item to assigned table
        self.prng = prng or random
    def incorporate(self, item, table):
        assert item not in self.assignments
        if table not in self.tables:
            self.tables[table] = set()
        self.tables[table].add(item)
        self.assignments[item] = table
        self.N += 1
    def unincorporate(self, item):
        table = self.assignments[item]
        self.tables[table].remove(item)
        if not self.tables[table]:
            del self.tables[table]
        del self.assignments[item]
        self.N -= 1
    def sample(self):
        crp_dist = self.tables_weights()
        tables = list(crp_dist.keys())
        weights = crp_dist.values()
        return self.prng.choices(tables, weights=weights)[0]
    def logp(self, table):
        dist = self.tables_weights()
        if table not in dist:
            return -float('inf')
        numer = dist[table]
        denom = self.N + self.alpha
        return math.log(numer) - math.log(denom)
    def logp_score(self):
        # http://gershmanlab.webfactional.com/pubs/GershmanBlei12.pdf#page=4 (eq 8)
        counts = [len(self.tables[t]) for t in self.tables]
        return len(self.tables) * math.log(self.alpha) \
            + sum(gammaln(counts)) \
            + gammaln(self.alpha) \
            - gammaln(self.N + self.alpha)
    def tables_weights(self):
        if self.N == 0:
            return {0: 1}
        crp_dist = {t : len(self.tables[t]) for t in self.tables}
        crp_dist[max(self.tables) + 1] = self.alpha
        return crp_dist
    def tables_weights_gibbs(self, table):
        assert 0 < self.N
        crp_dist = self.tables_weights()
        crp_dist[table] -= 1
        if crp_dist[table] == 0:
            crp_dist[table] = self.alpha
            del crp_dist[max(crp_dist)]
        return crp_dist
    def transition_alpha(self):
        grid = log_linspace(1/self.N, self.N+1, 30)
        log_weights = []
        for g in grid:
            self.alpha = g
            lp_g = self.logp_score()
            log_weights.append(lp_g)
        self.alpha = log_choices(grid, log_weights, prng=self.prng)[0]

    def __repr__(self):
        return 'CRP(alpha=%r, N=%r, tables=%r, assignments=%r)' \
            % (self.alpha, self.N, self.tables, self.assignments)
    def __str__(self):
        return repr(self)

class Domain:
    def __init__(self, name, prng=None):
        self.name = name             # Human-readable string name
        self.items = set()           # Set of items
        self.crp = CRP(prng=prng)    # Clustering model for items
        self.prng = self.crp.prng
    def incorporate(self, item, table=None):
        if item in self.items:
            assert table is None
        if item not in self.items:
            self.items.add(item)
            t = self.crp.sample() if table is None else table
            self.crp.incorporate(item, t)
    def unincorporate(self, item):
        raise NotImplementedError()
        # assert item in self.items
        # self.items[item].remove(relation)
        # if not self.items[item]:
        #     self.crp.unincorporate(item)
        #     del self.items[item]
    def get_cluster_assignment(self, item):
        assert item in self.items
        return self.crp.assignments[item]
    def set_cluster_assignment_gibbs(self, item, table):
        assert item in self.items
        assert self.crp.assignments[item] != table
        self.crp.unincorporate(item)
        self.crp.incorporate(item, table)
    def tables_weights(self):
        return self.crp.tables_weights()
    def tables_weights_gibbs(self, item):
        assert item in self.items
        table = self.get_cluster_assignment(item)
        return self.crp.tables_weights_gibbs(table)

    def __repr__(self):
        return 'Domain(name=%r)' % (self.name,)
    def __str__(self):
        return repr(self)

class Relation:
    def __init__(self, name, domains, prng=None):
        self.name = name                # Name of relation
        self.domains = tuple(domains)   # Domains it is defined over
        self.aux = BetaBernoulli        # TODO: Generalize
        self.clusters = {}              # Map from cluster id to BetaBernoulli
        self.data = {}                  # Map from items to observed value
        self.data_r = {domain.name : {} for domain in self.domains}
                                        # Map from domain name to reverse map
                                        # from item to set of incorporated
                                        # items that include that item
        self.prng = prng or random
    def incorporate(self, items, value):
        assert items not in self.data
        self.data[items] = value
        assert len(items) == len(self.domains)
        for domain, item in zip(self.domains, items):
            domain.incorporate(item)
            if item not in self.data_r[domain.name]:
                self.data_r[domain.name][item] = set()
            self.data_r[domain.name][item].add(items)
        cluster = self.get_cluster_assignment(items)
        if cluster not in self.clusters:
            self.clusters[cluster] = BetaBernoulli()
        self.clusters[cluster].incorporate(value)
    def unincorporate(self, items):
        raise NotImplementedError()
        # x = self.data[items]
        # z = self.get_cluster_assignment(items)
        # self.clusters[z].unincorporate(x)
        # if self.clusters[z].N == 0:
        #     del self.clusters[z]
        # for domain, item in zip(self.domains, items):
        #     if item in self.data_r[domain.name]:
        #         self.data_r[domain.name][item].discard(items)
        #         if len(self.data_r[domain.name][item]) == 0:
        #             del self.data_r[domain.name][item]
        #             domain.unincorporate(self.name, item)
        # del self.data[items]
    def get_cluster_assignment(self, items):
        return tuple((domain.get_cluster_assignment(item))
            for domain, item in zip(self.domains, items))
    def get_cluster_assignment_gibbs(self, items, domain, item, table):
        z = []
        assert len(items) == len(self.domains)
        hits = 0
        for domain_i, item_i in zip(self.domains, items):
            if (domain_i.name == domain.name) and (item_i == item):
                t = table
                hits += 1
            else:
                t = domain_i.get_cluster_assignment(item_i)
            z.append(t)
        assert hits
        return tuple(z)

    # Implementation of approximate Gibbs data probabilities (faster).
    def logp_gibbs_approx_current(self, domain, item):
        """Return approximate proposal probability for current table."""
        logp = 0
        for items in self.data_r[domain.name][item]:
            x = self.data[items]
            z = self.get_cluster_assignment(items)
            self.clusters[z].unincorporate(x)
            lp = self.clusters[z].logp(x)
            self.clusters[z].incorporate(x)
            logp += lp
        return logp
    def logp_gibbs_approx_variant(self, domain, item, table):
        """Return approximate proposal probability for non-current table."""
        logp = 0
        for items in self.data_r[domain.name][item]:
            x = self.data[items]
            z = self.get_cluster_assignment_gibbs(items, domain, item, table)
            cluster = self.clusters.get(z, self.aux())
            lp = cluster.logp(x)
            logp += lp
        return logp
    def logp_gibbs_approx(self, domain, item, table):
        """Return approximate proposal probability of domain.item at table."""
        table_current = domain.get_cluster_assignment(item)
        if table_current == table:
            logp = self.logp_gibbs_approx_current(domain, item)
        else:
            logp = self.logp_gibbs_approx_variant(domain, item, table)
        return logp

    # Implementation of exact Gibbs data probabilities.
    def get_cluster_to_items_list(self, domain, item):
        """Return mapping from cluster to all items in that cluster
        that have domain.item in at least one dimension."""
        cluster_to_items_list = {}
        for items in self.data_r[domain.name][item]:
            cluster = self.get_cluster_assignment(items)
            if cluster not in cluster_to_items_list:
                cluster_to_items_list[cluster] = []
            cluster_to_items_list[cluster].append(items)
        return cluster_to_items_list
    def logp_gibbs_exact_current(self, items_list):
        """Return exact proposal proposal probability for current table."""
        z = self.get_cluster_assignment(items_list[0])
        cluster = self.clusters[z]
        logp0 = cluster.logp_score()
        for items in items_list:
            x = self.data[items]
            # assert z == self.get_cluster_assignment(items)
            cluster.unincorporate(x)
        logp1 = cluster.logp_score()
        for items in items_list:
            x = self.data[items]
            cluster.incorporate(x)
        assert cluster.logp_score() == logp0
        return logp0 - logp1
    def logp_gibbs_exact_variant(self, domain, item, table, items_list):
        """Return exact proposal proposal probability for non-current table."""
        z = self.get_cluster_assignment_gibbs(items_list[0], domain, item, table)
        cluster = self.clusters.get(z, self.aux())
        logp0 = cluster.logp_score()
        for items in items_list:
            # assert z == self.get_cluster_assignment_gibbs(items, domain, item, table)
            x = self.data[items]
            cluster.incorporate(x)
        logp1 = cluster.logp_score()
        for items in items_list:
            # TODO: Skip this loop in case of cluster aux.
            x = self.data[items]
            cluster.unincorporate(x)
        assert cluster.logp_score() == logp0
        return logp1 - logp0
    def logp_gibbs_exact(self, domain, item, tables):
        """Return exact proposal probability of domain.item at tables."""
        # assert tables crp_dist = domain.tables_weights_gibbs(item)
        cluster_to_items_list = self.get_cluster_to_items_list(domain, item)
        table_current = domain.get_cluster_assignment(item)
        logps = []
        for table in tables:
            lp = 0
            for items_list in cluster_to_items_list.values():
                if table == table_current:
                    lp_cluster = self.logp_gibbs_exact_current(items_list)
                else:
                    lp_cluster = self.logp_gibbs_exact_variant(
                        domain, item, table, items_list)
                lp += lp_cluster
            logps.append(lp)
        return logps

    def logp(self, items, value):
        assert len(self.domains) == len(items)
        # TODO: Replace with call logp_observations.
        # XXX Formally, the following assertion is needed for this
        # algorithm to be correct: we should only one fresh item per
        # domain.  Otherwise, the CRP table probabilities are coupled in
        # the predictive.  However, we will assume that we have a
        # "truncated" version of the DPMM with only one auxiliary cluster,
        # where each fresh item has the same probability of belonging to the
        # cluster independently of previous fresh items from that domain.
        # domain_to_item = {}
        # for domain, item in zip(self.domains, items):
        #     assert domain.name not in domain_to_item or domain_to_item[domain.name] == item
        tabl_list = []
        wght_list = []
        indx_list = []
        for domain, item in zip(self.domains, items):
            if item in domain.items:
                t_list = [domain.get_cluster_assignment(item)]
                w_list = [0]
                i_list = [0]
            else:
                tables_weights = domain.tables_weights()
                Z = math.log(1 + domain.crp.N)
                t_list = tuple(tables_weights.keys())
                w_list = tuple(math.log(x) - Z for x in tables_weights.values())
                i_list = tuple(range(len(tables_weights)))
            tabl_list.append(t_list)
            wght_list.append(w_list)
            indx_list.append(i_list)
        logps = []
        for indexes in itertools.product(*indx_list):
            z = tuple(tabl_list[i][j] for i, j in enumerate(indexes))
            w = tuple(wght_list[i][j] for i, j in enumerate(indexes))
            cluster = self.clusters.get(z, self.aux())
            logp_data = cluster.logp(value)
            logp_clst = sum(w)
            logps.append(logp_clst + logp_data)
        return logsumexp(logps)
    def logp_score(self):
        return sum(cluster.logp_score() for cluster in self.clusters.values())
    def set_cluster_assignment_gibbs(self, domain, item, table):
        # More efficient than calling incorporate/unincorporate.
        table_current = domain.get_cluster_assignment(item)
        assert table != table_current
        for items in self.data_r[domain.name][item]:
            x = self.data[items]
            # Remove data point from current cluster.
            z_prev = self.get_cluster_assignment(items)
            cluster_prev = self.clusters[z_prev]
            cluster_prev.unincorporate(x)
            if cluster_prev.N == 0:
                del self.clusters[z_prev]
            # Add data point to new cluster.
            z_new = self.get_cluster_assignment_gibbs(items, domain, item, table)
            assert z_new not in self.clusters or self.clusters[z_new].N > 0
            if z_new not in self.clusters:
                self.clusters[z_new] = self.aux()
            self.clusters[z_new].incorporate(x)
    def has_observation(self, domain, item):
        return item in self.data_r[domain.name]

    def __repr__(self):
        return 'Relation(name=%s, domains=%r)' % (self.name, self.domains,)
    def __str__(self):
        return repr(self)

class IRM:
    def __init__(self, schema, prng=None):
        self.schema = {}
        self.domains = {}
        self.relations = {}
        self.domain_to_relations = {}
        self.prng = prng or random
        for (relation, domains) in schema.items():
            self.add_relation(relation, domains)
    def incorporate(self, r, items, value):
        self.relations[r].incorporate(items, value)
    def unincorporate(self, r, items):
        raise NotImplementedError()
        # self.relations[r].unincorporate(items)
    def transition_cluster_assignments(self, domains=None):
        if domains is None:
            domains = list(self.domains)
            self.prng.shuffle(domains)
        for d in domains:
            items = list(self.domains[d].items)
            self.prng.shuffle(items)
            for item in self.domains[d].items:
                self.transition_cluster_assignment_item(d, item)
    def transition_cluster_assignment_item(self, d, item):
        domain = self.domains[d]
        relations = [self.relations[r] for r in self.domain_to_relations[d]]
        crp_dist = domain.tables_weights_gibbs(item)
        # Compute probability of each table.
        tables = crp_dist.keys()
        logps = [math.log(crp_dist[t]) for t in tables]
        for relation in relations:
            if relation.has_observation(domain, item):
                lp_relation = relation.logp_gibbs_exact(domain, item, tables)
                assert len(lp_relation) == len(tables)
                logps = [x + y for x, y in zip(logps, lp_relation)]
        # Sample new table.
        choice = log_choices(list(crp_dist), logps, prng=self.prng)[0]
        if choice != domain.get_cluster_assignment(item):
            # Update the relations.
            for relation in relations:
                if relation.has_observation(domain, item):
                    relation.set_cluster_assignment_gibbs(domain, item, choice)
            # Update the domain.
            domain.set_cluster_assignment_gibbs(item, choice)
    def transition_crp_alphas(self, domains=None):
        if domains is None:
            domains = list(self.domains)
            self.prng.shuffle(domains)
        for d in domains:
            self.domains[d].crp.transition_alpha()
    def logp(self, observations):
        obs = [(self.relations[r], i, v) for (r, i, v) in observations]
        return logp_observations(obs)
    def logp_score(self):
        logp_score_crp = [self.domains[d].crp.logp_score() for d in self.domains]
        logp_score_relation = [self.relations[r].logp_score() for r in self.relations]
        return sum(logp_score_crp) + sum(logp_score_relation)
    def add_relation(self, r, domains):
        assert r not in self.schema
        assert r not in self.relations
        for d in domains:
            if d not in self.domains:
                self.domains[d] = Domain(d, prng=self.prng)
                self.domain_to_relations[d] = set()
            self.domain_to_relations[d].add(r)
        self.relations[r] = Relation(r, [self.domains[d] for d in domains],
            prng=self.prng)
        self.schema[r] = domains
    def remove_relation(self, r):
        domains = {d.name for d in self.relations[r].domains}
        for d in domains:
            self.domain_to_relations[d].discard(r)
            # TODO: Remove r from self.domains[d].items
            if len(self.domain_to_relations[d]) == 0:
                del self.domain_to_relations[d]
                del self.domains[d]
        del self.relations[r]
        del self.schema[r]

class HIRM:
    def __init__(self, schema, prng=None):
        self.crp = CRP(prng=prng)
        self.schema = {}
        self.irms = {}
        self.prng = prng or random
        for relation, domains in schema.items():
            self.add_relation(relation, domains)
    def incorporate(self, r, items, value):
        irm = self.relation_to_irm(r)
        irm.incorporate(r, items, value)
    def unincorporate(self, r, items):
        irm = self.relation_to_irm(r)
        irm.unincorporate(r, items)
    def relation_to_table(self, r):
        return self.crp.assignments[r]
    def relation_to_irm(self, r):
        table = self.crp.assignments[r]
        return self.irms[table]
    def relation(self, r):
        irm = self.relation_to_irm(r)
        return irm.relations[r]
    def transition_cluster_assignments(self):
        for r in list(self.crp.assignments):
            self.transition_cluster_assignment_relation(r)
    def transition_cluster_assignment_relation(self, r):
        table_current = self.crp.assignments[r]
        relation = self.irms[table_current].relations[r]
        signature = (r, [d.name for d in relation.domains])
        crp_dist = self.crp.tables_weights_gibbs(table_current)
        (table_aux, irm_aux) = (None, None)
        logps = []
        # Compute probabilities of each table.
        for table in crp_dist:
            irm = self.irms.get(table, None)
            if irm is None:
                irm = IRM({}, prng=self.prng)
                assert (table_aux, irm_aux) == (None, None)
                (table_aux, irm_aux) = (table, irm)
            if table != table_current:
                irm.add_relation(signature[0], signature[1])
                for items, value in relation.data.items():
                    irm.incorporate(r, items, value)
            lp_table = irm.relations[r].logp_score()
            logps.append(lp_table)
        # Sample new table.
        log_weights = [math.log(crp_dist[t]) + l for t, l in zip(crp_dist, logps)]
        choice = log_choices(list(crp_dist), log_weights, prng=self.prng)[0]
        # Remove relation from all other tables.
        for table in self.crp.tables:
            if table != choice:
                self.irms[table].remove_relation(r)
            if len(self.irms[table].relations) == 0:
                assert len(self.crp.tables[table]) == 1
                assert table == table_current
                del self.irms[table]
        # Add auxiliary table if necessary.
        if choice == table_aux:
            self.irms[choice] = irm_aux
        # Update the CRP.
        self.crp.unincorporate(r)
        self.crp.incorporate(r, choice)
        assert set(self.irms) == set(self.crp.tables)
    def set_cluster_assignment_gibbs(self, r, table):
        table_current = self.crp.assignments[r]
        assert table != table_current
        relation = self.irms[table_current].relations[r]
        # Remove from current IRM.
        self.irms[table_current].remove_relation(r)
        if len(self.irms[table_current].relations) == 0:
            del self.irms[table_current]
        # Add to target IRM.
        irm = self.irms.get(table, None)
        if irm is None:
            irm = IRM({}, prng=self.prng)
            self.irms[table] = irm
        irm.add_relation(r, [d.name for d in relation.domains])
        for items, value in relation.data.items():
            irm.incorporate(r, items, value)
        # Update CRP.
        self.crp.unincorporate(r)
        self.crp.incorporate(r, table)
        assert set(self.irms) == set(self.crp.tables)
    def transition_crp_alpha(self):
        self.crp.transition_alpha()
    def add_relation(self, r, domains):
        assert r not in self.schema
        self.schema[r] = domains
        table = self.crp.sample()
        self.crp.incorporate(r, table)
        if table in self.irms:
            self.irms[table].add_relation(r, domains)
        else:
            irm = IRM({r : domains}, prng=self.prng)
            self.irms[table] = irm
    def remove_relation(self, r):
        del self.schema[r]
        table = self.crp.assignments[r]
        self.crp.unincorporate(r)
        self.irms[table].remove_relation(r)
        if len(self.irms[table].relations) == 0:
            del self.irms[table]
    def logp(self, observations):
        obs_dict = {}
        for (relation, items, value) in observations:
            table = self.crp.assignments[relation]
            if table not in obs_dict:
                obs_dict[table] = []
            obs_dict[table].append((relation, items, value))
        logps = (self.irms[t].logp(obs_dict[t]) for t in obs_dict)
        return sum(logps)
    def logp_score(self):
        logp_score_crp = self.crp.logp_score()
        logp_score_irms = [irm.logp_score() for irm in self.irms.values()]
        return logp_score_crp + sum(logp_score_irms)

def logp_observations(observations):
    """Observations is a list of (relation, items, value) tuples."""
    # Compute all cluster combinations.
    item_universe = set()
    index_universe = []
    weight_universe = []
    cluster_universe = {}
    seen = set()
    for relation, items, value in observations:
        assert (relation.name, items) not in seen
        seen.add((relation.name, items))
        assert len(items) == len(relation.domains)
        for domain, item in zip(relation.domains, items):
            if (domain.name, item) in item_universe:
                assert (domain.name, item) in cluster_universe
                continue
            if item in domain.items:
                t_list = (domain.get_cluster_assignment(item),)
                w_list = (0,)
                i_list = (0,)
            else:
                tables_weights = domain.tables_weights()
                t_list = tuple(tables_weights.keys())
                Z = math.log(1 + domain.crp.N)
                w_list = tuple(math.log(x) - Z for x in tables_weights.values())
                i_list = tuple(range(len(tables_weights)))
            item_universe.add((domain.name, item))
            index_universe.append(i_list)
            weight_universe.append(w_list)
            loc = len(index_universe) - 1 # location of (domain.name, item)
                                          # within the index universe
            cluster_universe[(domain.name, item)] = (loc, t_list)
    assert len(item_universe) == len(index_universe)
    assert len(item_universe) == len(weight_universe)
    assert len(item_universe) == len(cluster_universe)
    # Compute data probabilities given each cluster combinations.
    # TODO: This implementation can be made more efficient by factoring
    # out relations that do not have any overlapping items.
    logps = []
    for indexes in itertools.product(*index_universe):
        logp_indexes = 0
        # Compute weight of cluster assignments.
        weight = [weight_universe[i][j] for i, j in enumerate(indexes)]
        logp_indexes += sum(weight)
        # Compute weight of data given cluster assignments.
        for relation, items, value in observations:
            z = []
            for domain, item in zip(relation.domains, items):
                loc, t_list = cluster_universe[(domain.name, item)]
                t = t_list[indexes[loc]]
                z.append(t)
            cluster = relation.clusters.get(tuple(z), relation.aux())
            logp_indexes += cluster.logp(value)
        # Add to global list of logps.
        logps.append(logp_indexes)
    return logsumexp(logps)
