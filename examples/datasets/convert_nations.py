# Copyright 2021 MIT Probabilistic Computing Project
# Apache License, Version 2.0, refer to LICENSE.txt

import json
import math

from scipy.io import loadmat

# Nations as two Relations.
x = loadmat('dnations.mat')
attnames = [y[0][0] for y in x['attnames'].T]
relnnames = [y[0][0] for y in x['relnnames'].T]
countrynames = [y[0][0] for y in x['countrynames'].T]
for i, r in enumerate(relnnames):
    if r in attnames:
        relnnames[i] += '_rel'

with open('nations.binary.schema', 'w') as f:
    f.write('bernoulli has feature country\n')
    f.write('bernoulli applies predicate country country\n')

with open('nations.binary.obs', 'w') as f:
    for i, country in enumerate(countrynames):
        for j, feature in enumerate(attnames):
            value = x['A'][i,j]
            if not math.isnan(value):
                f.write('%d has %s %s\n' % (value, feature, country))
    for k, predicate in enumerate(relnnames):
        for i, country0 in enumerate(countrynames):
            for j, country1 in enumerate(countrynames):
                value = x['R'][i,j,k]
                if not math.isnan(value):
                    f.write('%d applies %s %s %s\n' %
                        (value, predicate, country0, country1))

# Nations as multiple Relations.
with open('nations.unary.schema', 'w') as f:
    for feature in attnames:
        f.write('bernoulli %s country\n' % (feature,))
    for predicate in relnnames:
        f.write('bernoulli %s country country\n' % (predicate,))

with open('nations.unary.obs', 'w') as f:
    for j, feature in enumerate(attnames):
        for i, country in enumerate(countrynames):
            value = x['A'][i,j]
            if not math.isnan(value):
                f.write('%d %s %s\n' % (value, feature, country))
    for k, predicate in enumerate(relnnames):
        for i, country0 in enumerate(countrynames):
            for j, country1 in enumerate(countrynames):
                value = x['R'][i,j,k]
                if not math.isnan(value):
                    f.write('%d %s %s %s\n' %
                        (value, predicate, country0, country1))
