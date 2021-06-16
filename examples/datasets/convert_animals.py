# Copyright 2021 MIT Probabilistic Computing Project
# Apache License, Version 2.0, refer to LICENSE.txt

from scipy.io import loadmat

# Animals as a single binary relation"
#   has: Animals x Features -> {0,1}
x = loadmat('50animalbindat.mat')
features = [y[0][0] for y in x['features'].T]
animals = [y[0][0] for y in x['names'].T]
data = x['data']
with open('animals.binary.schema', 'w') as f:
    f.write('bernoulli has feature animal\n')
with open('animals.binary.obs', 'w') as f:
    for i, animal in enumerate(animals):
        for j, feature in enumerate(features):
            value = int(data[i,j])
            a = animal.replace(' ', '')
            f.write('%d has %s %s\n' % (value, feature, a))

with open('animals.unary.schema', 'w') as f:
    for feature in features:
        f.write('bernoulli %s animal\n' % (feature,))
with open('animals.unary.obs', 'w') as f:
    for j, feature in enumerate(features):
        for i, animal in enumerate(animals):
            value = data[i,j]
            a = animal.replace(' ', '')
            f.write('%d %s %s\n' % (value, feature, a))
