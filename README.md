# Hierarchical Infinite Relational Model

[![Actions Status](https://github.com/probcomp/hierarchical-irm/workflows/Python%20package/badge.svg)](https://github.com/probcomp/hierarchical-irm/actions)
[![pypi](https://img.shields.io/pypi/v/hirm.svg)](https://pypi.org/project/hirm/)

This repository contains implementations of the Hierarchical Infinite
Relational Model (HIRM), a Bayesian method for automatic structure discovery in
relational data. The method is described in:

Hierarchical Infinite Relational Model. Saad, Feras A. and Mansinghka, Vikash K.
In: Proc. 37th UAI, 2021.
<https://proceedings.mlr.press/v161/saad21a.html>

## Installation (Python)

This software is tested on Ubuntu 18.04+ and requires a Python 3.6+
environment. The library can be installed from the PyPI repository using

    $ python -m pip install hirm

The library will be available as a module named `hirm`.

The test suite can be invoked via

    $ python -m pytest --pyargs hirm

## Running Python examples

The [examples/](examples) are run using the Python (slower) backend, and in
several cases for fewer MCMC iterations (e.g., <=20) than are needed for
chains to converge. To invoke all the examples, first clone this repository
then run
    
    $ ./check.sh examples

The outputs and plots are written to [examples/assets](examples/assets).
To run a specific example

    $ cd examples
    $ python two_relations.py

## Installation (C++)

First obtain a GNU C++ compiler, version 7.5.0 or higher.
The binary can be installed by first cloning this repository and then writing

    $ cd cxx
    $ make hirm.out

The test suite can be invoked via

    $ make tests

A command-line interface to the HIRM is provided under `cxx/hirm.out`.

For an example of using the C++ library, refer to
[`cxx/tests/test_hirm_animals.cc`](cxx/tests/test_hirm_animals.cc).

## Usage: Command Line Interface

First build the C++ code as described above and then run the binary in
`cxx/hirm.out`. It shows the following synopsis

```
$ ./hirm.out --help
Run a hierarchical infinite relational model.
Usage:
  hirm.out [OPTION...] <path>

      --help         show help message
      --mode arg     options are {irm, hirm} (default: hirm)
      --seed arg     random seed (default: 10)
      --iters arg    number of inference iterations (default: 10)
      --verbose      report results to terminal
      --timeout arg  number of seconds of inference (default: 0)
      --load arg     path to .[h]irm file with initial clusters (default: "")
```

We will explain the usage by way of the following example

    $ cd cxx
    $ ./hirm.out assets/animals.unary
    setting seed to 10
    loading schema from assets/animals.unary.schema
    loading observations from assets/animals.unary.obs
    selected model is HIRM
    incorporating observations
    inferring 10 iters; timeout 0
    saving to assets/animals.unary.10.hirm

In this example we have specified `<path>` to be `assets/animals.unary`.
It is required for there to be two input files on disk:
    1. Schema file: of the form `<path>.schema`.
    2. Observation file: of the form `<path>.obs`.

The output file is `assets/animals.unary.10.hirm`.

We next describe the input and output files.

#### Schema file

The schema file `assets/animals.unary.schema` specifies the signature of
the relations in the system:

```
$ cat assets/animals.unary.schema
bernoulli black animal
bernoulli white animal
bernoulli blue animal
bernoulli brown animal
bernoulli gray animal
bernoulli orange animal
bernoulli red animal
bernoulli yellow animal
bernoulli patches animal
bernoulli spots animal
...
```

Each line specifies the signature of a relation in the system:

- The first entry is the observation type
  (only `bernoulli` is supported at the moment).
- The second entry is the name of the relation (e.g., `black`); all the
  relations names must be unique.
- The third entry is the domain of the relation (in this example, the only
  domain is `animal`).

Thus, for this schema, we have a list of unary relations that each specify
whether an `animal` has a given attribute.

Note that, in general a given relational system can be encoded in multiple
ways. See `assets/animals.binary.schema` for an encoding of this system using
a single higher-order relation with signature: `bernoulli has feature animal`.

#### Observation file

The observation file `assets/animals.unary.obs` specifies realizations of the relations

```
$ cat assets/animals.unary.obs
0 black antelope
1 black grizzlybear
1 black killerwhale
0 black beaver
1 black dalmatian
0 black persiancat
1 black horse
1 black germanshepherd
0 black bluewhale
1 black siamesecat
...
```

Each line specifies a single observation:

- The first entry is 0 or 1
- The second entry is the relation name (there must be a corresponding
  relation with the same name in the schema file)
- The third entry and afterwards are the names of domain entities; e.g,
  `antelope`, `grizzlybear`, etc., are entities in the `animals` domain.
  The number of domain entities must correspond to the arity of the
  relation from the schema file. Since all the relations in this example
  are unary, there is only one entity after each relation name.

Thus, for this observation file, we have observations `black(antelope) = 0`,
`black(grizzlybear) = 1`, and so on.

#### Output file

The output file `assets/animals.unary.10.hirm` specifies the learned
clusterings of relations and domain entities. The output file is comprised
of multiple sections, each delimited by a single blank line.

```
$ cat assets/animals.unary.10.hirm
0 oldworld black insects skimmer chewteeth agility bulbous fast lean orange inactive slow stripes tail red active
1 quadrapedal paws strainteeth pads meatteeth hooves longneck ocean coastal hunter hairless smart group nocturnal meat buckteeth plankton plains timid horns hibernate forager ground grazer furry fields brown solitary stalker toughskin water arctic blue smelly claws swims vegetation fish flippers walks
5 mountains jungle forest bipedal cave desert fierce nestspot tree tusks yellow hands scavenger flys
6 muscle longleg domestic tunnels newworld bush big gray spots strong weak patches white hops small

irm=0
animal 0 giraffe seal horse bat rabbit chimpanzee killerwhale dalmatian mole chihuahua zebra deer lion mouse raccoon dolphin collie bobcat tiger siamesecat germanshepherd otter weasel spidermonkey beaver leopard antelope gorilla fox hamster squirrel wolf rat
animal 1 skunk persiancat giantpanda polarbear moose pig buffalo elephant cow sheep grizzlybear ox humpbackwhale walrus rhinoceros bluewhale hippopotamus

irm=1
animal 0 mouse rabbit zebra moose antelope horse buffalo deer ox cow gorilla pig rhinoceros chimpanzee giraffe sheep spidermonkey elephant
animal 1 collie germanshepherd siamesecat giantpanda chihuahua lion raccoon squirrel grizzlybear dalmatian rat persiancat weasel leopard skunk bobcat mole tiger hamster fox wolf
animal 3 otter walrus humpbackwhale killerwhale bluewhale dolphin seal
animal 4 polarbear bat
animal 5 hippopotamus beaver

irm=5
animal 0 antelope germanshepherd elephant hippopotamus tiger rhinoceros zebra giraffe killerwhale sheep humpbackwhale mole hamster persiancat horse siamesecat chihuahua cow dolphin walrus collie polarbear mouse pig deer moose skunk bluewhale buffalo dalmatian rat beaver ox fox seal rabbit wolf weasel otter
animal 1 squirrel raccoon giantpanda gorilla lion bat spidermonkey chimpanzee grizzlybear bobcat leopard

irm=6
animal 0 horse killerwhale spidermonkey deer giraffe germanshepherd rhinoceros leopard moose fox wolf buffalo dolphin bluewhale grizzlybear chimpanzee walrus lion bobcat zebra beaver elephant ox antelope gorilla hippopotamus humpbackwhale polarbear tiger
animal 1 collie squirrel raccoon chihuahua sheep hamster rabbit rat mouse skunk persiancat weasel mole bat otter siamesecat
animal 2 dalmatian giantpanda cow pig
animal 3 seal
```

The first section in the file specifies the clustering of the relations.
Each line specifies a relation cluster, for example:

```
0 oldworld black insects skimmer chewteeth agility bulbous fast lean orange inactive slow stripes tail red active
```

Here, the first entry is a unique integer code for the cluster index and the
remaining entries are names of relations that belong to this cluster.
We see that there are four relation clusters with indexes `[0, 1, 5, 6]`.

All the remaining sections in the file start with `irm=x`, where `x` is an
integer code from the first section, for example:

```
irm=6
animal 0 horse killerwhale spidermonkey deer giraffe germanshepherd rhinoceros leopard moose fox wolf buffalo dolphin bluewhale grizzlybear chimpanzee walrus lion bobcat zebra beaver elephant ox antelope gorilla hippopotamus humpbackwhale polarbear tiger
animal 1 collie squirrel raccoon chihuahua sheep hamster rabbit rat mouse skunk persiancat weasel mole bat otter siamesecat
animal 2 dalmatian giantpanda cow pig
animal 3 seal
```

Each subsequent line in the `irm=6` section specifies a cluster for a given
domain, for example

```
animal 2 dalmatian giantpanda cow pig
```

Here, the first entry is the name of the domain, the second entry is a
unique integer for the cluster index, and the remaining entries are names
of entities within the domain that belong to this cluster. Recall that the
schema file `assets/animals.unary.schema` has only one domain, so all the
lines in the `irm` section start with `animal`.

## Citation

To cite this work, please use the following BibTeX.

```bibtex
@inproceedings{saad2021hirm,
title           = {Hierarchical Infinite Relational Model},
author          = {Saad, Feras A. and Mansinghka, Vikash K.},
booktitle       = {UAI 2021: Proceedings of the 37th Conference on Uncertainty in Artificial Intelligence},
fseries         = {Proceedings of Machine Learning Research},
year            = 2021,
location        = {Online},
publisher       = {AUAI Press},
address         = {Arlington, VA, USA},
}
```

## License

Copyright (c) 2021 MIT Probabilistic Computing Project

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
