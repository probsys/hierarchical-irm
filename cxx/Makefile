# Copyright 2021 MIT Probabilistic Computing Project
# Apache License, Version 2.0, refer to LICENSE.txt

GCC_VERSION := $(shell g++ -dumpversion)
MIN_VERSION := 7

ifeq ($(shell test $(GCC_VERSION) -lt $(MIN_VERSION); echo $$?),0)
$(error ERROR: g++ version $(VERSION) or higher is requried, found $(GCC_VERSION))
endif

CXX=g++

ifdef HIRMDEBUG
CXXFLAGS= -O0 -pg -g -std=c++17
else
CXXFLAGS= -O3 -std=c++17
endif

INCDIR=$(shell pwd)
INCFLAGS=-I$(INCDIR)

# https://www.gnu.org/software/make/manual/html_node/Catalogue-of-Rules.html
LHEADER 	= $(wildcard *.hh)
LSOURCE 	= $(filter-out hirm.cc,$(wildcard *.cc))
LOBJECT 	= $(LSOURCE:.cc=.o)

TEST_DIR=tests
LTEST		= $(wildcard $(TEST_DIR)/*.cc)

# Prevent LOBJECT from being removed.
# https://stackoverflow.com/a/29114706/
.SECONDARY: $(LOBJECT)

%.o : %.cc $(LHEADER)
	$(CXX) -c $(CXXFLAGS) -o $@ $<

%.out : %.cc $(LOBJECT)
	$(CXX) $(CXXFLAGS) $(INCFLAGS) -o $@ $(LOBJECT) $<

.PHONY: tests
tests: $(LTEST:.cc=.out) hirm.out
	./tests/test_misc.out
	./tests/test_hirm_animals.out
	./tests/test_irm_two_relations.out
	./hirm.out --mode=irm --iters=5 assets/animals.binary
	./hirm.out --seed=1 --iters=5 assets/animals.unary
	./hirm.out --iters=5 --load=assets/animals.unary.1.hirm assets/animals.unary

# Make sure -pg is not in CXXFLAGS
# since valgrind finds leaks from gmon
.PHONY: leak-check clean
leak-check: $(TEST_DIR)/test_misc.out
ifdef HIRMDEBUG
$(error ERROR: Cannot run Valgrind with HIRMDEBUG)
endif
	valgrind \
		--leak-check=full \
		--show-leak-kinds=all \
		--track-origins=yes \
		--verbose \
		./$<
clean:
	rm -rf **.o *.out *.gch
	rm -rf tests/*.out
	rm -rf assets/*.irm
	rm -rf assets/*.hirm
