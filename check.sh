#!/bin/sh

# Copyright 2021 MIT Probabilistic Computing Project
# Apache License, Version 2.0, refer to LICENSE.txt

set -Ceux

: ${PYTHON:=python}

root=$(cd -- "$(dirname -- "$0")" && pwd)

(
    set -Ceu
    cd -- "${root}"
    rm -rf build
    "$PYTHON" setup.py build
    if [ $# -eq 0 ]; then
        # (Default) Run tests/
        ./pythenv.sh "$PYTHON" -m pytest --pyargs hirm
        cd cxx && make tests
    elif [ ${1} = 'coverage' ]; then
        # Generate coverage report.
        ./pythenv.sh coverage run --source=build/ -m pytest --pyargs hirm
        coverage html
        coverage report
    elif [ ${1} = 'examples' ]; then
        # Run the .py files under examples/
        cd examples
        for x in *.py; do
            MPLBACKEND=agg python "${x}" || continue
        done
    elif [ ${1} = 'release' ]; then
        # Make a release to pypi
        rm -rf dist
        "$PYTHON" setup.py sdist bdist_wheel
        twine upload --repository pypi dist/*
    elif [ ${1} = 'tag' ]; then
        # Make a tagged release, e.g., ./check.sh 2.0.0
        status="$(git diff --stat && git diff --staged)"
        [ -z "${status}" ] || (echo 'fatal: tag dirty' && exit 1)
        tag="${2}"
        sed -i "s/__version__ = .*/__version__ = '${tag}'/g" -- src/__init__.py
        git add -- src/__init__.py
        git commit -m "Pin version ${tag}."
        git tag -a -m v"${tag}" v"${tag}"
    else
        # If args are specified delegate control to user.
        ./pythenv.sh "$PYTHON" -m pytest "$@"
    fi
)
