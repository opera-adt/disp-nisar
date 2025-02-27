# DISP-NISAR
[![Pytest and build docker image](https://github.com/opera-adt/disp-nisar/actions/workflows/test-build-push.yml/badge.svg?branch=main)](https://github.com/opera-adt/disp-nisar/actions/workflows/test-build-push.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/opera-adt/disp-nisar/main.svg)](https://results.pre-commit.ci/latest/github/opera-adt/disp-nisar/main)

Surface Displacement workflows for OPERA DISP-NISAR products.

Creates the science application software (SAS) using the [dolphin](https://github.com/opera-adt/dolphin) library

# disp-nisar

[![Actions Status][actions-badge]][actions-link]

[![Conda-Forge][conda-badge]][conda-link]
[![GitHub Discussion][github-discussions-badge]][github-discussions-link]


<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/opera-adt/disp-nisar/workflows/CI/badge.svg
[actions-link]:             https://github.com/opera-adt/disp-nisar/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/disp-nisar
[conda-link]:               https://github.com/conda-forge/disp-nisar-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/opera-adt/disp-nisar/discussions

<!-- prettier-ignore-end -->


## Development setup


### Prerequisite installs
1. Download source code:
```bash
git clone https://github.com/isce-framework/dolphin.git
git clone https://github.com/isce-framework/tophu.git
git clone https://github.com/opera-adt/disp-nisar.git
```
2. Install dependencies, either to a new environment:
```bash
mamba env create --name my-disp-env --file disp-nisar/conda-env.yml
conda activate my-disp-env
```
or install within your existing env with mamba.

3. Install `tophu, dolphin` and `disp-nisar` via pip in editable mode
```bash
python -m pip install --no-deps -e dolphin/ tophu/ disp-nisar/
```

### Setup for contributing


We use [pre-commit](https://pre-commit.com/) to automatically run linting, formatting, and [mypy type checking](https://www.mypy-lang.org/).
Additionally, we follow [`numpydoc` conventions for docstrings](https://numpydoc.readthedocs.io/en/latest/format.html).
To install pre-commit locally, run:

```bash
pre-commit install
```
This adds a pre-commit hooks so that linting/formatting is done automatically. If code does not pass the checks, you will be prompted to fix it before committing.
Remember to re-add any files you want to commit which have been altered by `pre-commit`. You can do this by re-running `git add` on the files.

Since we use [black](https://black.readthedocs.io/en/stable/) for formatting and [flake8](https://flake8.pycqa.org/en/latest/) for linting, it can be helpful to install these plugins into your editor so that code gets formatted and linted as you save.

### Running the unit tests

After making functional changes and/or have added new tests, you should run pytest to check that everything is working as expected.

First, install the extra test dependencies:
```bash
python -m pip install --no-deps -e .[test]
```

Then run the tests:

```bash
pytest
```

### Optional GPU setup

To enable GPU support (on aurora with CUDA 11.6 installed), install the following extra packages:
```bash
mamba install -c conda-forge "cudatoolkit=11.6" cupy "pynvml>=11.0"
```


### Building the docker image

To build the docker image, run:
```bash
./docker/build-docker-image.sh --tag my-tag
```
which will print out instructions for running the image.
