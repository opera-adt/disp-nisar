from __future__ import annotations

import importlib.metadata

import disp_nisar as m


def test_version():
    assert importlib.metadata.version("disp_nisar") == m.__version__
