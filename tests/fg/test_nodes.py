import numpy as np
import pytest

import pgmax.fg.nodes as nodes


def test_enumfactor_configints_error():
    v = nodes.Variable(3)
    configs = np.array([[1.0]])
    factor_configs_log_potentials = np.array([1.0])

    with pytest.raises(ValueError) as verror:
        nodes.EnumerationFactor(tuple([v]), configs, factor_configs_log_potentials)

    assert "Configurations" in str(verror.value)


def test_enumfactor_potentials_error():
    v = nodes.Variable(3)
    configs = np.array([[1]], dtype=int)
    factor_configs_log_potentials = np.array([1], dtype=int)

    with pytest.raises(ValueError) as verror:
        nodes.EnumerationFactor(tuple([v]), configs, factor_configs_log_potentials)

    assert "Potential" in str(verror.value)


def test_enumfactor_configsshape_error():
    v1 = nodes.Variable(3)
    v2 = nodes.Variable(3)
    configs = np.array([[1]], dtype=int)
    factor_configs_log_potentials = np.array([1.0])

    with pytest.raises(ValueError) as verror:
        nodes.EnumerationFactor(tuple([v1, v2]), configs, factor_configs_log_potentials)

    assert "Number of variables" in str(verror.value)


def test_enumfactor_potentialshape_error():
    v = nodes.Variable(3)
    configs = np.array([[1]], dtype=int)
    factor_configs_log_potentials = np.array([1.0, 2.0])

    with pytest.raises(ValueError) as verror:
        nodes.EnumerationFactor(tuple([v]), configs, factor_configs_log_potentials)

    assert "The potential array has" in str(verror.value)


def test_enumfactor_configvarsize_error():
    v1 = nodes.Variable(3)
    v2 = nodes.Variable(1)
    configs = np.array([[-1, 4]], dtype=int)
    factor_configs_log_potentials = np.array([1.0])

    with pytest.raises(ValueError) as verror:
        nodes.EnumerationFactor(tuple([v1, v2]), configs, factor_configs_log_potentials)

    assert "Invalid configurations for given variables" in str(verror.value)
