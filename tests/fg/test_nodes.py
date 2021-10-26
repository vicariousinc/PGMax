import re

import numpy as np
import pytest

from pgmax.fg import nodes


def test_enumeration_factor():
    variable = nodes.Variable(3)
    with pytest.raises(ValueError, match="Configurations should be integers. Got"):
        nodes.EnumerationFactor(
            variables=(variable,),
            configs=np.array([[1.0]]),
            log_potentials=np.array([0.0]),
        )

    with pytest.raises(ValueError, match="Potential should be floats. Got"):
        nodes.EnumerationFactor(
            variables=(variable,),
            configs=np.array([[1]]),
            log_potentials=np.array([0]),
        )

    with pytest.raises(ValueError, match="configs should be a 2D array"):
        nodes.EnumerationFactor(
            variables=(variable,),
            configs=np.array([1]),
            log_potentials=np.array([0.0]),
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Number of variables 1 doesn't match given configurations (1, 2)"
        ),
    ):
        nodes.EnumerationFactor(
            variables=(variable,),
            configs=np.array([[1, 2]]),
            log_potentials=np.array([0.0]),
        )

    with pytest.raises(
        ValueError, match=re.escape("Expected log potentials of shape (1,)")
    ):
        nodes.EnumerationFactor(
            variables=(variable,),
            configs=np.array([[1]]),
            log_potentials=np.array([0.0, 1.0]),
        )

    with pytest.raises(ValueError, match="Invalid configurations for given variables"):
        nodes.EnumerationFactor(
            variables=(variable,),
            configs=np.array([[10]]),
            log_potentials=np.array([0.0]),
        )
