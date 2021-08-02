import pytest

import pgmax.interface.datatypes as interface_datatypes


def test_vargroup_direct_instantiation():
    with pytest.raises(NotImplementedError):
        interface_datatypes.VariableGroup(3)


# TODO: Write more tests once interface has become fixed (issues like #36, #39 are resolved)
