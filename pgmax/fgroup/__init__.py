"""A sub-package defining factor groups and containing different types of factor groups."""

from .enum import EnumFactorGroup, PairwiseFactorGroup
from .fgroup import FactorGroup, SingleFactorGroup
from .logical import ANDFactorGroup, ORFactorGroup
