"""A sub-package defining different types of groups of factors."""

from .enum import EnumFactorGroup, PairwiseFactorGroup
from .fgroup import FactorGroup, SingleFactorGroup
from .logical import ANDFactorGroup, ORFactorGroup
