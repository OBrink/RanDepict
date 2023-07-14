# -*- coding: utf-8 -*-

"""
RanDepict Python Package.
This repository contains RanDepict,
an easy-to-use utility to generate a big variety of
chemical structure depictions (random depiction styles and image augmentations).


Example:
--------
>>> from RanDepict import RandomDepictor
>>> smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
>>> with RandomDepictor() as depictor:
>>>    image = depictor(smiles)

Have a look in the RanDepictNotebook.ipynb for more examples.

For comments, bug reports or feature ideas,
please raise an issue on the Github repository.

"""

__version__ = "1.2.1"

__all__ = [
    "RanDepict",
]

from .config import RandomDepictorConfig
from .depiction_feature_ranges import DepictionFeatureRanges
from .randepict import RandomDepictor
from .random_markush_structure_generator import RandomMarkushStructureCreator
