# RanDepict
This repository contains RanDepict, an easy-to-use utility to generate a big variety of chemical structure depictions (random depiction styles and image augmentations).

## Basic usage: 
```
from RanDepict import random_depictor

smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"

with random_depictor() as depictor:
    image = depictor(smiles)
``` 

Have a look in the RanDepictNotebook.ipynb for more examples and a more detailed documentation.


## TODO: Set up instructions to build an environment to run it.
