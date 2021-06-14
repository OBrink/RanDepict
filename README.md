# RanDepict
This repository contains RanDepict, an easy-to-use utility to generate a big variety of chemical structure depictions (random depiction styles and image augmentations).

## Usage
-  To use RanDepict, clone the repository to your local disk and make sure you install all the necessary requirements.

##### We recommend to use RanDepict inside a Conda environment to facilitate the installation of the dependencies.
- Conda can be downloaded as part of the [Anaconda](https://www.anaconda.com/) or the [Miniconda](https://conda.io/en/latest/miniconda.html) plattforms (Python 3.7). We recommend to install miniconda3. Using Linux you can get it with:
```
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash Miniconda3-latest-Linux-x86_64.sh
```
### Installation

```
$ git clone https://github.com/OBrink/RanDepict.git
$ cd RanDepict
$ conda create --name RanDepict python=3.7
$ conda activate RanDepict
$ conda install -c rdkit rdkit
$ conda install pip
$ python -m pip install -U pip #Upgrade pip
$ pip install numpy scikit-image epam.indigo jpype1
```

### Basic usage: 
```
from RanDepict import random_depictor

smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"

with random_depictor() as depictor:
    image = depictor(smiles)
``` 

Have a look in the RanDepictNotebook.ipynb for more examples and a more detailed documentation.


## TODO: Set up instructions to build an environment to run it.
