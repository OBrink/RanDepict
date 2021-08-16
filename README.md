[![License](https://img.shields.io/badge/License-MIT%202.0-blue.svg)](https://opensource.org/licenses/MIt)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-blue.svg)](https://github.com/OBrink/RanDepict/graphs/commit-activity)
[![GitHub issues](https://img.shields.io/github/issues/OBrink/RanDepict.svg)](https://GitHub.com/OBrink/RanDepict/issues/)
[![GitHub contributors](https://img.shields.io/github/contributors/OBrink/RanDepict.svg)](https://GitHub.com/OBrink/RanDepict/graphs/contributors/)
[![GitHub release](https://img.shields.io/github/release/OBrink/RanDepict.svg)](https://GitHub.com/OBrink/RanDepict/releases/)
[![PyPI version fury.io](https://badge.fury.io/py/RanDepict.svg)](https://pypi.python.org/pypi/RanDepict/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5205528.svg)](https://doi.org/10.5281/zenodo.5205528)
# RanDepict
This repository contains RanDepict, an easy-to-use utility to generate a big variety of chemical structure depictions (random depiction styles and image augmentations) based on RDKit, CDK and Indigo.

## Usage
-  To use RanDepict, clone the repository to your local disk and make sure you install all the necessary requirements.

##### We recommend to use RanDepict inside a Conda environment to facilitate the installation of the dependencies.
- Conda can be downloaded as part of the [Anaconda](https://www.anaconda.com/) or the [Miniconda](https://conda.io/en/latest/miniconda.html) plattforms (Python 3.7). We recommend to install miniconda3. Using Linux you can get it with:
```shell
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash Miniconda3-latest-Linux-x86_64.sh
```
### Installation

```shell
$ git clone https://github.com/OBrink/RanDepict.git
$ cd RanDepict
$ conda create --name RanDepict python=3.7
$ conda activate RanDepict
$ conda install -c rdkit rdkit
$ conda install pip
$ python -m pip install -U pip #Upgrade pip
$ pip install numpy scikit-image epam.indigo jpype1 ipyplot imagecorruptions imgaug
```
### Alternative
```shell
$ python -m pip install -U pip #Upgrade pip
$ pip install git+https://github.com/OBrink/RanDepict.git
```

### Install from PyPI
```shell
$ pip install RanDepict
```

### Basic usage: 
```python
from RanDepict import random_depictor

smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"

with random_depictor() as depictor:
    image = depictor(smiles)
``` 

Have a look in the RanDepictNotebook.ipynb for more examples and a more detailed documentation.

Here are some examples of depictions of caffeine without augmentations (left) and with augmentations (right) that were automatically created using RanDepict. 

![](caffeine_no_augmentations.gif)   ![](caffeine_augmentations.gif)












## Cite Us
```Bibtex
@misc{Brinkhaus2021,
  title={RanDepict},
  author={Brinkhaus, Henning Otto and Rajan, Kohulan},
  doi = {10.5281/zenodo.5205528},
  year={2021},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/OBrink/RanDepict}},
  url={https://doi.org/10.5281/zenodo.5205528},
 }
 ```
## More information about our research group

[![GitHub Logo](https://github.com/Kohulan/DECIMER-Image-to-SMILES/blob/master/assets/CheminfGit.png?raw=true)](https://cheminf.uni-jena.de)

