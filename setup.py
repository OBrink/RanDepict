#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="RanDepict",
    version="1.0.2",
    author="Otto Brinkhaus",
    author_email="henning.brinkhaus@uni-jena.de, kohulan.rajan@uni-jena.de",
    maintainer="Otto Brinkhaus, Kohulan Rajan",
    maintainer_email="henning.brinkhaus@uni-jena.de, kohulan.rajan@uni-jena.de",
    description="RanDepict is an easy-to-use utility to generate a big variety of chemical structure depictions (random depiction styles and image augmentations).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OBrink/RanDepict",
    packages=setuptools.find_packages(),
    license="MIT",
    install_requires=[
        "numpy>=1.19",
        "imgaug",
        "scikit-image",
        "epam.indigo",
        "jpype1",
        "ipyplot",
        "rdkit-pypi",
        "imagecorruptions",
        "pillow>=8.2.0",
    ],
    package_data={"RanDepict": ["assets/*.*", "assets/*/*.*", "assets/*/*/*.*"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.5",
)
