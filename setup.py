#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="RanDepict",
    version="1.0.0-dev",
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
    ],
    data_files=[('assets/', ['assets/superatom.txt']), ('assets/jar_files/', ['assets/jar_files/cdk_2_5.jar']), ('assets/arrow_images/straight_arrows/', ['assets/arrow_images/straight_arrows/01_right_arrow_9.png', 'assets/arrow_images/straight_arrows/01_right_arrow_3.png', 'assets/arrow_images/straight_arrows/01_right_arrow_2.png', 'assets/arrow_images/straight_arrows/01_left_right_eq_arrow_2.png', 'assets/arrow_images/straight_arrows/01_right_arrow_12.png', 'assets/arrow_images/straight_arrows/01_right_arrow_11.png', 'assets/arrow_images/straight_arrows/01_right_arrow_7.png', 'assets/arrow_images/straight_arrows/01_left_right_resonance_arrow_1.png', 'assets/arrow_images/straight_arrows/01_right_arrow_5.png', 'assets/arrow_images/straight_arrows/01_left_right_resonance_arrow_3.png', 'assets/arrow_images/straight_arrows/01_right_arrow_8.png', 'assets/arrow_images/straight_arrows/01_left_right_eq_arrow_1.png', 'assets/arrow_images/straight_arrows/01_right_arrow_6.png', 'assets/arrow_images/straight_arrows/01_right_arrow_1.png', 'assets/arrow_images/straight_arrows/01_left_right_eq_arrow_4.png', 'assets/arrow_images/straight_arrows/01_left_right_eq_arrow_5.png', 'assets/arrow_images/straight_arrows/01_left_right_eq_arrow_3.png', 'assets/arrow_images/straight_arrows/01_left_right_resonance_arrow_2.png', 'assets/arrow_images/straight_arrows/01_right_arrow_4.png', 'assets/arrow_images/straight_arrows/01_right_arrow_10.png']), ('assets/arrow_images/curved_arrows/', ['assets/arrow_images/curved_arrows/02_curved_arrow_two_heads_03.png', 'assets/arrow_images/curved_arrows/02_curved_arrow_one_head_02.png', 'assets/arrow_images/curved_arrows/02_curved_arrow_one_head_09.png', 'assets/arrow_images/curved_arrows/02_curved_arrow_two_heads_01.png', 'assets/arrow_images/curved_arrows/02_curved_arrow_two_heads_02.png', 'assets/arrow_images/curved_arrows/02_curved_arrow_one_head_07.png', 'assets/arrow_images/curved_arrows/02_curved_arrow_one_head_05.png', 'assets/arrow_images/curved_arrows/02_curved_arrow_one_head_01.png', 'assets/arrow_images/curved_arrows/02_curved_arrow_one_head_04.png', 'assets/arrow_images/curved_arrows/02_curved_arrow_one_head_06.png', 'assets/arrow_images/curved_arrows/02_curved_arrow_one_head_08.png', 'assets/arrow_images/curved_arrows/02_curved_arrow_one_head_03.png', 'assets/arrow_images/curved_arrows/02_curved_arrow_two_heads_04.png'])],
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
