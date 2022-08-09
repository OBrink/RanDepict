import os
import pathlib
import numpy as np
import io
from skimage import io as sk_io
from skimage.color import rgba2rgb, rgb2gray
from skimage.util import img_as_ubyte, img_as_float
from PIL import Image, ImageFont, ImageDraw, ImageStat, ImageEnhance
from multiprocessing import set_start_method, get_context
import imgaug.augmenters as iaa
import random
from copy import deepcopy
from typing import Tuple, List, Dict, Any, Callable
import re

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdAbbreviations import CondenseMolAbbreviations
from rdkit.Chem.rdAbbreviations import GetDefaultAbbreviations
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import DataStructs
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
from itertools import product

from indigo import Indigo
from indigo.renderer import IndigoRenderer
from jpype import startJVM, getDefaultJVMPath
from jpype import JClass, JVMNotFoundException, isJVMStarted
from pikachu.drawing import drawing
from pikachu.smiles.smiles import read_smiles
import base64

import cv2
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


class RandomDepictor:
    """
    This class contains everything necessary to generate a variety of
    random depictions with given SMILES strings. An instance of RandomDepictor
    can be called with a SMILES str and returns an np.array that represents
    the RGB image with the given chemical structure.
    """

    def __init__(self, seed: int = 42, hand_drawn: bool = False):
        """
        Load the JVM only once, load superatom list (OSRA),
        set context for multiprocessing
        """
        self.HERE = pathlib.Path(__file__).resolve().parent.joinpath("assets")

        # Start the JVM to access Java classes
        try:
            self.jvmPath = getDefaultJVMPath()
        except JVMNotFoundException:
            print(
                "If you see this message, for some reason JPype",
                "cannot find jvm.dll.",
                "This indicates that the environment varibale JAVA_HOME",
                "is not set properly.",
                "You can set it or set it manually in the code",
                "(see __init__() of RandomDepictor)",
            )
            self.jvmPath = "Define/path/or/set/JAVA_HOME/variable/properly"
        if not isJVMStarted():
            self.jar_path = self.HERE.joinpath("jar_files/cdk_2_5.jar")
            startJVM(self.jvmPath, "-ea", "-Djava.class.path=" + str(self.jar_path))

        self.seed = seed
        self.hand_drawn = hand_drawn
        random.seed(self.seed)

        # Load list of superatoms for label generation
        with open(self.HERE.joinpath("superatom.txt")) as superatoms:
            superatoms = superatoms.readlines()
            self.superatoms = [s[:-2] for s in superatoms]

        # Define PIL resizing methods to choose from:
        self.PIL_resize_methods = [
            Image.NEAREST,
            Image.BOX,
            Image.BILINEAR,
            Image.HAMMING,
            Image.BICUBIC,
            Image.LANCZOS,
        ]

        self.PIL_HQ_resize_methods = self.PIL_resize_methods[4:]

        self.from_fingerprint = False
        self.depiction_features = False

        # Set context for multiprocessing but make sure this only happens once
        try:
            set_start_method("spawn")
        except RuntimeError:
            pass

    def __call__(
        self,
        smiles: str,
        shape: Tuple[int, int, int] = (299, 299),
        grayscale: bool = False,
        hand_drawn: bool = False,
    ):
        # Depict structure with random parameters
        hand_drawn = self.hand_drawn
        if hand_drawn:
            depiction = self.random_depiction(smiles, shape)

        else:
            depiction = self.random_depiction(smiles, shape)
            # Add augmentations
            depiction = self.add_augmentations(depiction)

        if grayscale:
            return self.to_grayscale_float_img(depiction)
        return depiction

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        # I'd like to automatically close the JVM
        # But if it is closed once, you cannot reopen it
        # (for example when someone works in a IPython notebook)
        # Shutdown the JVM
        # shutdownJVM()
        pass

    def random_choice(self, iterable: List, log_attribute: str = False):
        """
        This function takes an iterable, calls random.choice() on it,
        increases random.seed by 1 and returns the result. This way, results
        produced by RanDepict are replicable.

        Additionally, this function handles the generation of depictions and
        augmentations from given fingerprints by handling all random decisions
        according to the fingerprint template.

        Args:
            iterable (List): iterable to pick from
            log_attribute (str, optional): ID for fingerprint.
                                           Defaults to False.

        Returns:
            Any: "Randomly" picked element
        """
        # Keep track of seed and change it with every pseudo-random decision.
        self.seed += 1
        random.seed(self.seed)

        # Generation from fingerprint:
        if self.from_fingerprint and log_attribute:
            # Get dictionaries that define positions and linked conditions
            pos_cond_dicts = self.active_scheme[log_attribute]
            for pos_cond_dict in pos_cond_dicts:
                pos = pos_cond_dict["position"]
                cond = pos_cond_dict["one_if"]
                if self.active_fingerprint[pos]:
                    # If the condition is a range: adapt iterable and go on
                    if isinstance(cond, tuple):
                        iterable = [
                            item
                            for item in iterable
                            if item > cond[0] - 0.001
                            if item < cond[1] + 0.001
                        ]
                        break
                    # Otherwise, simply return the condition value
                    else:
                        return cond
        # Pseudo-randomly pick an element from the iterable
        result = random.choice(iterable)

        return result

    def random_image_size(self, shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        This function takes a random image shape and returns an image shape
        where the first two dimensions are slightly distorted
        (90-110% of original value).

        Args:
            shape (Tuple[int, int]): original shape

        Returns:
            Tuple[int, int]: distorted shape
        """
        # Set random depiction image shape (to cause a slight distortion)
        y = int(shape[0] * self.random_choice(np.arange(0.9, 1.1, 0.02)))
        x = int(shape[1] * self.random_choice(np.arange(0.9, 1.1, 0.02)))
        return y, x

    def get_random_pikachu_rendering_settings(
        self, shape: Tuple[int, int] = (299, 299)
    ) -> drawing.Options:
        """
        This function defines random rendering options for the structure
        depictions created using PIKAChU.
        It returns an pikachu.drawing.drawing.Options object with the settings.

        Args:
            shape (Tuple[int, int], optional): im shape. Defaults to (299, 299)

        Returns:
            options: Options object that contains depictions settings
        """
        options = drawing.Options()
        options.height, options.width = shape
        options.bond_thickness = self.random_choice(np.arange(0.5, 2.2, 0.1))
        options.bond_length = self.random_choice(np.arange(10, 25, 1))
        options.chiral_bond_width = options.bond_length * self.random_choice(
            np.arange(0.05, 0.2, 0.01)
        )
        options.short_bond_length = self.random_choice(np.arange(0.2, 0.6, 0.05))
        options.double_bond_length = self.random_choice(np.arange(0.6, 0.8, 0.05))
        options.bond_spacing = options.bond_length * self.random_choice(
            np.arange(0.15, 0.28, 0.01)
        )
        options.padding = self.random_choice(np.arange(10, 50, 5))
        # options.font_size_large = 5
        # options.font_size_small = 3
        return options

    def hand_drawn_augment(self, img) -> np.array:
        """
        This function randomly applies different image augmentations with
        different probabilities to the input image.

        It has been modified from the original augment.py present on
        https://github.com/mtzgroup/ChemPixCH

        From the publication:
        https://pubs.rsc.org/en/content/articlelanding/2021/SC/D1SC02957F

        Args:
            img: the image to modify in array format.
        Returns:
            img: the augmented image.
        """
        # resize
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.5:
            img = self.resize_hd(img)
        # blur
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.4:
            img = self.blur(img)
        # erode
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.4:
            img = self.erode(img)
        # dilate
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.4:
            img = self.dilate(img)
        # aspect_ratio
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.7:
            img = self.aspect_ratio(img, "mol")
        # affine
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.7:
            img = self.affine(img, "mol")
        # distort
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.8:
            img = self.distort(img)
        if img.shape != (255, 255, 3):
            img = cv2.resize(img, (256, 256))
        return img

    def augment_bkg(self, img) -> np.array:
        """
        This function randomly applies different image augmentations with
        different probabilities to the input image.
        Args:
            img: the image to modify in array format.
        Returns:
            img: the augmented image.
        """
        # rotate
        rows, cols, _ = img.shape
        angle = self.random_choice(np.arange(0, 360))
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        img = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
        # resize
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.5:
            img = self.resize_hd(img)
        # blur
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.4:
            img = self.blur(img)
        # erode
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.2:
            img = self.erode(img)
        # dilate
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.2:
            img = self.dilate(img)
        # aspect_ratio
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.3:
            img = self.aspect_ratio(img, "bkg")
        # affine
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.3:
            img = self.affine(img, "bkg")
        # distort
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.8:
            img = self.distort(img)
        if img.shape != (255, 255, 3):
            img = cv2.resize(img, (256, 256))
        return img

    def resize_hd(self, img) -> np.array:
        """
        This function resizes the image randomly from between (200-300, 200-300)
        and then resizes it back to 256x256.
        Args:
            img: the image to modify in array format.
        Returns:
            img: the resized image.
        """
        interpolations = [
            cv2.INTER_NEAREST,
            cv2.INTER_AREA,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ]

        img = cv2.resize(
            img,
            (
                self.random_choice(np.arange(200, 300)),
                self.random_choice(np.arange(200, 300)),
            ),
            interpolation=self.random_choice(interpolations),
        )
        img = cv2.resize(
            img, (256, 256), interpolation=self.random_choice(interpolations)
        )

        return img

    def blur(self, img) -> np.array:
        """
        This function blurs the image randomly between 1-3.
        Args:
            img: the image to modify in array format.
        Returns:
            img: the blurred image.
        """
        n = self.random_choice(np.arange(1, 4))
        kernel = np.ones((n, n), np.float32) / n**2
        img = cv2.filter2D(img, -1, kernel)
        return img

    def erode(self, img) -> np.array:
        """
        This function bolds the image randomly between 1-2.
        Args:
           img: the image to modify in array format.
        Returns:
            img: the bold image.
        """
        n = self.random_choice(np.arange(1, 3))
        kernel = np.ones((n, n), np.float32) / n**2
        img = cv2.erode(img, kernel, iterations=1)
        return img

    def dilate(self, img) -> np.array:
        """
        This function dilates the image with a factor of 2.
        Args:
           img: the image to modify in array format.
        Returns:
            img: the dilated image.
        """
        n = 2
        kernel = np.ones((n, n), np.float32) / n**2
        img = cv2.dilate(img, kernel, iterations=1)
        return img

    def aspect_ratio(self, img, obj=None) -> np.array:
        """
        This function irregularly changes the size of the image
        and converts it back to (256,256).
        Args:
            img: the image to modify in array format.
            obj: "mol" or "bkg" to modify a chemical structure image or
                 a background image.
        Returns:
            image: the resized image.
        """
        n1 = self.random_choice(np.arange(0, 50))
        n2 = self.random_choice(np.arange(0, 50))
        n3 = self.random_choice(np.arange(0, 50))
        n4 = self.random_choice(np.arange(0, 50))
        if obj == "mol":
            image = cv2.copyMakeBorder(
                img, n1, n2, n3, n4, cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )
        elif obj == "bkg":
            image = cv2.copyMakeBorder(img, n1, n2, n3, n4, cv2.BORDER_REFLECT)

        image = cv2.resize(image, (256, 256))
        return image

    def affine(self, img, obj=None) -> np.array:
        """
        This function randomly applies affine transformation which consists
        of matrix rotations, translations and scale operations and converts
        it back to (256,256).
        Args:
            img: the image to modify in array format.
            obj: "mol" or "bkg" to modify a chemical structure image or
                 a background image.
        Returns:
            skewed: the transformed image.
        """
        rows, cols, _ = img.shape
        n = 20
        pts1 = np.float32([[5, 50], [200, 50], [50, 200]])
        pts2 = np.float32(
            [
                [
                    5 + self.random_choice(np.arange(-n, n)),
                    50 + self.random_choice(np.arange(-n, n)),
                ],
                [
                    200 + self.random_choice(np.arange(-n, n)),
                    50 + self.random_choice(np.arange(-n, n)),
                ],
                [
                    50 + self.random_choice(np.arange(-n, n)),
                    200 + self.random_choice(np.arange(-n, n)),
                ],
            ]
        )

        M = cv2.getAffineTransform(pts1, pts2)

        if obj == "mol":
            skewed = cv2.warpAffine(img, M, (cols, rows), borderValue=[255, 255, 255])
        elif obj == "bkg":
            skewed = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)

        skewed = cv2.resize(skewed, (256, 256))
        return skewed

    def elastic_transform(self, image, alpha_sigma) -> np.array:
        """
        Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.
        https://gist.github.com/erniejunior/601cdf56d2b424757de5
        This function distords an image randomly changing the alpha and gamma
        values.
        Args:
            image: the image to modify in array format.
            alpha_sigma: alpha and sigma values randomly selected as a list.
        Returns:
            distored_image: the image after the transformation with the same size
                            as it had originally.
        """
        alpha = alpha_sigma[0]
        sigma = alpha_sigma[1]
        random_state = np.random.RandomState(self.random_choice(np.arange(1, 1000)))

        shape = image.shape
        dx = (
            gaussian_filter(
                (random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0
            )
            * alpha
        )
        random_state = np.random.RandomState(self.random_choice(np.arange(1, 1000)))
        dy = (
            gaussian_filter(
                (random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0
            )
            * alpha
        )

        x, y, z = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2])
        )
        indices = (
            np.reshape(y + dy, (-1, 1)),
            np.reshape(x + dx, (-1, 1)),
            np.reshape(z, (-1, 1)),
        )

        distored_image = map_coordinates(
            image, indices, order=self.random_choice(np.arange(1, 5)), mode="reflect"
        )
        return distored_image.reshape(image.shape)

    def distort(self, img) -> np.array:
        """
        This function randomly selects a list with the shape [a, g] where
        a=alpha and g=gamma and passes them along with the input image
        to the elastic_transform function that will do the image distorsion.
        Args:
            img: the image to modify in array format.
        Returns:
            the output from elastic_transform function which is the image
            after the transformation with the same size as it had originally.
        """
        sigma_alpha = [
            (self.random_choice(np.arange(9, 11)), self.random_choice(np.arange(2, 4))),
            (self.random_choice(np.arange(80, 100)), 4),
            (self.random_choice(np.arange(150, 300)), 5),
            (
                self.random_choice(np.arange(800, 1200)),
                self.random_choice(np.arange(8, 10)),
            ),
            (
                self.random_choice(np.arange(1500, 2000)),
                self.random_choice(np.arange(10, 15)),
            ),
            (
                self.random_choice(np.arange(5000, 8000)),
                self.random_choice(np.arange(15, 25)),
            ),
            (
                self.random_choice(np.arange(10000, 15000)),
                self.random_choice(np.arange(20, 25)),
            ),
            (
                self.random_choice(np.arange(45000, 55000)),
                self.random_choice(np.arange(30, 35)),
            ),
        ]
        choice = self.random_choice(range(len(sigma_alpha)))
        sigma_alpha_chosen = sigma_alpha[choice]
        return self.elastic_transform(img, sigma_alpha_chosen)

    def degrade_img(self, img) -> np.array:
        """
        This function randomly degrades the input image by applying different
        degradation steps with different robabilities.
        Args:
            img: the image to modify in array format.
        Returns:
            img: the degraded image.
        """
        # s+p
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.1:
            img = self.s_and_p(img)

        # scale
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.5:
            img = self.scale(img)

        # brightness
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.7:
            img = self.brightness(img)

        # contrast
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.7:
            img = self.contrast(img)

        # sharpness
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.5:
            img = self.sharpness(img)

        # Modify the next line if you want a particular image size as output
        # img = cv2.resize(img, (256, 256))
        return img

    def contrast(self, img) -> np.array:
        """
        This function randomly changes the input image contrast.
        Args:
            img: the image to modify in array format.
        Returns:
            img: the image with the contrast changes.
        """
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.8:  # increase contrast
            f = self.random_choice(np.arange(1, 2, 0.01))
        else:  # decrease contrast
            f = self.random_choice(np.arange(0.5, 1, 0.01))
        im_pil = Image.fromarray(img)
        enhancer = ImageEnhance.Contrast(im_pil)
        im = enhancer.enhance(f)
        img = np.asarray(im)
        return np.asarray(im)

    def brightness(self, img) -> np.array:
        """
        This function randomly changes the input image brightness.
        Args:
            img: the image to modify in array format.
        Returns:
            img: the image with the brightness changes.
        """
        f = self.random_choice(np.arange(0.4, 1.1, 0.01))
        im_pil = Image.fromarray(img)
        enhancer = ImageEnhance.Brightness(im_pil)
        im = enhancer.enhance(f)
        img = np.asarray(im)
        return np.asarray(im)

    def sharpness(self, img) -> np.array:
        """
        This function randomly changes the input image sharpness.
        Args:
            img: the image to modify in array format.
        Returns:
            img: the image with the sharpness changes.
        """
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.5:  # increase sharpness
            f = self.random_choice(np.arange(0.1, 1, 0.01))
        else:  # decrease sharpness
            f = self.random_choice(np.arange(1, 10))
        im_pil = Image.fromarray(img)
        enhancer = ImageEnhance.Sharpness(im_pil)
        im = enhancer.enhance(f)
        img = np.asarray(im)
        return np.asarray(im)

    def s_and_p(self, img) -> np.array:
        """
        This function randomly adds salt and pepper to the input image.
        Args:
            img: the image to modify in array format.
        Returns:
            out: the image with the s&p changes.
        """
        amount = self.random_choice(np.arange(0.001, 0.01))
        # add some s&p
        s_vs_p = 0.5
        out = np.copy(img)
        # Salt mode
        num_salt = int(np.ceil(amount * img.size * s_vs_p))
        coords = []
        for i in img.shape:
            coordinates = []
            for n in range(num_salt):
                coordinates.append(self.random_choice(np.arange(0, i - 1)))
            coords.append(np.array(coordinates))
        out[tuple(coords)] = 1
        # pepper
        num_pepper = int(np.ceil(amount * img.size * (1.0 - s_vs_p)))
        coords = []
        for i in img.shape:
            coordinates = []
            for n in range(num_pepper):
                coordinates.append(self.random_choice(np.arange(0, i - 1)))
            coords.append(np.array(coordinates))
        out[tuple(coords)] = 0
        return out

    def scale(self, img) -> np.array:
        """
        This function randomly scales the input image.
        Args:
            img: the image to modify in array format.
        Returns:
            res: the scaled image.
        """
        f = self.random_choice(np.arange(0.5, 1.5, 0.01))
        res = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        res = cv2.resize(
            res, None, fx=1.0 / f, fy=1.0 / f, interpolation=cv2.INTER_CUBIC
        )
        return res

    def depict_and_resize_pikachu(
        self, smiles: str, shape: Tuple[int, int] = (299, 299)
    ) -> np.array:
        """
        This function takes a smiles str and an image shape.
        It renders the chemical structures using PIKAChU with random
        rendering/depiction settings and returns an RGB image (np.array)
        with the given image shape.

        Args:
            smiles (str): SMILES representation of molecule
            shape (Tuple[int, int], optional): im shape. Defaults to (299, 299)

        Returns:
            np.array: Chemical structure depiction
        """
        structure = read_smiles(smiles)
        depiction_settings = self.get_random_pikachu_rendering_settings()
        if "." in smiles:
            drawer = drawing.draw_multiple(structure, options=depiction_settings)
        else:
            drawer = drawing.Drawer(structure, options=depiction_settings)
        depiction = drawer.get_image_as_array()
        depiction = self.central_square_image(depiction)
        depiction = self.resize(depiction, (shape[0], shape[1]))
        return depiction

    def get_random_indigo_rendering_settings(
        self, shape: Tuple[int, int] = (299, 299)
    ) -> Indigo:
        """
        This function defines random rendering options for the structure
        depictions created using Indigo.
        It returns an Indigo object with the settings.

        Args:
            shape (Tuple[int, int], optional): im shape. Defaults to (299, 299)

        Returns:
            Indigo: Indigo object that contains depictions settings
        """
        # Define random shape for depiction (within boundaries);)
        indigo = Indigo()
        renderer = IndigoRenderer(indigo)
        # Get slightly distorted shape
        y, x = self.random_image_size(shape)
        indigo.setOption("render-image-width", x)
        indigo.setOption("render-image-height", y)
        # Set random bond line width
        bond_line_width = float(
            self.random_choice(
                np.arange(0.5, 2.5, 0.1), log_attribute="indigo_bond_line_width"
            )
        )
        indigo.setOption("render-bond-line-width", bond_line_width)
        # Set random relative thickness
        relative_thickness = float(
            self.random_choice(
                np.arange(0.5, 1.5, 0.1), log_attribute="indigo_relative_thickness"
            )
        )
        indigo.setOption("render-relative-thickness", relative_thickness)
        # Output_format: PNG
        indigo.setOption("render-output-format", "png")
        # Set random atom label rendering model
        # (standard is rendering terminal groups)
        if self.random_choice([True] + [False] * 19, log_attribute="indigo_labels_all"):
            # show all atom labels
            indigo.setOption("render-label-mode", "all")
        elif self.random_choice(
            [True] + [False] * 3, log_attribute="indigo_labels_hetero"
        ):
            indigo.setOption(
                "render-label-mode", "hetero"
            )  # only hetero atoms, no terminal groups
        # Render bold bond for Haworth projection
        if self.random_choice([True, False], log_attribute="indigo_render_bold_bond"):
            indigo.setOption("render-bold-bond-detection", "True")
        # Render labels for stereobonds
        stereo_style = self.random_choice(
            ["ext", "old", "none"], log_attribute="indigo_stereo_label_style"
        )
        indigo.setOption("render-stereo-style", stereo_style)
        # Collapse superatoms (default: expand)
        if self.random_choice(
            [True, False], log_attribute="indigo_collapse_superatoms"
        ):
            indigo.setOption("render-superatom-mode", "collapse")
        return indigo, renderer

    def depict_and_resize_indigo(
        self, smiles: str, shape: Tuple[int, int] = (299, 299)
    ) -> np.array:
        """
        This function takes a smiles str and an image shape.
        It renders the chemical structures using Indigo with random
        rendering/depiction settings and returns an RGB image (np.array)
        with the given image shape.

        Args:
            smiles (str): SMILES representation of molecule
            shape (Tuple[int, int], optional): im shape. Defaults to (299, 299)

        Returns:
            np.array: Chemical structure depiction
        """
        # Instantiate Indigo with random settings and IndigoRenderer
        indigo, renderer = self.get_random_indigo_rendering_settings()
        # Load molecule
        if not self.has_r_group(smiles):
            molecule = indigo.loadMolecule(smiles)
        else:
            mol_str = self.smiles_to_mol_str(smiles)
            molecule = indigo.loadMolecule(mol_str)
        # Kekulize in 67% of cases
        if not self.random_choice(
            [True, True, False], log_attribute="indigo_kekulized"
        ):
            molecule.aromatize()
        molecule.layout()
        # Write to buffer
        temp = renderer.renderToBuffer(molecule)
        temp = io.BytesIO(temp)
        depiction = sk_io.imread(temp)
        depiction = self.resize(depiction, (shape[0], shape[1]))
        depiction = rgba2rgb(depiction)
        depiction = img_as_ubyte(depiction)
        return depiction

    def get_random_rdkit_rendering_settings(
        self, smiles: str, shape: Tuple[int, int] = (299, 299)
    ) -> rdMolDraw2D.MolDraw2DCairo:
        """
        This function defines random rendering options for the structure
        depictions created using rdkit. It returns an MolDraw2DCairo object
        with the settings.

        Args:
            smiles (str): SMILES representation of molecule
            shape (Tuple[int, int], optional): im_shape. Defaults to (299, 299)

        Returns:
            rdMolDraw2D.MolDraw2DCairo: Object that contains depiction settings
        """
        # Get slightly distorted shape
        y, x = self.random_image_size(shape)
        # Instantiate object that saves the settings
        depiction_settings = rdMolDraw2D.MolDraw2DCairo(y, x)
        # Stereo bond annotation
        if self.random_choice(
            [True, False], log_attribute="rdkit_add_stereo_annotation"
        ):
            depiction_settings.drawOptions().addStereoAnnotation = True
        if self.random_choice(
            [True, False], log_attribute="rdkit_add_chiral_flag_labels"
        ):
            depiction_settings.drawOptions().includeChiralFlagLabel = True
        # Atom indices
        if self.random_choice(
            [True, False, False, False], log_attribute="rdkit_add_atom_indices"
        ):
            if not self.has_r_group(smiles):
                depiction_settings.drawOptions().addAtomIndices = True
        # Bond line width
        bond_line_width = self.random_choice(
            range(1, 5), log_attribute="rdkit_bond_line_width"
        )
        depiction_settings.drawOptions().bondLineWidth = bond_line_width
        # Draw terminal methyl groups
        if self.random_choice(
            [True, False], log_attribute="rdkit_draw_terminal_methyl"
        ):
            depiction_settings.drawOptions().explicitMethyl = True
        # Label font type and size
        font_dir = self.HERE.joinpath("fonts/")
        font_path = os.path.join(
            str(font_dir),
            self.random_choice(
                os.listdir(str(font_dir)), log_attribute="rdkit_label_font"
            ),
        )
        depiction_settings.drawOptions().fontFile = font_path
        min_font_size = self.random_choice(
            range(10, 20), log_attribute="rdkit_min_font_size"
        )
        depiction_settings.drawOptions().minFontSize = min_font_size
        depiction_settings.drawOptions().maxFontSize = 30
        # Rotate the molecule
        depiction_settings.drawOptions().rotate = self.random_choice(range(360))
        # Fixed bond length
        fixed_bond_length = self.random_choice(
            range(30, 45), log_attribute="rdkit_fixed_bond_length"
        )
        depiction_settings.drawOptions().fixedBondLength = fixed_bond_length
        # Comic mode (looks a bit hand drawn)
        if self.random_choice(
            [True, False, False, False, False], log_attribute="rdkit_comic_style"
        ):
            depiction_settings.drawOptions().comicMode = True
        # Keep it black and white
        depiction_settings.drawOptions().useBWAtomPalette()
        return depiction_settings

    def depict_and_resize_rdkit(
        self, smiles: str, shape: Tuple[int, int] = (299, 299)
    ) -> np.array:
        """
        This function takes a smiles str and an image shape.
        It renders the chemical structuresusing Rdkit with random
        rendering/depiction settings and returns an RGB image (np.array)
        with the given image shape.

        Args:
            smiles (str): SMILES representation of molecule_
            shape (Tuple[int, int], optional): im shape. Defaults to (299, 299)

        Returns:
            np.array: Chemical structure depiction
        """
        # Load molecule
        if not self.has_r_group(smiles):
            mol = Chem.MolFromSmiles(smiles)
        else:
            mol_str = self.smiles_to_mol_str(smiles)
            mol = Chem.MolFromMolBlock(mol_str)
        if mol:
            AllChem.Compute2DCoords(mol)
            # Abbreviate superatoms
            if self.random_choice(
                [True, False], log_attribute="rdkit_collapse_superatoms"
            ):
                abbrevs = GetDefaultAbbreviations()
                mol = CondenseMolAbbreviations(mol, abbrevs)
            # Get random depiction settings
            depiction_settings = self.get_random_rdkit_rendering_settings(smiles=smiles)
            # Create depiction
            # TODO: Figure out how to depict without kekulization here
            # The following line does not prevent the molecule from being
            # depicted kekulized:
            # mol = rdMolDraw2D.PrepareMolForDrawing(mol, kekulize = False)
            # The molecule must get kekulized somewhere "by accident"

            rdMolDraw2D.PrepareAndDrawMolecule(depiction_settings, mol)
            depiction = depiction_settings.GetDrawingText()
            depiction = sk_io.imread(io.BytesIO(depiction))
            # Resize image to desired shape
            depiction = self.resize(depiction, shape)
            depiction = img_as_ubyte(depiction)
            return np.asarray(depiction)
        else:
            print("RDKit was unable to read input SMILES: {}".format(smiles))

    def has_r_group(self, smiles: str) -> bool:
        """
        Determines whether or not a given SMILES str contains an R group

        Args:
            smiles (str): SMILES representation of molecule

        Returns:
            bool
        """
        if re.search("\[.*[RXYZ].*\]", smiles):
            return True

    def get_random_cdk_rendering_settings(self, rendererModel, molecule, smiles: str):
        """
        This function defines random rendering options for the structure
        depictions created using CDK.
        It takes a cdk.renderer.AtomContainerRenderer.2DModel
        and a cdk.AtomContainer and returns the 2DModel object with random
        rendering settings and the AtomContainer.
        I followed https://github.com/cdk/cdk/wiki/Standard-Generator while
        creating this.

        Args:
            rendererModel (cdk.renderer.AtomContainerRenderer.2DModel)
            molecule (cdk.AtomContainer): Atom container
            smiles (str): smiles representation of molecule

        Returns:
            rendererModel, molecule: Objects that hold depiction parameters
        """
        cdk_base = "org.openscience.cdk"

        StandardGenerator = JClass(
            cdk_base + ".renderer.generators.standard.StandardGenerator"
        )

        # Define visibility of atom/superatom labels
        symbol_visibility = self.random_choice(
            ["iupac_recommendation", "no_terminal_methyl", "show_all_atom_labels"],
            log_attribute="cdk_symbol_visibility",
        )
        SymbolVisibility = JClass("org.openscience.cdk.renderer.SymbolVisibility")
        if symbol_visibility == "iupac_recommendation":
            rendererModel.set(
                StandardGenerator.Visibility.class_,
                SymbolVisibility.iupacRecommendations(),
            )
        elif symbol_visibility == "no_terminal_methyl":
            # only hetero atoms, no terminal alkyl groups
            rendererModel.set(
                StandardGenerator.Visibility.class_,
                SymbolVisibility.iupacRecommendationsWithoutTerminalCarbon(),
            )
        elif symbol_visibility == "show_all_atom_labels":
            rendererModel.set(
                StandardGenerator.Visibility.class_, SymbolVisibility.all()
            )  # show all atom labels

        # Define bond line stroke width
        stroke_width = self.random_choice(
            np.arange(0.8, 2.0, 0.1), log_attribute="cdk_stroke_width"
        )
        rendererModel.set(StandardGenerator.StrokeRatio.class_, stroke_width)
        # Define symbol margin ratio
        margin_ratio = self.random_choice(
            [0, 1, 2, 2, 2, 3, 4], log_attribute="cdk_margin_ratio"
        )
        rendererModel.set(
            StandardGenerator.SymbolMarginRatio.class_,
            JClass("java.lang.Double")(margin_ratio),
        )
        # Define bond properties
        double_bond_dist = self.random_choice(
            np.arange(0.11, 0.25, 0.01), log_attribute="cdk_double_bond_dist"
        )
        rendererModel.set(StandardGenerator.BondSeparation.class_, double_bond_dist)
        wedge_ratio = self.random_choice(
            np.arange(4.5, 7.5, 0.1), log_attribute="cdk_wedge_ratio"
        )
        rendererModel.set(
            StandardGenerator.WedgeRatio.class_, JClass("java.lang.Double")(wedge_ratio)
        )
        if self.random_choice([True, False], log_attribute="cdk_fancy_bold_wedges"):
            rendererModel.set(StandardGenerator.FancyBoldWedges.class_, True)
        if self.random_choice([True, False], log_attribute="cdk_fancy_hashed_wedges"):
            rendererModel.set(StandardGenerator.FancyHashedWedges.class_, True)
        hash_spacing = self.random_choice(
            np.arange(4.0, 6.0, 0.2), log_attribute="cdk_hash_spacing"
        )
        rendererModel.set(StandardGenerator.HashSpacing.class_, hash_spacing)
        # Add CIP labels
        labels = False
        if self.random_choice([True, False], log_attribute="cdk_add_CIP_labels"):
            labels = True
            JClass("org.openscience.cdk.geometry.cip.CIPTool").label(molecule)
            for atom in molecule.atoms():
                label = atom.getProperty(
                    JClass("org.openscience.cdk.CDKConstants").CIP_DESCRIPTOR
                )
                atom.setProperty(StandardGenerator.ANNOTATION_LABEL, label)
            for bond in molecule.bonds():
                label = bond.getProperty(
                    JClass("org.openscience.cdk.CDKConstants").CIP_DESCRIPTOR
                )
                bond.setProperty(StandardGenerator.ANNOTATION_LABEL, label)
        # Add atom indices to the depictions
        if self.random_choice(
            [True, False, False, False], log_attribute="cdk_add_atom_indices"
        ):
            if not self.has_r_group(smiles):
                labels = True
                for atom in molecule.atoms():
                    label = JClass("java.lang.Integer")(
                        1 + molecule.getAtomNumber(atom)
                    )
                    atom.setProperty(StandardGenerator.ANNOTATION_LABEL, label)
        if labels:
            # We only need black
            rendererModel.set(
                StandardGenerator.AnnotationColor.class_,
                JClass("java.awt.Color")(0x000000),
            )
            # Font size of labels
            font_scale = self.random_choice(
                np.arange(0.5, 0.8, 0.1), log_attribute="cdk_label_font_scale"
            )
            rendererModel.set(StandardGenerator.AnnotationFontScale.class_, font_scale)
            # Distance between atom numbering and depiction
            annotation_distance = self.random_choice(
                np.arange(0.15, 0.30, 0.05), log_attribute="cdk_annotation_distance"
            )
            rendererModel.set(
                StandardGenerator.AnnotationDistance.class_, annotation_distance
            )
        # Abbreviate superatom labels in half of the cases
        # TODO: Find a way to define Abbreviations object as a class attribute.
        # Problem: can't be pickled.
        # Right now, this is loaded every time when a structure is depicted.
        # That seems inefficient.
        if self.random_choice([True, False], log_attribute="cdk_collapse_superatoms"):
            cdk_superatom_abrv = JClass("org.openscience.cdk.depict.Abbreviations")()
            abbreviation_path = str(self.HERE.joinpath("smiles_list.smi"))
            abbreviation_path = abbreviation_path.replace("\\", "/")
            abbreviation_path = JClass("java.lang.String")(abbreviation_path)
            cdk_superatom_abrv.loadFromFile(abbreviation_path)
            cdk_superatom_abrv.apply(molecule)
        return rendererModel, molecule

    def depict_and_resize_cdk(
        self, smiles: str, shape: Tuple[int, int] = (299, 299)
    ) -> np.array:
        """
        This function takes a smiles str and an image shape.
        It renders the chemical structures using CDK with random
        rendering/depiction settings and returns an RGB image (np.array)
        with the given image shape.
        The general workflow here is a JPype adaptation of code published
        by Egon Willighagen in 'Groovy Cheminformatics with the Chemistry
        Development Kit':
        https://egonw.github.io/cdkbook/ctr.html#depict-a-compound-as-an-image
        with additional adaptations to create all the different depiction
        types from
        https://github.com/cdk/cdk/wiki/Standard-Generator

        Args:
            smiles (str): SMILES representation of molecule
            shape (Tuple[int, int], optional): im shape. Defaults to (299, 299)

        Returns:
            np.array: Chemical structure depiction
        """
        cdk_base = "org.openscience.cdk"
        # Read molecule from SMILES str
        molecule = self.cdk_smiles_to_IAtomContainer(smiles)

        # Add hydrogens for coordinate generation (to make it look nicer/
        # avoid overlaps)
        matcher = JClass(cdk_base + ".atomtype.CDKAtomTypeMatcher").getInstance(
            molecule.getBuilder()
        )
        for atom in molecule.atoms():
            atom_type = matcher.findMatchingAtomType(molecule, atom)
            JClass(cdk_base + ".tools.manipulator.AtomTypeManipulator").configure(
                atom, atom_type
            )
        adder = JClass(cdk_base + ".tools.CDKHydrogenAdder").getInstance(
            molecule.getBuilder()
        )
        adder.addImplicitHydrogens(molecule)
        AtomContainerManipulator = JClass(
            cdk_base + ".tools.manipulator.AtomContainerManipulator"
        )
        AtomContainerManipulator.convertImplicitToExplicitHydrogens(molecule)

        # Instantiate StructureDiagramGenerator, determine coordinates
        sdg = JClass(cdk_base + ".layout.StructureDiagramGenerator")()
        sdg.setMolecule(molecule)
        sdg.generateCoordinates(molecule)
        molecule = sdg.getMolecule()

        # Remove explicit hydrogens again
        AtomContainerManipulator.suppressHydrogens(molecule)

        # Rotate molecule randomly
        point = JClass(cdk_base + ".geometry.GeometryTools").get2DCenter(molecule)
        rot_degrees = self.random_choice(range(360))
        JClass(cdk_base + ".geometry.GeometryTools").rotate(
            molecule, point, rot_degrees
        )

        # Get Generators
        generators = JClass("java.util.ArrayList")()
        BasicSceneGenerator = JClass(
            "org.openscience.cdk.renderer.generators.BasicSceneGenerator"
        )()
        generators.add(BasicSceneGenerator)
        font_size = self.random_choice(
            range(10, 20), log_attribute="cdk_atom_label_font_size"
        )
        Font = JClass("java.awt.Font")
        font_name = self.random_choice(
            ["Verdana", "Times New Roman", "Arial", "Gulliver Regular"],
            # log_attribute='cdk_atom_label_font'
        )
        font_style = self.random_choice(
            [Font.PLAIN, Font.BOLD],
            # log_attribute='cdk_atom_label_font_style'
        )
        font = Font(font_name, font_style, font_size)
        StandardGenerator = JClass(
            cdk_base + ".renderer.generators.standard.StandardGenerator"
        )(font)
        generators.add(StandardGenerator)

        # Instantiate renderer
        AWTFontManager = JClass(cdk_base + ".renderer.font.AWTFontManager")
        renderer = JClass(cdk_base + ".renderer.AtomContainerRenderer")(
            generators, AWTFontManager()
        )

        # Create an empty image of the right size
        y, x = self.random_image_size(shape)
        # Workaround for structures that are cut off at edged of images:
        # Make image twice as big, reduce Zoom factor, then remove white
        # areas at borders and resize to originally desired shape
        # TODO: Find out why the structures are cut off in the first place
        y = y * 4
        x = x * 4

        drawArea = JClass("java.awt.Rectangle")(x, y)
        BufferedImage = JClass("java.awt.image.BufferedImage")
        image = BufferedImage(x, y, BufferedImage.TYPE_INT_RGB)

        # Draw the molecule
        renderer.setup(molecule, drawArea)
        model = renderer.getRenderer2DModel()

        # Get random rendering settings
        model, molecule = self.get_random_cdk_rendering_settings(
            model, molecule, smiles
        )

        double = JClass("java.lang.Double")
        model.set(
            JClass(cdk_base + ".renderer.generators.BasicSceneGenerator.ZoomFactor"),
            double(1.0),
        )
        g2 = image.getGraphics()
        g2.setColor(JClass("java.awt.Color").WHITE)
        g2.fillRect(0, 0, x, y)
        AWTDrawVisitor = JClass("org.openscience.cdk.renderer.visitor.AWTDrawVisitor")

        renderer.paint(molecule, AWTDrawVisitor(g2))

        # Write the image into a format that can be read by skimage
        ImageIO = JClass("javax.imageio.ImageIO")
        os = JClass("java.io.ByteArrayOutputStream")()
        Base64 = JClass("java.util.Base64")
        ImageIO.write(
            image, JClass("java.lang.String")("PNG"), Base64.getEncoder().wrap(os)
        )
        depiction = bytes(os.toString("UTF-8"))
        depiction = base64.b64decode(depiction)

        # Read image in skimage
        depiction = sk_io.imread(depiction, plugin="imageio")
        # Normalise padding and get non-distorted image of right size
        depiction = self.normalise_padding(depiction)
        depiction = self.central_square_image(depiction)
        depiction = self.resize(depiction, shape, HQ=True)
        depiction = img_as_ubyte(depiction)
        return depiction

    def cdk_smiles_to_IAtomContainer(self, smiles: str):
        """
        This function takes a SMILES representation of a molecule and
        returns the corresponding IAtomContainer object.

        Args:
            smiles (str): SMILES representation of the molecule

        Returns:
            IAtomContainer: CDK IAtomContainer object that represents the molecule
        """
        cdk_base = "org.openscience.cdk"
        DCOB = JClass(cdk_base + ".DefaultChemObjectBuilder")
        SmilesParser = JClass(cdk_base + ".smiles.SmilesParser")(DCOB.getInstance())
        if self.random_choice([True, False, False], log_attribute="cdk_kekulized"):
            SmilesParser.kekulise(False)
        molecule = SmilesParser.parseSmiles(smiles)
        return molecule

    def smiles_to_mol_str(self, smiles: str) -> str:
        """
        This function takes a SMILES representation of a molecule and returns
        the content of the corresponding SD file using the CDK.
        ___
        The SMILES parser of the CDK is much more tolerant than the parsers of
        RDKit and Indigo.
        ___

        Args:
            smiles (str): SMILES representation of a molecule

        Returns:
            str: content of SD file of input molecule
        """
        i_atom_container = self.cdk_smiles_to_IAtomContainer(smiles)
        mol_str = self.cdk_IAtomContainer_to_mol_str(i_atom_container)
        return mol_str

    def cdk_IAtomContainer_to_mol_str(self, i_atom_container) -> str:
        """
        This function takes an IAtomContainer object and returns the content
        of the corresponding MDL MOL file as a string.

        Args:
            i_atom_container (CDK IAtomContainer)

        Returns:
            str: string content of MDL MOL file
        """
        string_writer = JClass("java.io.StringWriter")()
        mol_writer = JClass("org.openscience.cdk.io.MDLV2000Writer")(string_writer)
        mol_writer.write(i_atom_container)
        mol_writer.close()
        mol_str = string_writer.toString()
        return str(mol_str)

    def normalise_padding(self, im: np.array) -> np.array:
        """This function takes an RGB image (np.array) and deletes white space at
        the borders. Then 0-10% of the image width/height is added as padding
        again. The modified image is returned

        Args:
            im: input image (np.array)

        Returns:
            output: the modified image (np.array)
        """
        # Remove white space at borders
        mask = im > 200
        all_white = mask.sum(axis=2) > 0
        rows = np.flatnonzero((~all_white).sum(axis=1))
        cols = np.flatnonzero((~all_white).sum(axis=0))
        crop = im[rows.min(): rows.max() + 1, cols.min(): cols.max() + 1, :]
        # Add padding again.
        pad_range = np.arange(5, int(crop.shape[0] * 0.2), 1)
        if len(pad_range) > 0:
            pad = self.random_choice(np.arange(5, int(crop.shape[0] * 0.2), 1))
        else:
            pad = 5
        crop = np.pad(
            crop,
            pad_width=((pad, pad), (pad, pad), (0, 0)),
            mode="constant",
            constant_values=255,
        )
        return crop

    def central_square_image(self, im: np.array) -> np.array:
        """
        This function takes image (np.array) and will add white padding
        so that the image has a square shape with the width/height of the
        longest side of the original image.

        Args:
            im (np.array): Input image

        Returns:
            np.array: Output image
        """
        # Create new blank white image
        max_wh = max(im.shape)
        new_im = 255 * np.ones((max_wh, max_wh, 3), np.uint8)
        # Determine paste coordinates and paste image
        upper = int((new_im.shape[0] - im.shape[0]) / 2)
        lower = int((new_im.shape[0] - im.shape[0]) / 2) + im.shape[0]
        left = int((new_im.shape[1] - im.shape[1]) / 2)
        right = int((new_im.shape[1] - im.shape[1]) / 2) + im.shape[1]
        new_im[upper:lower, left:right] = im
        return new_im

    def random_depiction(
        self,
        smiles: str,
        shape: Tuple[int, int] = (299, 299),
        path_bkg="./backgrounds/",
    ) -> np.array:
        """
        This function takes a SMILES and depicts it using Rdkit, Indigo or CDK.
        The depiction method and the specific parameters for the depiction are
        chosen completely randomly. The purpose of this function is to enable
        depicting a diverse variety of chemical structure depictions.

        Args:
            smiles (str): SMILES representation of molecule
            shape (Tuple[int, int], optional): im shape. Defaults to (299, 299)

        Returns:
            np.array: Chemical structure depiction
        """
        depiction_functions = self.get_depiction_functions(smiles)
        # If nothing is returned, try different function
        for _ in range(3):
            if len(depiction_functions) != 0:
                # Pick random depiction function and call it
                depiction_function = self.random_choice(depiction_functions)
                depiction = depiction_function(smiles, shape)
                if depiction is False or depiction is None:
                    depiction_functions.remove(depiction_function)
                else:
                    break
            else:
                break

        if self.hand_drawn:
            # Augment molecule image
            mol_aug = self.hand_drawn_augment(depiction)

            # Randomly select background image and use is as it is
            backgroud_selected = self.random_choice(os.listdir(path_bkg))
            bkg = cv2.imread(path_bkg + backgroud_selected)
            bkg = cv2.resize(bkg, (256, 256))
            # Combine augmented molecule and augmented background
            p = 0.7
            mol_bkg = cv2.addWeighted(mol_aug, p, bkg, 1 - p, gamma=0)

            """
            If you want to randomly augment the background as well,
            simply comment the previous section and uncomment the next one.
            """

            """# Randomly select background image and augment it
            bkg_aug = self.augment_bkg(bkg)
            bkg_aug = cv2.resize(bkg_aug,(256,256))
            # Combine augmented molecule and augmented background
            p=0.7
            mol_bkg = cv2.addWeighted(mol_aug, p, bkg_aug, 1-p, gamma=0)"""

            # Degrade total image
            depiction = self.degrade_img(mol_bkg)
        return depiction

    def get_depiction_functions(self, smiles: str) -> List[Callable]:
        """
        PIKAChU, RDKit and Indigo can run into problems if certain R group variables
        are present in the input molecule, and PIKAChU cannot handle isotopes.
        Hence, the depiction functions that use their functionalities need to
        be removed based on the input smiles str (if necessary).

        Args:
            smiles (str): SMILES representation of a molecule

        Returns:
            List[Callable]: List of depiction functions
        """
        depiction_functions = [
            self.depict_and_resize_rdkit,
            self.depict_and_resize_indigo,
            self.depict_and_resize_cdk,
            self.depict_and_resize_pikachu,
        ]
        # Remove PIKAChU if there is an isotope
        if re.search("(\[\d\d\d?[A-Z])|(\[2H\])|(\[3H\])|(D)|(T)", smiles):
            depiction_functions.remove(self.depict_and_resize_pikachu)
        if self.has_r_group(smiles):
            # PIKAChU only accepts \[[RXZ]\d*\]
            squared_bracket_content = re.findall("\[.+?\]", smiles)
            for r_group in squared_bracket_content:
                if not re.search("\[[RXZ]\d*\]", r_group):
                    if self.depict_and_resize_pikachu in depiction_functions:
                        depiction_functions.remove(self.depict_and_resize_pikachu)
            # "R", "X", "Z" are not depicted by RDKit
            # The same is valid for X,Y,Z and a number
            if re.search("\[[RXZ]\]|\[[XYZ]\d+", smiles):
                depiction_functions.remove(self.depict_and_resize_rdkit)
            # "X", "R0" and indices above 32 are not depicted by Indigo
            if re.search("\[R0\]|\[X\]|[4-9][0-9]+|3[3-9]", smiles):
                depiction_functions.remove(self.depict_and_resize_indigo)
        return depiction_functions

    def resize(self, image: np.array, shape: Tuple[int], HQ: bool = False) -> np.array:
        """
        This function takes an image (np.array) and a shape and returns
        the resized image (np.array). It uses Pillow to do this, as it
        seems to have a bigger variety of scaling methods than skimage.
        The up/downscaling method is chosen randomly.

        Args:
            image (np.array): the input image
            shape (Tuple[int, int], optional): im shape. Defaults to (299, 299)
            HQ (bool): if true, only choose from Image.BICUBIC, Image.LANCZOS
        ___
        Returns:
            np.array: the resized image

        """
        image = Image.fromarray(image)
        shape = (shape[0], shape[1])
        if not HQ:
            image = image.resize(
                shape, resample=self.random_choice(self.PIL_resize_methods)
            )
        else:
            image = image = image.resize(
                shape, resample=self.random_choice(self.PIL_HQ_resize_methods)
            )

        return np.asarray(image)

    def imgaug_augment(
        self,
        image: np.array,
    ) -> np.array:
        """
        This function applies a random amount of augmentations to
        a given image (np.array) using and returns the augmented image
        (np.array).

        Args:
            image (np.array): input image

        Returns:
            np.array: output image (augmented)
        """
        original_shape = image.shape

        # Choose number of augmentations to apply (0-2);
        # return image if nothing needs to be done.
        aug_number = self.random_choice(range(0, 3))
        if not aug_number:
            return image

        # Add some padding to avoid weird artifacts after rotation
        image = np.pad(
            image, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=255
        )

        def imgaug_rotation():
            # Rotation between -10 and 10 degrees
            if not self.random_choice(
                [True, True, False], log_attribute="has_imgaug_rotation"
            ):
                return False
            rot_angle = self.random_choice(np.arange(-10, 10, 1))
            aug = iaa.Affine(rotate=rot_angle, mode="edge", fit_output=True)
            return aug

        def imgaug_black_and_white_noise():
            # Black and white noise
            if not self.random_choice(
                [True, True, False], log_attribute="has_imgaug_salt_pepper"
            ):
                return False
            coarse_dropout_p = self.random_choice(np.arange(0.0002, 0.0015, 0.0001))
            coarse_dropout_size_percent = self.random_choice(np.arange(1.0, 1.1, 0.01))
            replace_elementwise_p = self.random_choice(np.arange(0.01, 0.3, 0.01))
            aug = iaa.Sequential(
                [
                    iaa.CoarseDropout(
                        coarse_dropout_p, size_percent=coarse_dropout_size_percent
                    ),
                    iaa.ReplaceElementwise(replace_elementwise_p, 255),
                ]
            )
            return aug

        def imgaug_shearing():
            # Shearing
            if not self.random_choice(
                [True, True, False], log_attribute="has_imgaug_shearing"
            ):
                return False
            shear_param = self.random_choice(np.arange(-5, 5, 1))
            aug = self.random_choice(
                [
                    iaa.geometric.ShearX(shear_param, mode="edge", fit_output=True),
                    iaa.geometric.ShearY(shear_param, mode="edge", fit_output=True),
                ]
            )
            return aug

        def imgaug_imgcorruption():
            # Jpeg compression or pixelation
            if not self.random_choice(
                [True, True, False], log_attribute="has_imgaug_corruption"
            ):
                return False
            imgcorrupt_severity = self.random_choice(np.arange(1, 2, 1))
            aug = self.random_choice(
                [
                    iaa.imgcorruptlike.JpegCompression(severity=imgcorrupt_severity),
                    iaa.imgcorruptlike.Pixelate(severity=imgcorrupt_severity),
                ]
            )
            return aug

        def imgaug_brightness_adjustment():
            # Brightness adjustment
            if not self.random_choice(
                [True, True, False], log_attribute="has_imgaug_brightness_adj"
            ):
                return False
            brightness_adj_param = self.random_choice(np.arange(-50, 50, 1))
            aug = iaa.WithBrightnessChannels(iaa.Add(brightness_adj_param))
            return aug

        def imgaug_colour_temp_adjustment():
            # Colour temperature adjustment
            if not self.random_choice(
                [True, True, False], log_attribute="has_imgaug_col_adj"
            ):
                return False
            colour_temp = self.random_choice(np.arange(1100, 10000, 1))
            aug = iaa.ChangeColorTemperature(colour_temp)
            return aug

        # Define list of available augmentations
        aug_list = [
            imgaug_rotation,
            imgaug_black_and_white_noise,
            imgaug_shearing,
            imgaug_imgcorruption,
            imgaug_brightness_adjustment,
            imgaug_colour_temp_adjustment,
        ]

        # Every one of them has a 1/3 chance of returning False
        aug_list = [fun() for fun in aug_list]
        aug_list = [fun for fun in aug_list if fun]
        aug = iaa.Sequential(aug_list)
        augmented_image = aug.augment_images([image])[0]
        augmented_image = self.resize(augmented_image, original_shape)
        augmented_image = augmented_image.astype(np.uint8)
        return augmented_image

    def add_augmentations(self, depiction: np.array) -> np.array:
        """
        This function takes a chemical structure depiction (np.array)
        and returns the same image with added augmentation elements

        Args:
            depiction (np.array): chemical structure depiction

        Returns:
            np.array: chemical structure depiction with added augmentations
        """
        if self.random_choice(
            [True, False, False, False, False, False], log_attribute="has_curved_arrows"
        ):
            depiction = self.add_curved_arrows_to_structure(depiction)
        if self.random_choice(
            [True, False, False], log_attribute="has_straight_arrows"
        ):
            depiction = self.add_straight_arrows_to_structure(depiction)
        if self.random_choice(
            [True, False, False, False, False, False], log_attribute="has_id_label"
        ):
            depiction = self.add_chemical_label(depiction, "ID")
        if self.random_choice(
            [True, False, False, False, False, False], log_attribute="has_R_group_label"
        ):
            depiction = self.add_chemical_label(depiction, "R_GROUP")
        if self.random_choice(
            [True, False, False, False, False, False],
            log_attribute="has_reaction_label",
        ):
            depiction = self.add_chemical_label(depiction, "REACTION")
        if self.random_choice([True, False, False]):
            depiction = self.imgaug_augment(depiction)
        return depiction

    def get_random_label_position(self, width: int, height: int) -> Tuple[int, int]:
        """
        Given the width and height of an image (int), this function
        determines a random position in the outer 15% of the image and
        returns a tuple that contain the coordinates (y,x) of that position.

        Args:
            width (int): image width
            height (int): image height

        Returns:
            Tuple[int, int]: Random label position
        """
        if self.random_choice([True, False]):
            y_range = range(0, height)
            x_range = list(range(0, int(0.15 * width))) + list(
                range(int(0.85 * width), width)
            )
        else:
            y_range = list(range(0, int(0.15 * height))) + list(
                range(int(0.85 * height), height)
            )
            x_range = range(0, width)
        return self.random_choice(y_range), self.random_choice(x_range)

    def ID_label_text(self) -> str:
        """
        This function returns a string that resembles a typical
        chemical ID label

        Returns:
            str: Label text
        """
        label_num = range(1, 50)
        label_letters = [
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
        ]
        options = [
            "only_number",
            "num_letter_combination",
            "numtonum",
            "numcombtonumcomb",
        ]
        option = self.random_choice(options)
        if option == "only_number":
            return str(self.random_choice(label_num))
        if option == "num_letter_combination":
            return str(self.random_choice(label_num)) + self.random_choice(
                label_letters
            )
        if option == "numtonum":
            return (
                str(self.random_choice(label_num))
                + "-"
                + str(self.random_choice(label_num))
            )
        if option == "numcombtonumcomb":
            return (
                str(self.random_choice(label_num))
                + self.random_choice(label_letters)
                + "-"
                + self.random_choice(label_letters)
            )

    def new_reaction_condition_elements(self) -> Tuple[str, str, str]:
        """
        Randomly redefine reaction_time, solvent and other_reactand.

        Returns:
            Tuple[str, str, str]: Reaction time, solvent, reactand
        """
        reaction_time = self.random_choice(
            [str(num) for num in range(30)]
        ) + self.random_choice([" h", " min"])
        solvent = self.random_choice(
            [
                "MeOH",
                "EtOH",
                "CHCl3",
                "DCM",
                "iPrOH",
                "MeCN",
                "DMSO",
                "pentane",
                "hexane",
                "benzene",
                "Et2O",
                "THF",
                "DMF",
            ]
        )
        other_reactand = self.random_choice(
            [
                "HF",
                "HCl",
                "HBr",
                "NaOH",
                "Et3N",
                "TEA",
                "Ac2O",
                "DIBAL",
                "DIBAL-H",
                "DIPEA",
                "DMAP",
                "EDTA",
                "HOBT",
                "HOAt",
                "TMEDA",
                "p-TsOH",
                "Tf2O",
            ]
        )
        return reaction_time, solvent, other_reactand

    def reaction_condition_label_text(self) -> str:
        """
        This function returns a random string that looks like a
        reaction condition label.

        Returns:
            str: Reaction condition label text
        """
        reaction_condition_label = ""
        label_type = self.random_choice(["A", "B", "C", "D"])
        if label_type in ["A", "B"]:
            for n in range(self.random_choice(range(1, 5))):
                (
                    reaction_time,
                    solvent,
                    other_reactand,
                ) = self.new_reaction_condition_elements()
                if label_type == "A":
                    reaction_condition_label += (
                        str(n + 1)
                        + " "
                        + other_reactand
                        + ", "
                        + solvent
                        + ", "
                        + reaction_time
                        + "\n"
                    )
                elif label_type == "B":
                    reaction_condition_label += (
                        str(n + 1)
                        + " "
                        + other_reactand
                        + ", "
                        + solvent
                        + " ("
                        + reaction_time
                        + ")\n"
                    )
        elif label_type == "C":
            (
                reaction_time,
                solvent,
                other_reactand,
            ) = self.new_reaction_condition_elements()
            reaction_condition_label += (
                other_reactand + "\n" + solvent + "\n" + reaction_time
            )
        elif label_type == "D":
            reaction_condition_label += self.random_choice(
                self.new_reaction_condition_elements()
            )
        return reaction_condition_label

    def make_R_group_str(self) -> str:
        """
        This function returns a random string that looks like an R group label.
        It generates them by inserting randomly chosen elements into one of
        five templates.

        Returns:
            str: R group label text
        """
        rest_variables = [
            "X",
            "Y",
            "Z",
            "R",
            "R1",
            "R2",
            "R3",
            "R4",
            "R5",
            "R6",
            "R7",
            "R8",
            "R9",
            "R10",
            "Y2",
            "D",
        ]
        # Load list of superatoms (from OSRA)
        superatoms = self.superatoms
        label_type = self.random_choice(["A", "B", "C", "D", "E"])
        R_group_label = ""
        if label_type == "A":
            for _ in range(1, self.random_choice(range(2, 6))):
                R_group_label += (
                    self.random_choice(rest_variables)
                    + " = "
                    + self.random_choice(superatoms)
                    + "\n"
                )
        elif label_type == "B":
            R_group_label += "      " + self.random_choice(rest_variables) + "\n"
            for n in range(1, self.random_choice(range(2, 6))):
                R_group_label += str(n) + "    " + self.random_choice(superatoms) + "\n"
        elif label_type == "C":
            R_group_label += (
                "      "
                + self.random_choice(rest_variables)
                + "      "
                + self.random_choice(rest_variables)
                + "\n"
            )
            for n in range(1, self.random_choice(range(2, 6))):
                R_group_label += (
                    str(n)
                    + "  "
                    + self.random_choice(superatoms)
                    + "  "
                    + self.random_choice(superatoms)
                    + "\n"
                )
        elif label_type == "D":
            R_group_label += (
                "      "
                + self.random_choice(rest_variables)
                + "      "
                + self.random_choice(rest_variables)
                + "      "
                + self.random_choice(rest_variables)
                + "\n"
            )
            for n in range(1, self.random_choice(range(2, 6))):
                R_group_label += (
                    str(n)
                    + "  "
                    + self.random_choice(superatoms)
                    + "  "
                    + self.random_choice(superatoms)
                    + "  "
                    + self.random_choice(superatoms)
                    + "\n"
                )
        if label_type == "E":
            for n in range(1, self.random_choice(range(2, 6))):
                R_group_label += (
                    str(n)
                    + "  "
                    + self.random_choice(rest_variables)
                    + " = "
                    + self.random_choice(superatoms)
                    + "\n"
                )
        return R_group_label

    def add_chemical_label(
        self, image: np.array, label_type: str, foreign_fonts: bool = True
    ) -> np.array:
        """
        This function takes an image (np.array) and adds random text that
        looks like a chemical ID label, an R group label or a reaction
        condition label around the structure. It returns the modified image.
        The label type is determined by the parameter label_type (str),
        which needs to be 'ID', 'R_GROUP' or 'REACTION'

        Args:
            image (np.array): Chemical structure depiction
            label_type (str): 'ID', 'R_GROUP' or 'REACTION'
            foreign_fonts (bool, optional): Defaults to True.

        Returns:
            np.array: Chemical structure depiction with label
        """
        im = Image.fromarray(image)
        orig_image = deepcopy(im)
        width, height = im.size
        # Choose random font
        if self.random_choice([True, False]) or not foreign_fonts:
            font_dir = self.HERE.joinpath("fonts/")
        # In half of the cases: Use foreign-looking font to generate
        # bigger noise variety
        else:
            font_dir = self.HERE.joinpath("foreign_fonts/")

        fonts = os.listdir(str(font_dir))
        # Choose random font size
        font_sizes = range(10, 20)
        size = self.random_choice(font_sizes)
        # Generate random string that resembles the desired type of label
        if label_type == "ID":
            label_text = self.ID_label_text()
        if label_type == "R_GROUP":
            label_text = self.make_R_group_str()
        if label_type == "REACTION":
            label_text = self.reaction_condition_label_text()

        try:
            font = ImageFont.truetype(
                str(os.path.join(str(font_dir), self.random_choice(fonts))), size=size
            )
        except OSError:
            font = ImageFont.load_default()

        draw = ImageDraw.Draw(im, "RGBA")

        # Try different positions with the condition that the label´does not
        # overlap with non-white pixels (the structure)
        for _ in range(50):
            y_pos, x_pos = self.get_random_label_position(width, height)
            bounding_box = draw.textbbox(
                (x_pos, y_pos), label_text, font=font
            )  # left, up, right, low
            paste_region = orig_image.crop(bounding_box)
            try:
                mean = ImageStat.Stat(paste_region).mean
            except ZeroDivisionError:
                return np.asarray(im)
            if sum(mean) / len(mean) == 255:
                draw.text((x_pos, y_pos), label_text, font=font, fill=(0, 0, 0, 255))
                break
        return np.asarray(im)

    def add_curved_arrows_to_structure(self, image: np.array) -> np.array:
        """
        This function takes an image of a chemical structure (np.array)
        and adds between 2 and 4 curved arrows in random positions in the
        central part of the image.

        Args:
            image (np.array): Chemical structure depiction

        Returns:
            np.array: Chemical structure depiction with curved arrows
        """
        height, width, _ = image.shape
        image = Image.fromarray(image)
        orig_image = deepcopy(image)
        # Determine area where arrows are pasted.
        x_min, x_max = (int(0.1 * width), int(0.9 * width))
        y_min, y_max = (int(0.1 * height), int(0.9 * height))

        arrow_dir = os.path.normpath(
            str(self.HERE.joinpath("arrow_images/curved_arrows/"))
        )

        for _ in range(self.random_choice(range(2, 4))):
            # Load random curved arrow image, resize and rotate it randomly.
            arrow_image = Image.open(
                os.path.join(
                    str(arrow_dir), self.random_choice(os.listdir(str(arrow_dir)))
                )
            )
            new_arrow_image_shape = int(
                (x_max - x_min) / self.random_choice(range(3, 6))
            ), int((y_max - y_min) / self.random_choice(range(3, 6)))
            arrow_image = self.resize(np.asarray(arrow_image), new_arrow_image_shape)
            arrow_image = Image.fromarray(arrow_image)
            arrow_image = arrow_image.rotate(
                self.random_choice(range(360)),
                resample=self.random_choice(
                    [Image.BICUBIC, Image.NEAREST, Image.BILINEAR]
                ),
                expand=True,
            )
            # Try different positions with the condition that the arrows are
            # overlapping with non-white pixels (the structure)
            for _ in range(50):
                x_position = self.random_choice(
                    range(x_min, x_max - new_arrow_image_shape[0])
                )
                y_position = self.random_choice(
                    range(y_min, y_max - new_arrow_image_shape[1])
                )
                paste_region = orig_image.crop(
                    (
                        x_position,
                        y_position,
                        x_position + new_arrow_image_shape[0],
                        y_position + new_arrow_image_shape[1],
                    )
                )
                mean = ImageStat.Stat(paste_region).mean
                if sum(mean) / len(mean) < 252:
                    image.paste(arrow_image, (x_position, y_position), arrow_image)

                    break
        return np.asarray(image)

    def get_random_arrow_position(self, width: int, height: int) -> Tuple[int, int]:
        """
        Given the width and height of an image (int), this function determines
        a random position to paste a reaction arrow in the outer 15% frame of
        the image

        Args:
            width (_type_): image width
            height (_type_): image height

        Returns:
            Tuple[int, int]: Random arrow position
        """
        if self.random_choice([True, False]):
            y_range = range(0, height)
            x_range = list(range(0, int(0.15 * width))) + list(
                range(int(0.85 * width), width)
            )
        else:
            y_range = list(range(0, int(0.15 * height))) + list(
                range(int(0.85 * height), height)
            )
            x_range = range(0, int(0.5 * width))
        return self.random_choice(y_range), self.random_choice(x_range)

    def add_straight_arrows_to_structure(self, image: np.array) -> np.array:
        """
        This function takes an image of a chemical structure (np.array)
        and adds between 1 and 2 straight arrows in random positions in the
        image (no overlap with other elements)

        Args:
            image (np.array): Chemical structure depiction

        Returns:
            np.array: Chemical structure depiction with straight arrow
        """
        height, width, _ = image.shape
        image = Image.fromarray(image)

        arrow_dir = os.path.normpath(
            str(self.HERE.joinpath("arrow_images/straight_arrows/"))
        )

        for _ in range(self.random_choice(range(1, 3))):
            # Load random curved arrow image, resize and rotate it randomly.
            arrow_image = Image.open(
                os.path.join(
                    str(arrow_dir), self.random_choice(os.listdir(str(arrow_dir)))
                )
            )
            # new_arrow_image_shape = (int(width *
            # self.random_choice(np.arange(0.9, 1.5, 0.1))),
            # int(height/10 * self.random_choice(np.arange(0.7, 1.2, 0.1))))

            # arrow_image = arrow_image.resize(new_arrow_image_shape,
            # resample=Image.BICUBIC)
            # Rotate completely randomly in half of the cases and in 180° steps
            # in the other cases (higher probability that pasting works)
            if self.random_choice([True, False]):
                arrow_image = arrow_image.rotate(
                    self.random_choice(range(360)),
                    resample=self.random_choice(
                        [Image.BICUBIC, Image.NEAREST, Image.BILINEAR]
                    ),
                    expand=True,
                )
            else:
                arrow_image = arrow_image.rotate(self.random_choice([180, 360]))
            new_arrow_image_shape = arrow_image.size
            # Try different positions with the condition that the arrows are
            # overlapping with non-white pixels (the structure)
            for _ in range(50):
                y_position, x_position = self.get_random_arrow_position(width, height)
                x2_position = x_position + new_arrow_image_shape[0]
                y2_position = y_position + new_arrow_image_shape[1]
                # Make sure we only check a region inside of the image
                if x2_position > width:
                    x2_position = width - 1
                if y2_position > height:
                    y2_position = height - 1
                paste_region = image.crop(
                    (x_position, y_position, x2_position, y2_position)
                )
                try:
                    mean = ImageStat.Stat(paste_region).mean
                    if sum(mean) / len(mean) == 255:
                        image.paste(arrow_image, (x_position, y_position), arrow_image)
                        break
                except ZeroDivisionError:
                    pass
        return np.asarray(image)

    def to_grayscale_float_img(self, image: np.array) -> np.array:
        """
        This function takes an image (np.array), converts it to grayscale
        and returns it.

        Args:
            image (np.array): image

        Returns:
            np.array: grayscale float image
        """
        return img_as_float(rgb2gray(image))

    def depict_save(
        self,
        smiles: str,
        images_per_structure: int,
        output_dir: str,
        augment: bool,
        ID: str,
        shape: Tuple[int, int] = (299, 299),
        seed: int = 42,
    ):
        """
        This function takes a SMILES str, the amount of images to create
        per SMILES str and the path of an output directory. It then creates
        images_per_structure depictions of the chemical structure that is
        represented by the SMILES str and saves it as PNG images in output_dir.
        If augment == True, it adds augmentations to the structure depiction.
        If an ID is given, it is used as the base filename. Otherwise, the
        SMILES str is used.

        Args:
            smiles (str): SMILES representation of molecule
            images_per_structure (int): Number of images to create per SMILES
            output_dir (str): output directory path
            augment (bool): Add augmentations (if True)
            ID (str): ID (used for name of saved image)
            shape (Tuple[int, int], optional): im shape. Defaults to (299, 299)
            seed (int, optional): Seed. Defaults to 42.
        """

        depictor = RandomDepictor(seed + 13)

        if not ID:
            name = smiles
        else:
            name = ID
        for n in range(images_per_structure):
            if augment:
                image = depictor(smiles, shape)
            else:
                image = depictor.random_depiction(smiles, shape)
            output_file_path = os.path.join(output_dir, name + "_" + str(n) + ".png")
            sk_io.imsave(output_file_path, img_as_ubyte(image))

    def batch_depict_save(
        self,
        smiles_list: List[str],
        images_per_structure: int,
        output_dir: str,
        augment: bool,
        ID_list: List[str],
        shape: Tuple[int, int] = (299, 299),
        processes: int = 4,
        seed: int = 42,
    ) -> None:
        """
        Batch generation of chemical structure depictions without usage of
        fingerprints. The images are saved at a given path.

        Args:
            smiles_list (List[str]): List of SMILES str
            images_per_structure (int): Amount of images to create per SMILES
            output_dir (str): Output directory
            augment (bool):  indicates whether or not to use augmentations
            ID_list (List[str]): List of IDs (should be as long as smiles_list)
            shape (Tuple[int, int], optional): Defaults to (299, 299).
            processes (int, optional): Number of threads. Defaults to 4.
            seed (int, optional): Seed for random decisions. Defaults to 42.
        """
        starmap_tuple_generator = (
            (
                smiles_list[n],
                images_per_structure,
                output_dir,
                augment,
                ID_list[n],
                shape,
                (seed * n + 1) * len(smiles_list),  # individual seed
            )
            for n in range(len(smiles_list))
        )
        with get_context("spawn").Pool(processes) as p:
            p.starmap(self.depict_save, starmap_tuple_generator)

    def depict_from_fingerprint(
        self,
        smiles: str,
        fingerprints: List[np.array],
        schemes: List[Dict],
        shape: Tuple[int, int] = (299, 299),
        seed: int = 42,
        path_bkg="./backgrounds/",
    ) -> np.array:
        """
        This function takes a SMILES representation of a molecule,
        a list of one or two fingerprints and a list of the corresponding
        fingerprint schemes and generates a chemical structure depiction
        that fits the fingerprint.
        ___
        If only one fingerprint/scheme is given, we assume that they contain
        information for a depiction without augmentations. If two are given,
        we assume that the first one contains information about the depiction
        and the second one contains information about the augmentations.
        ___
        All this function does is set the class attributes in a manner that
        random_choice() knows to not to actually pick parameters randomly.

        Args:
            fingerprints (List[np.array]): List of one or two fingerprints
            schemes (List[Dict]): List of one or two fingerprint schemes
            shape (Tuple[int,int]): Desired output image shape

        Returns:
            np.array: Chemical structure depiction
        """
        # This needs to be done to ensure that the Java Virtual Machine is
        # running when working with multiproessing
        depictor = RandomDepictor(seed=seed)
        self.from_fingerprint = True
        self.active_fingerprint = fingerprints[0]
        self.active_scheme = schemes[0]
        # Depict molecule
        if "indigo" in list(schemes[0].keys())[0]:
            depiction = depictor.depict_and_resize_indigo(smiles, shape)
        elif "rdkit" in list(schemes[0].keys())[0]:
            depiction = depictor.depict_and_resize_rdkit(smiles, shape)
        elif "cdk" in list(schemes[0].keys())[0]:
            depiction = depictor.depict_and_resize_cdk(smiles, shape)

        # Add augmentations
        if len(fingerprints) == 2:
            self.active_fingerprint = fingerprints[1]
            self.active_scheme = schemes[1]
            depiction = self.add_augmentations(depiction)

        self.from_fingerprint, self.active_fingerprint, self.active_scheme = (
            False,
            False,
            False,
        )
        if self.hand_drawn:
            # Augment molecule image
            mol_aug = self.hand_drawn_augment(depiction)

            # Randomly select background image and use is as it is
            backgroud_selected = self.random_choice(os.listdir(path_bkg))
            bkg = cv2.imread(path_bkg + backgroud_selected)
            bkg = cv2.resize(bkg, (256, 256))
            # Combine augmented molecule and augmented background
            p = 0.7
            mol_bkg = cv2.addWeighted(mol_aug, p, bkg, 1 - p, gamma=0)

            """
            If you want to randomly augment the background as well,
            simply comment the previous section and uncomment the next one.
            """

            """# Randomly select background image and augment it
            bkg_aug = self.augment_bkg(bkg)
            bkg_aug = cv2.resize(bkg_aug,(256,256))
            # Combine augmented molecule and augmented background
            p=0.7
            mol_bkg = cv2.addWeighted(mol_aug, p, bkg_aug, 1-p, gamma=0)"""

            # Degrade total image
            depiction = self.degrade_img(mol_bkg)
        return depiction

    def depict_save_from_fingerprint(
        self,
        smiles: str,
        fingerprints: List[np.array],
        schemes: List[Dict],
        output_dir: str,
        filename: str,
        shape: Tuple[int, int] = (299, 299),
        seed: int = 42,
    ) -> None:
        """
        This function takes a SMILES representation of a molecule, a list
        of one or two fingerprints and a list of the corresponding fingerprint
        schemes, generates a chemical structure depiction that fits the
        fingerprint and saves the resulting image at a given path.
        ___
        If only one fingerprint/scheme is given, we assume that they contain
        information for a depiction without augmentations. If two are given,
        we assume that the first one contains information about the depiction
        and the second one contains information about the augmentations.
        ___
        All this function does is set the class attributes in a manner that
        random_choice() knows to not to actually pick parameters randomly.

        Args:
            smiles (str): SMILES representation of molecule
            fingerprints (List[np.array]): List of one or two fingerprints
            schemes (List[Dict]): List of one or two fingerprint schemes
            output_dir (str): output directory
            filename (str): filename
            shape (Tuple[int,int]): output image shape Defaults to (299,299).
            seed (int): Seed for remaining random decisions

        Returns:
            np.array: Chemical structure depiction
        """

        # Generate chemical structure depiction
        image = self.depict_from_fingerprint(smiles, fingerprints, schemes, shape, seed)
        # Save at given_path:
        output_file_path = os.path.join(output_dir, filename + ".png")
        sk_io.imsave(output_file_path, img_as_ubyte(image))

    def batch_depict_save_with_fingerprints(
        self,
        smiles_list: List[str],
        images_per_structure: int,
        output_dir: str,
        ID_list: List[str],
        indigo_proportion: float = 0.15,
        rdkit_proportion: float = 0.3,
        cdk_proportion: float = 0.55,
        aug_proportion: float = 0.5,
        shape: Tuple[int, int] = (299, 299),
        processes: int = 4,
        seed: int = 42,
    ) -> None:
        """
        Batch generation of chemical structure depictions with usage
        of fingerprints. This takes longer than the procedure with
        batch_depict_save but the diversity of the depictions and
        augmentations is ensured. The images are saved in the given
        output_directory

        Args:
            smiles_list (List[str]): List of SMILES str
            images_per_structure (int): Amount of images to create per SMILES
            output_dir (str): Output directory
            ID_list (List[str]): IDs (len: smiles_list * images_per_structure)
            indigo_proportion (float): Indigo proportion. Defaults to 0.15.
            rdkit_proportion (float): RDKit proportion. Defaults to 0.3.
            cdk_proportion (float): CDK proportion. Defaults to 0.55.
            aug_proportion (float): Augmentation proportion. Defaults to 0.5.
            shape (Tuple[int, int]): [description]. Defaults to (299, 299).
            processes (int, optional): Number of threads. Defaults to 4.
        """
        # Duplicate elements in smiles_list images_per_structure times
        smiles_list = [smi for smi in smiles_list for _ in range(images_per_structure)]
        # Generate corresponding amount of fingerprints
        dataset_size = len(smiles_list)
        FR = DepictionFeatureRanges()
        fingerprint_tuples = FR.generate_fingerprints_for_dataset(
            dataset_size,
            indigo_proportion,
            rdkit_proportion,
            cdk_proportion,
            aug_proportion,
        )
        starmap_tuple_generator = (
            (
                smiles_list[n],
                fingerprint_tuples[n],
                [
                    FR.FP_length_scheme_dict[len(element)]
                    for element in fingerprint_tuples[n]
                ],
                output_dir,
                ID_list[n],
                shape,
                n * 100 * seed,
            )
            for n in range(len(fingerprint_tuples))
        )
        with get_context("spawn").Pool(processes) as p:
            p.starmap(self.depict_save_from_fingerprint, starmap_tuple_generator)
        return None

    def batch_depict_with_fingerprints(
        self,
        smiles_list: List[str],
        images_per_structure: int,
        indigo_proportion: float = 0.15,
        rdkit_proportion: float = 0.3,
        cdk_proportion: float = 0.55,
        aug_proportion: float = 0.5,
        shape: Tuple[int, int] = (299, 299),
        processes: int = 4,
        seed: int = 42,
    ) -> None:
        """
        Batch generation of chemical structure depictions with usage
        of fingerprints. This takes longer than the procedure with
        batch_depict_save but the diversity of the depictions and
        augmentations is ensured. The images are saved in the given
        output_directory

        Args:
            smiles_list (List[str]): List of SMILES str
            images_per_structure (int): Amount of images to create per SMILES
            output_dir (str): Output directory
            ID_list (List[str]): IDs (len: smiles_list * images_per_structure)
            indigo_proportion (float): Indigo proportion. Defaults to 0.15.
            rdkit_proportion (float): RDKit proportion. Defaults to 0.3.
            cdk_proportion (float): CDK proportion. Defaults to 0.55.
            aug_proportion (float): Augmentation proportion. Defaults to 0.5.
            shape (Tuple[int, int]): [description]. Defaults to (299, 299).
            processes (int, optional): Number of threads. Defaults to 4.
        """
        # Duplicate elements in smiles_list images_per_structure times
        smiles_list = [smi for smi in smiles_list for _ in range(images_per_structure)]
        # Generate corresponding amount of fingerprints
        dataset_size = len(smiles_list)
        FR = DepictionFeatureRanges()
        fingerprint_tuples = FR.generate_fingerprints_for_dataset(
            dataset_size,
            indigo_proportion,
            rdkit_proportion,
            cdk_proportion,
            aug_proportion,
        )
        starmap_tuple_generator = (
            (
                smiles_list[n],
                fingerprint_tuples[n],
                [
                    FR.FP_length_scheme_dict[len(element)]
                    for element in fingerprint_tuples[n]
                ],
                shape,
                n * 100 * seed,
            )
            for n in range(len(fingerprint_tuples))
        )
        with get_context("spawn").Pool(processes) as p:
            depictions = p.starmap(
                self.depict_from_fingerprint, starmap_tuple_generator
            )
        return list(depictions)


class DepictionFeatureRanges(RandomDepictor):
    """Class for depiction feature fingerprint generation"""

    def __init__(self):
        super().__init__()
        # Fill ranges. By simply using all the depiction and augmentation
        # functions, the available features are saved by the overwritten
        # random_choice function. We just have to make sure to run through
        # every available decision once to get all the information about the
        # feature space that we need.
        smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        # Call every depiction function
        depiction = self(smiles)
        depiction = self.depict_and_resize_cdk(smiles)
        depiction = self.depict_and_resize_rdkit(smiles)
        depiction = self.depict_and_resize_indigo(smiles)
        # Call augmentation function
        depiction = self.add_augmentations(depiction)
        # Generate schemes for Fingerprint creation
        self.schemes = self.generate_fingerprint_schemes()
        (
            self.CDK_scheme,
            self.RDKit_scheme,
            self.Indigo_scheme,
            self.augmentation_scheme,
        ) = self.schemes
        # Generate the pool of all valid fingerprint combinations

        self.generate_all_possible_fingerprints()
        self.FP_length_scheme_dict = {
            len(self.CDK_fingerprints[0]): self.CDK_scheme,
            len(self.RDKit_fingerprints[0]): self.RDKit_scheme,
            len(self.Indigo_fingerprints[0]): self.Indigo_scheme,
            len(self.augmentation_fingerprints[0]): self.augmentation_scheme,
        }

    def random_choice(self, iterable: List, log_attribute: str = False) -> Any:
        """
        In RandomDepictor, this function  would take an iterable, call
        random_choice() on it,  increase the seed attribute by 1 and return
        the result.
        ___
        Here, this function is overwritten, so that it also sets the class
        attribute $log_attribute_range to contain the iterable.
        This way, a DepictionFeatureRanges object can easily be filled with
        all the iterables that define the complete depiction feature space.
        ___
        Args:
            iterable (List): iterable to pick from
            log_attribute (str, optional): ID for fingerprint.
                                           Defaults to False.

        Returns:
            Any: "Randomly" picked element
        """
        if log_attribute:
            setattr(self, "{}_range".format(log_attribute), iterable)
        self.seed += 1
        random.seed(self.seed)
        result = random.choice(iterable)
        # Add result(s) to augmentation_logger
        if log_attribute and self.depiction_features:
            found_logged_attribute = getattr(self.augmentation_logger, log_attribute)
            # If the attribute is not saved in a list, simply write it,
            # otherwise append it
            if not isinstance(found_logged_attribute, list):
                setattr(self.depiction_features, log_attribute, result)
            else:
                setattr(
                    self.depiction_features,
                    log_attribute,
                    found_logged_attribute + [result],
                )
        return result

    def generate_fingerprint_schemes(self) -> List[Dict]:
        """
        Generates fingerprint schemes (see generate_fingerprint_scheme())
        for the depictions with CDK, RDKit and Indigo as well as the
        augmentations.
        ___
         Returns:
            List[Dict]: [cdk_scheme: Dict, rdkit_scheme: Dict,
                         indigo_scheme: Dict, augmentation_scheme: Dict]
        """
        fingerprint_schemes = []
        range_IDs = [att for att in dir(self) if "range" in att]
        # Generate fingerprint scheme for our cdk, indigo and rdkit depictions
        depiction_toolkits = ["cdk", "rdkit", "indigo", ""]
        for toolkit in depiction_toolkits:
            toolkit_range_IDs = [att for att in range_IDs if toolkit in att]
            # Delete toolkit-specific ranges
            # (The last time this loop runs, only augmentation-related ranges
            # are left)
            for ID in toolkit_range_IDs:
                range_IDs.remove(ID)
            toolkit_range_dict = {
                attr[:-6]: list(set(getattr(self, attr))) for attr in toolkit_range_IDs
            }
            fingerprint_scheme = self.generate_fingerprint_scheme(toolkit_range_dict)
            fingerprint_schemes.append(fingerprint_scheme)
        return fingerprint_schemes

    def generate_fingerprint_scheme(self, ID_range_map: Dict) -> Dict:
        """
        This function takes the ID_range_map and returns a dictionary that
        defines where each feature is represented in the depiction feature
        fingerprint.
        ___
        Example:
        >> example_ID_range_map = {'thickness': [0, 1, 2, 3],
                                   'kekulized': [True, False]}
        >> generate_fingerprint_scheme(example_ID_range_map)
        >>>> {'thickness': [{'position': 0, 'one_if': 0},
                            {'position': 1, 'one_if': 1},
                            {'position': 2, 'one_if': 2},
                            {'position': 3, 'one_if': 3}],
            'kekulized': [{'position': 4, 'one_if': True}]}
        Args:
            ID_range_map (Dict): dict that maps an ID (str) of a feature range
                                to the feature range itself (iterable)

        Returns:
            Dict: Map of feature ID (str) and dictionaries that define the
                  fingerprint position and a condition
        """
        fingerprint_scheme = {}
        position = 0
        for feature_ID in ID_range_map.keys():
            feature_range = ID_range_map[feature_ID]
            # Make sure numeric ranges don't take up more than 5 positions
            # in the fingerprint
            if (
                type(feature_range[0]) in [int, float, np.float64, np.float32]
                and len(feature_range) > 5
            ):
                subranges = self.split_into_n_sublists(feature_range, n=3)
                position_dicts = []
                for subrange in subranges:
                    subrange_minmax = (min(subrange), max(subrange))
                    position_dict = {"position": position, "one_if": subrange_minmax}
                    position_dicts.append(position_dict)
                    position += 1
                fingerprint_scheme[feature_ID] = position_dicts
            # Bools take up only one position in the fingerprint
            elif isinstance(feature_range[0], bool):
                assert len(feature_range) == 2
                position_dicts = [{"position": position, "one_if": True}]
                position += 1
                fingerprint_scheme[feature_ID] = position_dicts
            else:
                # For other types of categorical data: Each category gets one
                # position in the FP
                position_dicts = []
                for feature in feature_range:
                    position_dict = {"position": position, "one_if": feature}
                    position_dicts.append(position_dict)
                    position += 1
                fingerprint_scheme[feature_ID] = position_dicts
        return fingerprint_scheme

    def split_into_n_sublists(self, iterable, n: int) -> List[List]:
        """
        Takes an iterable, sorts it, splits it evenly into n lists
        and returns the split lists.

        Args:
            iterable ([type]): Iterable that is supposed to be split
            n (int): Amount of sublists to return
        Returns:
            List[List]: Split list
        """
        iterable = sorted(iterable)
        iter_len = len(iterable)
        sublists = []
        for i in range(0, iter_len, int(np.ceil(iter_len / n))):
            sublists.append(iterable[i: i + int(np.ceil(iter_len / n))])
        return sublists

    def get_number_of_possible_fingerprints(self, scheme: Dict) -> int:
        """
        This function takes a fingerprint scheme (Dict) as returned by
        generate_fingerprint_scheme()
        and returns the number of possible fingerprints for that scheme.

        Args:
            scheme (Dict): Output of generate_fingerprint_scheme()

        Returns:
            int: Number of possible fingerprints
        """
        comb_count = 1
        for feature_key in scheme.keys():
            if len(scheme[feature_key]) != 1:
                # n fingerprint positions -> n options
                # (because only one position can be [1])
                # n = 3 --> [1][0][0] or [0][1][0] or [0][0][1]
                comb_count *= len(scheme[feature_key])
            else:
                # One fingerprint position -> two options: [0] or [1]
                comb_count *= 2
        return comb_count

    def get_FP_building_blocks(self, scheme: Dict) -> List[List[List]]:
        """
        This function takes a fingerprint scheme (Dict) as returned by
        generate_fingerprint_scheme()
        and returns a list of possible building blocks.
        Example:
            scheme = {'thickness': [{'position': 0, 'one_if': 0},
                                    {'position': 1, 'one_if': 1},
                                    {'position': 2, 'one_if': 2},
                                    {'position': 3, 'one_if': 3}],
                      'kekulized': [{'position': 4, 'one_if': True}]}

            --> Output: [[[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]],
                         [[1], [0]]]

        Args:
            scheme (Dict): Output of generate_fingerprint_scheme()

        Returns:
            List that contains the valid fingerprint parts that represent the
            different features

        """
        FP_building_blocks = []
        for feature_key in scheme.keys():
            position_condition_dicts = scheme[feature_key]
            FP_building_blocks.append([])
            # Add every single valid option to the building block
            for position_index in range(len(position_condition_dicts)):
                # Add list of zeros
                FP_building_blocks[-1].append([0] * len(position_condition_dicts))
                # Replace one zero with a one
                FP_building_blocks[-1][-1][position_index] = 1
            # If a feature is described by only one position in the FP,
            # make sure that 0 and 1 are listed options
            if FP_building_blocks[-1] == [[1]]:
                FP_building_blocks[-1].append([0])
        return FP_building_blocks

    def flatten_fingerprint(
        self,
        unflattened_list: List[List],
    ) -> List:
        """
        This function takes a list of lists and returns a list.
        ___
        Looks like this could be one line elsewhere but this function used for
        parallelisation of FP generation and consequently needs to be wrapped
        up in a separate function.

        Args:
            unflattened_list (List[List[X,Y,Z]])

        Returns:
            flattened_list (List[X,Y,Z]):
        """
        flattened_list = [
            element for sublist in unflattened_list for element in sublist
        ]
        return flattened_list

    def generate_all_possible_fingerprints_per_scheme(
        self,
        scheme: Dict,
    ) -> List[List[int]]:
        """
        This function takes a fingerprint scheme (Dict) as returned by
        generate_fingerprint_scheme()
        and returns a List of all possible fingerprints for that scheme.

        Args:
            scheme (Dict): Output of generate_fingerprint_scheme()
            name (str): name that is used for filename of saved FPs

        Returns:
            List[List[int]]: List of fingerprints
        """
        # Determine valid building blocks for fingerprints
        FP_building_blocks = self.get_FP_building_blocks(scheme)
        # Determine cartesian product of valid building blocks to get all
        # valid fingerprints
        FP_generator = product(*FP_building_blocks)
        flattened_fingerprints = list(map(self.flatten_fingerprint, FP_generator))
        return flattened_fingerprints

    def generate_all_possible_fingerprints(self) -> None:
        """
        This function generates all possible valid fingerprint combinations
        for the four available fingerprint schemes if they have not been
        created already. Otherwise, they are loaded from files.
        This function returns None but saves the fingerprint pools as a
        class attribute $ID_fingerprints
        """
        exists_already = False
        FP_names = ["CDK", "RDKit", "Indigo", "augmentation"]
        for scheme_index in range(len(self.schemes)):
            n_FP = self.get_number_of_possible_fingerprints(self.schemes[scheme_index])
            # Load fingerprint pool from file (if it exists)
            FP_filename = "{}_fingerprints.npz".format(FP_names[scheme_index])
            FP_file_path = self.HERE.joinpath(FP_filename)
            if os.path.exists(FP_file_path):
                fps = np.load(FP_file_path)["arr_0"]
                if len(fps) == n_FP:
                    exists_already = True
            # Otherwise, generate the fingerprint pool
            if not exists_already:
                print("No saved fingerprints found. This may take a minute.")
                fps = self.generate_all_possible_fingerprints_per_scheme(
                    self.schemes[scheme_index]
                )
                np.savez_compressed(FP_file_path, fps)
                print(
                    "{} fingerprints were saved in {}.".format(
                        FP_names[scheme_index], FP_file_path
                    )
                )
            setattr(self, "{}_fingerprints".format(FP_names[scheme_index]), fps)
        return

    def convert_to_int_arr(
        self, fingerprints: List[List[int]]
    ) -> List[DataStructs.cDataStructs.ExplicitBitVect]:
        """
        Takes a list of fingerprints (List[int]) and returns them as a list of
        rdkit.DataStructs.cDataStructs.ExplicitBitVect so that they can be
        processed by RDKit's MaxMinPicker.

        Args:
            fingerprints (List[List[int]]): List of fingerprints

        Returns:
            List[DataStructs.cDataStructs.ExplicitBitVect]: Converted arrays
        """
        converted_fingerprints = []
        for fp in fingerprints:
            bitstring = "".join(np.array(fp).astype(str))
            fp_converted = DataStructs.cDataStructs.CreateFromBitString(bitstring)
            converted_fingerprints.append(fp_converted)
        return converted_fingerprints

    def pick_fingerprints(
        self,
        fingerprints: List[List[int]],
        n: int,
    ) -> np.array:
        """
        Given a list of fingerprints and a number n of fingerprints to pick,
        this function uses RDKit's MaxMin Picker to pick n fingerprints and
        returns them.

        Args:
            fingerprints (List[List[int]]): List of fingerprints
            n (int): Number of fingerprints to pick

        Returns:
            np.array: Picked fingerprints
        """

        converted_fingerprints = self.convert_to_int_arr(fingerprints)

        """TODO: I don't like this function definition in the function but
        according to the RDKit Documentation, the fingerprints need to be
        given in the distance function as the default value."""

        def dice_dist(
            fp_index_1: int,
            fp_index_2: int,
            fingerprints: List[
                DataStructs.cDataStructs.ExplicitBitVect
            ] = converted_fingerprints,
        ) -> float:
            """
            Returns the dice similarity between two fingerprints.
            Args:
                fp_index_1 (int): index of first fingerprint in fingerprints
                fp_index_2 (int): index of second fingerprint in fingerprints
                fingerprints (List[cDataStructs.ExplicitBitVect]): fingerprints

            Returns:
                float: Dice similarity between the two fingerprints
            """
            return 1 - DataStructs.DiceSimilarity(
                fingerprints[fp_index_1], fingerprints[fp_index_2]
            )

        # If we want to pick more fingerprints than there are in the pool,
        # simply distribute the complete pool as often as possible and pick
        # the amount that is not dividable by the size of the pool
        picked_fingerprints, n = self.correct_amount_of_FP_to_pick(fingerprints, n)

        picker = MaxMinPicker()
        pick_indices = picker.LazyPick(dice_dist, len(fingerprints), n, seed=42)
        if isinstance(picked_fingerprints, bool):
            picked_fingerprints = np.array([fingerprints[i] for i in pick_indices])
        else:
            picked_fingerprints = np.concatenate(
                (np.array(picked_fingerprints), np.array(([fingerprints[i] for i in pick_indices])))
            )
        return picked_fingerprints

    def correct_amount_of_FP_to_pick(self, fingerprints: List, n: int) -> Tuple[List, int]:
        """
        When picking n elements from a list of fingerprints, if the amount of fingerprints is
        bigger than n, there is no need to pick n fingerprints. Instead, the complete fingerprint
        list is added to the picked fingerprints as often as possible while only the amount
        that is not dividable by the fingerprint pool size is picked.
        ___
        Given a list of fingerprints and the amount of fingerprints to pick n, this function
        returns a list of "picked" fingerprints and (in the ideal case) a corrected lower number
        of fingerprints to be picked

        Args:
            fingerprints (List): _description_
            n (int): _description_

        Returns:
            Tuple[List, int]: _description_
        """
        if n > len(fingerprints):
            oversize_factor = int(n / len(fingerprints))
            picked_fingerprints = np.concatenate([fingerprints for _
                                                  in range(oversize_factor)])
            n = n - len(fingerprints) * oversize_factor
        else:
            picked_fingerprints = False
        return picked_fingerprints, n

    def generate_fingerprints_for_dataset(
        self,
        size: int,
        indigo_proportion: float = 0.15,
        rdkit_proportion: float = 0.3,
        cdk_proportion: float = 0.55,
        aug_proportion: float = 0.5,
    ) -> List[List[int]]:
        """Given a dataset size (int) and (optional) proportions for the
        different types of fingerprints, this function returns

        Args:
            size (int): Desired dataset size, number of returned  fingerprints
            indigo_proportion (float): Indigo proportion. Defaults to 0.15.
            rdkit_proportion (float):  RDKit proportion. Defaults to 0.3.
            cdk_proportion (float):  CDK proportion. Defaults to 0.55.
            aug_proportion (float):  Augmentation proportion. Defaults to 0.5.

        Raises:
            ValueError:
                - If the sum of Indigo, RDKit and CDK proportions is not 1
                - If the augmentation proportion is > 1

        Returns:
            List[List[int]]: List of lists containing the fingerprints.
            ___
            Depending on augmentation_proportion, the depiction fingerprints
            are paired with augmentation fingerprints or not.

        Example output:
            [[$some_depiction_fingerprint, $some augmentation_fingerprint],
             [$another_depiction_fingerprint]
             [$yet_another_depiction_fingerprint]]

        """
        # Make sure that the given proportion arguments make sense
        if sum((indigo_proportion, rdkit_proportion, cdk_proportion)) != 1:
            raise ValueError(
                "Sum of Indigo, CDK and RDKit proportion arguments has to be 1"
            )
        if aug_proportion > 1:
            raise ValueError(
                "The proportion of augmentation fingerprints can't be > 1."
            )
        # Pick and return diverse fingerprints
        picked_Indigo_fingerprints = self.pick_fingerprints(
            self.Indigo_fingerprints, int(size * indigo_proportion)
        )
        picked_RDKit_fingerprints = self.pick_fingerprints(
            self.RDKit_fingerprints, int(size * rdkit_proportion)
        )
        picked_CDK_fingerprints = self.pick_fingerprints(
            self.CDK_fingerprints, int(size * cdk_proportion)
        )
        picked_augmentation_fingerprints = self.pick_fingerprints(
            self.augmentation_fingerprints, int(size * aug_proportion)
        )
        # Distribute augmentation_fingerprints over depiction fingerprints
        fingerprint_tuples = self.distribute_elements_evenly(
            picked_augmentation_fingerprints,
            picked_Indigo_fingerprints,
            picked_RDKit_fingerprints,
            picked_CDK_fingerprints,
        )
        # Shuffle fingerprint tuples randomly to avoid the same smiles
        # always being depicted with the same cheminformatics toolkit
        random.seed(self.seed)
        random.shuffle(fingerprint_tuples)
        return fingerprint_tuples

    def distribute_elements_evenly(
        self, elements_to_be_distributed: List[Any], *args: List[Any]
    ) -> List[List[Any]]:
        """
        This function distributes the elements from elements_to_be_distributed
        evenly over the lists of elements given in args. It can be used to link
        augmentation fingerprints to given lists of depiction fingerprints.

        Example:
        distribute_elements_evenly(["A", "B", "C", "D"], [1, 2, 3], [4, 5, 6])
        Output: [[1, "A"], [2, "B"], [3], [4, "C"], [5, "D"], [6]]
        --> see test_distribute_elements_evenly() in ../Tests/test_functions.py

        Args:
            elements_to_be_distributed (List[Any]): elements to be distributed
            args: Every arg is a list of elements (B)

        Returns:
            List[List[Any]]: List of Lists (B, A) where the elements A are
                             distributed evenly over the elements B according
                             to the length of the list of elements B
        """
        # Make sure that the input is valid
        args_total_len = len([element for sublist in args for element in sublist])
        if len(elements_to_be_distributed) > args_total_len:
            raise ValueError("Can't take more elements to be distributed than in args.")

        output = []
        start_index = 0
        for element_list in args:
            # Define part of elements_to_be_distributed that belongs to this
            # element_sublist
            sublist_len = len(element_list)
            end_index = start_index + int(
                sublist_len / args_total_len * len(elements_to_be_distributed)
            )
            select_elements_to_be_distributed = elements_to_be_distributed[
                start_index:end_index
            ]
            for element_index in range(len(element_list)):
                if element_index < len(select_elements_to_be_distributed):
                    output.append(
                        [
                            element_list[element_index],
                            select_elements_to_be_distributed[element_index],
                        ]
                    )
                else:
                    output.append([element_list[element_index]])
            start_index = start_index + int(
                sublist_len / args_total_len * len(elements_to_be_distributed)
            )
        return output


class RandomMarkushStructureCreator:
    def __init__(self, *args):
        """
        RandomMarkushStructureCreator objects are instantiated with the desired
        inserted R group variables. Otherwise, "R", "X" and "Z" are used.
        """
        # Instantiate RandomDepictor for reproducible random decisions
        self.depictor = RandomDepictor()
        # Define R group variables
        if not args:
            self.r_group_variables = ["R", "X", "Z"]
        else:
            self.r_group_variables = list(args)

        self.potential_indices = range(21)

    def generate_markush_structure_dataset(self, smiles_list: List[str]) -> List[str]:
        """
        This function takes a list of SMILES, replaces 1-4 carbon or hydrogen atoms per
        molecule with R groups and returns the resulting list of SMILES.

        Args:
            smiles_list (List[str]): SMILES representations of molecules

        Returns:
            List[str]: SMILES reprentations of markush structures
        """
        numbers = [self.depictor.random_choice(range(1, 5)) for _ in smiles_list]
        r_group_smiles = [
            self.insert_R_group_var(smiles_list[index], numbers[index])
            for index in range(len(smiles_list))
        ]
        return r_group_smiles

    def insert_R_group_var(self, smiles: str, num: int) -> str:
        """
        This function takes a smiles string and a number of R group variables. It then
        replaces the given number of H or C atoms with R groups and returns the SMILES str.

        Args:
            smiles (str): SMILES (absolute) representation of a molecule
            num (int): number of R group variables to be inserted

        Returns:
            smiles (str): input SMILES with $num inserted R group variables
        """
        smiles = self.add_explicite_hydrogen_to_smiles(smiles)
        potential_replacement_positions = self.get_valid_replacement_positions(smiles)
        r_groups = []
        # Replace C or H in SMILES with *
        # If we would directly insert the R group variables, CDK would replace them with '*'
        # later when removing the explicite hydrogen atoms
        smiles = list(smiles)
        for _ in range(num):
            if len(potential_replacement_positions) > 0:
                position = self.depictor.random_choice(potential_replacement_positions)
                smiles[position] = "*"
                potential_replacement_positions.remove(position)
                r_groups.append(self.get_r_group_smiles())
            else:
                break
        # Remove explicite hydrogen again and get absolute SMILES
        smiles = "".join(smiles)
        smiles = self.remove_explicite_hydrogen_from_smiles(smiles)
        # Replace * with R groups
        for r_group in r_groups:
            smiles = smiles.replace("*", r_group, 1)
        return smiles

    def get_r_group_smiles(self) -> str:
        """
        This function returns a random R group substring that can be inserted
        into an existing SMILES str.

        Returns:
            str: SMILES compatible of R group str
        """
        has_indices = self.depictor.random_choice([True, True, True, True, False])
        r_group_var = self.depictor.random_choice(self.r_group_variables)
        if has_indices:
            index = self.depictor.random_choice(self.potential_indices)
            return f"[{r_group_var}{index}]"
        else:
            return f"[{r_group_var}]"

    def get_valid_replacement_positions(self, smiles: str) -> List[int]:
        """
        Returns positions in a SMILES str where elements in the str can be replaced with
        R groups without endangering its validity

        Args:
            smiles (str): SMILES representation of a molecule

        Returns:
            replacement_positions (List[int]): valid replacement positions for R group variables
        """
        # Add space char to represent insertion position at the end of smiles str
        smiles = f"{smiles} "
        replacement_positions = []
        for index in range(len(smiles)):
            # Be aware of digits --> don't destroy ring syntax
            if not smiles[index].isdigit():
                if smiles[index - 1] == "H":
                    if smiles[index] == "]":
                        replacement_positions.append(index - 1)
                # Only replace "C" and "H"
                elif smiles[index - 1] == "C":
                    # Don't replace "C" in "Cl", "Ca", Cu", etc...
                    if smiles[index] not in [
                        "s",
                        "a",
                        "e",
                        "o",
                        "u",
                        "r",
                        "l",
                        "f",
                        "d",
                        "n",
                    ]:
                        replacement_positions.append(index - 1)
        return replacement_positions

    def add_explicite_hydrogen_to_smiles(self, smiles: str) -> str:
        """
        This function takes a SMILES str and uses CDK to add explicite hydrogen atoms.
        It returns an adapted version of the SMILES str.

        Args:
            smiles (str): SMILES representation of a molecule

        Returns:
            smiles (str): SMILES representation of a molecule with explicite H
        """
        i_atom_container = self.depictor.cdk_smiles_to_IAtomContainer(smiles)

        # Add explicite hydrogen atoms
        cdk_base = "org.openscience.cdk."
        manipulator = JClass(cdk_base + "tools.manipulator.AtomContainerManipulator")
        manipulator.convertImplicitToExplicitHydrogens(i_atom_container)

        # Create absolute SMILES
        smi_flavor = JClass("org.openscience.cdk.smiles.SmiFlavor").Absolute
        smiles_generator = JClass("org.openscience.cdk.smiles.SmilesGenerator")(
            smi_flavor
        )
        smiles = smiles_generator.create(i_atom_container)
        return str(smiles)

    def remove_explicite_hydrogen_from_smiles(self, smiles: str) -> str:
        """
        This function takes a SMILES str and uses CDK to remove explicite hydrogen atoms.
        It returns an adapted version of the SMILES str.

        Args:
            smiles (str): SMILES representation of a molecule

        Returns:
            smiles (str): SMILES representation of a molecule with explicite H
        """
        i_atom_container = self.depictor.cdk_smiles_to_IAtomContainer(smiles)
        # Remove explicite hydrogen atoms
        cdk_base = "org.openscience.cdk."
        manipulator = JClass(cdk_base + "tools.manipulator.AtomContainerManipulator")
        i_atom_container = manipulator.copyAndSuppressedHydrogens(i_atom_container)
        # Create absolute SMILES
        smi_flavor = JClass("org.openscience.cdk.smiles.SmiFlavor").Absolute
        smiles_generator = JClass("org.openscience.cdk.smiles.SmilesGenerator")(
            smi_flavor
        )
        smiles = smiles_generator.create(i_atom_container)
        return str(smiles)
