from __future__ import annotations
import copy
import cv2
from indigo import Indigo
from jpype import startJVM, getDefaultJVMPath
from jpype import JVMNotFoundException, isJVMStarted
from multiprocessing import set_start_method, get_context
import numpy as np
from omegaconf import OmegaConf
import os
from pathlib import Path
from PIL import Image
import random
import re
from rdkit import Chem
from rdkit.Chem import AllChem
from skimage import io as sk_io
from skimage.util import img_as_ubyte
from typing import Callable, Dict, List, Optional, Tuple

from .augmentations import Augmentations
from .cdk_functionalities import CDKFunctionalities
from .config import RandomDepictorConfig
from .indigo_functionalities import IndigoFunctionalities
from .pikachu_functionalities import PikachuFunctionalities
from .rdkit_functionalities import RDKitFuntionalities


# Below version 9.0, PIL stores resampling methods differently
if not hasattr(Image, 'Resampling'):
    Image.Resampling = Image


class RandomDepictor(Augmentations,
                     CDKFunctionalities,
                     IndigoFunctionalities,
                     PikachuFunctionalities,
                     RDKitFuntionalities):
    """
    This class contains everything necessary to generate a variety of
    random depictions with given SMILES strings. An instance of RandomDepictor
    can be called with a SMILES str and returns an np.array that represents
    the RGB image with the given chemical structure.
    """

    def __init__(self, seed: Optional[int] = None, hand_drawn: Optional[bool] = None, *, config: RandomDepictorConfig = None):
        """
        Load the JVM only once, load superatom list (OSRA),
        set context for multiprocessing

        Parameters
        ----------
        seed : int
            seed for random number generator
        hand_drawn : bool
            Whether to augment with hand drawn features
        config : Path object to configuration file in yaml format.
            RandomDepictor section is expected.

        Returns
        -------

        """

        if config is None:
            self._config = RandomDepictorConfig()
        else:
            self._config = copy.deepcopy(config)
        if seed is not None:
            self._config.seed = seed
        if hand_drawn is not None:
            self._config.hand_drawn = hand_drawn

        self.seed = self._config.seed
        self.hand_drawn = self._config.hand_drawn

        self.HERE = Path(__file__).resolve().parent.joinpath("assets")

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
            self.jar_path = self.HERE.joinpath("jar_files/cdk-2.8.jar")
            startJVM(self.jvmPath,
                     "-ea",
                     "-Djava.class.path=" + str(self.jar_path),
                     "-Xmx4096M")

        random.seed(self.seed)

        # Load list of superatoms for label generation
        with open(self.HERE.joinpath("superatom.txt")) as superatoms:
            superatoms = superatoms.readlines()
            self.superatoms = [s[:-2] for s in superatoms]

        # Define PIL resizing methods to choose from:
        self.PIL_resize_methods = [
            Image.Resampling.NEAREST,
            Image.Resampling.BOX,
            Image.Resampling.BILINEAR,
            Image.Resampling.HAMMING,
            Image.Resampling.BICUBIC,
            Image.Resampling.LANCZOS,
        ]

        self.PIL_HQ_resize_methods = self.PIL_resize_methods[4:]

        self.from_fingerprint = False
        self.depiction_features = False

        # Set context for multiprocessing but make sure this only happens once
        try:
            set_start_method("spawn")
        except RuntimeError:
            pass

    @classmethod
    def from_config(cls, config_file: Path) -> 'RandomDepictor':
        try:
            # TODO Needs documentation of config_file yaml format...
            """
            # randepict.yaml
            RandomDepictorConfig:
                seed: 42
                augment: False
                styles:
                    - cdk
            """
            config: RandomDepictorConfig = RandomDepictorConfig.from_config(OmegaConf.load(config_file)[RandomDepictorConfig.__name__])
        except Exception as e:
            print(f"Error loading from {config_file}. Make sure it has {cls.__name__} section. {e}")
            print("Using default config.")
            config = RandomDepictorConfig()
        return RandomDepictor(config=config)

    def __call__(
        self,
        smiles: str,
        shape: Tuple[int, int, int] = (299, 299),
        grayscale: bool = False,
        hand_drawn: bool = False,
    ):
        # Depict structure with random parameters
        # TODO hand_drawn to this call is ignored. Decide which one to keep
        hand_drawn = self.hand_drawn
        if hand_drawn:
            depiction = self.random_depiction(smiles, shape)
            # TODO is call to hand_drawn_augment missing?
        else:
            depiction = self.random_depiction(smiles, shape)
            # Add augmentations
            if self._config.augment:
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

    def random_depiction(
        self,
        smiles: str,
        shape: Tuple[int, int] = (299, 299),
    ) -> np.array:
        """
        This function takes a SMILES and depicts it using Rdkit, Indigo, CDK or PIKACHU.
        The depiction method and the specific parameters for the depiction are
        chosen completely randomly. The purpose of this function is to enable
        depicting a diverse variety of chemical structure depictions.

        Args:
            smiles (str): SMILES representation of molecule
            shape (Tuple[int, int], optional): im shape. Defaults to (299, 299)

        Returns:
            np.array: Chemical structure depiction
        """
        orig_styles = self._config.styles
        # TODO: add this to depiction feature fingerprint
        if self.random_choice([True] + [False] * 5):
            smiles = self._cdk_add_explicite_hydrogen_to_smiles(smiles)
            self._config.styles = [style for style in orig_styles if style != 'pikachu']
        depiction_functions = self.get_depiction_functions(smiles)
        self._config.styles = orig_styles
        for _ in range(3):
            if len(depiction_functions) != 0:
                # Pick random depiction function and call it
                depiction_function = self.random_choice(depiction_functions)
                depiction = depiction_function(smiles=smiles, shape=shape)
                if depiction is False or depiction is None:
                    depiction_functions.remove(depiction_function)
                else:
                    break
            else:
                return None

        if self.hand_drawn:
            path_bkg = self.HERE.joinpath("backgrounds/")
            # Augment molecule image
            mol_aug = self.hand_drawn_augment(depiction)

            # Randomly select background image and use is as it is
            backgroud_selected = self.random_choice(os.listdir(path_bkg))
            bkg = cv2.imread(os.path.join(os.path.normpath(path_bkg), backgroud_selected))
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

    def random_depiction_with_coordinates(
        self,
        smiles: str,
        augment: bool = False,
        shape: Tuple[int, int] = (512, 512),
    ) -> Tuple[np.array, str]:
        """
        This function takes a SMILES and depicts it using Rdkit, Indigo or CDK.
        We cannot use PIKAChU here, as it does not depict given coordinates, but it
        always generates them during the prediction process.
        The depiction method and the specific parameters for the depiction are
        chosen completely randomly. The purpose of this function is to enable
        depicting a diverse variety of chemical structure depictions.

        The depiction (np.array) and the cxSMILES (str) that encodes the coordinates of
        the depicted molecule are returned.

        Args:
            smiles (str): SMILES representation of a molecule
            augment (bool, optional): Whether add augmentations to the image. Defaults to False.
            shape (Tuple[int, int], optional): Image shape. Defaults to (512, 512).

        Returns:
            Tuple[np.array, str]: structure depiction, cxSMILES
        """
        orig_styles = self._config.styles
        self._config.styles = [style for style in orig_styles if style != 'pikachu']
        depiction_functions = self.get_depiction_functions(smiles)
        fun = self.random_choice(depiction_functions)
        self._config.styles = orig_styles
        # TODO: add this to depiction feature fingerprint
        if self.random_choice([True] + [False] * 5):
            smiles = self._cdk_add_explicite_hydrogen_to_smiles(smiles)
        mol_block = self._smiles_to_mol_block(smiles,
                                              self.random_choice(['rdkit',
                                                                  'indigo',
                                                                  'cdk']))
        cxsmiles = self._cdk_mol_block_to_cxsmiles(mol_block,
                                                   ignore_explicite_hydrogens=True)
        depiction = fun(mol_block=mol_block, shape=shape)
        if augment:
            depiction = self.add_augmentations(depiction)
        return depiction, cxsmiles

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

        depiction_functions_registry = {
            'rdkit': self.rdkit_depict,
            'indigo': self.indigo_depict,
            'cdk': self.cdk_depict,
            'pikachu': self.pikachu_depict,
        }
        depiction_functions = [depiction_functions_registry[k]
                               for k in self._config.styles]

        # Remove PIKAChU if there is an isotope
        if re.search("(\[\d\d\d?[A-Z])|(\[2H\])|(\[3H\])|(D)|(T)", smiles):
            if self.pikachu_depict in depiction_functions:
                depiction_functions.remove(self.pikachu_depict)
        if self.has_r_group(smiles):
            # PIKAChU only accepts \[[RXZ]\d*\]
            squared_bracket_content = re.findall("\[.+?\]", smiles)
            for r_group in squared_bracket_content:
                if not re.search("\[[RXZ]\d*\]", r_group):
                    if self.pikachu_depict in depiction_functions:
                        depiction_functions.remove(self.pikachu_depict)
            # "R", "X", "Z" are not depicted by RDKit
            # The same is valid for X,Y,Z and a number
            if self.rdkit_depict in depiction_functions:
                if re.search("\[[RXZ]\]|\[[XYZ]\d+", smiles):
                    depiction_functions.remove(self.rdkit_depict)
            # "X", "R0", [RXYZ]\d+[a-f] and indices above 32 are not depicted by Indigo
            if self.indigo_depict in depiction_functions:
                if re.search("\[R0\]|\[X\]|[4-9][0-9]+|3[3-9]|[XYZR]\d+[a-f]", smiles):
                    depiction_functions.remove(self.indigo_depict)
        # Workaround because PIKAChU fails to depict large structures
        # TODO: Delete workaround when problem is fixed in PIKAChU
        # https://github.com/BTheDragonMaster/pikachu/issues/11
        if len(smiles) > 100:
            if self.pikachu_depict in depiction_functions:
                depiction_functions.remove(self.pikachu_depict)
        return depiction_functions

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
        # path_bkg="./backgrounds/",
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
            depiction = depictor.indigo_depict(smiles, shape)
        elif "rdkit" in list(schemes[0].keys())[0]:
            depiction = depictor.rdkit_depict(smiles, shape)
        elif "pikachu" in list(schemes[0].keys())[0]:
            depiction = depictor.pikachu_depict(smiles, shape)
        elif "cdk" in list(schemes[0].keys())[0]:
            depiction = depictor.cdk_depict(smiles, shape)

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
        # if self.hand_drawn:
        #     # Augment molecule image
        #     mol_aug = self.hand_drawn_augment(depiction)

        #     # Randomly select background image and use is as it is
        #     backgroud_selected = self.random_choice(os.listdir(path_bkg))
        #     bkg = cv2.imread(path_bkg + backgroud_selected)
        #     bkg = cv2.resize(bkg, (256, 256))
        #     # Combine augmented molecule and augmented background
        #     p = 0.7
        #     mol_bkg = cv2.addWeighted(mol_aug, p, bkg, 1 - p, gamma=0)

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
        # depiction = self.degrade_img(mol_bkg)
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
        rdkit_proportion: float = 0.25,
        pikachu_proportion: float = 0.25,
        cdk_proportion: float = 0.35,
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
            rdkit_proportion (float): RDKit proportion. Defaults to 0.25.
            pikachu_proportion (float): PIKAChU proportion. Defaults to 0.25.
            cdk_proportion (float): CDK proportion. Defaults to 0.35.
            aug_proportion (float): Augmentation proportion. Defaults to 0.5.
            shape (Tuple[int, int]): [description]. Defaults to (299, 299).
            processes (int, optional): Number of threads. Defaults to 4.
        """
        # Duplicate elements in smiles_list images_per_structure times
        smiles_list = [smi for smi in smiles_list for _ in range(images_per_structure)]
        # Generate corresponding amount of fingerprints
        dataset_size = len(smiles_list)
        from .depiction_feature_ranges import DepictionFeatureRanges
        FR = DepictionFeatureRanges()
        fingerprint_tuples = FR.generate_fingerprints_for_dataset(
            dataset_size,
            indigo_proportion,
            rdkit_proportion,
            pikachu_proportion,
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
        rdkit_proportion: float = 0.25,
        pikachu_proportion: float = 0.25,
        cdk_proportion: float = 0.35,
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
        from .depiction_feature_ranges import DepictionFeatureRanges
        FR = DepictionFeatureRanges()
        fingerprint_tuples = FR.generate_fingerprints_for_dataset(
            dataset_size,
            indigo_proportion,
            rdkit_proportion,
            pikachu_proportion,
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
            log_attribute (str, optional): ID for fingerprint. Defaults to False.

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

    def _smiles_to_mol_block(
        self,
        smiles: str,
        generate_2d: bool = False,
    ) -> str:
        """
        This function takes a SMILES representation of a molecule and returns
        the content of the corresponding SD file using the CDK.
        ___
        The SMILES parser of the CDK is much more tolerant than the parsers of
        RDKit and Indigo.
        ___

        Args:
            smiles (str): SMILES representation of a molecule
            generate_2d (bool or str, optional): False if no coordinates are created
                                                 Otherwise pick tool for coordinate
                                                 generation:
                                                 "rdkit", "cdk" or "indigo"
                                                 If rdkit or Indigo cannot handle
                                                 certain Markush SMILES, the CDK is used

        Returns:
            mol_block (str): content of SD file of input molecule
        """
        if not generate_2d:
            molecule = self._cdk_smiles_to_IAtomContainer(smiles)
            return self._cdk_iatomcontainer_to_mol_block(molecule)
        elif generate_2d == "cdk":
            molecule = self._cdk_smiles_to_IAtomContainer(smiles)
            molecule = self._cdk_generate_2d_coordinates(molecule)
            molecule = self._cdk_rotate_coordinates(molecule)
            return self._cdk_iatomcontainer_to_mol_block(molecule)
        elif generate_2d == "rdkit":
            if re.search("\[[RXZ]\]|\[[XYZ]\d+", smiles):
                return self._smiles_to_mol_block(smiles, generate_2d="cdk")
            mol_block = self._smiles_to_mol_block(smiles)
            molecule = Chem.MolFromMolBlock(mol_block, sanitize=False)
            if molecule:
                AllChem.Compute2DCoords(molecule)
                mol_block = Chem.MolToMolBlock(molecule)
                atom_container = self._cdk_mol_block_to_iatomcontainer(mol_block)
                atom_container = self._cdk_rotate_coordinates(atom_container)
                return self._cdk_iatomcontainer_to_mol_block(atom_container)
            else:
                raise ValueError(f"RDKit could not read molecule: {smiles}")
        elif generate_2d == "indigo":
            if re.search("\[R0\]|\[X\]|[4-9][0-9]+|3[3-9]|[XYZR]\d+[a-f]", smiles):
                return self._smiles_to_mol_block(smiles, generate_2d="cdk")
            indigo = Indigo()
            mol_block = self._smiles_to_mol_block(smiles)
            molecule = indigo.loadMolecule(mol_block)
            molecule.layout()
            buf = indigo.writeBuffer()
            buf.sdfAppend(molecule)
            mol_block = buf.toString()
            atom_container = self._cdk_mol_block_to_iatomcontainer(mol_block)
            atom_container = self._cdk_rotate_coordinates(atom_container)
            return self._cdk_iatomcontainer_to_mol_block(atom_container)
        elif generate_2d == "pikachu":
            pass

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
