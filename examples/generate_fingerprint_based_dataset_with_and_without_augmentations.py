import sys
import os
from typing import List, Tuple, Dict
import numpy as np
from skimage import io as sk_io
from skimage.util import img_as_ubyte
import pickle

from multiprocessing import get_context
from RanDepict import RandomDepictor, DepictionFeatureRanges


class FingerprintDatasetWithAndWithoutAugmentationsCreator(RandomDepictor):
    """
    RandomDepictor with method to depict a fingerprint-based dataset with and without
    augmentations
    --> Evaluation of the influence of the augmentations on an OCSR model
    --> Result: Dataset of images without augmentations and the same images with augmentations

    """
    def __init__(self,):
        super().__init__()

    def batch_depict_save_with_and_without_aug_with_fingerprints(
        self,
        smiles_list: List[str],
        images_per_structure: int,
        output_dir: str,
        ID_list: List[str],
        indigo_proportion: float = 0.2,
        rdkit_proportion: float = 0.3,
        cdk_proportion: float = 0.5,
        shape: Tuple[int, int] = (299, 299),
        processes: int = 4,
        seed: int = 42,
    ) -> None:
        """
        Batch generation of chemical structure depictions with usage
        of fingerprints. Every image is saved without augmentations
        and with augmentations.
        This takes longer than the procedure with
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
            shape (Tuple[int, int]): [description]. Defaults to (299, 299).
            processes (int, optional): Number of threads. Defaults to 4.
        """
        # Duplicate elements in smiles_list images_per_structure times
        smiles_list = [
            smi for smi in smiles_list for _ in range(images_per_structure)]
        # Generate corresponding amount of fingerprints
        dataset_size = len(smiles_list)
        FR = DepictionFeatureRanges()
        fingerprint_tuples = FR.generate_fingerprints_for_dataset(
            dataset_size,
            indigo_proportion,
            rdkit_proportion,
            cdk_proportion,
            aug_proportion=1,
        )
        with open("fingerprint_tuples.pkl", "wb") as fingerprint_file:
            pickle.dump(fingerprint_tuples, fingerprint_file)

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
            p.starmap(
                self.depict_save_from_fingerprint_with_and_without_aug,
                starmap_tuple_generator)
        return None

    def depict_save_from_fingerprint_with_and_without_aug(
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
        of two fingerprints and a list of the corresponding fingerprint
        schemes, generate a chemical structure depiction that fits the
        fingerprint first fingerprint and saves the resulting image at a given path.
        It then applies augmentations according to the second fingerprint
        and also saves the augmented structure depiction.
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
        try:
            depiction, augmented_depiction = self.depict_from_fingerprint(
                smiles, fingerprints, schemes, shape, seed)
            # Save at given_path:
            output_file_path = os.path.join(output_dir, filename + ".png")
            sk_io.imsave(output_file_path, img_as_ubyte(depiction))
            # Save at given_path:
            output_file_path = os.path.join(output_dir, filename + "_aug.png")
            sk_io.imsave(output_file_path, img_as_ubyte(augmented_depiction))
        except IndexError:
            with open("error_log.txt", "a") as error_log:
                error_message = f"Could not depict SMILES {smiles} due to IndexError.\n"
                error_log.write(error_message)

    def depict_from_fingerprint(
        self,
        smiles: str,
        fingerprints: List[np.array],
        schemes: List[Dict],
        shape: Tuple[int, int] = (299, 299),
        seed: int = 42,
    ) -> Tuple[np.array, np.array]:
        """
        This function takes a SMILES representation of a molecule,
        a list of two fingerprints (depiction fingerprint, augmentation fingerprint)
        and a list of the corresponding fingerprint schemes and generates a chemical
        structure depiction that fits the fingerprint. It then adds augmentations
        according to the second fingerprints and returns both the non-augmented and
        the augmented structure depictions (np.array)
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
        # running when working with multiprocessing
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

        if depiction is False or depiction is None:
            # For the rare case: Use CDK
            self.from_fingerprint, self.active_fingerprint, self.active_scheme = (
                False,
                False,
                False,
            )
            depiction = depictor.depict_and_resize_cdk(smiles, shape)
            with open('error_log.txt', 'a') as error_log:
                error_log.write(f'Failed depicting SMILES: {smiles}\n')
                error_log.write('It was depicted using CDK WITHOUT fingerprints.\n')
        # Add augmentations
        self.active_fingerprint = fingerprints[1]
        self.active_scheme = schemes[1]
        augmented_depiction = self.add_augmentations(depiction)

        self.from_fingerprint, self.active_fingerprint, self.active_scheme = (
            False,
            False,
            False,
        )
        return depiction, augmented_depiction


def main():
    """
    This script generates a dataset of structure depictions from a file that contains
    IDs and SMILES (structure of file per line: "$ID,$SMILES\n"). Every chemical structure
    depiction is saved with and without added augmentations.
    """
    input_file_path = sys.argv[1]
    output_path = sys.argv[2]
    # Read input file
    with open(input_file_path, 'r') as input_file:
        ids: List = []
        smiles: List = []
        for line in input_file.readlines()[:]:
            id, smi = line[:-1].split(',')
            ids.append(id)
            smiles.append(smi)
    # Generate balanced dataset of non-augmented and augmented depictions
    depictor = FingerprintDatasetWithAndWithoutAugmentationsCreator()
    depictor.batch_depict_save_with_and_without_aug_with_fingerprints(
        smiles_list=smiles,
        images_per_structure=1,
        output_dir=output_path,
        ID_list=ids,
        processes=15,
        seed=42,
    )


if __name__ == '__main__':
    if len(sys.argv) == 3:
        main()
    else:
        print(f'Usage: {sys.argv[0]} ID_SMILES_dataset output_dir')
