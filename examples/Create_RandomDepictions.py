import os
from typing import List, Tuple
import argparse
from RanDepict import RandomDepictor
from multiprocessing.pool import MaybeEncodingError


def parse_arguments() -> argparse.Namespace:
    """
    Parse arguments with argparse.

    Returns:
        Returns an namespace object that holds the given arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="+")
    return parser.parse_args()


def read_input_file(filename: str) -> Tuple[List, List]:
    """
    Read IDs and SMILES from a file with expected structure:
    ID1,SMILES1\n
    ID2,SMILES2\n
    (...)

    Args:
        filename (str): Input file name

    Returns:
        Tuple[List, List]: List of IDs, List of SMILES
    """
    ids = []
    smiles = []
    with open(filename, "r") as fp:
        for line in fp.readlines():
            ids.append(line.strip("\n").split(",")[0])
            smiles.append(line.strip("\n").split(",")[1])
    return ids, smiles


def coordinate_depiction(
        ids: List[str],
        smiles: List[str],
        image_dir: str,
        im_per_smiles: int,
        augment: bool,
        depiction_img_shape: Tuple[int, int],
        ) -> None:
    """
    Coordination function for batch depiction. If an error occurs,
    the process is restarted without re-depicting the already depicted
    structures.

    Args:
        ids (List[str])
        smiles (List[str])
        image_dir (str)
        im_per_smiles (int)
        augment (bool)
        depiction_img_shape (Tuple[int, int])
    """
    # Generate depictions without augmentations
    unprocessed_IDs = ids
    while len(unprocessed_IDs) != 0:
        # If the process has been aborted before,
        # it can be restarted without re-depicting already depicted structures:
        unprocessed_IDs, unprocessed_SMILES = update_SMILES_list(zip(
                                                            ids, smiles),
                                                            image_dir)
        if len(unprocessed_IDs) != 0:
            print('Start depicting {} structures (no augmentations).'.format(
                len(unprocessed_IDs)
                ))
            try:
                with RandomDepictor() as depictor:
                    depictor.batch_depict_save(
                        unprocessed_SMILES,
                        im_per_smiles,
                        image_dir,
                        augment,
                        unprocessed_IDs,
                        depiction_img_shape,
                        10
                    )
            except MaybeEncodingError as e:
                print(e)
                print('The error is ignored for now and the depiction',
                      'processes is restarted (already depicted structures',
                      'are not depicted again)')
        else:
            print('Done. All structures have been depicted',
                  'Augmentations: {}'.format(augment))


def write_paths_to_file(
        input_filename: str,
        ids: List[str],
        smiles: List[str],
        non_aug_image_path: str,
        im_per_smiles_non_aug: int,
        aug_image_path: str,
        im_per_smiles_aug: int
        ) -> None:
    """
    Write file with paths of saved images

    Args:
        input_filename (str)
        ids (List[str])
        smiles (List[str])
        non_aug_image_path (str)
        im_per_smiles_non_aug (int)
        aug_image_path (str)
        im_per_smiles_aug (int)
    """
    with open(input_filename + "_paths", "w") as fp:
        for i in range(len(ids)):
            for j in range(im_per_smiles_non_aug):
                fp.write(
                    non_aug_image_path
                    + ids[i]
                    + "_"
                    + str(j)
                    + ","
                    + smiles[i]
                    + "\n"
                )
            for k in range(im_per_smiles_aug):
                fp.write(
                    aug_image_path
                    + ids[i]
                    + "_"
                    + str(k)
                    + ","
                    + smiles[i]
                    + "\n")


def update_SMILES_list(
        ID_SMILES_tuples: List[Tuple[str, str]],
        path: str,
        ) -> Tuple[List[str], List[str]]:
    '''
    This function takes a List of tuples containing IDs (str) and SMILES (str)
    and the output path (str). It checks which of the corresponding structures
    already have been depicted and returns two lists of IDs and SMILES of
    undepicted structures.
    This way, we don't have to start from scratch if the process was aborted.
    '''
    # Get list of IDs of already depicted structures
    already_processed = [img_name.split('_')[0]
                         for img_name in os.listdir(path)]
    # Get list of SMILES of not-yet depicted structures
    updated_ID_SMILES_tuples = [(tup[0], tup[1]) for tup in ID_SMILES_tuples
                                if tup[0] not in already_processed]
    if len(updated_ID_SMILES_tuples) != 0:
        IDs, SMILES = zip(*updated_ID_SMILES_tuples)
    else:
        IDs = []
        SMILES = []
    return IDs, SMILES


def main() -> None:
    '''
    This script reads SMILES corresponding IDs from a file and generates three
    augmented and three non-augmented depictions per chemical structure.
    The images are saved in $filename_augmented/$filename_not_augmented.
    ___
    Structure of input file:
    ID1,SMILES1\n
    ID2,SMILES2\n
    (...)
    ___
    '''

    args = parse_arguments()
    ids, smiles = read_input_file(args.file[0])

    # Set desired image shape/number of depictions per SMILES and output paths
    im_per_smiles_non_aug = 3
    im_per_smiles_aug = 3
    depiction_img_shape = (299, 299)
    aug_image_path = os.path.normpath(args.file[0] + "_augmented")
    not_aug_image_path = os.path.normpath(args.file[0] + "_not_augmented")

    # Create output directories if necessary
    if not os.path.exists(aug_image_path):
        os.mkdir(aug_image_path)
    if not os.path.exists(not_aug_image_path):
        os.mkdir(not_aug_image_path)

    # Generate  depictions without augmentations
    coordinate_depiction(
        ids,
        smiles,
        not_aug_image_path,
        im_per_smiles_non_aug,
        augment=False,
        depiction_img_shape=depiction_img_shape,
        )

    # Generate depictions with augmentations
    coordinate_depiction(
        ids,
        smiles,
        aug_image_path,
        im_per_smiles_aug,
        augment=True,
        depiction_img_shape=depiction_img_shape,
        )

    write_paths_to_file(
        args.file[0],
        ids,
        smiles,
        not_aug_image_path,
        im_per_smiles_non_aug,
        aug_image_path,
        im_per_smiles_aug
    )


if __name__ == "__main__":
    main()
