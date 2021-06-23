import sys
import os
from copy import deepcopy
from typing import List, Tuple
import argparse
sys.path.append('../RanDepict/')
from RanDepict import random_depictor
from multiprocessing.pool import MaybeEncodingError


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
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="+")
    args = parser.parse_args()


    # Read input data from file
    for file_in in args.file:
        chembl = []
        smiles = []
        with open(file_in, "r") as fp:
            for line in fp.readlines():
                chembl.append(line.strip("\n").split(",")[0])
                smiles.append(line.strip("\n").split(",")[1])


    # Set desired image shape and number of depictions per SMILES and output paths
    im_per_SMILES = 3
    depiction_img_shape = (299, 299)
    aug_image_path = os.path.normpath(file_in + "_augmented")
    not_aug_image_path = os.path.normpath(file_in + "_not_augmented")


    # Create output directories if necessary
    if not os.path.exists(aug_image_path):
        os.mkdir(aug_image_path)
    if not os.path.exists(not_aug_image_path):
        os.mkdir(not_aug_image_path)


    # Generate depictions without augmentations
    unprocessed_IDs = chembl
    while len(unprocessed_IDs) != 0:   
        # If the process has been aborted before, it can be restarted without re-depicting already depicted structures:
        unprocessed_IDs, unprocessed_SMILES = update_SMILES_list(zip(chembl, smiles), not_aug_image_path)
        if len(unprocessed_IDs)!= 0:
            print('Start depicting {} structures (no augmentations).'.format(len(unprocessed_IDs)))
            try:
                with random_depictor() as depictor:
                    depictor.batch_depict_save(
                        unprocessed_SMILES, im_per_SMILES, not_aug_image_path, depiction_img_shape, unprocessed_IDs, 10
                    )
            except MaybeEncodingError as e:
                print(e)
                print('The error is ignored for now and the depiction processes is restarted (already depicted structures are not depicted again)')
        else:
            print("Done. All structures have been depicted without augmentations")


    # Generate depictions with augmentations
    unprocessed_IDs = chembl
    while len(unprocessed_IDs) != 0:
        # If the process has been aborted before, it can be restarted without re-depicting already depicted structures:
        unprocessed_IDs, unprocessed_SMILES = update_SMILES_list(zip(chembl, smiles), aug_image_path)
        if len(unprocessed_IDs)!= 0:
            print('Start depicting {} structures (no augmentations).'.format(len(unprocessed_IDs)))
            try:
                with random_depictor() as depictor:
                    depictor.batch_depict_augment_save(
                        unprocessed_SMILES, im_per_SMILES, aug_image_path, depiction_img_shape, unprocessed_IDs, 10
                    )
            except MaybeEncodingError as e:
                print(e)
                print('The error is ignored for now and the depiction processes is restarted (already depicted structures are not depicted again)')
        else:
            print("Done. All structures have been depicted with augmentations")


    # Write file with paths of saved images
    with open(file_in + "_paths", "w") as fp:
        for i in range(len(chembl)):
            for j in range(im_per_SMILES):
                fp.write(
                    not_aug_image_path
                    + chembl[i]
                    + "_"
                    + str(j)
                    + ","
                    + smiles[i]
                    + "\n"
                )
            for k in range(im_per_SMILES):
                fp.write(
                    aug_image_path
                    + chembl[i]
                    + "_"
                    + str(k)
                    + ","
                    + smiles[i]
                    + "\n"
                )


def update_SMILES_list(
        ID_SMILES_tuples: List[Tuple[str, str]],
        path: str,
        ) -> Tuple[List[str], List[str]]:
    '''
    This function takes a List of tuples containing IDs (str) and SMILES (str) and
    the output path (str). It checks which of the corresponding structures already
    have been depicted and returns two lists of IDs and SMILES of undepicted structures.
    This way, we don't have to start from scratch if the process was aborted.
    '''
    # Get list of IDs of already depicted structures
    already_processed = [img_name.split('_')[0] for img_name in os.listdir(path)]
    # Get list of SMILES of not-yet depicted structures
    updated_ID_SMILES_tuples = [(tup[0], tup[1]) for tup in ID_SMILES_tuples
                                if tup[0] not in already_processed]
    if len(updated_ID_SMILES_tuples) != 0:
        IDs, SMILES = zip(*updated_ID_SMILES_tuples)
    else:
        IDs = []
        SMILES = []
    return IDs, SMILES

if __name__ == "__main__":
    main()
