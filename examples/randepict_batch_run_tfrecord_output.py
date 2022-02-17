import os
import io
from typing import Tuple, List
import argparse
import numpy as np
from multiprocessing import get_context
from PIL import Image
import random
import tensorflow as tf
from RanDepict import RandomDepictor


class RandomDepictor(RandomDepictor):
    """This function is a child of RanDepict.random_depictor and contains some
    additional functions to write the images into tfrecord files."""

    def __init__(self, seed=42) -> None:
        super().__init__(seed=seed)

    def batch_depict_save(
        self,
        smiles_list: List[str],
        n_non_augmented: int,
        n_augmented: int,
        tokens: List[np.array],
        ID_list: List[str],
        chunksize: int,
        shape: Tuple[int, int] = (299, 299),
        processes: int = 20,
        seed: int = 42,
    ) -> None:
        """This function takes a list of SMILES str, the amount of images to
        create per SMILES str and the path of an output directory.
        It then creates images_per_structure depictions of each chemical
        structure that is represented by the SMILES strings and saves them
        in tfrecord files in output_dir.
        If an ID list (list with names of same length as smiles_list that
        contains unique IDs), the IDs will be used to name the files."""

        counter = (n for n in range(int(len(smiles_list) / chunksize)))

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        smiles_list = chunks(smiles_list, chunksize)
        tokens = chunks(tokens, chunksize)
        IDs = chunks(ID_list, chunksize)

        starmap_tuple_generator = (
            (
                next(smiles_list),
                n_non_augmented,
                n_augmented,
                next(tokens),
                shape,
                next(IDs),
                (seed * n + 1),  # individual seed
            )
            for n in counter
        )

        with get_context("spawn").Pool(processes) as p:
            p.starmap(self.depict_save, starmap_tuple_generator)

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = (
                value.numpy()
            )  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def depict_save(
        self,
        smiles: List[str],
        n_non_augmented: int,
        n_augmented: int,
        tokens: List[np.array],
        shape: Tuple[int, int],
        ID: List[int],
        seed: int = 0,
    ):
        """This function takes a list of SMILES str, the amount of images to
        create per SMILES str and the path of an output directory.
        It then creates images_per_structure depictions of the chemical
        structure that is represented by the SMILES strings and saves it
        in a tfrecord file in output_dir.
        The given ID is used as the base filename."""
        depictor = RandomDepictor(seed + 13)

        tfrecord_name = "tfrecord_ds/dataset_DS-{}.tfrecord".format(ID[0])
        writer = tf.io.TFRecordWriter(tfrecord_name)
        for smiles_index in range(len(smiles)):
            smi = smiles[smiles_index]
            token_list = tokens[smiles_index].astype(np.uint32)
            for _ in range(n_augmented):
                for _ in range(5):
                    try:
                        self.save_in_tfrecord(token_list, depictor(smi, shape), writer)
                        break
                    except Exception as e:
                        print(
                            "An exception occured:\n{}\n".format(e),
                            "It is ignored for now and we",
                            "simply try to depict this structure again...",
                        )
                else:
                    print(
                        "FAILED depicting structure: {}.".format(smi),
                        "Tried five times.",
                    )

            for _ in range(n_non_augmented):
                for _ in range(5):
                    try:
                        self.save_in_tfrecord(
                            token_list, depictor.random_depiction(smi, shape), writer
                        )
                        break
                    except Exception as e:
                        print(
                            "An exception occured:\n{}\n".format(e),
                            "It is ignored for now and we",
                            "simply try to depict this structure again...",
                        )
                else:
                    print(
                        "FAILED depicting structure: {}.".format(smi),
                        "Tried five times.",
                    )
        return

    def save_in_tfrecord(
        self, train_caption: np.array, image: np.array, writer: tf.io.TFRecordWriter
    ) -> None:
        """
        This function takes a list of captions (token list, np.array)
        and a list of images (np.array) as well as a file index.
        It saves the images and captions in a tfrecord file.

        Args:
            train_captions (List[np.array]): np.array([0,1,2,3,1,1,2,0,...])
            images (List[np.array]): array representing the structure depiction
            file_index (int): index for naming tfrecord files
        """
        if not os.path.exists("tfrecord_ds"):
            os.mkdir("tfrecord_ds")
        image = Image.fromarray(image)
        with io.BytesIO() as output:
            image.save(output, format="PNG")
            raw = output.getvalue()

            feature = {
                "image_raw": self._bytes_feature(raw),
                "caption": self._bytes_feature(train_caption.tobytes()),
            }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        serialized = example.SerializeToString()
        writer.write(serialized)
        return None


def parse_arguments() -> argparse.Namespace:
    """
    Parse arguments with argparse.

    Returns:
        Returns an namespace object that holds the given arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="+")
    return parser.parse_args()


def main() -> None:
    """
    This script reads a SMILES, the annotationa and IDs from a file and
    generates three augmented and three non-augmented depictions per chemical
    structure. The images are saved in tfrecord files.
    ___
    Structure of input file:
    ID1,SMILES1,annotation\n
    ID2,SMILES2,annotation\n
    (...)
    Here, the annotation is an array saved as a str ([1  2  3  4  ..  .. X])
    ___
    """

    # Parse arguments
    args = parse_arguments()

    # Read input data from file
    for file_in in args.file:
        ID_list = []
        smiles_list = []
        tokens_list = []
        with open(file_in, "r") as fp:
            for line in fp.readlines():
                if line[-1] == "\n":
                    line = line[:-1]
                line = line.replace("  ", " ").replace(" ", ",")
                ID, smiles, tokens = line.split(";")
                ID_list.append(ID)
                smiles_list.append(smiles)
                tokens_list.append(np.array(eval(tokens)))

    # Set image shape and number of depictions per SMILES and output paths
    im_per_SMILES_noaug = 1
    im_per_SMILES_aug = 3
    depiction_img_shape = (299, 299)

    if not os.path.exists("tfrecord_ds"):
        os.mkdir("tfrecord_ds")

    # If SMILES_chunksize is 100,
    # then 100*im_per_SMILES_noaug*im_per_SMILES_aug are
    # saved in one tfrecord file
    SMILES_chunksize = 1500
    with RandomDepictor() as depictor:
        depictor.batch_depict_save(
            smiles_list,
            im_per_SMILES_noaug,
            im_per_SMILES_aug,
            tokens_list,
            ID_list,
            SMILES_chunksize,
            depiction_img_shape,
            None,
            random.randint(0, 100),
        )


if __name__ == "__main__":
    main()
