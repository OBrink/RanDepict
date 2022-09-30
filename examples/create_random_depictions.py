import os
from typing import Tuple, List
import argparse
from multiprocessing import Process
import time
import random
from skimage.util import img_as_ubyte
from skimage import io as sk_io

from RanDepict import RandomDepictor

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class RandomDepictor(RandomDepictor):
    """
    This class is a child of RanDepict.RandomDepictor and contains some
    additional functions to write the images.
    """
    def __init__(
        self,
        seed=(random.randint(0, 100))
    ) -> None:
        super().__init__(seed=seed)

    def batch_depict_save(
        self,
        smiles_list: List[str],
        n_non_augmented: int,
        n_augmented: int,
        ID_list: List[str],
        chunksize: int,
        shape: Tuple[int, int] = (299, 299),
        num_processes: int = 20,
        seed: int = (random.randint(0, 100)),
        timeout_limit: int = 1800
    ) -> None:
        """
        This function takes a list of SMILES str, the amount of images
        to create per SMILES str and the path of an output directory.
        It then creates images_per_structure depictions of each chemical
        structure that is represented by the SMILES strings and saves them
        in in output_dir. If an ID list (list with names of
        same length as smiles_list that contains unique IDs), the IDs will
        be used to name the files.
        """

        counter = (n for n in range(int(len(smiles_list)/chunksize)))

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        smiles_list = chunks(smiles_list, chunksize)
        IDs = chunks(ID_list, chunksize)

        async_proc_args = (
            (
                next(smiles_list),
                n_non_augmented,
                n_augmented,
                shape,
                next(IDs),
                (seed * n + 1),  # individual seed
            )
            for n in counter
        )
        process_list = []
        while True:
            for proc, init_time in process_list:
                # Remove finished processes
                if not proc.is_alive():
                    process_list.remove((proc, init_time))
                # Remove timed out processes
                elif time.time() - init_time >= timeout_limit:
                    process_list.remove((proc, init_time))
                    proc.terminate()
                    proc.join()
            if len(process_list) < num_processes:
                # Start new processes
                for _ in range(num_processes-len(process_list)):
                    try:
                        p = Process(target=self.depict_save,
                                    args=next(async_proc_args))
                        process_list.append((p, time.time()))
                        p.start()
                    except StopIteration:
                        break
            if len(process_list) == 0:
                break


    def depict_save(
        self,
        smiles: List[str],
        n_non_augmented: int,
        n_augmented: int,
        shape: Tuple[int, int],
        ID: List[int],
        seed: int = (random.randint(0, 100)),
    ):
        """
        This function takes a list of SMILES str, the amount of
        images to create per SMILES str and the path of an output
        directory. It then creates images_per_structure depictions
        of the chemical structure that is represented by the SMILES
        strings and saves it in output_dir.
        The given ID is used as the base filename.
        """
        depictor = RandomDepictor(seed + 13)
        for smiles_index in range(len(smiles)):
            smi = smiles[smiles_index]
            for _ in range(n_augmented):
                for n in range(5):
                    try:
                        output_path = f"RanDepict_dataset/{ID[smiles_index]}_aug_{n}.png"
                        sk_io.imsave(output_path, img_as_ubyte(depictor(smi, shape)))
                        break
                    except Exception as e:
                        print(f'An exception occured:\n{e}',
                              'It is ignored for now and we simply try to depict this structure again...')
                else:
                    print('FAILED depicting structure: {}. Tried five times.'.format(smi))
            for _ in range(n_non_augmented):
                for _ in range(5):
                    try:
                        output_path = f"RanDepict_dataset/{ID[smiles_index]}_aug_{n}.png"
                        sk_io.imsave(output_path, img_as_ubyte(depictor(smi, shape)))
                        break
                    except Exception as e:
                        print(f'An exception occured:\n{e}',
                              'It is ignored for now and we simply try to depict this structure again...')
                else:
                    print('FAILED depicting structure: {}. Tried five times.'.format(smi))
        return


def main() -> None:
    '''
    This script reads a SMILES, the annotationa and IDs from a file and
    generates three augmented and three non-augmented depictions per chemical structure.
    The images are saved.
    ___
    Structure of input file:
    ID1,SMILES1,annotation\n
    ID2,SMILES2,annotation\n
    (...)
    The annotation is an array saved as a str ([1  2  3  4  ..  .. X])
    ___
    '''
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="+")
    args = parser.parse_args()

    # Read input data from file
    ids = []
    smiles = []
    with open(args.file[0], "r") as fp:
        for line in fp.readlines():
            ids.append(line.strip("\n").split("\t")[0])
            smiles.append(line.strip("\n").split("\t")[1])

    if not os.path.exists('RanDepict_dataset'):
        os.mkdir('RanDepict_dataset')

    # Set desired image shape and number of depictions per SMILES and output paths
    im_per_SMILES_noaug = 1
    im_per_SMILES_aug = 1
    depiction_img_shape = (299, 299)

    SMILES_chunksize = 1500
    with RandomDepictor() as depictor:
        depictor.batch_depict_save(
            smiles,
            im_per_SMILES_noaug,
            im_per_SMILES_aug,
            ids,
            SMILES_chunksize,
            depiction_img_shape,
            20,
            random.randint(0, 100),
            1800)


if __name__ == '__main__':
    main()
