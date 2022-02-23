from RanDepict import RandomDepictor
import time
import os
import shutil


def main():
    """
    This script depicts caffeine n times and writes the runtime into a file.
        - without augmentations with fingerprint picking
        - with augmentations with fingerprint picking
        - without augmentations without fingerprint picking
        - with augmentations without fingerprint picking
    for n in [100*2**n for n in range(1, 11)]
    """
    smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    with RandomDepictor() as depictor:
        for number in [100*2**n for n in range(0, 11)]:
            with open('time_report.txt', 'a') as time_report:
                tmp_dir = "tmp"
                if not os.path.exists(tmp_dir):
                    os.mkdir(tmp_dir)
                # Depict SMILES n times without augmentation with FP picking
                start = time.time()
                depictor.batch_depict_save_with_fingerprints([smiles],
                                                             number,
                                                             tmp_dir,
                                                             [f"{num}.png" for num in range(number)],
                                                             aug_proportion=0,
                                                             processes=1)
                end = time.time()
                time_report.write('{}\t'.format(number))
                time_report.write('{}\t'.format(end-start))
                shutil.rmtree(tmp_dir, ignore_errors=True)

                # Depict SMILES n times with augmentation with FP picking
                if not os.path.exists(tmp_dir):
                    os.mkdir(tmp_dir)
                start = time.time()
                depictor.batch_depict_save_with_fingerprints([smiles],
                                                             number,
                                                             tmp_dir,
                                                             [f"{num}.png" for num in range(number)],
                                                             aug_proportion=1,
                                                             processes=1)
                end = time.time()
                time_report.write('{}\t'.format(end-start))
                shutil.rmtree(tmp_dir, ignore_errors=True)

                # Depict SMILES n times without augmentation without FP picking
                if not os.path.exists(tmp_dir):
                    os.mkdir(tmp_dir)
                start = time.time()
                depictor.batch_depict_save([smiles], number, tmp_dir, False, ['caffeine'], (299, 299), processes=1)
                end = time.time()
                time_report.write('{}\t'.format(end-start))
                shutil.rmtree(tmp_dir, ignore_errors=True)

                # Depict SMILES n times with augmentation without FP picking
                if not os.path.exists(tmp_dir):
                    os.mkdir(tmp_dir)
                start = time.time()
                depictor.batch_depict_save([smiles], number, tmp_dir, True, ['caffeine'], (299, 299), processes=1)
                end = time.time()
                time_report.write('{}\t'.format(end-start))
                time_report.write('\n')
                shutil.rmtree(tmp_dir, ignore_errors=True)
    return


if __name__ == '__main__':
    main()
