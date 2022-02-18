from RanDepict import RandomDepictor
import time


def main():
    """
    This script depicts caffeine 1000 times and writes the runtime into a file.
        - without augmentations with fingerprint picking
        - with augmentations with fingerprint picking
        - without augmentations without fingerprint picking
        - with augmentations without fingerprint picking
    """
    smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    with RandomDepictor() as depictor:
        with open('time_report.txt', 'w') as time_report:
            # Depict SMILES 1000 times without augmentation with FP picking
            start = time.time()
            _ = depictor.batch_depict_with_fingerprints([smiles],
                                                        1000,
                                                        aug_proportion=0,
                                                        processes=None)
            end = time.time()
            time_report.write('Depiction of 1000 structures without ' +
                              'augmentations with fingerprint picking\n')
            time_report.write('{} seconds \n ___\n\n'.format(end-start))

            # Depict SMILES 1000 times with augmentation with FP picking
            start = time.time()
            _ = depictor.batch_depict_with_fingerprints([smiles],
                                                        1000,
                                                        aug_proportion=1,
                                                        processes=None)
            end = time.time()
            time_report.write('Depiction of 1000 structures with ' +
                              'augmentations with fingerprint picking\n')
            time_report.write('{} seconds \n ___\n\n'.format(end-start))

            # Depict SMILES 1000 times without augmentation without FP picking
            start = time.time()
            for _ in 1000:
                _ = depictor.random_depiction(smiles)
            end = time.time()
            time_report.write('Depiction of 1000 structures without ' +
                              'augmentations with fingerprint picking\n')
            time_report.write('{} seconds \n ___\n\n'.format(end-start))

            # Depict SMILES 1000 times with augmentation without FP picking
            start = time.time()
            for _ in 1000:
                _ = depictor(smiles)
            end = time.time()
            time_report.write('Depiction of 1000 structures without ' +
                              'augmentations with fingerprint picking\n')
            time_report.write('{} seconds \n ___\n\n'.format(end-start))
    return


if __name__ == '__main__':
    main()
