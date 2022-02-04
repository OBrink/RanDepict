import numpy as np
from RanDepict import DepictionFeatureRanges

def main() -> None:
    """
    This script generates all possible valid fingerprint combinations for the four
    available fingerprint schemes. They are saved in $NAME_FP_pool.npz in the working directory.
    """
    FP_names = ['CDK', 'RDKit', 'Indigo', 'augmentation']
    FR = DepictionFeatureRanges()
    for scheme_index in range(len(FR.schemes)):
        n_FP = FR.get_number_of_possible_fingerprints(FR.schemes[scheme_index])
        print('There are {} {} fingerprint combinations.'.format(n_FP, FP_names[scheme_index]))
        FPs = FR.generate_all_possible_fingerprints(FR.schemes[scheme_index])
        filename = '{}_FP_pool.npz'
        np.savez_compressed(filename, FPs)
    return

if __name__ == '__main__':
    main()