# RanDepict Examples

## Usage

- RanDepictNotebook.ipynb: A complete set of examples which demonstrate how to use the RanDepict package.
- create_random_depictions.py: A script for the generation of random depictions with and without augmentations using a test file containing IDs and SMILES. This script does not use the depiction feature fingerprints.
    - python3 create_random_depictions.py id_smiles_sample.txt
- randepict_batch_run_tfrecord_output.py: A script for the generation of random depictions with and without augmentations using a test file containing an ID,
the SMILES and some sort of annotation per line. This script saves batches of images and annotations in tfrecord files instead of generating image files. This makes handling big datasets a lot easier when working with Tensorflow. 
    - python3 randepict_batch_run_tfrecord_output.py tfrecord_creation_data_sample.txt
