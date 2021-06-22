import os
import argparse
from RanDepict import random_depictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="+")

    args = parser.parse_args()
    print(args)
    depiction_img_shape = (299, 299)
    for file_in in args.file:
        chembl = []
        smiles = []
        with open(file_in, "r") as fp:
            for i, line in enumerate(fp):
                try:
                    chembl.append(line.strip("\n").split(",")[0])
                    smiles.append(line.strip("\n").split(",")[1])
                except Exception as e:
                    print(line)
    print(len(smiles))
    print(len(chembl))

    if not os.path.exists(file_in + "_not_augmented"):
        os.mkdir(file_in + "_not_augmented")

    if not os.path.exists(file_in + "_augmented"):
        os.mkdir(file_in + "_augmented")

    with random_depictor() as depictor:
        depictor.batch_depict_save(
            smiles, 3, file_in + "_not_augmented", depiction_img_shape, chembl, 10
        )

    with random_depictor() as depictor:
        depictor.batch_depict_augment_save(
            smiles, 3, file_in + "_augmented", depiction_img_shape, chembl, 10
        )

    with open(file_in + "_paths", "w") as fp:
        for i in range(len(chembl)):
            for j in range(3):
                fp.write(
                    file_in
                    + "_not_augmented/"
                    + chembl[i]
                    + "_"
                    + str(j)
                    + ","
                    + smiles[i]
                    + "\n"
                )
            for k in range(3):
                fp.write(
                    file_in
                    + "_augmented/"
                    + chembl[i]
                    + "_"
                    + str(k)
                    + ","
                    + smiles[i]
                    + "\n"
                )


if __name__ == "__main__":
    main()
