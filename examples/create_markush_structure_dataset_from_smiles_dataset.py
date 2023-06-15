import sys
from typing import List, Tuple
from multiprocessing import Pool

from RanDepict import RandomMarkushStructureCreator


def helper(lines: List[str], seed=42) -> List[str]:
    """
    Take list of "$id,$smiles" str, separate IDs and SMILES,
    replace SMILES with R group SMILES, return output list in
    input format "$id,$smiles"

    Args:
        lines (List[str]): as read by readlines() from input file
        seed (_type_): seed for pseudo-random decisions
    """
    id_list, smiles_list = split_id_from_smiles(lines)
    markush_creator = RandomMarkushStructureCreator()
    markush_creator.depictor.seed = seed
    numbers = [markush_creator.depictor.random_choice(range(1, 5))
               for _ in smiles_list]
    output_list = []
    for index in range(len(smiles_list)):
        try:
            r_group_smiles = markush_creator.insert_R_group_var(smiles_list[index], numbers[index])
            output_list.append(f"{id_list[index]},{r_group_smiles}")
        except:
            with open("SMILES_ds_creation_error_log", "a") as error_log:
                print(f"Problem with SMILES: {smiles_list[index]}")
                exc = sys.exc_info()[0]
                print(exc)
                error_log.write(f"Problem with SMILES: {smiles_list[index]}\n{exc}")
    with open("SMILES_with_R_groups", "a") as output_file:
        for line in output_list:
            output_file.write(line + '\n')

def split_id_from_smiles(lines: List[str]) -> Tuple[List[str], List[str]]:
    # Split IDs from SMILES in input_format
    id_list = []
    smiles_list = []
    for line in lines:
        line = line[:-1]
        id, smiles = line.split("\t")
        id_list.append(id)
        smiles_list.append(smiles)
    return id_list, smiles_list

def split_smiles_list(smiles_list: List[str], n: int) -> List[List[str]]:
    """
    Split up given list into n chunks and return list of n lists

    Args:
        smiles_list (List[str]): input list
        n (int): desired amount of output lists

    Returns:
        List[List[str]]: List of n chunks
    """
    # Code from
    # https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    k, m = divmod(len(smiles_list), n)
    return [smiles_list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
            for i in range(n)]


def main():
    """
    This script reads IDs/SMILES from a file ("$ID,$SMILES\n" per line), and
    automatically inserts random R groups using RanDepict's RandomMarkushStructureCreator.
    It writes a file with the IDs and the adapted SMILES in the same structure.
    """
    with open(sys.argv[1], "r") as input_file:
        smiles_lists = split_smiles_list(input_file.readlines(), 1000)
    starmap_tuples = [(smiles_lists[index], index * 1000000)
                      for index in range(len(smiles_lists))]
    with Pool(40) as pool:
        _ = pool.starmap(helper, starmap_tuples)


if __name__ == "__main__":
    main()
