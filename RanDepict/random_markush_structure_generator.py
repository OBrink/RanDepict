from copy import deepcopy
from itertools import cycle
from typing import List
from .randepict import RandomDepictor

class RandomMarkushStructureCreator:
    def __init__(
        self,
        variables_list=None,
        max_index=20,
        max_num=4,
        enumerate_indices=False,
    ):
        """
        RandomMarkushStructureCreator class for the generation of random Markush
        structures from given SMILES strings

        Args:
            variables_list (List[str]): List of R group variables that can be used to
                replace C or H atoms in a SMILES str. desired .
                Otherwise, "R", "X" and "Z" are used.
            max_index (int): Maximum index that is used for R group variables.
            max_num (int): Maximum number of R group variables that are inserted
            enumerate_indices (bool): If True, R group indices are enumerates (eg.
                R1, R2, R3, ...) instead of randomly picked (eg. R1, R20, R4, ...).
        """
        self.depictor = RandomDepictor()
        if variables_list is None:
            self.r_group_variables = ["R", "X", "Y", "Z"]
        else:
            self.r_group_variables = variables_list

        self.potential_indices = range(1, max_index + 1)
        self.max_num = max_num
        self.enumerate_indices = enumerate_indices
        if enumerate_indices:
            self.potential_indices = cycle(self.potential_indices)

    def generate_markush_structure_dataset(self, smiles_list: List[str]) -> List[str]:
        """
        This function takes a list of SMILES, replaces 1-4 carbon or hydrogen atoms per
        molecule with R groups and returns the resulting list of SMILES.

        Args:
            smiles_list (List[str]): SMILES representations of molecules

        Returns:
            List[str]: SMILES reprentations of markush structures
        """
        numbers = [self.depictor.random_choice(range(1, self.max_num + 1))
                   for _ in smiles_list]
        r_group_smiles = [
            self.insert_R_group_var(smiles_list[index], numbers[index])
            for index in range(len(smiles_list))
        ]
        return r_group_smiles

    def insert_R_group_var(self, smiles: str, num: int) -> str:
        """
        This function takes a smiles string and a number of R group variables. It then
        replaces the given number of H or C atoms with R groups and returns the SMILES str.

        Args:
            smiles (str): SMILES (absolute) representation of a molecule
            num (int): number of R group variables to be inserted

        Returns:
            smiles (str): input SMILES with $num inserted R group variables
        """
        smiles = self.depictor._cdk_add_explicite_hydrogen_to_smiles(smiles)
        potential_replacement_positions = self.get_valid_replacement_positions(smiles)
        r_groups = []
        # Replace C or H in SMILES with *
        # If we would directly insert the R group variables, CDK would replace them with '*'
        # later when removing the explicite hydrogen atoms
        smiles = list(smiles)
        orig_potential_indices = deepcopy(self.potential_indices)
        for _ in range(num):
            if len(potential_replacement_positions) > 0:
                position = self.depictor.random_choice(potential_replacement_positions)
                smiles[position] = "*"
                potential_replacement_positions.remove(position)
                r_groups.append(self.get_r_group_smiles())
            else:
                break
        self.potential_indices = orig_potential_indices
        # Remove explicite hydrogen again and get absolute SMILES
        smiles = "".join(smiles)
        smiles = self.depictor._cdk_remove_explicite_hydrogen_from_smiles(smiles)
        # Replace * with R groups
        for r_group in r_groups:
            smiles = smiles.replace("*", r_group, 1)
        return smiles

    def get_r_group_smiles(self) -> str:
        """
        This function returns a random R group substring that can be inserted
        into an existing SMILES str.

        Returns:
            str: SMILES compatible of R group str
        """
        has_indices = self.depictor.random_choice([True, True, True, True, False])
        r_group_var = self.depictor.random_choice(self.r_group_variables)
        if has_indices:
            if self.enumerate_indices:
                index = next(self.potential_indices)
            else:
                index = self.depictor.random_choice(self.potential_indices)
            if self.depictor.random_choice([True, False, False]):
                index_char = self.depictor.random_choice(["a", "b", "c", "d", "e", "f"])
            else:
                index_char = ""
            return f"[{r_group_var}{index}{index_char}]"
        else:
            return f"[{r_group_var}]"

    def get_valid_replacement_positions(self, smiles: str) -> List[int]:
        """
        Returns positions in a SMILES str where elements in the str can be replaced with
        R groups without endangering its validity

        Args:
            smiles (str): SMILES representation of a molecule

        Returns:
            replacement_positions (List[int]): valid replacement positions for R group variables
        """
        # Add space char to represent insertion position at the end of smiles str
        smiles = f"{smiles} "
        replacement_positions = []
        for index in range(len(smiles)):
            # Be aware of digits --> don't destroy ring syntax
            if smiles[index].isdigit():
                continue
            # Don't replace isotopes to not to end up with [13R]
            elif index >= 2 and smiles[index - 2].isdigit() and smiles[index] == "]":
                continue
            # Don't produce charged R groups (eg. "R+")
            elif smiles[index] in ["+", "-"]:
                continue
            elif smiles[index - 1] == "H" and smiles[index] == "]":
                replacement_positions.append(index - 1)
            # Only replace "C" and "H"
            elif smiles[index - 1] == "C":
                # Don't replace "C" in "Cl", "Ca", Cu", etc...
                if smiles[index] not in [
                    "s",
                    "a",
                    "e",
                    "o",
                    "u",
                    "r",
                    "l",
                    "f",
                    "d",
                    "n",
                    "@"  # replacing chiral C leads to invalid SMILES
                ]:
                    replacement_positions.append(index - 1)
        return replacement_positions
