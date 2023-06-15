from jpype import JClass
# import sys
from typing import List
from .randepict import RandomDepictor

class RandomMarkushStructureCreator:
    def __init__(self, *, variables_list=None, max_index=20):
        """
        RandomMarkushStructureCreator objects are instantiated with the desired
        inserted R group variables. Otherwise, "R", "X" and "Z" are used.
        """
        # Instantiate RandomDepictor for reproducible random decisions
        self.depictor = RandomDepictor()
        # Define R group variables
        if variables_list is None:
            self.r_group_variables = ["R", "X", "Y", "Z"]
        else:
            self.r_group_variables = variables_list

        self.potential_indices = range(1, max_index + 1)

    def generate_markush_structure_dataset(self, smiles_list: List[str]) -> List[str]:
        """
        This function takes a list of SMILES, replaces 1-4 carbon or hydrogen atoms per
        molecule with R groups and returns the resulting list of SMILES.

        Args:
            smiles_list (List[str]): SMILES representations of molecules

        Returns:
            List[str]: SMILES reprentations of markush structures
        """
        numbers = [self.depictor.random_choice(range(1, 5)) for _ in smiles_list]
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
        smiles = self.add_explicite_hydrogen_to_smiles(smiles)
        potential_replacement_positions = self.get_valid_replacement_positions(smiles)
        r_groups = []
        # Replace C or H in SMILES with *
        # If we would directly insert the R group variables, CDK would replace them with '*'
        # later when removing the explicite hydrogen atoms
        smiles = list(smiles)
        for _ in range(num):
            if len(potential_replacement_positions) > 0:
                position = self.depictor.random_choice(potential_replacement_positions)
                smiles[position] = "*"
                potential_replacement_positions.remove(position)
                r_groups.append(self.get_r_group_smiles())
            else:
                break
        # Remove explicite hydrogen again and get absolute SMILES
        smiles = "".join(smiles)
        smiles = self.remove_explicite_hydrogen_from_smiles(smiles)
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

    def add_explicite_hydrogen_to_smiles(self, smiles: str) -> str:
        """
        This function takes a SMILES str and uses CDK to add explicite hydrogen atoms.
        It returns an adapted version of the SMILES str.

        Args:
            smiles (str): SMILES representation of a molecule

        Returns:
            smiles (str): SMILES representation of a molecule with explicite H
        """
        i_atom_container = self.depictor._cdk_smiles_to_IAtomContainer(smiles)

        # Add explicite hydrogen atoms
        cdk_base = "org.openscience.cdk."
        manipulator = JClass(cdk_base + "tools.manipulator.AtomContainerManipulator")
        manipulator.convertImplicitToExplicitHydrogens(i_atom_container)

        # Create absolute SMILES
        smi_flavor = JClass("org.openscience.cdk.smiles.SmiFlavor").Absolute
        smiles_generator = JClass("org.openscience.cdk.smiles.SmilesGenerator")(
            smi_flavor
        )
        smiles = smiles_generator.create(i_atom_container)
        return str(smiles)

    def remove_explicite_hydrogen_from_smiles(self, smiles: str) -> str:
        """
        This function takes a SMILES str and uses CDK to remove explicite hydrogen atoms.
        It returns an adapted version of the SMILES str.

        Args:
            smiles (str): SMILES representation of a molecule

        Returns:
            smiles (str): SMILES representation of a molecule with explicite H
        """
        i_atom_container = self.depictor._cdk_smiles_to_IAtomContainer(smiles)
        # Remove explicite hydrogen atoms
        cdk_base = "org.openscience.cdk."
        manipulator = JClass(cdk_base + "tools.manipulator.AtomContainerManipulator")
        i_atom_container = manipulator.copyAndSuppressedHydrogens(i_atom_container)
        # Create absolute SMILES
        smi_flavor = JClass("org.openscience.cdk.smiles.SmiFlavor").Absolute
        smiles_generator = JClass("org.openscience.cdk.smiles.SmilesGenerator")(
            smi_flavor
        )
        smiles = smiles_generator.create(i_atom_container)
        return str(smiles)
