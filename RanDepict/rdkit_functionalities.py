from __future__ import annotations

import os
import numpy as np
import io
from skimage import io as sk_io
from skimage.util import img_as_ubyte

from typing import Tuple, List, Dict

from rdkit import Chem
from rdkit.Chem.rdAbbreviations import CondenseMolAbbreviations
from rdkit.Chem.rdAbbreviations import GetDefaultAbbreviations
from rdkit.Chem.Draw import rdMolDraw2D


class RDKitFuntionalities:
    """
    Child class of RandomDepictor that contains all RDKit-related functions.
    ___
    This class does not work on its own. It is meant to be used as a child class.
    """
    def rdkit_depict(
        self,
        smiles: str = None,
        mol_block: str = None,
        has_R_group: bool = False,
        shape: Tuple[int, int] = (512, 512)
    ) -> np.array:
        """
        This function takes a mol_block str and an image shape.
        It renders the chemical structures using Rdkit with random
        rendering/depiction settings and returns an RGB image (np.array)
        with the given image shape.

        Args:
            smiles (str, Optional): SMILES representation of molecule
            mol block (str, Optional): mol block representation of molecule
            has_R_group (bool): Whether the molecule has R groups (used to determine
                                whether or not to use atom numbering as it can be
                                confusing with R groups indices) for SMILES, this is
                                checked using a simple regex. This argument only has
                                an effect if the mol_block is provided.
                                # TODO: check this in mol_block
            shape (Tuple[int, int], optional): im shape. Defaults to (512, 512)

        Returns:
            np.array: Chemical structure depiction
        """
        if not smiles and not mol_block:
            raise ValueError("Either smiles or mol_block must be provided")
        if smiles:
            has_R_group = self.has_r_group(smiles)
            mol_block = self._smiles_to_mol_block(smiles,
                                                  generate_2d=self.random_choice(
                                                      ["rdkit", "cdk", "indigo"]
                                                  ))

        mol = Chem.MolFromMolBlock(mol_block, sanitize=True)
        if mol:
            return self.rdkit_depict_from_mol_object(mol, has_R_group, shape)
        else:
            print("RDKit was unable to read input:\n{}\n{}\n".format(smiles, mol_block))
            return None

    def rdkit_depict_from_mol_object(
        self,
        mol: Chem.rdchem.Mol,
        has_R_group: bool = False,
        shape: Tuple[int, int] = (512, 512),
    ) -> np.array:
        """
        This function takes a mol object and an image shape.
        It renders the chemical structures using Rdkit with random
        rendering/depiction settings and returns an RGB image (np.array)
        with the given image shape.

        Args:
            mol (Chem.rdchem.Mol): RDKit mol object
            has_R_group (bool): Whether the molecule has R groups (used to determine
                                whether or not to use atom numbering as it can be
                                confusing with R groups indices)
                                # TODO: check this in mol object
            shape (Tuple[int, int], optional): im shape. Defaults to (512, 512)

        Returns:
            np.array: Chemical structure depiction
        """
        # Get random depiction settings
        if self.random_choice([True, False], log_attribute="rdkit_collapse_superatoms"):
            abbrevs = self.random_choice(self.get_all_rdkit_abbreviations())
            mol = CondenseMolAbbreviations(mol, abbrevs)

        depiction_settings = self.get_random_rdkit_rendering_settings(
            has_R_group=has_R_group)
        rdMolDraw2D.PrepareAndDrawMolecule(depiction_settings, mol)
        depiction = depiction_settings.GetDrawingText()
        depiction = sk_io.imread(io.BytesIO(depiction))
        # Resize image to desired shape
        depiction = self.resize(depiction, shape)
        depiction = img_as_ubyte(depiction)
        return np.asarray(depiction)

    def get_all_rdkit_abbreviations(
        self,
    ) -> List[Chem.rdAbbreviations._vectstruct]:
        """
        This function returns the Default abbreviations for superatom and functional
        group collapsing in RDKit as well as alternative abbreviations defined in
        rdkit_alternative_superatom_abbreviations.txt.

        Returns:
            Chem.rdAbbreviations._vectstruct: RDKit's data structure that contains the
            abbreviations
        """
        abbreviations = []
        abbreviations.append(GetDefaultAbbreviations())
        abbr_path = self.HERE.joinpath("rdkit_alternative_superatom_abbreviations.txt")

        with open(abbr_path) as alternative_abbreviations:
            split_lines = [line[:-1].split(",")
                           for line in alternative_abbreviations.readlines()]
            swap_dict = {line[0]: line[1:] for line in split_lines}

        abbreviations.append(self.get_modified_rdkit_abbreviations(swap_dict))
        for key in swap_dict.keys():
            new_labels = []
            for label in swap_dict[key]:
                if label[:2] in ["n-", "i-", "t-"]:
                    label = f"{label[2:]}-{label[0]}"
                elif label[-2:] in ["-n", "-i", "-t"]:
                    label = f"{label[-1]}-{label[:-2]}"
                new_labels.append(label)
            swap_dict[key] = new_labels
        abbreviations.append(self.get_modified_rdkit_abbreviations(swap_dict))
        return abbreviations

    def get_modified_rdkit_abbreviations(
        self,
        swap_dict: Dict
    ) -> Chem.rdAbbreviations._vectstruct:
        """
        This function takes a dictionary that maps the original superatom/FG label in
        the RDKit abbreviations to the desired labels, replaces them as defined in the
        dictionary and returns the abbreviations in RDKit's preferred format.

        Args:
            swap_dict (Dict): Dictionary that maps the original label (eg. "Et") to the
                              desired label (eg. "C2H5"), a displayed label (eg.
                              "C<sub>2</sub>H<sub>5</sub>") and a reversed display label
                              (eg. "H<sub>5</sub>C<sub>2</sub>").
                              Example:
                              {"Et": [
                                  "C2H5",
                                  "C<sub>2</sub>H<sub>5</sub>"
                                  "H<sub>5</sub>C<sub>2</sub>"
                              ]}

        Returns:
            Chem.rdAbbreviations._vectstruct: Modified abbreviations
        """
        alt_abbreviations = GetDefaultAbbreviations()
        for abbr in alt_abbreviations:
            alt_label = swap_dict.get(abbr.label)
            if alt_label:
                abbr.label, abbr.displayLabel, abbr.displayLabelW = alt_label
        return alt_abbreviations

    def get_random_rdkit_rendering_settings(
        self,
        has_R_group: (bool) = False,
        shape: Tuple[int, int] = (299, 299)
    ) -> rdMolDraw2D.MolDraw2DCairo:
        """
        This function defines random rendering options for the structure
        depictions created using rdkit. It returns an MolDraw2DCairo object
        with the settings.

        Args:
            has_R_group (bool): SMILES representation of molecule
            shape (Tuple[int, int], optional): im_shape. Defaults to (299, 299)

        Returns:
            rdMolDraw2D.MolDraw2DCairo: Object that contains depiction settings
        """
        y, x = shape
        # Instantiate object that saves the settings
        depiction_settings = rdMolDraw2D.MolDraw2DCairo(y, x)
        # Stereo bond annotation
        if self.random_choice(
            [True, False], log_attribute="rdkit_add_stereo_annotation"
        ):
            depiction_settings.drawOptions().addStereoAnnotation = True
        if self.random_choice(
            [True, False], log_attribute="rdkit_add_chiral_flag_labels"
        ):
            depiction_settings.drawOptions().includeChiralFlagLabel = True
        # Atom indices
        if self.random_choice(
            [True, False, False, False], log_attribute="rdkit_add_atom_indices"
        ):
            if not has_R_group:
                depiction_settings.drawOptions().addAtomIndices = True
        # Bond line width
        bond_line_width = self.random_choice(
            range(1, 5), log_attribute="rdkit_bond_line_width"
        )
        depiction_settings.drawOptions().bondLineWidth = bond_line_width
        # Draw terminal methyl groups
        if self.random_choice(
            [True, False], log_attribute="rdkit_draw_terminal_methyl"
        ):
            depiction_settings.drawOptions().explicitMethyl = True
        # Label font type and size
        font_dir = self.HERE.joinpath("fonts/")
        font_path = os.path.join(
            str(font_dir),
            self.random_choice(
                os.listdir(str(font_dir)), log_attribute="rdkit_label_font"
            ),
        )
        depiction_settings.drawOptions().fontFile = font_path
        min_font_size = self.random_choice(
            range(10, 20), log_attribute="rdkit_min_font_size"
        )
        depiction_settings.drawOptions().minFontSize = min_font_size
        depiction_settings.drawOptions().maxFontSize = 30
        # Rotate the molecule
        # depiction_settings.drawOptions().rotate = self.random_choice(range(360))
        # Fixed bond length
        fixed_bond_length = self.random_choice(
            range(30, 45), log_attribute="rdkit_fixed_bond_length"
        )
        depiction_settings.drawOptions().fixedBondLength = fixed_bond_length
        # Comic mode (looks a bit hand drawn)
        if self.random_choice(
            [True, False, False, False, False], log_attribute="rdkit_comic_style"
        ):
            depiction_settings.drawOptions().comicMode = True
        # Keep it black and white
        depiction_settings.drawOptions().useBWAtomPalette()
        return depiction_settings
