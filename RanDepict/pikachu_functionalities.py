
import numpy as np
from pikachu.drawing import drawing
from pikachu.smiles.smiles import read_smiles
from typing import Tuple


class PikachuFunctionalities:
    def get_random_pikachu_rendering_settings(
        self, shape: Tuple[int, int] = (512, 512)
    ) -> drawing.Options:
        """
        This function defines random rendering options for the structure
        depictions created using PIKAChU.
        It returns an pikachu.drawing.drawing.Options object with the settings.

        Args:
            shape (Tuple[int, int], optional): im shape. Defaults to (512, 512)

        Returns:
            options: Options object that contains depictions settings
        """
        options = drawing.Options()
        options.height, options.width = shape
        options.bond_thickness = self.random_choice(np.arange(0.5, 2.2, 0.1),
                                                    log_attribute="pikachu_bond_line_width")
        options.bond_length = self.random_choice(np.arange(10, 25, 1),
                                                 log_attribute="pikachu_bond_length")
        options.chiral_bond_width = options.bond_length * self.random_choice(
            np.arange(0.05, 0.2, 0.01)
        )
        options.short_bond_length = self.random_choice(np.arange(0.2, 0.6, 0.05),
                                                       log_attribute="pikachu_short_bond_length")
        options.double_bond_length = self.random_choice(np.arange(0.6, 0.8, 0.05),
                                                        log_attribute="pikachu_double_bond_length")
        options.bond_spacing = options.bond_length * self.random_choice(
            np.arange(0.15, 0.28, 0.01),
            log_attribute="pikachu_bond_spacing"

        )
        options.padding = self.random_choice(np.arange(10, 50, 5),
                                             log_attribute="pikachu_padding")
        # options.font_size_large = 5
        # options.font_size_small = 3
        return options

    def pikachu_depict(
        self,
        smiles: str = None,
        shape: Tuple[int, int] = (512, 512)
    ) -> np.array:
        """
        This function takes a mol block str and an image shape.
        It renders the chemical structures using PIKAChU with random
        rendering/depiction settings and returns an RGB image (np.array)
        with the given image shape.

        Args:
            smiles (str, optional): smiles representation of molecule
            shape (Tuple[int, int], optional): im shape. Defaults to (512, 512)

        Returns:
            np.array: Chemical structure depiction
        """
        structure = read_smiles(smiles)

        depiction_settings = self.get_random_pikachu_rendering_settings()
        if "." in smiles:
            drawer = drawing.draw_multiple(structure, options=depiction_settings)
        else:
            drawer = drawing.Drawer(structure, options=depiction_settings)
        depiction = drawer.get_image_as_array()
        depiction = self.central_square_image(depiction)
        depiction = self.resize(depiction, (shape[0], shape[1]))
        return depiction
