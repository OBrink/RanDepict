from indigo import Indigo
from indigo import IndigoException
from indigo.renderer import IndigoRenderer
import io
import numpy as np
from skimage import io as sk_io
from skimage.color import rgba2rgb
from skimage.util import img_as_ubyte
from typing import Tuple


class IndigoFunctionalities:
    """
    Child class of RandomDepictor that contains all RDKit-related functions.
    ___
    This class does not work on its own. It is meant to be used as a child class.
    """

    def indigo_depict(
        self,
        smiles: str = None,
        mol_block: str = None,
        shape: Tuple[int, int] = (512, 512)
    ) -> np.array:
        """
        This function takes a mol block str and an image shape.
        It renders the chemical structures using Indigo with random
        rendering/depiction settings and returns an RGB image (np.array)
        with the given image shape.

        Args:
            smiles (str): SMILES representation of molecule
            mol_block (str): mol block representation of molecule
            shape (Tuple[int, int], optional): im shape. Defaults to (512, 512)

        Returns:
            np.array: Chemical structure depiction
        """
        # Instantiate Indigo with random settings and IndigoRenderer
        indigo, renderer = self.get_random_indigo_rendering_settings()
        if not smiles and not mol_block:
            raise ValueError("Either smiles or mol_block must be provided")
        if smiles:
            mol_block = self._smiles_to_mol_block(smiles,
                                                  generate_2d=self.random_choice(
                                                      ["rdkit", "cdk", "indigo"]
                                                  ))
        try:
            molecule = indigo.loadMolecule(mol_block)
        except IndigoException:
            return None
        # Kekulize in 67% of cases
        if not self.random_choice(
            [True, True, False], log_attribute="indigo_kekulized"
        ):
            molecule.aromatize()
        temp = renderer.renderToBuffer(molecule)
        temp = io.BytesIO(temp)
        depiction = sk_io.imread(temp)
        depiction = self.resize(depiction, (shape[0], shape[1]))
        depiction = rgba2rgb(depiction)
        depiction = img_as_ubyte(depiction)
        return depiction

    def get_random_indigo_rendering_settings(
        self, shape: Tuple[int, int] = (299, 299)
    ) -> Indigo:
        """
        This function defines random rendering options for the structure
        depictions created using Indigo.
        It returns an Indigo object with the settings.

        Args:
            shape (Tuple[int, int], optional): im shape. Defaults to (299, 299)

        Returns:
            Indigo: Indigo object that contains depictions settings
        """
        # Define random shape for depiction (within boundaries);)
        indigo = Indigo()
        renderer = IndigoRenderer(indigo)
        # Get slightly distorted shape
        y, x = shape
        indigo.setOption("render-image-width", x)
        indigo.setOption("render-image-height", y)
        # Set random bond line width
        bond_line_width = float(
            self.random_choice(
                np.arange(0.5, 2.5, 0.1), log_attribute="indigo_bond_line_width"
            )
        )
        indigo.setOption("render-bond-line-width", bond_line_width)
        # Set random relative thickness
        relative_thickness = float(
            self.random_choice(
                np.arange(0.5, 1.5, 0.1), log_attribute="indigo_relative_thickness"
            )
        )
        indigo.setOption("render-relative-thickness", relative_thickness)
        # Output_format: PNG
        indigo.setOption("render-output-format", "png")
        # Set random atom label rendering model
        # (standard is rendering terminal groups)
        if self.random_choice([True] + [False] * 19, log_attribute="indigo_labels_all"):
            # show all atom labels
            indigo.setOption("render-label-mode", "all")
        elif self.random_choice(
            [True] + [False] * 3, log_attribute="indigo_labels_hetero"
        ):
            indigo.setOption(
                "render-label-mode", "hetero"
            )  # only hetero atoms, no terminal groups
        # Render bold bond for Haworth projection
        if self.random_choice([True, False], log_attribute="indigo_render_bold_bond"):
            indigo.setOption("render-bold-bond-detection", "True")
        # Render labels for stereobonds
        stereo_style = self.random_choice(
            ["ext", "old", "none"], log_attribute="indigo_stereo_label_style"
        )
        indigo.setOption("render-stereo-style", stereo_style)
        # Collapse superatoms (default: expand)
        if self.random_choice(
            [True, False], log_attribute="indigo_collapse_superatoms"
        ):
            indigo.setOption("render-superatom-mode", "collapse")
        return indigo, renderer
