from __future__ import annotations
import base64
from jpype import JClass, JException
import numpy as np
from skimage import io as sk_io
from skimage.util import img_as_ubyte
from typing import Tuple


class CDKFunctionalities:
    """
    Child class of RandomDepictor that contains all CDK-related functions.
    ___
    This class does not work on its own. It is meant to be used as a child class.
    """
    def cdk_depict(
        self,
        smiles: str = None,
        mol_block: str = None,
        has_R_group: bool = False,
        shape: Tuple[int, int] = (512, 512)
    ) -> np.array:
        """
        This function takes a mol block str and an image shape.
        It renders the chemical structures using CDK with random
        rendering/depiction settings and returns an RGB image (np.array)
        with the given image shape.
        The general workflow here is a JPype adaptation of code published
        by Egon Willighagen in 'Groovy Cheminformatics with the Chemistry
        Development Kit':
        https://egonw.github.io/cdkbook/ctr.html#depict-a-compound-as-an-image
        with additional adaptations to create all the different depiction
        types from
        https://github.com/cdk/cdk/wiki/Standard-Generator

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
        molecule = self._cdk_mol_block_to_iatomcontainer(mol_block)
        depiction = self._cdk_render_molecule(molecule, has_R_group, shape)
        return depiction

    def _cdk_mol_block_to_cxsmiles(
        self,
        mol_block: str,
        ignore_explicite_hydrogens: bool = True,
    ) -> str:
        """
        This function takes a mol block str and returns the corresponding CXSMILES
        with coordinates using the CDK.

        Args:
            mol_block (str): mol block str
            ignore_explicite_hydrogens (bool, optional): whether or not to ignore H

        Returns:
            str: CXSMILES
        """
        atom_container = self._cdk_mol_block_to_iatomcontainer(mol_block)
        if ignore_explicite_hydrogens:
            cdk_base = "org.openscience.cdk."
            manipulator = JClass(cdk_base + "tools.manipulator.AtomContainerManipulator")
            atom_container = manipulator.copyAndSuppressedHydrogens(atom_container)
        smi_gen = JClass(cdk_base + "smiles.SmilesGenerator")
        flavor = JClass(cdk_base + "smiles.SmiFlavor")
        smi_gen = smi_gen(flavor.CxSmilesWithCoords)
        cxsmiles = smi_gen.create(atom_container)
        return cxsmiles

    def _cdk_smiles_to_IAtomContainer(self, smiles: str):
        """
        This function takes a SMILES representation of a molecule and
        returns the corresponding IAtomContainer object.

        Args:
            smiles (str): SMILES representation of the molecule

        Returns:
            IAtomContainer: CDK IAtomContainer object that represents the molecule
        """
        cdk_base = "org.openscience.cdk"
        SCOB = JClass(cdk_base + ".silent.SilentChemObjectBuilder")
        SmilesParser = JClass(cdk_base + ".smiles.SmilesParser")(SCOB.getInstance())
        if self.random_choice([True, False, False], log_attribute="cdk_kekulized"):
            SmilesParser.kekulise(False)
        molecule = SmilesParser.parseSmiles(smiles)
        return molecule

    def _cdk_mol_block_to_iatomcontainer(self, mol_block: str):
        """
        Given a mol block, this function returns an IAtomContainer (JClass) object.

        Args:
            mol_block (str): content of MDL MOL file

        Returns:
            IAtomContainer: CDK IAtomContainer object that represents the molecule
        """
        scob = JClass("org.openscience.cdk.silent.SilentChemObjectBuilder")
        bldr = scob.getInstance()
        iac_class = JClass("org.openscience.cdk.interfaces.IAtomContainer").class_
        string_reader = JClass("java.io.StringReader")(mol_block)
        mdlr = JClass("org.openscience.cdk.io.MDLV2000Reader")(string_reader)
        iatomcontainer = mdlr.read(bldr.newInstance(iac_class))
        mdlr.close()
        return iatomcontainer

    def _cdk_iatomcontainer_to_mol_block(self, i_atom_container) -> str:
        """
        This function takes an IAtomContainer object and returns the content
        of the corresponding MDL MOL file as a string.

        Args:
            i_atom_container (CDK IAtomContainer (JClass object))

        Returns:
            str: string content of MDL MOL file
        """
        string_writer = JClass("java.io.StringWriter")()
        mol_writer = JClass("org.openscience.cdk.io.MDLV2000Writer")(string_writer)
        mol_writer.setWriteAromaticBondTypes(True)
        mol_writer.write(i_atom_container)
        mol_writer.close()
        mol_str = string_writer.toString()
        return str(mol_str)

    def _cdk_add_explicite_hydrogen_to_molblock(self, mol_block: str) -> str:
        """
        This function takes a mol block and returns the mol block with explicit
        hydrogen atoms.

        Args:
            mol_block (str): mol block that describes a molecule

        Returns:
            str: The same mol block with explicit hydrogen atoms
        """
        i_atom_container = self._cdk_mol_block_to_iatomcontainer(mol_block)
        cdk_base = "org.openscience.cdk."
        manipulator = JClass(cdk_base + "tools.manipulator.AtomContainerManipulator")
        manipulator.convertImplicitToExplicitHydrogens(i_atom_container)
        mol_block = self._cdk_iatomcontainer_to_mol_block(i_atom_container)
        return mol_block

    def _cdk_add_explicite_hydrogen_to_smiles(self, smiles: str) -> str:
        """
        This function takes a SMILES str and uses CDK to add explicite hydrogen atoms.
        It returns an adapted version of the SMILES str.

        Args:
            smiles (str): SMILES representation of a molecule

        Returns:
            smiles (str): SMILES representation of a molecule with explicite H
        """
        i_atom_container = self._cdk_smiles_to_IAtomContainer(smiles)
        cdk_base = "org.openscience.cdk."
        manipulator = JClass(cdk_base + "tools.manipulator.AtomContainerManipulator")
        manipulator.convertImplicitToExplicitHydrogens(i_atom_container)
        smi_flavor = JClass("org.openscience.cdk.smiles.SmiFlavor").Absolute
        smiles_generator = JClass("org.openscience.cdk.smiles.SmilesGenerator")(
            smi_flavor
        )
        smiles = smiles_generator.create(i_atom_container)
        return str(smiles)

    def _cdk_remove_explicite_hydrogen_from_smiles(self, smiles: str) -> str:
        """
        This function takes a SMILES str and uses CDK to remove explicite hydrogen atoms.
        It returns an adapted version of the SMILES str.

        Args:
            smiles (str): SMILES representation of a molecule

        Returns:
            smiles (str): SMILES representation of a molecule with explicite H
        """
        i_atom_container = self._cdk_smiles_to_IAtomContainer(smiles)
        cdk_base = "org.openscience.cdk."
        manipulator = JClass(cdk_base + "tools.manipulator.AtomContainerManipulator")
        i_atom_container = manipulator.copyAndSuppressedHydrogens(i_atom_container)
        smi_flavor = JClass("org.openscience.cdk.smiles.SmiFlavor").Absolute
        smiles_generator = JClass("org.openscience.cdk.smiles.SmilesGenerator")(
            smi_flavor
        )
        smiles = smiles_generator.create(i_atom_container)
        return str(smiles)

    def _cdk_get_depiction_generator(self, molecule, has_R_group: bool = False):
        """
        This function defines random rendering options for the structure
        depictions created using CDK.
        It takes an iAtomContainer and a SMILES string and returns the iAtomContainer
        and the DepictionGenerator
        with random rendering settings and the AtomContainer.
        I followed https://github.com/cdk/cdk/wiki/Standard-Generator to adjust the
        depiction parameters.

        Args:
            molecule (cdk.AtomContainer): Atom container
            smiles (str): smiles representation of molecule
            has_R_group (bool): Whether the molecule has R groups (used to determine
                                whether or not to use atom numbering as it can be
                                confusing with R groups indices)
                                # TODO: check this in atomcontainer

        Returns:
            DepictionGenerator, molecule: Objects that hold depiction parameters
        """
        cdk_base = "org.openscience.cdk"
        dep_gen = JClass("org.openscience.cdk.depict.DepictionGenerator")(
            self._cdk_get_random_java_font()
        )
        StandardGenerator = JClass(
            cdk_base + ".renderer.generators.standard.StandardGenerator"
        )

        # Define visibility of atom/superatom labels
        symbol_visibility = self.random_choice(
            ["iupac_recommendation", "no_terminal_methyl", "show_all_atom_labels"],
            log_attribute="cdk_symbol_visibility",
        )
        SymbolVisibility = JClass("org.openscience.cdk.renderer.SymbolVisibility")
        if symbol_visibility == "iupac_recommendation":
            dep_gen = dep_gen.withParam(
                StandardGenerator.Visibility.class_,
                SymbolVisibility.iupacRecommendations(),
            )
        elif symbol_visibility == "no_terminal_methyl":
            # only hetero atoms, no terminal alkyl groups
            dep_gen = dep_gen.withParam(
                StandardGenerator.Visibility.class_,
                SymbolVisibility.iupacRecommendationsWithoutTerminalCarbon(),
            )
        elif symbol_visibility == "show_all_atom_labels":
            dep_gen = dep_gen.withParam(
                StandardGenerator.Visibility.class_, SymbolVisibility.all()
            )  # show all atom labels

        # Define bond line stroke width
        stroke_width = self.random_choice(
            np.arange(0.8, 2.0, 0.1), log_attribute="cdk_stroke_width"
        )
        dep_gen = dep_gen.withParam(StandardGenerator.StrokeRatio.class_,
                                    stroke_width)
        # Define symbol margin ratio
        margin_ratio = self.random_choice(
            [0, 1, 2, 2, 2, 3, 4], log_attribute="cdk_margin_ratio"
        )
        dep_gen = dep_gen.withParam(
            StandardGenerator.SymbolMarginRatio.class_,
            JClass("java.lang.Double")(margin_ratio),
        )
        # Define bond properties
        double_bond_dist = self.random_choice(
            np.arange(0.11, 0.25, 0.01), log_attribute="cdk_double_bond_dist"
        )
        dep_gen = dep_gen.withParam(StandardGenerator.BondSeparation.class_,
                                    double_bond_dist)
        wedge_ratio = self.random_choice(
            np.arange(4.5, 7.5, 0.1), log_attribute="cdk_wedge_ratio"
        )
        dep_gen = dep_gen.withParam(
            StandardGenerator.WedgeRatio.class_, JClass("java.lang.Double")(wedge_ratio)
        )
        if self.random_choice([True, False], log_attribute="cdk_fancy_bold_wedges"):
            dep_gen = dep_gen.withParam(StandardGenerator.FancyBoldWedges.class_, True)
        if self.random_choice([True, False], log_attribute="cdk_fancy_hashed_wedges"):
            dep_gen = dep_gen.withParam(StandardGenerator.FancyHashedWedges.class_,
                                        True)
        hash_spacing = self.random_choice(
            np.arange(4.0, 6.0, 0.2), log_attribute="cdk_hash_spacing"
        )
        dep_gen = dep_gen.withParam(StandardGenerator.HashSpacing.class_, hash_spacing)
        # Add CIP labels
        labels = False
        if self.random_choice([True, False], log_attribute="cdk_add_CIP_labels"):
            labels = True
            JClass("org.openscience.cdk.geometry.cip.CIPTool").label(molecule)
            for atom in molecule.atoms():
                label = atom.getProperty(
                    JClass("org.openscience.cdk.CDKConstants").CIP_DESCRIPTOR
                )
                atom.setProperty(StandardGenerator.ANNOTATION_LABEL, label)
            for bond in molecule.bonds():
                label = bond.getProperty(
                    JClass("org.openscience.cdk.CDKConstants").CIP_DESCRIPTOR
                )
                bond.setProperty(StandardGenerator.ANNOTATION_LABEL, label)
        # Add atom indices to the depictions
        if self.random_choice(
            [True, False, False, False], log_attribute="cdk_add_atom_indices"
        ):
            if not has_R_group:
                # Avoid confusion with R group indices and atom numbering
                labels = True
                for atom in molecule.atoms():
                    label = JClass("java.lang.Integer")(
                        1 + molecule.getAtomNumber(atom)
                    )
                    atom.setProperty(StandardGenerator.ANNOTATION_LABEL, label)
        if labels:
            # We only need black
            dep_gen = dep_gen.withParam(
                StandardGenerator.AnnotationColor.class_,
                JClass("java.awt.Color")(0x000000),
            )
            # Font size of labels
            font_scale = self.random_choice(
                np.arange(0.5, 0.8, 0.1), log_attribute="cdk_label_font_scale"
            )
            dep_gen = dep_gen.withParam(
                StandardGenerator.AnnotationFontScale.class_,
                font_scale)
            # Distance between atom numbering and depiction
            annotation_distance = self.random_choice(
                np.arange(0.15, 0.30, 0.05), log_attribute="cdk_annotation_distance"
            )
            dep_gen = dep_gen.withParam(
                StandardGenerator.AnnotationDistance.class_, annotation_distance
            )
        # Abbreviate superatom labels in half of the cases
        # TODO: Find a way to define Abbreviations object as a class attribute.
        # Problem: can't be pickled.
        # Right now, this is loaded every time when a structure is depicted.
        # That seems inefficient.
        if self.random_choice([True, False], log_attribute="cdk_collapse_superatoms"):
            cdk_superatom_abrv = JClass("org.openscience.cdk.depict.Abbreviations")()
            abbr_filename = self.random_choice([
                "cdk_superatom_abbreviations.smi",
                "cdk_alt_superatom_abbreviations.smi"])
            abbreviation_path = str(self.HERE.joinpath(abbr_filename))
            abbreviation_path = abbreviation_path.replace("\\", "/")
            abbreviation_path = JClass("java.lang.String")(abbreviation_path)
            cdk_superatom_abrv.loadFromFile(abbreviation_path)
            try:
                cdk_superatom_abrv.apply(molecule)
            except JException:
                pass
        return dep_gen, molecule

    def _cdk_get_random_java_font(self):
        """
        This function returns a random java.awt.Font (JClass) object

        Returns:
        font: java.awt.Font (JClass object)
        """
        font_size = self.random_choice(
            range(10, 20), log_attribute="cdk_atom_label_font_size"
        )
        Font = JClass("java.awt.Font")
        font_name = self.random_choice(
            ["Verdana",
             "Times New Roman",
             "Arial",
             "Gulliver Regular",
             "Helvetica",
             "Courier",
             "architectural",
             "Geneva",
             "Lucida Sans",
             "Teletype"],
            # log_attribute='cdk_atom_label_font'
        )
        font_style = self.random_choice(
            [Font.PLAIN, Font.BOLD],
            # log_attribute='cdk_atom_label_font_style'
        )
        font = Font(font_name, font_style, font_size)
        return font

    def _cdk_rotate_coordinates(self, molecule):
        """
        Given an IAtomContainer (JClass object), this function rotates the molecule
        and adapts the coordinates of accordingly. The IAtomContainer is then returned.#

        Args:
        molecule: IAtomContainer (JClass object)

        Returns:
        molecule: IAtomContainer (JClass object)
        """
        cdk_base = "org.openscience.cdk"
        point = JClass(cdk_base + ".geometry.GeometryTools").get2DCenter(molecule)
        rot_degrees = self.random_choice(range(360))
        JClass(cdk_base + ".geometry.GeometryTools").rotate(
            molecule, point, rot_degrees
        )
        return molecule

    def _cdk_generate_2d_coordinates(self, molecule):
        """
        Given an IAtomContainer (JClass object), this function adds 2D coordinate to
        the molecule. The modified IAtomContainer is then returned.

        Args:
        molecule: IAtomContainer (JClass object)

        Returns:
        molecule: IAtomContainer (JClass object)
        """
        cdk_base = "org.openscience.cdk"
        sdg = JClass(cdk_base + ".layout.StructureDiagramGenerator")()
        sdg.setMolecule(molecule)
        sdg.generateCoordinates(molecule)
        molecule = sdg.getMolecule()
        return molecule

    def _convert_rgba2rgb(self, rgba: np.array, background=(255, 255, 255)):
        """
        Convert an RGBA image (np.array) to an RGB image (np.array).
        https://stackoverflow.com/questions/50331463/convert-rgba-to-rgb-in-python

        Args:
            rgba (np.array): RGBA image
            background (tuple, optional): . Defaults to (255, 255, 255).

        Returns:
            np.array: RGB image
        """
        row, col, ch = rgba.shape

        if ch == 3:
            return rgba

        assert ch == 4, 'RGBA image has 4 channels.'

        rgb = np.zeros((row, col, 3), dtype='float32')
        r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

        a = np.asarray(a, dtype='float32') / 255.0

        R, G, B = background

        rgb[:, :, 0] = r * a + (1.0 - a) * R
        rgb[:, :, 1] = g * a + (1.0 - a) * G
        rgb[:, :, 2] = b * a + (1.0 - a) * B

        return np.asarray(rgb, dtype='uint8')

    def _cdk_bufferedimage_to_numpyarray(
        self,
        image
    ) -> np.ndarray:
        """
        This function converts a BufferedImage (JClass object) into a numpy array.

        Args:
            image (BufferedImage (JClass object))

        Returns:
            image (np.ndarray)
        """
        # Write the image into a format that can be read by skimage
        ImageIO = JClass("javax.imageio.ImageIO")
        os = JClass("java.io.ByteArrayOutputStream")()
        Base64 = JClass("java.util.Base64")
        ImageIO.write(
            image, JClass("java.lang.String")("PNG"), Base64.getEncoder().wrap(os)
        )
        image = bytes(os.toString("UTF-8"))
        image = base64.b64decode(image)
        image = sk_io.imread(image, plugin="imageio")
        image = img_as_ubyte(image)
        image = self._convert_rgba2rgb(image)
        return image

    def _cdk_render_molecule(
        self,
        molecule,
        has_R_group: bool = False,
        shape: Tuple[int, int] = (512, 512)
    ):
        """
        This function takes an IAtomContainer (JClass object), the corresponding SMILES
        string and an image shape and returns a BufferedImage (JClass object) with the
        rendered molecule.

        Args:
            molecule (IAtomContainer (JClass object)): molecule
            has_R_group (bool): Whether the molecule has R groups (used to determine
                                whether or not to use atom numbering as it can be
                                confusing with R groups indices)
                                # TODO: check this in atomcontainer
            smiles (str): SMILES string
            shape (Tuple[int, int]): y, x
        Returns:
            depiction (np.ndarray): chemical structure depiction
        """
        dep_gen, molecule = self._cdk_get_depiction_generator(molecule, has_R_group)
        dep_gen = dep_gen.withSize(shape[1], shape[0])
        dep_gen = dep_gen.withFillToFit()
        depiction = dep_gen.depict(molecule).toImg()
        depiction = self._cdk_bufferedimage_to_numpyarray(depiction)
        return depiction
