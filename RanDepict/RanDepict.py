import os
import pathlib
import numpy as np
import io
from skimage import io as sk_io
from skimage.color import rgba2rgb, rgb2gray
from skimage.util import img_as_ubyte, img_as_float
from PIL import Image, ImageFont, ImageDraw, ImageStat
from multiprocessing import set_start_method, get_context
import imgaug.augmenters as iaa
import random
from copy import deepcopy
from typing import Tuple, List, Dict

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdAbbreviations import CondenseMolAbbreviations, GetDefaultAbbreviations
from rdkit.Chem.Draw import rdMolDraw2D
from indigo import Indigo
from indigo.renderer import IndigoRenderer
from jpype import *
import base64

#from depiction_feature_diversity import DepictionFeatures, DepictionFeatureRanges

HERE = pathlib.Path(__file__).resolve().parent.joinpath("assets")


class RandomDepictor:
    """This class contains everything necessary to generate a variety of random depictions
    with given SMILES strings.
    An instance of RandomDepictor can be called with a SMILES str and returns an np.array
    that represents the RGB image with the given chemical structure."""

    def __init__(self, seed: int = 42):
        """Load the JVM only once, load superatom list (OSRA), set context for multiprocessing"""
        # Start the JVM to access Java classes
        try:
            self.jvmPath = getDefaultJVMPath()
        except JVMNotFoundException:
            print(
                "If you see this message, for some reason JPype cannot find jvm.dll.",
                "This indicates that the environment varibale JAVA_HOME is not set properly.",
                "You can set it or set it manually in the code (see __init__() of RandomDepictor)",
            )
            self.jvmPath = "Define/your/path/or/set/your/JAVA_HOME/variable/properly"
        if not isJVMStarted():
            self.jar_path = HERE.joinpath("jar_files/cdk_2_5.jar")
            startJVM(self.jvmPath, "-ea", "-Djava.class.path=" + str(self.jar_path))

        self.seed = seed
        random.seed(self.seed)

        # Load list of superatoms for label generation
        with open(HERE.joinpath("superatom.txt")) as superatoms:
            superatoms = superatoms.readlines()
            self.superatoms = [s[:-2] for s in superatoms]

        # Define PIL resizing methods to choose from:
        self.PIL_resize_methods = [
        Image.NEAREST, 
        Image.BOX, 
        Image.BILINEAR, 
        Image.HAMMING, 
        Image.BICUBIC, 
        Image.LANCZOS
        ]
        
        self.from_fingerprint = False
        self.depiction_features = False

        # Set context for multiprocessing but make sure this only happens once
        try:
            set_start_method("spawn")
        except RuntimeError:
            pass

    def __call__(
        self,
        smiles: str,
        shape: Tuple[int, int, int] = (299, 299),
        grayscale: bool = False,
    ):
        # Depict structure with random parameters
        depiction = self.random_depiction(smiles, shape)
        # Add augmentations
        depiction = self.add_augmentations(depiction)
        
        if grayscale:
            return self.to_grayscale_float_img(depiction)
        return depiction

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        # I am not really happy with this, I'd like to automatically close the JVM
        # But if it is closed once, you cannot reopen it.
        # Shutdown the JVM
        # shutdownJVM()
        pass

    def random_choice(
        self, 
        iterable: List, 
        log_attribute: str = False):
        """This function takes an iterable, calls random.choice() on it,
        increases random.seed by 1 and returns the result. This way, results
        produced by RanDepict are replicable.
        
        Additionally, this function handles the generation of depictions and
        augmentations from given fingerprints by handling all random decisions 
        according to the fingerprint template.
        """
        # Keep track of seed and change it with every pseudo-random decision.
        self.seed += 1
        random.seed(self.seed)
        
        # Generation from fingerprint:
        if self.from_fingerprint and log_attribute:
            # Get dictionaries that define positions and linked conditions
            pos_cond_dicts = self.active_scheme[log_attribute]
            for pos_cond_dict in pos_cond_dicts:
                pos = pos_cond_dict['position']
                cond = pos_cond_dict['one_if']
                if self.active_fingerprint[pos]:
                    # If the condition is a range: adapt iterable accordingly and go on
                    if type(cond) == tuple:
                        iterable = [item for item in iterable
                                    if item > cond[0] - 0.001
                                    if item < cond[1] + 0.001]
                        break
                    # Otherwise, simply return the condition value
                    else:
                        return cond
        # Pseudo-randomly pick an element from the iterable
        result = random.choice(iterable)
        
        # This block takes care of the automated fingerprint generation.
        # Add result(s) to depiction_features
        if log_attribute and self.depiction_features:
            found_logged_attribute = getattr(self.depiction_features, log_attribute)
            # If the attribute is not saved in a list, simply write it, otherwise append it
            if type(found_logged_attribute) != list:
                setattr(self.depiction_features, log_attribute, result)
            else:
                setattr(self.depiction_features, log_attribute, found_logged_attribute + [result])
        return result

    def random_choices(self, iterable: List, k: int) -> List:
        """
        This function takes an iterable, calls self.random_choices() on it to take k
        elements from it,increases random.seed by 1 and returns the result. This way, results
        produced by RanDepict are replicable.
        """
        self.seed += 1
        random.seed(self.seed)
        return random.choices(iterable, k=k)

    def get_nonexisting_image_name(
        self, path: str = "./temp/", format: str = "PNG"
    ) -> str:
        """This function returns a random file name that does not already exist at path"""
        for _ in range(100):
            file_name = str(self.random_choice(range(10000))) + "." + format
            if not os.path.exists(os.path.join(path, file_name)):
                return file_name

    def random_image_size(self, shape: Tuple[int, int]) -> Tuple[int, int]:
        """This function takes a random image shape and returns an image shape where the
        first two dimensions are slightly distorted."""
        # Set random depiction image shape (to cause a slight distortion)
        y = int(shape[0] * self.random_choice(np.arange(0.9, 1.1, 0.02)))
        x = int(shape[1] * self.random_choice(np.arange(0.9, 1.1, 0.02)))
        return y, x

    def get_random_indigo_rendering_settings(
        self, shape: Tuple[int, int] = (299, 299)
    ) -> Indigo:
        """This function defines random rendering options for the structure depictions created
        using Indigo. It returns an Indigo object with the settings."""
        # Define random shape for depiction (within boundaries); image is resized later)
        indigo = Indigo()
        renderer = IndigoRenderer(indigo)
        # Get slightly distorted shape
        y, x = self.random_image_size(shape)
        indigo.setOption("render-image-width", x)
        indigo.setOption("render-image-height", y)
        # Set random bond line width
        bond_line_width = float(self.random_choice(np.arange(0.5, 2.5, 0.1), log_attribute='indigo_bond_line_width'))
        indigo.setOption("render-bond-line-width", bond_line_width)
        # Set random relative thickness
        relative_thickness = float(self.random_choice(np.arange(0.5, 1.5, 0.1), log_attribute='indigo_relative_thickness'))
        indigo.setOption("render-relative-thickness", relative_thickness)
        # Set random bond length
        # CAREFUL! Changing the bond length does not change the bond length relative to other
        # elements. Instead, the whole molecule is scaled down!
        # bond_length = self.random_choice(range(int(shape[0]/19), int(shape[0]/6))) #19
        # indigo.setOption("render-bond-length", bond_length)
        # Output_format: PNG
        indigo.setOption("render-output-format", "png")
        # Set random atom label rendering model (standard is rendering terminal groups)
        if self.random_choice([True] + [False] * 19, log_attribute='indigo_labels_all'):
            indigo.setOption("render-label-mode", "all")  # show all atom labels
        elif self.random_choice([True] + [False] * 3, log_attribute='indigo_labels_hetero'):
            indigo.setOption(
                "render-label-mode", "hetero"
            )  # only hetero atoms, no terminal groups
        # Set random depiction colour / not necessary for us as we binarise everything anyway
        # if self.random_choice([True, False, False, False, False]):
        #    R = str(self.random_choice(np.arange(0.1, 1.0, 0.1)))
        #    G = str(self.random_choice(np.arange(0.1, 1.0, 0.1)))
        #    B = str(self.random_choice(np.arange(0.1, 1.0, 0.1)))
        #    indigo.setOption("render-base-color", ", ".join([R,G,B]))
        # Render bold bond for Haworth projection
        if self.random_choice([True, False], log_attribute='indigo_render_bold_bond'):
            indigo.setOption("render-bold-bond-detection", "True")
        # Render labels for stereobonds
        stereo_style = self.random_choice(["ext", "old", "none"], log_attribute='indigo_stereo_label_style')
        indigo.setOption("render-stereo-style", stereo_style)
        # Collapse superatoms (default: expand)
        if self.random_choice([True, False], log_attribute='indigo_collapse_superatoms'):
            indigo.setOption("render-superatom-mode", "collapse")
        return indigo, renderer

    def depict_and_resize_indigo(
        self, smiles: str, shape: Tuple[int, int] = (299, 299)
    ) -> np.array:
        """This function takes a smiles str and an image shape. It renders the chemical structures
        using Indigo with random rendering/depiction settings and returns an RGB image (np.array)
        with the given image shape."""
        # Instantiate Indigo with random settings and IndigoRenderer
        indigo, renderer = self.get_random_indigo_rendering_settings()
        # Load molecule
        molecule = indigo.loadMolecule(smiles)
        # Do not kekulize in 20% of cases
        if self.random_choice([True, False, False], log_attribute='indigo_not_kekulized'):
            molecule.aromatize()
        molecule.layout()
        # Write to buffer
        temp = renderer.renderToBuffer(molecule)
        temp = io.BytesIO(temp)
        depiction = sk_io.imread(temp) 
        depiction = self.resize(depiction, (shape[0], shape[1]))
        depiction = rgba2rgb(depiction)
        depiction = img_as_ubyte(depiction)
        return depiction

    def get_random_rdkit_rendering_settings(
        self, shape: Tuple[int, int] = (299, 299)
    ) -> rdMolDraw2D.MolDraw2DCairo:
        """This function defines random rendering options for the structure depictions created
        using rdkit. It returns an MolDraw2DCairo object with the settings."""
        # Get slightly distorted shape
        y, x = self.random_image_size(shape)
        # Instantiate object that saves the settings
        depiction_settings = rdMolDraw2D.MolDraw2DCairo(y, x)
        # Stereo bond annotation
        if self.random_choice([True, False], log_attribute='rdkit_add_stereo_annotation'):
            depiction_settings.drawOptions().addStereoAnnotation = True
        if self.random_choice([True, False], log_attribute='rdkit_add_chiral_flag_labels'):
            depiction_settings.drawOptions().includeChiralFlagLabel = True
        # Atom indices
        if self.random_choice([True, False, False, False],  log_attribute='rdkit_add_atom_indices'):
            depiction_settings.drawOptions().addAtomIndices = True
        # Bond line width
        bond_line_width = self.random_choice(range(1, 5), log_attribute='rdkit_bond_line_width')
        depiction_settings.drawOptions().bondLineWidth = bond_line_width
        # Draw terminal methyl groups
        if self.random_choice([True, False], log_attribute='rdkit_draw_terminal_methyl'):
            depiction_settings.drawOptions().explicitMethyl = True
        # Label font type and size
        font_dir = HERE.joinpath("fonts/")
        font_path = os.path.join(
            str(font_dir), self.random_choice(os.listdir(str(font_dir)), log_attribute='rdkit_label_font')
        )
        depiction_settings.drawOptions().fontFile = font_path
        min_font_size = self.random_choice(range(10, 20), log_attribute='rdkit_min_font_size')
        depiction_settings.drawOptions().minFontSize = min_font_size
        depiction_settings.drawOptions().maxFontSize = 30
        # Rotate the molecule
        depiction_settings.drawOptions().rotate = self.random_choice(range(360))
        # Fixed bond length
        fixed_bond_length = self.random_choice(range(30, 45), log_attribute='rdkit_fixed_bond_length')
        depiction_settings.drawOptions().fixedBondLength = fixed_bond_length
        # Comic mode (looks a bit hand drawn)
        if self.random_choice([True, False, False, False, False], log_attribute='rdkit_comic_style'):
            depiction_settings.drawOptions().comicMode = True
        # Keep it black and white
        depiction_settings.drawOptions().useBWAtomPalette()
        return depiction_settings

    def depict_and_resize_rdkit(
        self, smiles: str, shape: Tuple[int, int] = (299, 299)
    ) -> np.array:
        """This function takes a smiles str and an image shape. It renders the chemical structures
        using Rdkit with random rendering/depiction settings and returns an RGB image (np.array)
        with the given image shape."""
        # Generate mol object from smiles str
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            AllChem.Compute2DCoords(mol)
            # Abbreviate superatoms
            if self.random_choice([True, False], log_attribute='rdkit_collapse_superatoms'):
                abbrevs = GetDefaultAbbreviations()
                mol = CondenseMolAbbreviations(mol, abbrevs)
            # Get random depiction settings
            depiction_settings = self.get_random_rdkit_rendering_settings()
            # Create depiction
            # TODO: Figure out how to depict molecules without kekulization here.
            # The following line does not prevent the molecule from being depicted kekulized:
            # mol = rdMolDraw2D.PrepareMolForDrawing(mol, kekulize = False)
            # The molecule must get kekulized somewhere "by accident"

            rdMolDraw2D.PrepareAndDrawMolecule(depiction_settings, mol)
            depiction = depiction_settings.GetDrawingText()
            depiction = sk_io.imread(io.BytesIO(depiction))
            # Resize image to desired shape
            depiction = self.resize(depiction, shape)
            depiction = img_as_ubyte(depiction)
            return np.asarray(depiction)
        else: 
            print('RDKit was unable to read SMILES: {}'.format(smiles))

    def get_random_cdk_rendering_settings(self, rendererModel, molecule):
        """This function defines random rendering options for the structure depictions created
        using CDK. It takes a org.openscience.cdk.renderer.AtomContainerRenderer.2DModel and a
        org.openscience.cdk.AtomContainer and returns the 2DModel object with random rendering settings
        and the AtomContainer with superatom abbreviations (50% of the cases).
        I followed https://github.com/cdk/cdk/wiki/Standard-Generator while creating this."""

        StandardGenerator = JClass(
            "org.openscience.cdk.renderer.generators.standard.StandardGenerator"
        )

        # Define visibility of atom/superatom labels
        SymbolVisibility = JClass("org.openscience.cdk.renderer.SymbolVisibility")
        if self.random_choice([True] + [False] * 19, log_attribute='cdk_show_all_atom_labels'):
            rendererModel.set(
                StandardGenerator.Visibility.class_, SymbolVisibility.all()
            )  # show all atom labels
        elif self.random_choice([True, False, False, False], log_attribute='cdk_no_terminal_methyl'):
            # only hetero atoms, no terminal alkyl groups
            rendererModel.set(
                StandardGenerator.Visibility.class_,
                SymbolVisibility.iupacRecommendationsWithoutTerminalCarbon(),
            )
        elif self.random_choice([True, True, False], log_attribute='cdk_symbol_visibility_iupac_recommendation'):
            rendererModel.set(
                StandardGenerator.Visibility.class_,
                SymbolVisibility.iupacRecommendations(),
            )
        # Define bond line stroke width
        stroke_width = self.random_choice(np.arange(0.8, 2.0, 0.1), log_attribute='cdk_stroke_width')
        rendererModel.set(StandardGenerator.StrokeRatio.class_, stroke_width)
        # Define symbol margin ratio
        margin_ratio = self.random_choice([0, 1, 2, 2, 2, 3, 4], log_attribute='cdk_margin_ratio')
        rendererModel.set(
            StandardGenerator.SymbolMarginRatio.class_,
            JClass("java.lang.Double")(margin_ratio),
        )
        # Define bond properties
        double_bond_dist = self.random_choice(np.arange(0.11, 0.25, 0.01), log_attribute='cdk_double_bond_dist')
        rendererModel.set(StandardGenerator.BondSeparation.class_, double_bond_dist)
        wedge_ratio = self.random_choice(np.arange(4.5, 7.5, 0.1), log_attribute='cdk_wedge_ratio')
        rendererModel.set(
            StandardGenerator.WedgeRatio.class_, JClass("java.lang.Double")(wedge_ratio)
        )
        if self.random_choice([True, False], log_attribute='cdk_fancy_bold_wedges'):
            rendererModel.set(StandardGenerator.FancyBoldWedges.class_, True)
        if self.random_choice([True, False], log_attribute='cdk_fancy_hashed_wedges'):
            rendererModel.set(StandardGenerator.FancyHashedWedges.class_, True)
        hash_spacing = self.random_choice(np.arange(4.0, 6.0, 0.2), log_attribute='cdk_hash_spacing')
        rendererModel.set(StandardGenerator.HashSpacing.class_, hash_spacing)
        # Add CIP labels
        labels = False
        if self.random_choice([True, False], log_attribute='cdk_add_CIP_labels'):
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
        if self.random_choice([True, False, False, False], log_attribute='cdk_add_atom_indices'):
            labels = True
            for atom in molecule.atoms():
                label = JClass("java.lang.Integer")(1 + molecule.getAtomNumber(atom))
                atom.setProperty(StandardGenerator.ANNOTATION_LABEL, label)
        if labels:
            # We only need black
            rendererModel.set(
                StandardGenerator.AnnotationColor.class_,
                JClass("java.awt.Color")(0x000000),
            )
            # Font size of labels
            font_scale = self.random_choice(np.arange(0.5, 0.8, 0.1), log_attribute='cdk_label_font_scale')
            rendererModel.set(StandardGenerator.AnnotationFontScale.class_, font_scale)
            # Distance between atom numbering and depiction
            annotation_distance = self.random_choice(np.arange(0.15, 0.30, 0.05), log_attribute='cdk_annotation_distance')
            rendererModel.set(
                StandardGenerator.AnnotationDistance.class_, annotation_distance
            )
        # Abbreviate superatom labels in half of the cases
        # TODO: Find a way to define Abbreviations object as a class attribute. Problem: can't be pickled.
        # Right now, this is loaded every time when a structure is depicted. That seems inefficient.
        if self.random_choice([True, False], log_attribute='cdk_collapse_superatoms'):
            cdk_superatom_abrv = JClass("org.openscience.cdk.depict.Abbreviations")()
            abbreviation_path = str(HERE.joinpath("smiles_list.smi")).replace(
                "\\", "/"
            )
            abbreviation_path = JClass("java.lang.String")(abbreviation_path)
            cdk_superatom_abrv.loadFromFile(abbreviation_path)
            cdk_superatom_abrv.apply(molecule)
        return rendererModel, molecule

    def depict_and_resize_cdk(
        self, smiles: str, shape: Tuple[int, int] = (299, 299)
    ) -> np.array:
        """This function takes a smiles str and an image shape. It renders the chemical structures
        using CDK with random rendering/depiction settings and returns an RGB image (np.array)
        with the given image shape.
        The general workflow here is a JPype adaptation of code published by Egon Willighagen
        in 'Groovy Cheminformatics with the Chemistry Development Kit':
        https://egonw.github.io/cdkbook/ctr.html#depict-a-compound-as-an-image
        with additional adaptations to create all the different depiction types from
        https://github.com/cdk/cdk/wiki/Standard-Generator"""

        # Read molecule from SMILES str
        SCOB = JClass("org.openscience.cdk.silent.SilentChemObjectBuilder")
        SmilesParser = JClass("org.openscience.cdk.smiles.SmilesParser")(
            SCOB.getInstance()
        )
        if self.random_choice([True, False, False], log_attribute='cdk_kekulized'):
            SmilesParser.kekulise(False)
        molecule = SmilesParser.parseSmiles(smiles)
        
        # Add hydrogens for coordinate generation (to make it look nicer/ avoid overlaps)
        matcher = JClass("org.openscience.cdk.atomtype.CDKAtomTypeMatcher").getInstance(molecule.getBuilder())
        for atom in molecule.atoms():
            atom_type = matcher.findMatchingAtomType(molecule, atom)
            JClass("org.openscience.cdk.tools.manipulator.AtomTypeManipulator").configure(atom, atom_type)
        adder = JClass("org.openscience.cdk.tools.CDKHydrogenAdder").getInstance(molecule.getBuilder())
        adder.addImplicitHydrogens(molecule)
        AtomContainerManipulator = JClass("org.openscience.cdk.tools.manipulator.AtomContainerManipulator")
        AtomContainerManipulator.convertImplicitToExplicitHydrogens(molecule)

        # Instantiate StructureDiagramGenerator, determine coordinates
        sdg = JClass("org.openscience.cdk.layout.StructureDiagramGenerator")()
        sdg.setMolecule(molecule)
        sdg.generateCoordinates(molecule)
        molecule = sdg.getMolecule()

        # Remove explicit hydrogens again
        AtomContainerManipulator.suppressHydrogens(molecule)

        # Rotate molecule randomly
        point = JClass("org.openscience.cdk.geometry.GeometryTools").get2DCenter(
            molecule
        )
        rot_degrees = self.random_choice(range(360))
        JClass("org.openscience.cdk.geometry.GeometryTools").rotate(
            molecule, point, rot_degrees
        )

        # Get Generators
        generators = JClass("java.util.ArrayList")()
        BasicSceneGenerator = JClass(
            "org.openscience.cdk.renderer.generators.BasicSceneGenerator"
        )()
        generators.add(BasicSceneGenerator)
        font_size = self.random_choice(range(10, 20), log_attribute='cdk_atom_label_font_size')
        Font = JClass("java.awt.Font")
        font_name = self.random_choice(
            ["Verdana", "Times New Roman", "Arial", "Gulliver Regular"],
            #log_attribute='cdk_atom_label_font'
        )
        font_style = self.random_choice([Font.PLAIN, Font.BOLD], 
                                        #log_attribute='cdk_atom_label_font_style'
                                        )
        font = Font(font_name, font_style, font_size)
        StandardGenerator = JClass(
            "org.openscience.cdk.renderer.generators.standard.StandardGenerator"
        )(font)
        generators.add(StandardGenerator)

        # Instantiate renderer
        AWTFontManager = JClass("org.openscience.cdk.renderer.font.AWTFontManager")
        renderer = JClass("org.openscience.cdk.renderer.AtomContainerRenderer")(
            generators, AWTFontManager()
        )

        # Create an empty image of the right size
        y, x = self.random_image_size(shape)
        # Workaround for structures that are cut off at edged of images:
        # Make image twice as big, reduce Zoom factor, then remove white 
        # areas at borders and resize to originally desired shape
        # TODO: Find out why the structures are cut off in the first place
        y = y * 3
        x = x * 3

        drawArea = JClass("java.awt.Rectangle")(x, y)
        BufferedImage = JClass("java.awt.image.BufferedImage")
        image = BufferedImage(x, y, BufferedImage.TYPE_INT_RGB)

        # Draw the molecule
        renderer.setup(molecule, drawArea)
        model = renderer.getRenderer2DModel()

        # Get random rendering settings
        model, molecule = self.get_random_cdk_rendering_settings(model, molecule)

        double = JClass("java.lang.Double")
        model.set(
            JClass(
                "org.openscience.cdk.renderer.generators.BasicSceneGenerator.ZoomFactor"
            ),
            double(0.75),
        )
        g2 = image.getGraphics()
        g2.setColor(JClass("java.awt.Color").WHITE)
        g2.fillRect(0, 0, x, y)
        AWTDrawVisitor = JClass("org.openscience.cdk.renderer.visitor.AWTDrawVisitor")

        renderer.paint(molecule, AWTDrawVisitor(g2))

        # Write the image into a format that can be read by skimage
        ImageIO = JClass("javax.imageio.ImageIO")
        os = JClass("java.io.ByteArrayOutputStream")()
        Base64 = JClass("java.util.Base64")
        ImageIO.write(
            image, JClass("java.lang.String")("PNG"), Base64.getEncoder().wrap(os)
        )
        depiction = bytes(os.toString("UTF-8"))
        depiction = base64.b64decode(depiction)

        # Read image in skimage
        depiction = sk_io.imread(depiction, plugin="imageio")
        # Normalise padding and get non-distorted image of right size
        depiction = self.normalise_padding(depiction)
        depiction = self.central_square_image(depiction)
        depiction = self.resize(depiction, shape)
        depiction = img_as_ubyte(depiction)
        return depiction

    def normalise_padding(self, im: np.array) -> np.array:
        """This function takes an RGB image (np.array) and deletes white space at 
        the borders. Then 0-10% of the image width/height is added as padding again.
        The modified image is returned
        
        Args:
            im: input image (np.array)
        
        Returns:
            output: the modified image (np.array)
        """
        # Remove white space at borders
        mask = im > 200 
        all_white = mask.sum(axis=2) > 0
        rows = np.flatnonzero((~all_white).sum(axis=1))
        cols = np.flatnonzero((~all_white).sum(axis=0))
        crop = im[rows.min():rows.max()+1, cols.min():cols.max()+1, :]
        # Add padding again.
        pad_range = np.arange(5, int(crop.shape[0]*0.2), 1)
        if len(pad_range) > 0:
            pad = self.random_choice(np.arange(5, int(crop.shape[0]*0.2), 1))
        else:
            pad = 5
        crop = np.pad(crop, pad_width=((pad,pad), (pad,pad), (0,0)), mode='constant', constant_values=255)
        return crop

    def central_square_image(
        self, 
        im: np.array
        ) -> np.array:
        """
        This function takes image (np.array) and will add white padding 
        so that the image has a square shape with the width/height of the longest side 
        of the original image.
        ___
        im: np.array
        ___
        output: np.array
        """
        # Create new blank white image
        max_wh = max(im.shape)
        new_im = 255 * np.ones((max_wh, max_wh, 3), np.uint8)
        # Determine paste coordinates and paste image
        upper = int((new_im.shape[0]-im.shape[0])/2)
        lower = int((new_im.shape[0]-im.shape[0])/2) + im.shape[0]
        left = int((new_im.shape[1]-im.shape[1])/2)
        right = int((new_im.shape[1]-im.shape[1])/2) + im.shape[1]
        new_im[upper:lower, left:right] = im
        return new_im


    def random_depiction(
        self, 
        smiles: str, 
        shape: Tuple[int, int] = (299, 299)
    ) -> np.array:
        """This function takes a SMILES str and depicts it using Rdkit, Indigo or CDK.
        The depiction method and the specific parameters for the depiction are chosen
        completely randomly. The purpose of this function is to enable depicting a diverse
        variety of chemical structure depictions."""
        depiction_functions = [
            self.depict_and_resize_rdkit,
            self.depict_and_resize_indigo,
            self.depict_and_resize_cdk,
        ]
        depiction_function = self.random_choice(depiction_functions)
        depiction = depiction_function(smiles, shape)
        # RDKit sometimes has troubles reading SMILES. If that happens, use Indigo or CDK
        if type(depiction) == bool or type(depiction) == type(None):
            depiction_function = self.random_choice([self.depict_and_resize_indigo,self.depict_and_resize_cdk])
            depiction = depiction_function(smiles, shape)
        return depiction


    def resize(
        self, 
        image: np.array, 
        shape:Tuple[int]
        ) -> np.array:
        """
        This function takes an image (np.array) and a shape and returns the resized image (np.array).
        It uses Pillow to do this, as it seems to have a bigger variety of scaling methods than skimage.
        The up/downscaling method is chosen randomly.
        ___
        image: np.array; the input image
        shape: tuple that describes desired output shape
        ___
        Output: np.array; the resized image

        """
        image = Image.fromarray(image)
        shape = (shape[0], shape[1])
        image = image.resize(shape, resample = self.random_choice(self.PIL_resize_methods))
        return np.asarray(image)


    def depict_from_fingerprint(
        self,
        smiles: str,
        fingerprints: List[np.array], 
        schemes: List[Dict]
        ) -> np.array:
        """
        This function takes a SMILES representation of a molecule, a list of one or two fingerprints 
        and a list of the corresponding fingerprint schemes and generates a chemical structure depiction
        that fits the fingerprint. 
        ___
        If only one fingerprint/scheme is given, we assume that they contain information for
        a depiction without augmentations. If two are given, we assume that the first one
        contains information about the depiction and the second one contains information about the 
        augmentations.
        ___
        All this function does is set the class attributes in a manner that random_choice() knows to not to
        actually pick parameters randomly.

        Args:
            fingerprints (List[np.array]): List of one or two fingerprints
            schemes (List[Dict]): List of one or two fingerprint schemes

        Returns:
            np.array: Chemical structure depiction
        """
        self.from_fingerprint = True
        self.active_fingerprint = fingerprints[0]
        self.active_scheme = schemes[0]
        # Depict molecule
        if 'indigo' in list(schemes[0].keys())[0]:
            depiction = self.depict_and_resize_indigo(smiles)
        elif 'rdkit' in list(schemes[0].keys())[0]:
            depiction = self.depict_and_resize_rdkit(smiles)
        elif 'cdk' in list(schemes[0].keys())[0]:
            depiction = self.depict_and_resize_cdk(smiles)
        
        # Add augmentations
        if len(fingerprints) == 2:
            self.active_fingerprint = fingerprints[1]
            self.active_scheme = schemes[1]
            depiction = self.add_augmentations(depiction)
        
        self.from_fingerprint, self.active_fingerprint, self.active_scheme = False, False, False
        return depiction
    

    def imgaug_augment(
        self, 
        image: np.array, 
        call_all: bool = False
        ) -> np.array:
        """This function applies a random amount of augmentations to a given image (np.array)
        using and returns the augmented image (np.array).
        If call_all = True, all augmentation functions will be applied"""
        # Add some padding to make sure rotation does not lead to funny bits at the edges of the image
        # print(image)
        original_shape = image.shape

        # Choose number of augmentations to apply (0-2); return image if nothiing needs to be done.
        aug_number = self.random_choice(range(0, 3))
        if not aug_number:
            return image

        # Add some padding to avoid weird artifacts after rotation
        image = np.pad(
            image, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=255
        )
        
        def imgaug_rotation():
            # Rotation between -10 and 10 degrees
            if not self.random_choice([True, True, False], log_attribute='has_imgaug_rotation'):
                return False
            rot_angle = self.random_choice(np.arange(-10, 10, 1))
            aug = iaa.Affine(rotate=rot_angle, mode="edge", fit_output=True)
            return aug
        
        def imgaug_black_and_white_noise():
            # Black and white noise
            if not self.random_choice([True, True, False], log_attribute='has_imgaug_salt_pepper'):
                return False
            coarse_dropout_p = self.random_choice(np.arange(0.0002, 0.0015, 0.0001))
            coarse_dropout_size_percent = self.random_choice(np.arange(1.0, 1.1, 0.01))
            replace_elementwise_p = self.random_choice(np.arange(0.01, 0.3, 0.01))
            aug = iaa.Sequential(
                [
                    iaa.CoarseDropout(
                        coarse_dropout_p, size_percent=coarse_dropout_size_percent
                    ),
                    iaa.ReplaceElementwise(replace_elementwise_p, 255),
                ]
            )
            return aug
        
        def imgaug_shearing():
            # Shearing
            if not self.random_choice([True, True, False], log_attribute='has_imgaug_shearing'):
                return False
            shear_param = self.random_choice(np.arange(-5, 5, 1))
            aug = self.random_choice(
                [
                    iaa.geometric.ShearX(shear_param, mode="edge", fit_output=True),
                    iaa.geometric.ShearY(shear_param, mode="edge", fit_output=True),
                ]
            )
            return aug
        
        def imgaug_imgcorruption():
            # Jpeg compression or pixelation
            if not self.random_choice([True, True, False], log_attribute='has_imgaug_corruption'):
                return False
            imgcorrupt_severity = self.random_choice(np.arange(1, 2, 1))
            aug = self.random_choice(
                [
                    iaa.imgcorruptlike.JpegCompression(severity=imgcorrupt_severity),
                    iaa.imgcorruptlike.Pixelate(severity=imgcorrupt_severity),
                ]
            )
            return aug
        
        def imgaug_brightness_adjustment():
            # Brightness adjustment
            if not self.random_choice([True, True, False], log_attribute='has_imgaug_brightness_adj'):
                return False
            brightness_adj_param = self.random_choice(np.arange(-50, 50, 1))
            aug = iaa.WithBrightnessChannels(iaa.Add(brightness_adj_param))
            return aug
        
        def imgaug_colour_temp_adjustment():
            # Colour temperature adjustment
            if not self.random_choice([True, True, False], log_attribute='has_imgaug_col_adj'):
                return False
            colour_temp = self.random_choice(np.arange(1100, 10000, 1))
            aug = iaa.ChangeColorTemperature(colour_temp)
            return aug
        
        # Define list of available augmentations
        aug_list = [imgaug_rotation,
                    imgaug_black_and_white_noise,
                    imgaug_shearing,
                    imgaug_imgcorruption,
                    imgaug_brightness_adjustment,
                    imgaug_colour_temp_adjustment]
        
        # Every one of them has a 1/3 chance of returning False
        aug_list = [fun() for fun in aug_list]
        aug_list = [fun for fun in aug_list
                             if fun]
        aug = iaa.Sequential(aug_list)
        augmented_image = aug.augment_images([image])[0]
        augmented_image = self.resize(augmented_image, original_shape)
        augmented_image = augmented_image.astype(np.uint8)
        return augmented_image


    def add_augmentations(
        self, 
        depiction: np.array
        ) -> np.array:
        """
        This function takes a chemical structure depiction (np.array) and returns the same image
        with added augmentation elements
 
        Args:
            depiction (np.array): chemical structure depiction

        Returns:
            np.array: chemical structure depiction with added augmentations
        """
        if self.random_choice([True, False, False, False, False, False], log_attribute='has_curved_arrows'):
            depiction = self.add_curved_arrows_to_structure(depiction)
        if self.random_choice([True, False, False], log_attribute='has_straight_arrows'):
            depiction = self.add_straight_arrows_to_structure(depiction)
        if self.random_choice([True, False, False, False, False, False], log_attribute='has_id_label'):
            depiction = self.add_chemical_label(depiction, "ID")
        if self.random_choice([True, False, False, False, False, False], log_attribute='has_R_group_label'):
            depiction = self.add_chemical_label(depiction, "R_GROUP")
        if self.random_choice([True, False, False, False, False, False], log_attribute='has_reaction_label'):
            depiction = self.add_chemical_label(depiction, "REACTION")
        if self.random_choice([True, False, False]):
            depiction = self.imgaug_augment(depiction)
        return depiction
        


    def get_random_label_position(
        self, 
        width: int, 
        height: int
        ) -> Tuple:
        """Given the width and height of an image (int), this function determines a random
        position in the outer 15% of the image and returns a tuple that contain the coordinates
        (y,x) of that position."""
        if self.random_choice([True, False]):
            y_range = range(0, height)
            x_range = list(range(0, int(0.15 * width))) + list(
                range(int(0.85 * width), width)
            )
        else:
            y_range = list(range(0, int(0.15 * height))) + list(
                range(int(0.85 * height), height)
            )
            x_range = range(0, width)
        return self.random_choice(y_range), self.random_choice(x_range)

    def ID_label_text(self) -> str:
        """This function returns a string that resembles a typical chemica ID label"""
        label_num = range(1, 50)
        label_letters = [
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
        ]
        options = [
            "only_number",
            "num_letter_combination",
            "numtonum",
            "numcombtonumcomb",
        ]
        option = self.random_choice(options)
        if option == "only_number":
            return str(self.random_choice(label_num))
        if option == "num_letter_combination":
            return str(self.random_choice(label_num)) + self.random_choice(
                label_letters
            )
        if option == "numtonum":
            return (
                str(self.random_choice(label_num))
                + "-"
                + str(self.random_choice(label_num))
            )
        if option == "numcombtonumcomb":
            return (
                str(self.random_choice(label_num))
                + self.random_choice(label_letters)
                + "-"
                + self.random_choice(label_letters)
            )

    def new_reaction_condition_elements(self) -> Tuple[str, str, str]:
        """Randomly redefine reaction_time, solvent and other_reactand."""
        reaction_time = self.random_choice(
            [str(num) for num in range(30)]
        ) + self.random_choice([" h", " min"])
        solvent = self.random_choice(
            [
                "MeOH",
                "EtOH",
                "CHCl3",
                "DCM",
                "iPrOH",
                "MeCN",
                "DMSO",
                "pentane",
                "hexane",
                "benzene",
                "Et2O",
                "THF",
                "DMF",
            ]
        )
        other_reactand = self.random_choice(
            [
                "HF",
                "HCl",
                "HBr",
                "NaOH",
                "Et3N",
                "TEA",
                "Ac2O",
                "DIBAL",
                "DIBAL-H",
                "DIPEA",
                "DMAP",
                "EDTA",
                "HOBT",
                "HOAt",
                "TMEDA",
                "p-TsOH",
                "Tf2O",
            ]
        )
        return reaction_time, solvent, other_reactand

    def reaction_condition_label_text(self) -> str:
        """This function returns a random string that looks like a reaction condition label."""
        reaction_condition_label = ""
        label_type = self.random_choice(["A", "B", "C", "D"])
        if label_type in ["A", "B"]:
            for n in range(self.random_choice(range(1, 5))):
                (
                    reaction_time,
                    solvent,
                    other_reactand,
                ) = self.new_reaction_condition_elements()
                if label_type == "A":
                    reaction_condition_label += (
                        str(n + 1)
                        + " "
                        + other_reactand
                        + ", "
                        + solvent
                        + ", "
                        + reaction_time
                        + "\n"
                    )
                elif label_type == "B":
                    reaction_condition_label += (
                        str(n + 1)
                        + " "
                        + other_reactand
                        + ", "
                        + solvent
                        + " ("
                        + reaction_time
                        + ")\n"
                    )
        elif label_type == "C":
            (
                reaction_time,
                solvent,
                other_reactand,
            ) = self.new_reaction_condition_elements()
            reaction_condition_label += (
                other_reactand + "\n" + solvent + "\n" + reaction_time
            )
        elif label_type == "D":
            reaction_condition_label += self.random_choice(
                self.new_reaction_condition_elements()
            )
        return reaction_condition_label

    def make_R_group_str(self) -> str:
        """This function returns a random string that looks like an R group label.
        It generates them by inserting randomly chosen elements into one of five
        templates."""
        rest_variables = [
            "X",
            "Y",
            "Z",
            "R",
            "R1",
            "R2",
            "R3",
            "R4",
            "R5",
            "R6",
            "R7",
            "R8",
            "R9",
            "R10",
            "Y2",
            "D",
        ]
        # Load list of superatoms (from OSRA)
        superatoms = self.superatoms
        label_type = self.random_choice(["A", "B", "C", "D", "E"])
        R_group_label = ""
        if label_type == "A":
            for _ in range(1, self.random_choice(range(2, 6))):
                R_group_label += (
                    self.random_choice(rest_variables)
                    + " = "
                    + self.random_choice(superatoms)
                    + "\n"
                )
        elif label_type == "B":
            R_group_label += "      " + self.random_choice(rest_variables) + "\n"
            for n in range(1, self.random_choice(range(2, 6))):
                R_group_label += str(n) + "    " + self.random_choice(superatoms) + "\n"
        elif label_type == "C":
            R_group_label += (
                "      "
                + self.random_choice(rest_variables)
                + "      "
                + self.random_choice(rest_variables)
                + "\n"
            )
            for n in range(1, self.random_choice(range(2, 6))):
                R_group_label += (
                    str(n)
                    + "  "
                    + self.random_choice(superatoms)
                    + "  "
                    + self.random_choice(superatoms)
                    + "\n"
                )
        elif label_type == "D":
            R_group_label += (
                "      "
                + self.random_choice(rest_variables)
                + "      "
                + self.random_choice(rest_variables)
                + "      "
                + self.random_choice(rest_variables)
                + "\n"
            )
            for n in range(1, self.random_choice(range(2, 6))):
                R_group_label += (
                    str(n)
                    + "  "
                    + self.random_choice(superatoms)
                    + "  "
                    + self.random_choice(superatoms)
                    + "  "
                    + self.random_choice(superatoms)
                    + "\n"
                )
        if label_type == "E":
            for n in range(1, self.random_choice(range(2, 6))):
                R_group_label += (
                    str(n)
                    + "  "
                    + self.random_choice(rest_variables)
                    + " = "
                    + self.random_choice(superatoms)
                    + "\n"
                )
        return R_group_label

    def add_chemical_label(self, image: np.array, label_type: str, foreign_fonts: bool= True) -> np.array:
        '''This function takes an image (np.array) and adds random text that looks like a chemical ID label ,
        an R group label or a reaction condition label around the structure. It returns the modified image.
        The label type is determined by the parameter label_type (str), which needs to be "ID", R_GROUP" or
        "REACTION"'''
        
        im = Image.fromarray(image)
        orig_image = deepcopy(im)
        width, height = im.size
        # Choose random font
        if self.random_choice([True, False]) or not foreign_fonts:
            font_dir = HERE.joinpath("fonts/")
        #In half of the cases: Use foreign-looking font to generate bigger noise variety
        else:
            font_dir = HERE.joinpath("foreign_fonts/")

        fonts = os.listdir(str(font_dir))
        # Choose random font size
        font_sizes = range(10, 20)
        size = self.random_choice(font_sizes)
        # Generate random string that resembles the desired type of chemical label
        if label_type == "ID":
            label_text = self.ID_label_text()
        if label_type == "R_GROUP":
            label_text = self.make_R_group_str()
        if label_type == "REACTION":
            label_text = self.reaction_condition_label_text()

        try:
            font = ImageFont.truetype(
                str(os.path.join(str(font_dir), self.random_choice(fonts))), size=size
            )
        except OSError:
            font = ImageFont.load_default()

        draw = ImageDraw.Draw(im, "RGBA")

        # Try different positions with the condition that the label´does not overlap with non-white pixels (the structure)
        for _ in range(50):
            y_pos, x_pos = self.get_random_label_position(width, height)
            bounding_box = draw.textbbox(
                (x_pos, y_pos), label_text, font=font
            )  # left, up, right, low
            paste_region = orig_image.crop(bounding_box)
            try:
                mean = ImageStat.Stat(paste_region).mean
            except ZeroDivisionError:
                return np.asarray(im)
            if sum(mean) / len(mean) == 255:
                draw.text((x_pos, y_pos), label_text, font=font, fill=(0, 0, 0, 255))
                break

        return np.asarray(im)

    def add_curved_arrows_to_structure(
        self, 
        image: np.array
        ) -> np.array:
        """This function takes an image of a chemical structure (np.array) and adds between 2 and 4 curved arrows
        in random positions in the central part of the image."""
        height, width, _ = image.shape
        image = Image.fromarray(image)
        orig_image = deepcopy(image)
        # Determine area where arrows are pasted.
        x_min, x_max = (int(0.1 * width), int(0.9 * width))
        y_min, y_max = (int(0.1 * height), int(0.9 * height))

        arrow_dir = os.path.normpath(str(HERE.joinpath("arrow_images/curved_arrows/")))

        for _ in range(self.random_choice(range(2, 4))):
            # Load random curved arrow image, resize and rotate it randomly.
            arrow_image = Image.open(
                os.path.join(
                    str(arrow_dir), self.random_choice(os.listdir(str(arrow_dir)))
                )
            )
            new_arrow_image_shape = int(
                (x_max - x_min) / self.random_choice(range(3, 6))
            ), int((y_max - y_min) / self.random_choice(range(3, 6)))
            arrow_image = self.resize(np.asarray(arrow_image), new_arrow_image_shape)
            arrow_image = Image.fromarray(arrow_image)
            arrow_image = arrow_image.rotate(
                self.random_choice(range(360)), 
                resample=self.random_choice([Image.BICUBIC, Image.NEAREST, Image.BILINEAR]), 
                expand=True
            )
            # Try different positions with the condition that the arrows are overlapping with non-white pixels (the structure)
            for _ in range(50):
                x_position = self.random_choice(
                    range(x_min, x_max - new_arrow_image_shape[0])
                )
                y_position = self.random_choice(
                    range(y_min, y_max - new_arrow_image_shape[1])
                )
                paste_region = orig_image.crop(
                    (
                        x_position,
                        y_position,
                        x_position + new_arrow_image_shape[0],
                        y_position + new_arrow_image_shape[1],
                    )
                )
                mean = ImageStat.Stat(paste_region).mean
                if sum(mean) / len(mean) < 252:
                    image.paste(arrow_image, (x_position, y_position), arrow_image)

                    break
        return np.asarray(image)

    def get_random_straight_arrow_position(self, width, height):
        """Given the width and height of an image (int), this function determines a random
        position to paste a reaction arrow."""
        if self.random_choice([True, False]):
            y_range = range(0, height)
            x_range = list(range(0, int(0.15 * width))) + list(
                range(int(0.85 * width), width)
            )
        else:
            y_range = list(range(0, int(0.15 * height))) + list(
                range(int(0.85 * height), height)
            )
            x_range = range(0, int(0.5 * width))
        return self.random_choice(y_range), self.random_choice(x_range)

    def add_straight_arrows_to_structure(
        self, 
        image: np.array
        ) -> np.array:
        """This function takes an image of a chemical structure (np.array) and adds between 1 and 2 straight arrows
        in random positions in the image (no overlap with other elements)"""
        height, width, _ = image.shape
        image = Image.fromarray(image)

        arrow_dir = os.path.normpath(
            str(HERE.joinpath("arrow_images/straight_arrows/"))
        )

        for _ in range(self.random_choice(range(1, 3))):
            # Load random curved arrow image, resize and rotate it randomly.
            arrow_image = Image.open(
                os.path.join(
                    str(arrow_dir), self.random_choice(os.listdir(str(arrow_dir)))
                )
            )
            # new_arrow_image_shape = (int(width * self.random_choice(np.arange(0.9, 1.5, 0.1))), int(height/10 * self.random_choice(np.arange(0.7, 1.2, 0.1))))

            # arrow_image = arrow_image.resize(new_arrow_image_shape, resample=Image.BICUBIC)
            # Rotate completely randomly in half of the cases and in   180° steps in the other cases (higher probability that pasting works)
            if self.random_choice([True, False]):
                arrow_image = arrow_image.rotate(
                    self.random_choice(range(360)), 
                    resample=self.random_choice([Image.BICUBIC, Image.NEAREST, Image.BILINEAR]), 
                    expand=True
                )
            else:
                arrow_image = arrow_image.rotate(self.random_choice([180, 360]))
            new_arrow_image_shape = arrow_image.size
            # Try different positions with the condition that the arrows are overlapping with non-white pixels (the structure)
            for _ in range(50):
                y_position, x_position = self.get_random_straight_arrow_position(width, height)
                x2_position = x_position + new_arrow_image_shape[0]
                y2_position = y_position + new_arrow_image_shape[1]
                # Make sure we only check a region inside of the image
                if x2_position > width:
                    x2_position = width - 1
                if y2_position > height:
                    y2_position = height - 1
                paste_region = image.crop(
                    (x_position, y_position, x2_position, y2_position)
                )
                try:
                    mean = ImageStat.Stat(paste_region).mean
                    if sum(mean) / len(mean) == 255:
                        image.paste(arrow_image, (x_position, y_position), arrow_image)
                        if self.depiction_features:
                            self.depiction_features.straight_arrow_paste_pos_x.append(x_position)
                            self.depiction_features.straight_arrow_paste_pos_y.append(y_position)
                        break
                except ZeroDivisionError:
                    pass
        return np.asarray(image)

    def to_grayscale_float_img(self, image: np.array) -> np.array:
        """This function takes an image (np.array), converts it to grayscale and returns it."""
        return img_as_float(rgb2gray(image))

    def depict_save(
        self,
        smiles: str,
        images_per_structure: int,
        output_dir: str,
        shape: Tuple[int, int] = (299, 299),
        ID=False,
        seed: int = 0,
    ):
        """This function takes a SMILES str, the amount of images to create per SMILES str and the path
        of an output directory. It then creates images_per_structure depictions of the chemical structure
        that is represented by the SMILES str and saves it as PNG images in output_dir.
        If an ID is given, it is used as the base filename. Otherwise, the SMILES str is used."""
        # This seems a bit odd but it appears that it is the only way to make the seed tracking work
        # with multiprocessing
        depictor = RandomDepictor(seed + 13)

        if not ID:
            name = smiles
        else:
            name = ID
        for n in range(images_per_structure):
            image = depictor.random_depiction(smiles, shape)
            output_file_path = os.path.join(output_dir, name + "_" + str(n) + ".png")
            sk_io.imsave(output_file_path, img_as_ubyte(image))

    def depict_augment_save(
        self,
        smiles: str,
        images_per_structure: int,
        output_dir: str,
        shape: Tuple[int, int] = (299, 299),
        ID=False,
        seed: int = 0,
    ) -> None:
        """This function takes a SMILES str, the amount of images to create per SMILES str and the path
        of an output directory. It then creates images_per_structure augmented depictions of the chemical structure
        that is represented by the SMILES str and saves it as PNG images in output_dir.
        If an ID is given, it is used as the base filename. Otherwise, the SMILES str is used."""
        # This seems a bit odd but it appears that it is the only way to make the seed tracking work
        # with multiprocessing
        depictor = RandomDepictor(seed + 13)

        if not ID:
            name = smiles
        else:
            name = ID
        for n in range(images_per_structure):
            image = depictor(smiles, shape)
            output_file_path = os.path.join(output_dir, name + "_" + str(n) + ".png")
            sk_io.imsave(output_file_path, img_as_ubyte(image))

    def batch_depict_save(
        self,
        smiles_list: List[str],
        images_per_structure: int,
        output_dir: str,
        shape: Tuple[int, int] = (299, 299),
        ID_list: List[str] = False,
        processes: int = 4,
        seed: int = 42,
    ) -> None:
        """This function takes a list of SMILES str, the amount of images to create per SMILES str and the path
        of an output directory. It then creates images_per_structure depictions of each chemical structure
        that is represented by a SMILES str and saves them as PNG images in output_dir.
        If an ID list (list with names of same length as smiles_list that contains unique IDs), the IDs will be used
        to name the files. Otherwise, the SMILES str is used as a filename."""
        if ID_list:
            starmap_tuple_generator = (
                (
                    smiles_list[n],
                    images_per_structure,
                    output_dir,
                    shape,
                    ID_list[n],
                    (seed * n + 1) * len(smiles_list),  # individual seed
                )
                for n in range(len(smiles_list))
            )
        else:
            starmap_tuple_generator = (
                (smiles, images_per_structure, output_dir, shape)
                for smiles in smiles_list
            )
        with get_context("spawn").Pool(processes) as p:
            p.starmap(self.depict_save, starmap_tuple_generator)

    def batch_depict_augment_save(
        self,
        smiles_list: List[str],
        images_per_structure: int,
        output_dir: str,
        shape: Tuple[int, int] = (299, 299),
        ID_list: List[str] = False,
        processes: int = 4,
        seed: int = 42,
    ) -> None:
        """This function takes a list of SMILES str, the amount of images to create per SMILES str and the path
        of an output directory. It then creates images_per_structure augmented depictions of each chemical structure
        that is represented by a SMILES str and saves them as PNG images in output_dir.
        If an ID list (list with names of same length as smiles_list that contains unique IDs), the IDs will be used
        to name the files. Otherwise, the SMILES str is used as a filename."""
        if ID_list:
            starmap_tuple_generator = (
                (
                    smiles_list[n],
                    images_per_structure,
                    output_dir,
                    shape,
                    ID_list[n],
                    (seed * n + 1) * len(smiles_list),  # individual seed
                )
                for n in range(len(smiles_list))
            )

        else:
            starmap_tuple_generator = (
                (smiles, images_per_structure, output_dir, shape)
                for smiles in smiles_list
            )
        with get_context("spawn").Pool(processes) as p:
            p.starmap(self.depict_augment_save, starmap_tuple_generator)
