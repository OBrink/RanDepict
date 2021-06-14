import os
import numpy as np
import io
from skimage import io as sk_io
from skimage.transform import resize
from skimage.color import rgba2rgb
from skimage.util import img_as_ubyte
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageStat

import imgaug.augmenters as iaa
import random
from copy import deepcopy
from typing import Tuple

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from indigo import Indigo, IndigoException
from indigo.renderer import IndigoRenderer
from jpype import *
import base64


class random_depictor:
    '''This class contains everything necessary to generate a variety of random depictions
    with given SMILES strings. 
    An instance of random_depictor can be called with a SMILES str and returns an np.array
    that represents the RGB image with the given chemical structure.'''


    def __init__(self):
        '''Load the JVM only once, otherwise it produces errors.'''
        # Start the JVM to access Java classes
        try:
            jvmPath = getDefaultJVMPath() 
        except JVMNotFoundException:
            print("If you see this message, for some reason JPype cannot find jvm.dll.", 
                  "This indicates that the environment varibale JAVA_HOME is not set properly.",
                  "You can set it or set it manually in the code (see __init__() of random_depictor)")
            jvmPath ="Define/your/path/or/set/your/JAVA_HOME/variable/properly"
        if not isJVMStarted():
            jar_path = './jar_files/'
            startJVM(jvmPath, "-ea", "-Djava.class.path=./jar_files/cdk_2_5.jar")
            
            
    def __call__(self, smiles: str):
        # Depict structure with random parameters
        depiction = self.random_depiction(smiles)
        # In 50 percent of the cases add a chemical ID label or curved arrows
        if random.choice([True, False]):
            if random.choice([True, False]):
                depiction = self.add_chemical_ID(depiction)
            else:
                depiction = self.add_arrows_to_structure(depiction)
        # Add augmentations
        depiction = self.imgaug_augment(depiction)
        return depiction
    

    def __enter__(self):
        return self


    def __exit__(self, type, value, tb):
        # I am not really happy with this, I'd like to automatically close the JVM
        # But if it is closed once, you cannot reopen it.
        # Shutdown the JVM
        #shutdownJVM()
        pass
        

    def get_nonexisting_image_name(self, path: str = './tmp/', format: str = 'PNG') -> str:
        '''This function returns a random file name that does not already exist at path'''
        for _ in range(100):
            file_name = str(random.choice(range(10000))) + '.' + format
            if not os.path.exists(os.path.join(path, file_name)):
                return file_name
    
    
    def random_image_size(self, shape: Tuple[int,int,int]) -> Tuple[int,int,int]:  
        '''This function takes a random image shape and returns an image shape where the 
        first two dimensions are slightly distorted.'''
        # Set random depiction image shape (to cause a slight distortion)
        y = int(shape[0] * random.choice(np.arange(0.9 ,1.1, 0.02)))
        x = int(shape[1] * random.choice(np.arange(0.9 ,1.1, 0.02)))
        z = shape[2]
        return y, x, z
        

    def get_random_indigo_rendering_settings(self, shape: Tuple[int,int,int] = (299,299,3)) -> Indigo:
        '''This function defines random rendering options for the structure depictions created
        using Indigo. It returns an Indigo object with the settings.'''
        # Define random shape for depiction (within boundaries); image is resized later)
        indigo = Indigo()
        # Get slightly distorted shape
        y, x, _ = self.random_image_size(shape)
        indigo.setOption("render-image-width", x)
        indigo.setOption("render-image-height", y)
        # Set random bond line width
        bond_line_width = float(random.choice(np.arange(0.5, 2.5, 0.5)))
        indigo.setOption("render-bond-line-width", bond_line_width)
        # Set random relative thickness
        relative_thickness = float(random.choice(np.arange(1, 2.5, 0.5)))
        indigo.setOption("render-relative-thickness", relative_thickness)
        # Set random bond length 
        # CAREFUL! Changing the bond length does not change the bond length relative to other
        # elements. Instead, the whole molecule is scaled down!
        #bond_length = random.choice(range(int(shape[0]/19), int(shape[0]/6))) #19
        #indigo.setOption("render-bond-length", bond_length)
        # Output_format: PNG
        indigo.setOption("render-output-format", "png")
        # Set random atom label rendering model (standard is rendering terminal groups)
        if random.choice([True] + [False] * 19):
            indigo.setOption('render-label-mode', 'all') # show all atom labels
        elif random.choice([True] + [False] * 3):
            indigo.setOption('render-label-mode', 'hetero') # only hetero atoms, no terminal groups
        # Set random depiction colour / not necessary for us as we binarise everything anyway
        #if random.choice([True, False, False, False, False]):
        #    R = str(random.choice(np.arange(0.1, 1.0, 0.1)))
        #    G = str(random.choice(np.arange(0.1, 1.0, 0.1)))
        #    B = str(random.choice(np.arange(0.1, 1.0, 0.1)))
        #    indigo.setOption("render-base-color", ", ".join([R,G,B]))
        # Render bold bond for Haworth projection
        if random.choice([True, False]):
            indigo.setOption("render-bold-bond-detection", "True")
        # Render labels for stereobonds
        stereo_style = random.choice(["ext", "old", "none"])
        indigo.setOption("render-stereo-style", stereo_style)
        # Collapse superatoms (default: expand)
        if random.choice([True, False]):
            indigo.setOption("render-superatom-mode", 'collapse')
        return indigo

    
    def depict_and_resize_indigo(self, smiles: str, shape: Tuple[int,int,int]=(299, 299, 3)) -> np.array:
        '''This function takes a smiles str and an image shape. It renders the chemical structures
        using Indigo with random rendering/depiction settings and returns an RGB image (np.array) 
        with the given image shape.'''
        # Instantiate Indigo with random settings and IndigoRenderer
        try:
            indigo = self.get_random_indigo_rendering_settings()
        except IndigoException:
             # This happens the first time something is depicted, I don't know why.
            indigo = Indigo() 
        renderer = IndigoRenderer(indigo)
        # Load molecule
        molecule = indigo.loadMolecule(smiles)
        # Do not kekulize in 20% of cases
        if random.choice([True, False, False, False, False]):
            molecule.aromatize()
        molecule.layout()
        # Create structure depiction, save in temporary file and load as Pillow image
        # TODO: Do this without saving file (renderToBuffer format does not seem to work)
        tmp_filename = self.get_nonexisting_image_name(path= './tmp/', format= 'png')
        renderer.renderToFile(molecule, os.path.join('./tmp/', tmp_filename))
        depiction = sk_io.imread(os.path.join('./tmp/', tmp_filename))
        os.remove(os.path.join('./tmp/', tmp_filename))
        depiction = resize(depiction, (shape[0], shape[1], 4))
        depiction = rgba2rgb(depiction)
        depiction = img_as_ubyte(depiction)
        return depiction


    def get_random_rdkit_rendering_settings(self, shape: Tuple[int,int,int] = (299,299,3)) -> rdMolDraw2D.MolDraw2DCairo:
        '''This function defines random rendering options for the structure depictions created
        using rdkit. It returns an MolDraw2DCairo object with the settings.'''
        # Get slightly distorted shape
        y, x, _ = self.random_image_size(shape)
        # Instantiate object that saves the settings
        depiction_settings = rdMolDraw2D.MolDraw2DCairo(y,x)
        # Stereo bond annotation
        if random.choice([True, False]):
            depiction_settings.drawOptions().addStereoAnnotation = True
        if random.choice([True, False]):
            depiction_settings.drawOptions().includeChiralFlagLabel = True
        # Atom indices
        if random.choice([True, False, False, False]):
            depiction_settings.drawOptions().addAtomIndices = True
        # Bond line width
        bond_line_width = random.choice(range(1, 5))
        depiction_settings.drawOptions().bondLineWidth = bond_line_width
        # Draw terminal methyl groups
        if random.choice([True, False]):
            depiction_settings.drawOptions().explicitMethyl = True
        # Label font type and size
        font_dir = os.path.normpath('./fonts/')
        font_path = os.path.join(font_dir, random.choice(os.listdir(font_dir)))
        depiction_settings.drawOptions().fontFile = font_path
        min_font_size = random.choice(range(10, 20))
        depiction_settings.drawOptions().minFontSize = min_font_size
        depiction_settings.drawOptions().maxFontSize = 30
        # Rotate the molecule
        depiction_settings.drawOptions().rotate = random.choice(range(360))
        # Fixed bond length
        fixed_bond_length = random.choice(range(30,45))
        depiction_settings.drawOptions().fixedBondLength = fixed_bond_length
        # Comic mode (looks a bit hand drawn)
        if random.choice([True, False, False, False, False]):
            depiction_settings.drawOptions().comicMode = True
        # Keep it black and white
        depiction_settings.drawOptions().useBWAtomPalette()
        return depiction_settings

    
    def depict_and_resize_rdkit(self, smiles: str, shape: Tuple[int,int]=(299, 299, 3)) -> np.array:
        '''This function takes a smiles str and an image shape. It renders the chemical structures
        using Rdkit with random rendering/depiction settings and returns an RGB image (np.array) 
        with the given image shape.'''
        # Generate mol object from smiles str
        mol = Chem.MolFromSmiles(smiles)
        # Get random depiction settings
        depiction_settings = self.get_random_rdkit_rendering_settings()
        # Create depiction
        # TODO: Figure out how to depict molecules without kekulization here.
        # This does not prevent the molecule from being depicted kekulized
        #mol = rdMolDraw2D.PrepareMolForDrawing(mol, kekulize = False) 
        rdMolDraw2D.PrepareAndDrawMolecule(depiction_settings, mol)
        depiction = depiction_settings.GetDrawingText() 
        depiction = sk_io.imread(io.BytesIO(depiction))
        # Resize image to desired shape
        depiction = resize(depiction, shape)
        depiction = img_as_ubyte(depiction)
        return np.asarray(depiction)

    
    def get_random_cdk_rendering_settings(self, rendererModel, molecule):
        '''This function defines random rendering options for the structure depictions created
        using CDK. It takes a org.openscience.cdk.renderer.AtomContainerRenderer.2DModel and a
        org.openscience.cdk.AtomContainer and returns the 2DModel object with random rendering settings.
        I followed https://github.com/cdk/cdk/wiki/Standard-Generator while creating this.'''
        
        StandardGenerator = JClass("org.openscience.cdk.renderer.generators.standard.StandardGenerator")
        
        # Define visibility of atom/superatom labels
        SymbolVisibility = JClass("org.openscience.cdk.renderer.SymbolVisibility")
        if random.choice([True] + [False] * 19):
            rendererModel.set(StandardGenerator.Visibility.class_, SymbolVisibility.all()) # show all atom labels
        elif random.choice([True] + [False] * 3):
            # only hetero atoms, no terminal alkyl groups
            rendererModel.set(StandardGenerator.Visibility.class_, SymbolVisibility.iupacRecommendationsWithoutTerminalCarbon())
        else:
            rendererModel.set(StandardGenerator.Visibility.class_, SymbolVisibility.iupacRecommendations())
        # Define bond line stroke width
        stroke_width = random.choice(np.arange(0.8, 2.0, 0.1))
        rendererModel.set(StandardGenerator.StrokeRatio.class_, stroke_width);
        # Define symbol margin ratio
        margin_ratio = random.choice([0,1,2,2,2,3,4])
        rendererModel.set(StandardGenerator.SymbolMarginRatio.class_, JClass("java.lang.Double")(margin_ratio))
        # Define bond properties
        double_bond_dist = random.choice(np.arange(0.11, 0.25, 0.01))
        rendererModel.set(StandardGenerator.BondSeparation.class_, double_bond_dist)
        wedge_ratio = random.choice(np.arange(4.5, 7.5, 0.1))
        rendererModel.set(StandardGenerator.WedgeRatio.class_, JClass("java.lang.Double")(wedge_ratio))
        if random.choice([True, False]):
            rendererModel.set(StandardGenerator.FancyBoldWedges.class_, False)
        if random.choice([True, False]):
            rendererModel.set(StandardGenerator.FancyHashedWedges.class_, False);
        hash_spacing = random.choice(np.arange(4.0, 6.0, 0.2))
        rendererModel.set(StandardGenerator.HashSpacing.class_, hash_spacing)
        # Add CIP labels
        labels = False
        if random.choice([True, True]):
            labels = True
            JClass("org.openscience.cdk.geometry.cip.CIPTool").label(molecule)
            for atom in molecule.atoms():
                label = atom.getProperty(JClass("org.openscience.cdk.CDKConstants").CIP_DESCRIPTOR)
                atom.setProperty(StandardGenerator.ANNOTATION_LABEL, label)
            for bond in molecule.bonds():
                label = bond.getProperty(JClass("org.openscience.cdk.CDKConstants").CIP_DESCRIPTOR)
                bond.setProperty(StandardGenerator.ANNOTATION_LABEL, label)
        # Add atom indices to the depictions
        if random.choice([True, False, False, False]):
            labels = True
            for atom in molecule.atoms():
                label = JClass("java.lang.Integer")(1 + molecule.getAtomNumber(atom))
                atom.setProperty(StandardGenerator.ANNOTATION_LABEL, label)
        if labels:
            # We only need black
            rendererModel.set(StandardGenerator.AnnotationColor.class_, JClass("java.awt.Color")(0x000000))
            # Font size of labels
            font_scale = random.choice(np.arange(0.5, 0.8, 0.1))
            rendererModel.set(StandardGenerator.AnnotationFontScale.class_, font_scale)
            # Distance between atom numbering and depiction
            annotation_distance = random.choice(np.arange(0.15, 0.30, 0.05))
            rendererModel.set(StandardGenerator.AnnotationDistance.class_, annotation_distance)
        return rendererModel
    

    def depict_and_resize_cdk(self, smiles: str, shape: Tuple[int,int]=(299, 299, 3)) -> np.array:
        '''This function takes a smiles str and an image shape. It renders the chemical structures
        using CDK with random rendering/depiction settings and returns an RGB image (np.array) 
        with the given image shape.
        The general workflow here is a JPype adaptation of code published by Egon Willighagen
        in 'Groovy Cheminformatics with the Chemistry Development Kit':
        https://egonw.github.io/cdkbook/ctr.html#depict-a-compound-as-an-image 
        with additional adaptations to create all the different depiction types from
        https://github.com/cdk/cdk/wiki/Standard-Generator'''

        # Read molecule from SMILES str
        SCOB = JClass("org.openscience.cdk.silent.SilentChemObjectBuilder")
        SmilesParser = JClass("org.openscience.cdk.smiles.SmilesParser")(SCOB.getInstance())
        if random.choice([True, False, False, False, False]):
            SmilesParser.kekulise(False)
        molecule = SmilesParser.parseSmiles(smiles)
        # Instantiate StructureDiagramGenerator, determine coordinates
        sdg = JClass("org.openscience.cdk.layout.StructureDiagramGenerator")()
        sdg.setMolecule(molecule)
        sdg.generateCoordinates()
        molecule = sdg.getMolecule()

        # Rotate molecule randomly
        point = JClass("org.openscience.cdk.geometry.GeometryTools").get2DCenter(molecule)
        rot_degrees = random.choice(range(360))
        JClass("org.openscience.cdk.geometry.GeometryTools").rotate(molecule, point, rot_degrees)

        # Get Generators
        generators = JClass("java.util.ArrayList")()
        BasicSceneGenerator = JClass("org.openscience.cdk.renderer.generators.BasicSceneGenerator")()
        generators.add(BasicSceneGenerator)
        font_size = random.choice(range(10, 20))
        Font = JClass("java.awt.Font")
        font_name = random.choice(["Verdana", "Times New Roman", "Arial", "Gulliver Regular"])
        font_style = random.choice([Font.PLAIN, Font.BOLD])
        font = Font(font_name, font_style, font_size)
        #font = Font("Verdana", Font.PLAIN, font_size) # FONTFONTFONTFONTFONTFONTFONTFONTFONTFONTFONTFONTFONTFONTFONTFONTFONTFONTFONTFONTFONTFONTFONTFONTFONTFONTFONTFONTFONTFONTFONTFONTFONTFONTFONTFONTFONTFONTFONTFONT
        StandardGenerator = JClass("org.openscience.cdk.renderer.generators.standard.StandardGenerator")(font)
        generators.add(StandardGenerator)
        
        # Instantiate renderer
        AWTFontManager = JClass("org.openscience.cdk.renderer.font.AWTFontManager")
        renderer = JClass("org.openscience.cdk.renderer.AtomContainerRenderer")(generators, AWTFontManager())
        
        # Create an empty image of the right size
        y, x, _ = self.random_image_size(shape)
        drawArea = JClass("java.awt.Rectangle")(x, y)
        BufferedImage = JClass("java.awt.image.BufferedImage")
        image = BufferedImage(x, y, BufferedImage.TYPE_INT_RGB)

        # Draw the molecule
        renderer.setup(molecule, drawArea)
        model = renderer.getRenderer2DModel()
        
        # Get random rendering settings
        model = self.get_random_cdk_rendering_settings(model, molecule)
        
        double = JClass("java.lang.Double")
        model.set(JClass("org.openscience.cdk.renderer.generators.BasicSceneGenerator.ZoomFactor"), double(0.9))
        g2 = image.getGraphics()
        g2.setColor(JClass("java.awt.Color").WHITE)
        g2.fillRect(0, 0, x, y)
        AWTDrawVisitor = JClass("org.openscience.cdk.renderer.visitor.AWTDrawVisitor")
        renderer.paint(molecule, AWTDrawVisitor(g2))
        
        # Write the image into a format that can be read by skimage
        ImageIO = JClass("javax.imageio.ImageIO")
        os = JClass("java.io.ByteArrayOutputStream")()
        Base64 = JClass("java.util.Base64")
        ImageIO.write(image, JClass("java.lang.String")("PNG"), Base64.getEncoder().wrap(os))
        depiction = bytes(os.toString("UTF-8"))
        depiction = base64.b64decode(depiction)
        
        # Read image in skimage
        depiction = sk_io.imread(depiction, plugin='imageio')
        depiction = resize(depiction, shape)
        depiction = img_as_ubyte(depiction)
        return depiction


    def random_depiction(self, smiles: str, shape: Tuple[int, int, int] = (299, 299, 3)) -> np.array:
        '''This function takes a SMILES str and depicts it using Rdkit, Indigo or CDK.
        The depiction method and the specific parameters for the depiction are chosen
        completely randomly. The purpose of this function is to enable depicting a diverse
        variety of chemical structure depictions.'''
        depiction_functions = [self.depict_and_resize_rdkit, 
                               self.depict_and_resize_indigo,
                               self.depict_and_resize_cdk,
                              ]
        depiction_function = random.choice(depiction_functions)
        depiction = depiction_function(smiles, shape)
        return depiction


    def imgaug_augment(self, image: np.array) -> np.array:
        '''This function applies a random amount of augmentations to a given image (np.array) 
        using and returns the augmented image (np.array).'''
        aug_list = [
            # Rotation between -10 and 10 degrees
            iaa.Affine(rotate=(-10, 10), mode = 'edge', fit_output = True),
            # Black and white noise
            iaa.Sequential([
                            iaa.CoarseDropout((0.0002, 0.0015), size_percent=(1.0, 1.1)), 
                            iaa.ReplaceElementwise((0.01, 0.3), 255)]),
            # Shearing
            iaa.OneOf([
                        iaa.geometric.ShearX((-5, 5), mode = 'edge', fit_output = True),
                        iaa.geometric.ShearY((-5, 5), mode = 'edge', fit_output = True)]),
            # Jpeg compression or pixelation
            iaa.OneOf([
                iaa.imgcorruptlike.JpegCompression(severity=(1,2)),
                iaa.imgcorruptlike.Pixelate(severity=(1,2))]),
            # Brightness adjustment
            iaa.WithBrightnessChannels(iaa.Add((-50, 50))),
            # Colour temperature adjustment
            iaa.ChangeColorTemperature((1100, 10000)), # colour temperature adjustment
        ]
        # Apply zero to all augmentations
        aug_number = random.choice(range(0, 3))
        aug = iaa.SomeOf(aug_number, aug_list)
        augmented_image = aug.augment_images([image])[0]
        augmented_image = resize(augmented_image, image.shape)
        return augmented_image


    def get_random_label_position(self, width: int, height: int) -> Tuple:
        '''Given the width and height of an image (int), this function determines a random 
        position in the outer 5% of the image and returns a tuple that contain the coordinates
        (y,x) of that position.'''
        if random.choice([True, False]):
            y_range = range(0, height)
            x_range = list(range(0, int(0.05*width))) + list(range(int(0.95*width), width))
        else:
            y_range = list(range(0, int(0.05*height))) + list(range(int(0.95*height), height))
            x_range = range(0, width)
        return random.choice(y_range), random.choice(x_range)
            

    def ID_label_text(self)->str:
        '''This function returns a string that resembles a typical chemica ID label'''
        label_num = range(1,50)
        label_letters = ["a", "b", "c", "d", "e", "f", "g", "i", "j", "k", "l", "m", "n", "o"]
        options = ["only_number", "num_letter_combination", "numtonum", "numcombtonumcomb"]
        option = random.choice(options)
        if option == "only_number":
            return str(random.choice(label_num))
        if option == "num_letter_combination":
            return str(random.choice(label_num)) + random.choice(label_letters)
        if option == "numtonum":
            return str(random.choice(label_num)) + "-" + str(random.choice(label_num))
        if option == "numcombtonumcomb":
            return str(random.choice(label_num)) + random.choice(label_letters) + "-" + random.choice(label_letters)


    def add_chemical_ID(self, image: np.array)-> np.array:
        '''This function takes an image (np.array) and adds random text that looks like a chemical ID label 
        around the structure. It returns the modified image.'''
        
        im = Image.fromarray(image)
        # Choose random font
        font_dir = os.path.abspath("./fonts/")
        fonts = os.listdir(font_dir)
        font_sizes = range(12, 24)
        # Define random position for text element around the chemical structure (if it can be placed around it)
        width, height = im.size
        y_pos, x_pos = self.get_random_label_position(width, height)

        # Choose random font size
        size = random.choice(font_sizes)
        label_text = self.ID_label_text()
        
        try:
            font = ImageFont.truetype(str(os.path.join(font_dir, random.choice(fonts))), size = size)
        except OSError:
            font = ImageFont.load_default()
        draw = ImageDraw.Draw(im, 'RGBA')
        bb = draw.textbbox((x_pos, y_pos), label_text, font = font) # left, up, right, low

        # Add text element to image
        draw = ImageDraw.Draw(im, 'RGBA')
        draw.text((x_pos,y_pos), label_text, font = font, fill=(0,0,0,255))
        return np.asarray(im)


    def add_arrows_to_structure(self, image: np.array) -> np.array:
        '''This function takes an image of a chemical structure (np.array) and adds between 2 and 6 curved arrows 
        in random positions in the central part of the image.'''
        height, width, _ = image.shape
        image = Image.fromarray(image)
        orig_image = deepcopy(image)
        # Determine area where arrows are pasted.
        x_min, x_max = (int(0.1 * width), int(0.9 * width))
        y_min, y_max = (int(0.1 * height), int(0.9 * height))
        
        arrow_dir = os.path.normpath('./arrow_images/')
        
        for _ in range(random.choice(range(2, 4))):
            # Load random curved arrow image, resize and rotate it randomly.
            arrow_image = Image.open(os.path.join(arrow_dir, random.choice(os.listdir(arrow_dir))))
            new_arrow_image_shape = int((x_max - x_min) / random.choice(range(3,6))), int((y_max - y_min) / random.choice(range(3,6)))
            arrow_image = arrow_image.resize(new_arrow_image_shape, resample=Image.BICUBIC)
            arrow_image = arrow_image.rotate(random.choice(range(360)), resample=Image.BICUBIC, expand=True)
            # Try different positions with the condition that the arrows are overlapping with non-white pixels (the structure)
            for _ in range(50):
                x_position = random.choice(range(x_min, x_max - new_arrow_image_shape[0]))
                y_position = random.choice(range(y_min, y_max - new_arrow_image_shape[1]))
                paste_region = orig_image.crop((x_position, y_position, x_position + new_arrow_image_shape[0], y_position + new_arrow_image_shape[1]))
                mean = ImageStat.Stat(paste_region).mean
                if sum(mean)/len(mean) < 252:
                    image.paste(arrow_image, (x_position, y_position), arrow_image)
                    break
        return np.asarray(image)