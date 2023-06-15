from copy import deepcopy
import cv2
import imgaug.augmenters as iaa
import numpy as np
import os
from PIL import Image, ImageEnhance, ImageFont, ImageDraw, ImageStat
from scipy.ndimage import gaussian_filter
from scipy.ndimage import map_coordinates
from skimage.color import rgb2gray
from skimage.util import img_as_float
from typing import Tuple


class Augmentations:
    def resize(self, image: np.array, shape: Tuple[int], HQ: bool = False) -> np.array:
        """
        This function takes an image (np.array) and a shape and returns
        the resized image (np.array). It uses Pillow to do this, as it
        seems to have a bigger variety of scaling methods than skimage.
        The up/downscaling method is chosen randomly.

        Args:
            image (np.array): the input image
            shape (Tuple[int, int], optional): im shape. Defaults to (299, 299)
            HQ (bool): if true, only choose from Image.BICUBIC, Image.LANCZOS
        ___
        Returns:
            np.array: the resized image

        """
        image = Image.fromarray(image)
        shape = (shape[0], shape[1])
        if not HQ:
            image = image.resize(
                shape, resample=self.random_choice(self.PIL_resize_methods)
            )
        else:
            image = image = image.resize(
                shape, resample=self.random_choice(self.PIL_HQ_resize_methods)
            )

        return np.asarray(image)

    def imgaug_augment(
        self,
        image: np.array,
    ) -> np.array:
        """
        This function applies a random amount of augmentations to
        a given image (np.array) using and returns the augmented image
        (np.array).

        Args:
            image (np.array): input image

        Returns:
            np.array: output image (augmented)
        """
        original_shape = image.shape

        # Choose number of augmentations to apply (0-2);
        # return image if nothing needs to be done.
        aug_number = self.random_choice(range(0, 3))
        if not aug_number:
            return image

        # Add some padding to avoid weird artifacts after rotation
        image = np.pad(
            image, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=255
        )

        def imgaug_rotation():
            # Rotation between -10 and 10 degrees
            if not self.random_choice(
                [True, True, False], log_attribute="has_imgaug_rotation"
            ):
                return False
            rot_angle = self.random_choice(np.arange(-10, 10, 1))
            aug = iaa.Affine(rotate=rot_angle, mode="edge", fit_output=True)
            return aug

        def imgaug_black_and_white_noise():
            # Black and white noise
            if not self.random_choice(
                [True, True, False], log_attribute="has_imgaug_salt_pepper"
            ):
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
            if not self.random_choice(
                [True, True, False], log_attribute="has_imgaug_shearing"
            ):
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
            if not self.random_choice(
                [True, True, False], log_attribute="has_imgaug_corruption"
            ):
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
            if not self.random_choice(
                [True, True, False], log_attribute="has_imgaug_brightness_adj"
            ):
                return False
            brightness_adj_param = self.random_choice(np.arange(-50, 50, 1))
            aug = iaa.WithBrightnessChannels(iaa.Add(brightness_adj_param))
            return aug

        def imgaug_colour_temp_adjustment():
            # Colour temperature adjustment
            if not self.random_choice(
                [True, True, False], log_attribute="has_imgaug_col_adj"
            ):
                return False
            colour_temp = self.random_choice(np.arange(1100, 10000, 1))
            aug = iaa.ChangeColorTemperature(colour_temp)
            return aug

        # Define list of available augmentations
        aug_list = [
            imgaug_rotation,
            imgaug_black_and_white_noise,
            imgaug_shearing,
            imgaug_imgcorruption,
            imgaug_brightness_adjustment,
            imgaug_colour_temp_adjustment,
        ]

        # Every one of them has a 1/3 chance of returning False
        aug_list = [fun() for fun in aug_list]
        aug_list = [fun for fun in aug_list if fun]
        aug = iaa.Sequential(aug_list)
        augmented_image = aug.augment_images([image])[0]
        augmented_image = self.resize(augmented_image, original_shape)
        augmented_image = augmented_image.astype(np.uint8)
        return augmented_image

    def add_augmentations(self, depiction: np.array) -> np.array:
        """
        This function takes a chemical structure depiction (np.array)
        and returns the same image with added augmentation elements

        Args:
            depiction (np.array): chemical structure depiction

        Returns:
            np.array: chemical structure depiction with added augmentations
        """
        if self.random_choice(
            [True, False, False, False, False, False], log_attribute="has_curved_arrows"
        ):
            depiction = self.add_curved_arrows_to_structure(depiction)
        if self.random_choice(
            [True, False, False], log_attribute="has_straight_arrows"
        ):
            depiction = self.add_straight_arrows_to_structure(depiction)
        if self.random_choice(
            [True, False, False, False, False, False], log_attribute="has_id_label"
        ):
            depiction = self.add_chemical_label(depiction, "ID")
        if self.random_choice(
            [True, False, False, False, False, False], log_attribute="has_R_group_label"
        ):
            depiction = self.add_chemical_label(depiction, "R_GROUP")
        if self.random_choice(
            [True, False, False, False, False, False],
            log_attribute="has_reaction_label",
        ):
            depiction = self.add_chemical_label(depiction, "REACTION")
        depiction = self.imgaug_augment(depiction)
        return depiction

    def get_random_label_position(self, width: int, height: int) -> Tuple[int, int]:
        """
        Given the width and height of an image (int), this function
        determines a random position in the outer 15% of the image and
        returns a tuple that contain the coordinates (y,x) of that position.

        Args:
            width (int): image width
            height (int): image height

        Returns:
            Tuple[int, int]: Random label position
        """
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
        """
        This function returns a string that resembles a typical
        chemical ID label

        Returns:
            str: Label text
        """
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
        """
        Randomly redefine reaction_time, solvent and other_reactand.

        Returns:
            Tuple[str, str, str]: Reaction time, solvent, reactand
        """
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
        """
        This function returns a random string that looks like a
        reaction condition label.

        Returns:
            str: Reaction condition label text
        """
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
        """
        This function returns a random string that looks like an R group label.
        It generates them by inserting randomly chosen elements into one of
        five templates.

        Returns:
            str: R group label text
        """
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

    def add_chemical_label(
        self, image: np.array, label_type: str, foreign_fonts: bool = True
    ) -> np.array:
        """
        This function takes an image (np.array) and adds random text that
        looks like a chemical ID label, an R group label or a reaction
        condition label around the structure. It returns the modified image.
        The label type is determined by the parameter label_type (str),
        which needs to be 'ID', 'R_GROUP' or 'REACTION'

        Args:
            image (np.array): Chemical structure depiction
            label_type (str): 'ID', 'R_GROUP' or 'REACTION'
            foreign_fonts (bool, optional): Defaults to True.

        Returns:
            np.array: Chemical structure depiction with label
        """
        im = Image.fromarray(image)
        orig_image = deepcopy(im)
        width, height = im.size
        # Choose random font
        if self.random_choice([True, False]) or not foreign_fonts:
            font_dir = self.HERE.joinpath("fonts/")
        # In half of the cases: Use foreign-looking font to generate
        # bigger noise variety
        else:
            font_dir = self.HERE.joinpath("foreign_fonts/")

        fonts = os.listdir(str(font_dir))
        # Choose random font size
        font_sizes = range(10, 20)
        size = self.random_choice(font_sizes)
        # Generate random string that resembles the desired type of label
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

        # Try different positions with the condition that the label´does not
        # overlap with non-white pixels (the structure)
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

    def add_curved_arrows_to_structure(self, image: np.array) -> np.array:
        """
        This function takes an image of a chemical structure (np.array)
        and adds between 2 and 4 curved arrows in random positions in the
        central part of the image.

        Args:
            image (np.array): Chemical structure depiction

        Returns:
            np.array: Chemical structure depiction with curved arrows
        """
        height, width, _ = image.shape
        image = Image.fromarray(image)
        orig_image = deepcopy(image)
        # Determine area where arrows are pasted.
        x_min, x_max = (int(0.1 * width), int(0.9 * width))
        y_min, y_max = (int(0.1 * height), int(0.9 * height))

        arrow_dir = os.path.normpath(
            str(self.HERE.joinpath("arrow_images/curved_arrows/"))
        )

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
                resample=self.random_choice(
                    [Image.BICUBIC, Image.NEAREST, Image.BILINEAR]
                ),
                expand=True,
            )
            # Try different positions with the condition that the arrows are
            # overlapping with non-white pixels (the structure)
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

    def get_random_arrow_position(self, width: int, height: int) -> Tuple[int, int]:
        """
        Given the width and height of an image (int), this function determines
        a random position to paste a reaction arrow in the outer 15% frame of
        the image

        Args:
            width (_type_): image width
            height (_type_): image height

        Returns:
            Tuple[int, int]: Random arrow position
        """
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

    def add_straight_arrows_to_structure(self, image: np.array) -> np.array:
        """
        This function takes an image of a chemical structure (np.array)
        and adds between 1 and 2 straight arrows in random positions in the
        image (no overlap with other elements)

        Args:
            image (np.array): Chemical structure depiction

        Returns:
            np.array: Chemical structure depiction with straight arrow
        """
        height, width, _ = image.shape
        image = Image.fromarray(image)

        arrow_dir = os.path.normpath(
            str(self.HERE.joinpath("arrow_images/straight_arrows/"))
        )

        for _ in range(self.random_choice(range(1, 3))):
            # Load random curved arrow image, resize and rotate it randomly.
            arrow_image = Image.open(
                os.path.join(
                    str(arrow_dir), self.random_choice(os.listdir(str(arrow_dir)))
                )
            )
            # new_arrow_image_shape = (int(width *
            # self.random_choice(np.arange(0.9, 1.5, 0.1))),
            # int(height/10 * self.random_choice(np.arange(0.7, 1.2, 0.1))))

            # arrow_image = arrow_image.resize(new_arrow_image_shape,
            # resample=Image.BICUBIC)
            # Rotate completely randomly in half of the cases and in 180° steps
            # in the other cases (higher probability that pasting works)
            if self.random_choice([True, False]):
                arrow_image = arrow_image.rotate(
                    self.random_choice(range(360)),
                    resample=self.random_choice(
                        [Image.Resampling.BICUBIC, Image.Resampling.NEAREST, Image.Resampling.BILINEAR]
                    ),
                    expand=True,
                )
            else:
                arrow_image = arrow_image.rotate(self.random_choice([180, 360]))
            new_arrow_image_shape = arrow_image.size
            # Try different positions with the condition that the arrows are
            # overlapping with non-white pixels (the structure)
            for _ in range(50):
                y_position, x_position = self.get_random_arrow_position(width, height)
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
                        break
                except ZeroDivisionError:
                    pass
        return np.asarray(image)

    def to_grayscale_float_img(self, image: np.array) -> np.array:
        """
        This function takes an image (np.array), converts it to grayscale
        and returns it.

        Args:
            image (np.array): image

        Returns:
            np.array: grayscale float image
        """
        return img_as_float(rgb2gray(image))

    def hand_drawn_augment(self, img) -> np.array:
        """
        This function randomly applies different image augmentations with
        different probabilities to the input image.

        It has been modified from the original augment.py present on
        https://github.com/mtzgroup/ChemPixCH

        From the publication:
        https://pubs.rsc.org/en/content/articlelanding/2021/SC/D1SC02957F

        Args:
            img: the image to modify in array format.
        Returns:
            img: the augmented image.
        """
        # resize
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.5:
            img = self.resize_hd(img)
        # blur
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.4:
            img = self.blur(img)
        # erode
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.4:
            img = self.erode(img)
        # dilate
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.4:
            img = self.dilate(img)
        # aspect_ratio
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.7:
            img = self.aspect_ratio(img, "mol")
        # affine
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.7:
            img = self.affine(img, "mol")
        # distort
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.8:
            img = self.distort(img)
        if img.shape != (255, 255, 3):
            img = cv2.resize(img, (256, 256))
        return img

    def augment_bkg(self, img) -> np.array:
        """
        This function randomly applies different image augmentations with
        different probabilities to the input image.
        Args:
            img: the image to modify in array format.
        Returns:
            img: the augmented image.
        """
        # rotate
        rows, cols, _ = img.shape
        angle = self.random_choice(np.arange(0, 360))
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        img = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
        # resize
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.5:
            img = self.resize_hd(img)
        # blur
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.4:
            img = self.blur(img)
        # erode
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.2:
            img = self.erode(img)
        # dilate
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.2:
            img = self.dilate(img)
        # aspect_ratio
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.3:
            img = self.aspect_ratio(img, "bkg")
        # affine
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.3:
            img = self.affine(img, "bkg")
        # distort
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.8:
            img = self.distort(img)
        if img.shape != (255, 255, 3):
            img = cv2.resize(img, (256, 256))
        return img

    def resize_hd(self, img) -> np.array:
        """
        This function resizes the image randomly from between (200-300, 200-300)
        and then resizes it back to 256x256.
        Args:
            img: the image to modify in array format.
        Returns:
            img: the resized image.
        """
        interpolations = [
            cv2.INTER_NEAREST,
            cv2.INTER_AREA,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ]

        img = cv2.resize(
            img,
            (
                self.random_choice(np.arange(200, 300)),
                self.random_choice(np.arange(200, 300)),
            ),
            interpolation=self.random_choice(interpolations),
        )
        img = cv2.resize(
            img, (256, 256), interpolation=self.random_choice(interpolations)
        )

        return img

    def blur(self, img) -> np.array:
        """
        This function blurs the image randomly between 1-3.
        Args:
            img: the image to modify in array format.
        Returns:
            img: the blurred image.
        """
        n = self.random_choice(np.arange(1, 4))
        kernel = np.ones((n, n), np.float32) / n**2
        img = cv2.filter2D(img, -1, kernel)
        return img

    def erode(self, img) -> np.array:
        """
        This function bolds the image randomly between 1-2.
        Args:
           img: the image to modify in array format.
        Returns:
            img: the bold image.
        """
        n = self.random_choice(np.arange(1, 3))
        kernel = np.ones((n, n), np.float32) / n**2
        img = cv2.erode(img, kernel, iterations=1)
        return img

    def dilate(self, img) -> np.array:
        """
        This function dilates the image with a factor of 2.
        Args:
           img: the image to modify in array format.
        Returns:
            img: the dilated image.
        """
        n = 2
        kernel = np.ones((n, n), np.float32) / n**2
        img = cv2.dilate(img, kernel, iterations=1)
        return img

    def aspect_ratio(self, img, obj=None) -> np.array:
        """
        This function irregularly changes the size of the image
        and converts it back to (256,256).
        Args:
            img: the image to modify in array format.
            obj: "mol" or "bkg" to modify a chemical structure image or
                 a background image.
        Returns:
            image: the resized image.
        """
        n1 = self.random_choice(np.arange(0, 50))
        n2 = self.random_choice(np.arange(0, 50))
        n3 = self.random_choice(np.arange(0, 50))
        n4 = self.random_choice(np.arange(0, 50))
        if obj == "mol":
            image = cv2.copyMakeBorder(
                img, n1, n2, n3, n4, cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )
        elif obj == "bkg":
            image = cv2.copyMakeBorder(img, n1, n2, n3, n4, cv2.BORDER_REFLECT)

        image = cv2.resize(image, (256, 256))
        return image

    def affine(self, img, obj=None) -> np.array:
        """
        This function randomly applies affine transformation which consists
        of matrix rotations, translations and scale operations and converts
        it back to (256,256).
        Args:
            img: the image to modify in array format.
            obj: "mol" or "bkg" to modify a chemical structure image or
                 a background image.
        Returns:
            skewed: the transformed image.
        """
        rows, cols, _ = img.shape
        n = 20
        pts1 = np.float32([[5, 50], [200, 50], [50, 200]])
        pts2 = np.float32(
            [
                [
                    5 + self.random_choice(np.arange(-n, n)),
                    50 + self.random_choice(np.arange(-n, n)),
                ],
                [
                    200 + self.random_choice(np.arange(-n, n)),
                    50 + self.random_choice(np.arange(-n, n)),
                ],
                [
                    50 + self.random_choice(np.arange(-n, n)),
                    200 + self.random_choice(np.arange(-n, n)),
                ],
            ]
        )

        M = cv2.getAffineTransform(pts1, pts2)

        if obj == "mol":
            skewed = cv2.warpAffine(img, M, (cols, rows), borderValue=[255, 255, 255])
        elif obj == "bkg":
            skewed = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)

        skewed = cv2.resize(skewed, (256, 256))
        return skewed

    def elastic_transform(self, image, alpha_sigma) -> np.array:
        """
        Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.
        https://gist.github.com/erniejunior/601cdf56d2b424757de5
        This function distords an image randomly changing the alpha and gamma
        values.
        Args:
            image: the image to modify in array format.
            alpha_sigma: alpha and sigma values randomly selected as a list.
        Returns:
            distored_image: the image after the transformation with the same size
                            as it had originally.
        """
        alpha = alpha_sigma[0]
        sigma = alpha_sigma[1]
        random_state = np.random.RandomState(self.random_choice(np.arange(1, 1000)))

        shape = image.shape
        dx = (
            gaussian_filter(
                (random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0
            )
            * alpha
        )
        random_state = np.random.RandomState(self.random_choice(np.arange(1, 1000)))
        dy = (
            gaussian_filter(
                (random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0
            )
            * alpha
        )

        x, y, z = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2])
        )
        indices = (
            np.reshape(y + dy, (-1, 1)),
            np.reshape(x + dx, (-1, 1)),
            np.reshape(z, (-1, 1)),
        )

        distored_image = map_coordinates(
            image, indices, order=self.random_choice(np.arange(1, 5)), mode="reflect"
        )
        return distored_image.reshape(image.shape)

    def distort(self, img) -> np.array:
        """
        This function randomly selects a list with the shape [a, g] where
        a=alpha and g=gamma and passes them along with the input image
        to the elastic_transform function that will do the image distorsion.
        Args:
            img: the image to modify in array format.
        Returns:
            the output from elastic_transform function which is the image
            after the transformation with the same size as it had originally.
        """
        sigma_alpha = [
            (self.random_choice(np.arange(9, 11)), self.random_choice(np.arange(2, 4))),
            (self.random_choice(np.arange(80, 100)), 4),
            (self.random_choice(np.arange(150, 300)), 5),
            (
                self.random_choice(np.arange(800, 1200)),
                self.random_choice(np.arange(8, 10)),
            ),
            (
                self.random_choice(np.arange(1500, 2000)),
                self.random_choice(np.arange(10, 15)),
            ),
            (
                self.random_choice(np.arange(5000, 8000)),
                self.random_choice(np.arange(15, 25)),
            ),
            (
                self.random_choice(np.arange(10000, 15000)),
                self.random_choice(np.arange(20, 25)),
            ),
            (
                self.random_choice(np.arange(45000, 55000)),
                self.random_choice(np.arange(30, 35)),
            ),
        ]
        choice = self.random_choice(range(len(sigma_alpha)))
        sigma_alpha_chosen = sigma_alpha[choice]
        return self.elastic_transform(img, sigma_alpha_chosen)

    def degrade_img(self, img) -> np.array:
        """
        This function randomly degrades the input image by applying different
        degradation steps with different robabilities.
        Args:
            img: the image to modify in array format.
        Returns:
            img: the degraded image.
        """
        # s+p
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.1:
            img = self.s_and_p(img)

        # scale
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.5:
            img = self.scale(img)

        # brightness
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.7:
            img = self.brightness(img)

        # contrast
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.7:
            img = self.contrast(img)

        # sharpness
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.5:
            img = self.sharpness(img)

        return img

    def contrast(self, img) -> np.array:
        """
        This function randomly changes the input image contrast.
        Args:
            img: the image to modify in array format.
        Returns:
            img: the image with the contrast changes.
        """
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.8:  # increase contrast
            f = self.random_choice(np.arange(1, 2, 0.01))
        else:  # decrease contrast
            f = self.random_choice(np.arange(0.5, 1, 0.01))
        im_pil = Image.fromarray(img)
        enhancer = ImageEnhance.Contrast(im_pil)
        im = enhancer.enhance(f)
        img = np.asarray(im)
        return np.asarray(im)

    def brightness(self, img) -> np.array:
        """
        This function randomly changes the input image brightness.
        Args:
            img: the image to modify in array format.
        Returns:
            img: the image with the brightness changes.
        """
        f = self.random_choice(np.arange(0.4, 1.1, 0.01))
        im_pil = Image.fromarray(img)
        enhancer = ImageEnhance.Brightness(im_pil)
        im = enhancer.enhance(f)
        img = np.asarray(im)
        return np.asarray(im)

    def sharpness(self, img) -> np.array:
        """
        This function randomly changes the input image sharpness.
        Args:
            img: the image to modify in array format.
        Returns:
            img: the image with the sharpness changes.
        """
        if self.random_choice(np.arange(0, 1, 0.01)) < 0.5:  # increase sharpness
            f = self.random_choice(np.arange(0.1, 1, 0.01))
        else:  # decrease sharpness
            f = self.random_choice(np.arange(1, 10))
        im_pil = Image.fromarray(img)
        enhancer = ImageEnhance.Sharpness(im_pil)
        im = enhancer.enhance(f)
        img = np.asarray(im)
        return np.asarray(im)

    def s_and_p(self, img) -> np.array:
        """
        This function randomly adds salt and pepper to the input image.
        Args:
            img: the image to modify in array format.
        Returns:
            out: the image with the s&p changes.
        """
        amount = self.random_choice(np.arange(0.001, 0.01))
        # add some s&p
        s_vs_p = 0.5
        out = np.copy(img)
        # Salt mode
        num_salt = int(np.ceil(amount * img.size * s_vs_p))
        coords = []
        for i in img.shape:
            coordinates = []
            for _ in range(num_salt):
                coordinates.append(self.random_choice(np.arange(0, i - 1)))
            coords.append(np.array(coordinates))
        out[tuple(coords)] = 1
        # pepper
        num_pepper = int(np.ceil(amount * img.size * (1.0 - s_vs_p)))
        coords = []
        for i in img.shape:
            coordinates = []
            for _ in range(num_pepper):
                coordinates.append(self.random_choice(np.arange(0, i - 1)))
            coords.append(np.array(coordinates))
        out[tuple(coords)] = 0
        return out

    def scale(self, img) -> np.array:
        """
        This function randomly scales the input image.
        Args:
            img: the image to modify in array format.
        Returns:
            res: the scaled image.
        """
        f = self.random_choice(np.arange(0.5, 1.5, 0.01))
        res = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        res = cv2.resize(
            res, None, fx=1.0 / f, fy=1.0 / f, interpolation=cv2.INTER_CUBIC
        )
        return res
