import random
from typing import List, Dict, Tuple 
import numpy as np
from RanDepict import RandomDepictor

class DepictionFeatures:
    """
    A DepictionFeatures objects simply holds all depiction parameters
    of a chemical structure depiction generated with RanDepict
    """
    def __init__(self):
        # Depiction parameters for Indigo depictions
        self.indigo_bond_line_width = None
        self.indigo_relative_thickness = None
        self.indigo_labels_all = None
        self.indigo_labels_hetero = None
        self.indigo_render_bold_bond = None
        self.indigo_stereo_label_style = None
        self.indigo_collapse_superatoms = None
        self.indigo_not_kekulized = None
        # Depiction parameters for RDKit depictions
        self.rdkit_add_stereo_annotation = None
        self.rdkit_add_chiral_flag_labels = None
        self.rdkit_add_atom_indices = None
        self.rdkit_bond_line_width = None
        self.rdkit_draw_terminal_methyl = None
        self.rdkit_label_font = None
        self.rdkit_min_font_size = None
        self.rdkit_molecule_rotation = None
        self.rdkit_fixed_bond_length = None
        self.rdkit_comic_style = None
        self.rdkit_collapse_superatoms = None
        # Depiction parameters for CDK depictions
        self.cdk_kekulized = None
        self.cdk_molecule_rotation = None
        self.cdk_atom_label_font_size = None
        self.cdk_atom_label_font = None
        self.cdk_atom_label_font_style = None
        self.cdk_show_all_atom_labels = None
        self.cdk_no_terminal_methyl = None
        self.cdk_stroke_width = None
        self.cdk_margin_ratio = None
        self.cdk_double_bond_dist = None
        self.cdk_wedge_ratio = None
        self.cdk_fancy_bold_wedges = None
        self.cdk_fancy_hashed_wedges = None
        self.cdk_hash_spacing = None
        self.cdk_add_CIP_labels = None
        self.cdk_add_atom_indices = None
        self.cdk_label_font_scale = None
        self.cdk_annotation_distance = None
        self.cdk_collapse_superatoms = None
        # Everything related to curved arrows
        self.has_curved_arrows = None
        self.curved_arrow_image_type = []
        self.curved_arrow_shape_x = []
        self.curved_arrow_shape_y = []
        self.curved_arrow_rot_angles = []
        self.curved_arrow_rot_resampling_methods = []
        self.curved_arrow_paste_pos_x = []
        self.curved_arrow_paste_pos_y = []
        # Everything related to straight_arrows
        self.has_straight_arrows = None
        self.straight_arrow_images = []
        self.straight_arrow_rot_angles = []
        self.straight_arrow_rot_resampling_methods = []
        self.straight_arrow_paste_pos_x = [] 
        self.straight_arrow_paste_pos_y = []
        # Everything related to chemical labels
        self.label_types = []
        self.label_texts = [] # not used for fingerprint
        self.label_font_types = []
        self.label_font_sizes = []
        self.label_paste_x_positions = []
        self.label_paste_y_position = []
        
        # Everything related to imgaug_augmentations
        self.imgaug_rotation_angle = None
        self.imgaug_coarse_dropout_p = None
        self.imgaug_coarse_dropout_size_percent = None
        self.imgaug_replace_elementwise_p = None
        self.imgaug_shear_param = None
        self.imgaug_imgcorrupt_severity = None
        self.imgaug_brightness_adj = None
        self.imgaug_colour_temp_change = None


class DepictionFeatureRanges(RandomDepictor):
    """Class for depiction feature fingerprint generation"""
    def __init__(self):
        super().__init__()
        # Fill ranges. By simply using all the depiction and augmentation
        # functions, the available features are saved by the overwritten random_choice
        # function. We just have to make sure to run through every available 
        # decision once to get all the information about the feature space that we need.
        smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        # Call every depiction function
        depiction = self(smiles)
        depiction = self.depict_and_resize_cdk(smiles)
        depiction = self.depict_and_resize_rdkit(smiles)
        depiction = self.depict_and_resize_indigo(smiles)
        # Call every augmentation function
        depiction = self.add_curved_arrows_to_structure(depiction)
        depiction = self.imgaug_augment(depiction, call_all=True)
        depiction = self.add_straight_arrows_to_structure(depiction)
        depiction = self.add_chemical_label(depiction, "ID")
        depiction = self.add_chemical_label(depiction, "R_GROUP")
        depiction = self.depiction = self.add_chemical_label(depiction, "REACTION")
    
    
    def random_choice(self, iterable: List, log_attribute: str = False):
        """
        In RandomDepictor, this function  would take an iterable, call random_choice() on it,
        increase random.seed by 1 and return the result.
        ___
        Here, this function is overwritten, so that it also sets the class attribute
        $log_attribute_range to contain the iterable.
        This way, a DepictionFeatureRanges object can easily be filled with all the iterables 
        that define the complete depiction feature space.
        ___
        """
        if log_attribute:
            setattr(self, '{}_range'.format(log_attribute), iterable)
        self.seed += 1
        random.seed(self.seed)
        result = random.choice(iterable)
        # Add result(s) to augmentation_logger
        if log_attribute and self.depiction_features:
            found_logged_attribute = getattr(self.augmentation_logger, log_attribute)
            # If the attribute is not saved in a list, simply write it, otherwise append it
            if type(found_logged_attribute) != list:
                setattr(self.depiction_features, log_attribute, result)
            else:
                setattr(self.augmentation_logger, log_attribute, found_logged_attribute + [result])
        return result
    
    
    def generate_fingerprint_schemes(self) -> List[Dict]:
        """
        Generates fingerprint schemes (see generate_fingerprint_scheme()) for the depictions
        with CDK, RDKit and Indigo as well as the augmentations.
        ___
        --> Returns [cdk_scheme: Dict, rdkit_scheme: Dict, indigo_scheme: Dict, augmentation_scheme: Dict]
        """
        fingerprint_schemes = []
        range_IDs = [att for att in dir(self) 
                  if 'range' in att]
        # Generate fingerprint scheme for our cdk, indigo and rdkit depictions
        depiction_toolkits = ['cdk', 'rdkit', 'indigo', '']
        for toolkit in depiction_toolkits:
            toolkit_range_IDs = [att for att in range_IDs if toolkit in att]
            # Delete toolkit-specific ranges 
            # (The last time this loop runs, only augmentation-related ranges are left)
            for ID in toolkit_range_IDs:
                range_IDs.remove(ID)
            toolkit_range_dict = {attr[:-6]: list(set(getattr(self, attr))) for attr in toolkit_range_IDs}
            fingerprint_scheme = self.generate_fingerprint_scheme(toolkit_range_dict)
            fingerprint_schemes.append(fingerprint_scheme)
        return fingerprint_schemes
        
            
    def generate_fingerprint_scheme(self, ID_range_map: Dict) -> Dict:
        """
        This function takes the ID_range_map and returns a dictionary that defines
        where each feature is represented in the depiction feature fingerprint.
        ___
        Example:
        >> example_ID_range_map = {'thickness': [0, 1, 2, 3], 'kekulized': [True, False]}
        >> DepictionFeatures().generate_fingerprint_scheme(example_ID_range_map)
        >>>> {'thickness': [{'position': 0, 'one_if': 0}, {'position': 1, 'one_if': 1}, 
            {'position': 2, 'one_if': 2}, {'position': 3, 'one_if': 3}], 
            'kekulized': [{'position': 4, 'one_if': True}]}
        Args:
            ID_range_map (Dict): dictionary that maps an ID (str) of a feature range
                                to the feature range itself (some kind of iterable)

        Returns:
            Dict: [description]
        """
        
        fingerprint_scheme = {}
        position = 0
        for feature_ID in ID_range_map.keys():
            feature_range = ID_range_map[feature_ID]
            # Make sure numeric ranges don't take up more than 5 positions in the fingerprint
            if type(feature_range[0]) in [int, float, np.float64, np.float32] and len(feature_range) > 5:
                subranges = self.split_into_n_sublists(feature_range, n=5)
                position_dicts = []
                for subrange in subranges:
                    subrange_minmax = (min(subrange), max(subrange))
                    position_dict = {"position": position,
                                        "one_if": subrange_minmax}
                    position_dicts.append(position_dict)
                    position += 1
                fingerprint_scheme[feature_ID] = position_dicts                        
            # Bools take up only one position in the fingerprint
            elif type(feature_range[0]) == bool:
                assert len(feature_range) == 2
                position_dicts = [{"position": position,
                                    "one_if": True}]
                position += 1
                fingerprint_scheme[feature_ID] = position_dicts
            else:
                # For other types of categorical data: Each category gets one position in the FP
                position_dicts = []
                for feature in feature_range:
                    position_dict = {"position": position,
                                        "one_if": feature}
                    position_dicts.append(position_dict)
                    position += 1
                fingerprint_scheme[feature_ID] = position_dicts
        return fingerprint_scheme
    
    
    def split_into_n_sublists(self, iterable, n: int):
        """
        Takes an iterable, sorts it, splits it evenly into n lists
        and returns the split lists.

        Args:
            iterable ([type]): Iterable that is supposed to be split
            n (int): Amount of sublists to return
        """
        iterable = sorted(iterable)
        iter_len = len(iterable)
        sublists = []
        for i in range(0, iter_len, int(np.ceil(iter_len/n))):
            sublists.append(iterable[i:i + int(np.ceil(iter_len/n))])
        return sublists

    
    def count_combinations(self, feature_ranges: List):
        """
        Takes a list of lists and returns the number of possible combinations
        of elements.
        Example: 
        Input: [[1, 2][3, 4]]
        Combinations: (1,3), (1,4), (2,3), (2,4) 
        Output: 4
        
        Args:
            feature_ranges (List): List of lists of features to choose from when depicting a molecule

        Returns:
            [int]: Number of possible feature combinations
        """
        possible_combination_count = 1
        for feature_range in feature_ranges:
            possible_combination_count *= len(set(feature_range))
        return possible_combination_count
    
    
    
    
    
# # Ranges that the parameters above are picked from
# self.indigo_bond_line_width_range = None
# self.indigo_relative_thickness_range = None
# self.indigo_labels_hetero_range = None
# self.indigo_render_bold_bond_range = None
# self.indigo_collapse_superatoms_range = None
# self.indigo_not_kekulized_range = None
# # Depiction parameters for RDKit depictions
# self.rdkit_bond_line_width_range = None
# self.rdkit_draw_terminal_methyl_range = None
# self.rdkit_label_font_range = None
# self.rdkit_min_font_size_range = None
# self.rdkit_molecule_rotation_range = None
# self.rdkit_fixed_bond_length_range = None
# self.rdkit_only_black_and_white_range = None
# self.rdkit_collapse_superatoms_range = None
# self.rdkit_default_abbreviations_range = None
# # Depiction parameters for CDK depictions
# self.cdk_kekulized_range = None
# self.cdk_molecule_rotation_range = None
# self.cdk_label_font_size_range = None
# self.cdk_label_font_range = None
# self.cdk_label_font_style_range = None
# self.cdk_no_terminal_methyl_range = None
# self.cdk_stroke_width_range = None
# self.cdk_margin_ratio_range = None
# self.cdk_double_bond_dist_range = None
# self.cdk_wedge_ratio_range = None
# self.cdk_fancy_bold_wedges_range = None
# self.cdk_fancy_hashed_wedges_range = None
# self.cdk_collapse_superatoms_range = None
# self.cdk_default_abbreviations_range = None
# # Everything related to curved arrows
# self.has_curved_arrows_range = None
# self.curved_arrow_images_range = None
# self.curved_arrow_shape_x_range = None
# self.curved_arrow_shape_y_range = None
# self.curved_arrow_rot_angles_range = None
# self.curved_arrow_rot_resampling_methods_range = None
# self.curved_arrow_paste_pos_x_range = None
# self.curved_arrow_paste_pos_y_range = None
# # Everything related to straight_arrows
# self.has_straight_arrows_range = None
# self.straight_arrow_images_range = None
# self.straight_arrow_rot_angles_range = None
# self.straight_arrow_rot_resampling_methods_range = None
# self.straight_arrow_paste_pos_x_range = None
# self.straight_arrow_paste_pos_y_range = None
# # Everything related to chemical labels
# self.has_chemical_label_range = None
# self.label_types_range = None
# self.label_texts_range = None
# self.label_font_types_range = None
# self.label_font_sizes_range = None
# # Everything related to atom numbers and chirality labels
# self.has_atom_numbers_chiral_labels_range = None
# self.has_chiral_labels_range = None
# self.atom_number_chiral_label_font_size_range = None
# self.atom_numbering_chiral_label_font_type_range = None
# self.atom_number_chiral_label_x_pos_range = None
# self.atom_number_chiral_label_y_pos_range = None
# self.atom_numbering_chiral_label_text_range = None

# # Everything related to imgaug_augmentations
# self.imgaug_brightness_adj_range = None
# self.imgaug_colour_temp_change_range = None
        