class depiction_features:
    """
    A depiction_features objects simply holds all depiction parameters
    of a chemical structure depiction generated with RanDepict
    """
    def __init__(self):
        # Depiction parameters for Indigo depictions
        self.indigo_bond_line_width = None
        self.indigo_relative_thickness = None
        self.indigo_labels_hetero = None
        self.indigo_render_bold_bond = None
        self.indigo_collapse_superatoms = None
        self.indigo_not_kekulized = None
        # Depiction parameters for RDKit depictions
        self.rdkit_bond_line_width = None
        self.rdkit_draw_terminal_methyl = None
        self.rdkit_label_font = None
        self.rdkit_min_font_size = None
        self.rdkit_molecule_rotation = None
        self.rdkit_fixed_bond_length = None
        self.rdkit_only_black_and_white = None
        self.rdkit_collapse_superatoms = None
        self.rdkit_default_abbreviations = None
        # Depiction parameters for CDK depictions
        self.cdk_kekulized = None
        self.cdk_molecule_rotation = None
        self.cdk_label_font_size = None
        self.cdk_label_font = None
        self.cdk_label_font_style = None
        self.cdk_no_terminal_methyl = None
        self.cdk_stroke_width = None
        self.cdk_margin_ratio = None
        self.cdk_double_bond_dist = None
        self.cdk_wedge_ratio = None
        self.cdk_fancy_bold_wedges = None
        self.cdk_fancy_hashed_wedges = None
        self.cdk_collapse_superatoms = None
        self.cdk_default_abbreviations = None
        # Everything related to curved arrows
        self.has_curved_arrows = None
        self.curved_arrow_images = []
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
        self.has_chemical_label = None
        self.label_types = []
        self.label_texts = []
        self.label_font_types = []
        self.label_font_sizes = []
        # Everything related to atom numbers and chirality labels
        self.has_atom_numbers_chiral_labels = None
        self.has_chiral_labels = None
        self.atom_number_chiral_label_font_size = None
        self.atom_numbering_chiral_label_font_type = None
        self.atom_number_chiral_label_x_pos = []
        self.atom_number_chiral_label_y_pos = []
        self.atom_numbering_chiral_label_text = []
        # Everything related to imgaug_augmentations
        self.imgaug_brightness_adj = None
        self.imgaug_colour_temp_change = None
        
        
        
        
class depiction_feature_ranges:
    def __init__(self):       
        pass
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
        