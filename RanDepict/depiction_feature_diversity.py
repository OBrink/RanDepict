import os
import random
from typing import List, Dict, Tuple
import numpy as np
from itertools import product
from RanDepict import RandomDepictor

from rdkit import DataStructs
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

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
        depiction = self.imgaug_augment(depiction)
        depiction = self.add_straight_arrows_to_structure(depiction)
        depiction = self.add_chemical_label(depiction, "ID")
        depiction = self.add_chemical_label(depiction, "R_GROUP")
        depiction = self.depiction = self.add_chemical_label(depiction, "REACTION")
        # Generate schemes for Fingerprint creation
        self.schemes = self.generate_fingerprint_schemes()
        self.CDK_scheme, self.RDKit_scheme, self.Indigo_scheme, self.augmentation_scheme = self.schemes
        # Generate the pool of all valid fingerprint combinations
        self.generate_all_possible_fingerprints()
    
    
    def random_choice(
        self, 
        iterable: List, 
        log_attribute: str = False
        ):
        """
        In RandomDepictor, this function  would take an iterable, call random_choice() on it,
        increase the seed attribute by 1 and return the result.
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
    
    
    def generate_fingerprint_schemes(
        self
        ) -> List[Dict]:
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
        
            
    def generate_fingerprint_scheme(
        self, 
        ID_range_map: Dict
        ) -> Dict:
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
            Dict: Mapping of feature ID (str) and dictionaries that define the fingerprint position and a condition
        """
        
        fingerprint_scheme = {}
        position = 0
        for feature_ID in ID_range_map.keys():
            feature_range = ID_range_map[feature_ID]
            # Make sure numeric ranges don't take up more than 5 positions in the fingerprint
            if type(feature_range[0]) in [int, float, np.float64, np.float32] and len(feature_range) > 5:
                subranges = self.split_into_n_sublists(feature_range, n=3)
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
    
    
    def split_into_n_sublists(self, 
                              iterable, 
                              n: int
                              ) -> List[List]:
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

    
    def count_combinations(
        self, 
        feature_ranges: List
        ) -> int:
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
    
    
    def get_number_of_possible_fingerprints(
        self, 
        scheme: Dict
        ) -> int:
        """
        This function takes a fingerprint scheme (Dict) as returned by generate_fingerprint_scheme()
        and returns the number of possible fingerprints for that scheme.

        Args:
            scheme (Dict): Output of generate_fingerprint_scheme()

        Returns:
            int: Number of possible fingerprints
        """
        comb_count = 1
        for feature_key in scheme.keys():
            if len(scheme[feature_key]) != 1:
                # n fingerprint positions -> n options (because only one position can be [1])
                # n = 3 --> [1][0][0] or [0][1][0] or [0][0][1]
                comb_count *= len(scheme[feature_key])
            else:
                # One fingerprint position -> two options: [0] or [1]
                comb_count *= 2
        return comb_count
    
    
    def get_FP_building_blocks(
        self, 
        scheme: Dict
        ) -> List[List[List]]:
        """
        This function takes a fingerprint scheme (Dict) as returned by generate_fingerprint_scheme()
        and returns a list of possible building blocks.
        Example:
            scheme = {'thickness': [{'position': 0, 'one_if': 0}, {'position': 1, 'one_if': 1}, 
                {'position': 2, 'one_if': 2}, {'position': 3, 'one_if': 3}], 
                'kekulized': [{'position': 4, 'one_if': True}]}
                
            --> Output: [((1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1)), ((1), (0))]
        
        Args: 
            scheme (Dict): Output of generate_fingerprint_scheme()
        
        Returns:
            List that contains the valid fingerprint parts that represent the different features
        
        """
        FP_building_blocks = []
        for feature_key in scheme.keys():
            position_condition_dicts = scheme[feature_key]
            FP_building_blocks.append([])
            # Add every single valid option to the building block
            for position_index in range(len(position_condition_dicts)):
                # Add list of zeros
                FP_building_blocks[-1].append([0]*len(position_condition_dicts))
                # Replace one zero with a one
                FP_building_blocks[-1][-1][position_index] = 1
            # If a feature is described by only one position in the FP, make sure that 0 and 1 are listed options
            if FP_building_blocks[-1] == [[1]]:
                FP_building_blocks[-1].append([0])
        return FP_building_blocks
        
    def flatten_fingerprint(
        self, 
        unflattened_list: List[List],
        ) -> List:
        """
        This function takes a list of lists and returns a list.
        ___
        Looks like this could be one line elsewhere but this function used for parallelisation 
        of FP generation and consequently needs to be wrapped up in a separate function.
        

        Args:
            unflattened_list (List[List[X,Y,Z]])

        Returns:
            flattened_list (List[X,Y,Z]): 
        """
        flattened_list = [element for sublist in unflattened_list for element in sublist]
        return flattened_list
    
    
    def generate_all_possible_fingerprints_per_scheme(
        self, 
        scheme: Dict,
        ) -> List[List[int]]:
        """
        This function takes a fingerprint scheme (Dict) as returned by generate_fingerprint_scheme()
        and returns a List of all possible fingerprints for that scheme.

        Args:
            scheme (Dict): Output of generate_fingerprint_scheme()
            name (str): name that is used for filename of saved FPs
            
        Returns:
            List of fingerprints
        """        
        # Determine valid building blocks for fingerprints
        FP_building_blocks = self.get_FP_building_blocks(scheme)
        # Determine cartesian product of valid building blocks to get all valid fingerprints
        FP_generator = product(*FP_building_blocks)
        flattened_fingerprints = list(map(self.flatten_fingerprint, FP_generator))
        return flattened_fingerprints
    
    
    def generate_all_possible_fingerprints(self) -> None:
        """
        This function generates all possible valid fingerprint combinations for the four
        available fingerprint schemes if they have not been created already. Otherwise, 
        they are loaded from files.
        This function returns None but saves the fingerprint pools as a class attribute $ID_fingerprints 
        """
        exists_already = False
        FP_names = ['CDK', 'RDKit', 'Indigo', 'augmentation']
        for scheme_index in range(len(self.schemes)):
            n_FP = self.get_number_of_possible_fingerprints(self.schemes[scheme_index])
            #print('There are {} {} fingerprints.'.format(n_FP, FP_names[scheme_index]))
            # Load fingerprint pool from file (if it exists)
            FP_filename = '{}_fingerprints.npz'.format(FP_names[scheme_index])
            FP_file_path = self.HERE.joinpath(FP_filename)
            if os.path.exists(FP_file_path):
                fingerprints = np.load(FP_filename)['arr_0']
                if len(fingerprints) == n_FP:
                    exists_already = True
            # Otherwise, generate the fingerprint pool
            if not exists_already:
                print('No pre-computed fingerprints found. The generation may take a minute.')
                fingerprints = self.generate_all_possible_fingerprints_per_scheme(self.schemes[scheme_index])
                np.savez_compressed(FP_file_path, fingerprints)
                print('{} fingerprints were saved in {}.'.format(FP_names[scheme_index], FP_file_path))    
            setattr(self, "{}_fingerprints".format(FP_names[scheme_index]), fingerprints)
        return
    
    
    def convert_to_int_arr(
        self,
        fingerprints: List[List[int]]
        ) -> List[DataStructs.cDataStructs.ExplicitBitVect]:
        """
        Takes a list of fingerprints (List[int]) and returns them as a list of
        rdkit.DataStructs.cDataStructs.ExplicitBitVect so that they can be 
        processed by RDKit's MaxMinPicker.

        Args:
            fingerprints (List[List[int]]): List of fingerprints

        Returns:
            List[rdkit.DataStructs.cDataStructs.ExplicitBitVect]: Converted arrays
        """
        converted_fingerprints = []
        for fp in fingerprints:
            bitstring = "".join(np.array(fp).astype(str))
            fp_converted = DataStructs.cDataStructs.CreateFromBitString(bitstring)
            converted_fingerprints.append(fp_converted)
        return converted_fingerprints


    def pick_fingerprints(
        self,
        fingerprints: List[List[int]],
        n: int,
        ) -> List[np.array]:
        """
        Given a list of fingerprints and a number n of fingerprints to pick, this function uses 
        RDKit's MaxMin Picker to pick n fingerprints and returns them.
        
        Args:
            fingerprints (List[List[int]]): List of fingerprints
            n (int): Number of fingerprints to pick
            
        Returns:
            Picked fingerprints (List[List[int]])
        """
        
        converted_fingerprints = self.convert_to_int_arr(fingerprints)
        
        """TODO: I don't like this function definition in the function but according to the 
        RDKit Documentation, the fingerprints need to be given in the distance function as the
        default value."""
        
        def dice_dist(
            fp_index_1: int, 
            fp_index_2: int, 
            fingerprints: List[DataStructs.cDataStructs.ExplicitBitVect] = converted_fingerprints
            ) -> float:
            """
            Returns the dice similarity between two fingerprints.
            Args:
                fp_index_1 (int): index of first fingerprint in fingerprints
                fp_index_2 (int): index of second fingerprint in fingerprints
                fingerprints (List[DataStructs.cDataStructs.ExplicitBitVect]): List of fingerprints

            Returns:
                float: Dice similarity between the two fingerprints
            """
            return 1-DataStructs.DiceSimilarity(fingerprints[fp_index_1], fingerprints[fp_index_2])
        
        n_fingerprints = len(fingerprints)
        picker = MaxMinPicker()
        pick_indices = picker.LazyPick(dice_dist, n_fingerprints, n, seed=42)
        picked_fingerprints = [fingerprints[i]
                            for i in pick_indices]
        return picked_fingerprints  


    def generate_fingerprints_for_dataset(
        self,
        size: int,
        indigo_proportion: float = 0.15,
        rdkit_proportion: float = 0.3,
        cdk_proportion: float = 0.55,
        aug_proportion: float = 0.5,
        ) -> Tuple[List[int]]:
        """Given a dataset size (int) and (optional) proportions for the different types of fingerprints

        Args:
            size (int): Desired dataset size, number of returned Indigo/RDKit/CDK fingerprints
            indigo_proportion (float, optional): Proportion of Indigo fingerprints. Defaults to 0.15.
            rdkit_proportion (float, optional): Proportion of RDKit fingerprints. Defaults to 0.3.
            cdk_proportion (float, optional): Proportion of CDK fingerprints. Defaults to 0.55.
            aug_proportion (float, optional): Proportion of Augmentation fingerprints. Defaults to 0.5.

        Raises:
            ValueError: 
                - If the sum of Indigo, RDKit and CDK proportions is not 1
                - If the augmentation proportion is > 1
        
        Returns:
            Tuple[List[int]]: Tuple of lists of indigo, rdkit, cdk and augmentation fingerprints

        """
        # Make sure that the given proportion arguments make sense
        if sum(indigo_proportion, rdkit_proportion, cdk_proportion) != 1:
            raise ValueError("Sum of Indigo, CDK and RDKitproportion arguments needs to be 1.")
        if aug_proportion > 1:
            raise ValueError("The proportion of augmentation fingerprints can't be > 1.")
        # Pick and return diverse fingerprints
        picked_Indigo_fingerprints = self.pick_fingerprints(self.Indigo_fingerprints, int(size*indigo_proportion))
        picked_RDKit_fingerprints = self.pick_fingerprints(self.RDKit_fingerprints, int(size*rdkit_proportion))
        picked_CDK_fingerprints = self.pick_fingerprints(self.CDK_fingerprints, int(size*cdk_proportion))
        picked_augmentation_fingerprints = self.pick_fingerprints(self.augmentation_fingerprints, int(size*aug_proportion)) 
        return (picked_Indigo_fingerprints,
                picked_RDKit_fingerprints,
                picked_CDK_fingerprints,
                picked_augmentation_fingerprints)

class DepictionFeatures:
    """
    A DepictionFeatures objects simply holds all depiction parameters
    of a chemical structure depiction generated with RanDepict
    """
    def __init__(self):
        pass
