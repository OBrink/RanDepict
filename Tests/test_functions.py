from RanDepict import RandomDepictor, DepictionFeatureRanges, RandomMarkushStructureCreator
from rdkit import DataStructs
import re
import numpy as np
from omegaconf import OmegaConf 

import pytest

from RanDepict import RandomDepictorConfig

class TestDepictionFeatureRanges:
    DFR = DepictionFeatureRanges()

    def test_random_choice(self):
        chosen_element = self.DFR.random_choice(
            range(4), log_attribute="thickness")
        # Assert that random_choice picks an element from the given iterable
        assert chosen_element in range(4)
        # Assert that random_choice saves the range of values it picks from
        assert getattr(self.DFR, "thickness_range") == range(4)

    def test_generate_fingerprint_schemes(self):
        # This cannot really be tested so we simply assert that it generates
        # every scheme when running generate_fingerprint_schemes()
        fingerprint_schemes = self.DFR.generate_fingerprint_schemes()
        assert len(fingerprint_schemes) == 5
        cdk_scheme, rdkit_scheme, indigo_scheme, _, _ = fingerprint_schemes
        assert "indigo" in list(indigo_scheme.keys())[0]
        assert "rdkit" in list(rdkit_scheme.keys())[0]
        assert "cdk" in list(cdk_scheme.keys())[0]

    def test_generate_fingerprint_scheme(self):
        # Assert that generate_fingerprint_scheme generates takes a map
        # of keywords and ranges and returns a map of keyword and every
        # fingerprint position to a condition
        example_ID_range_map = {"thickness": [
            0, 1, 2, 3], "kekulized": [True, False]}
        expected_result = {
            "thickness": [
                {"position": 0, "one_if": 0},
                {"position": 1, "one_if": 1},
                {"position": 2, "one_if": 2},
                {"position": 3, "one_if": 3},
            ],
            "kekulized": [{"position": 4, "one_if": True}],
        }
        result = self.DFR.generate_fingerprint_scheme(example_ID_range_map)
        assert result == expected_result

    def test_get_FP_building_blocks(self):
        example_scheme = {
            "thickness": [
                {"position": 0, "one_if": 0},
                {"position": 1, "one_if": 1},
                {"position": 2, "one_if": 2},
                {"position": 3, "one_if": 3},
            ],
            "kekulized": [{"position": 4, "one_if": True}],
        }
        # For the key 'thickness', there are 4 valid building blocks:
        # [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]
        # For the key 'kekulized', there are 2 valid building blocks:
        # [0] and [1]
        # The expected output is a list of lists of the building blocks
        expected_building_blocks = [
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            [[1], [0]],
        ]
        actual_building_blocks = self.DFR.get_FP_building_blocks(
            example_scheme)
        assert actual_building_blocks == expected_building_blocks

    def test_split_into_n_sublists(self):
        # Assert that an iterable is split into even lists
        actual_result = self.DFR.split_into_n_sublists(range(10), 2)
        expected_result = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        assert actual_result == expected_result
        # The same for an even split if the length of the iterable does
        # not allow a clean split
        actual_result = self.DFR.split_into_n_sublists(range(9), 2)
        expected_result = [[0, 1, 2, 3, 4], [5, 6, 7, 8]]
        assert actual_result == expected_result

    def test_get_number_of_possible_fingerprints(self):
        example_scheme = {
            "thickness": [
                {"position": 0, "one_if": 0},
                {"position": 1, "one_if": 1},
                {"position": 2, "one_if": 2},
                {"position": 3, "one_if": 3},
            ],
            "kekulized": [{"position": 4, "one_if": True}],
        }
        # For the key 'thickness', there are 4 valid building blocks:
        # [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]
        # For the key 'kekulized', there are 2 valid building blocks:
        # [0] and [1]
        # --> 8 valid fingerprint combinations
        expected_number = 8
        actual_number = self.DFR.get_number_of_possible_fingerprints(
            example_scheme)
        assert expected_number == actual_number

    def test_flatten_fingerprint(self):
        # itertools.products applied to the building blocks gives us a
        # generator for all valid fingerprints
        # But they are not flattened.
        example_input = [[1, 0, 0, 0], [1]]
        expected_output = [1, 0, 0, 0, 1]
        actual_output = self.DFR.flatten_fingerprint(example_input)
        assert actual_output == expected_output

    def test_generate_all_possible_fingerprints_per_scheme(self):
        example_scheme = {
            "thickness": [
                {"position": 0, "one_if": 0},
                {"position": 1, "one_if": 1},
                {"position": 2, "one_if": 2},
                {"position": 3, "one_if": 3},
            ],
            "kekulized": [{"position": 4, "one_if": True}],
        }
        # For the key 'thickness', there are 4 valid building blocks:
        # [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]
        # For the key 'kekulized', there are 2 valid building blocks:
        # [0] and [1]
        # Based on that, we expect 8 fingerprints
        expected_output = [
            [1, 0, 0, 0, 1],
            [0, 1, 0, 0, 1],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
        ]
        actual_output = self.DFR.generate_all_possible_fingerprints_per_scheme(
            example_scheme
        )
        # The order is irrelevant here --> both lists are sorted
        assert sorted(actual_output) == sorted(expected_output)

    def test_generate_all_possible_fingerprints(self):
        # Not really much that can be tested here in a meaningful way
        # Assert that fingerprints are generated and saved as attributes of DFR
        self.DFR.CDK_fingerprints = False
        self.DFR.RDKit_fingerprints = False
        self.DFR.Indigo_fingerprints = False
        self.DFR.augmentation_fingerprints = False
        # Bit weird to assert the type but np.arrays raise a ValueError when
        # "assert array" is run
        self.DFR.generate_all_possible_fingerprints()
        assert type(self.DFR.CDK_fingerprints) != bool
        assert type(self.DFR.RDKit_fingerprints) != bool
        assert type(self.DFR.Indigo_fingerprints) != bool
        assert type(self.DFR.augmentation_fingerprints) != bool

    def test_convert_to_int_arr(self):
        example_input = [[1, 0], [0, 1]]
        output = self.DFR.convert_to_int_arr(example_input)
        for arr_index in range(len(output)):
            # Assert that a list of DataStructs.cDataStructs.ExplicitBitVect
            # is returned
            assert type(output[arr_index]
                        ) == DataStructs.cDataStructs.ExplicitBitVect
            # Assert that all values are the same
            for value_index in range(len(output[arr_index])):
                assert (
                    output[arr_index][value_index]
                    == example_input[arr_index][value_index]
                )

    def test_pick_fingerprints_small_number(self):
        # Assert that a diverse subset is picked when less than the
        # available amount of fingerprints is picked
        example_pool = np.array(
            [[1, 0, 0],
             [1, 0, 0],
             [1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]])
        number = 3
        expected_subset = np.array(
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]])
        actual_subset = self.DFR.pick_fingerprints(example_pool, number)

        assert np.array_equal(actual_subset, expected_subset)

    def test_pick_fingerprints_big_number(self):
        # Assert that a diverse subset is picked when more than the
        # available amount of fingerprints is picked
        example_pool = np.array(
            [[1, 0, 0],
             [1, 0, 0],
             [1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]])
        number = 8
        expected_subset = np.array(
            [[1, 0, 0],
             [1, 0, 0],
             [1, 0, 0],
             [0, 1, 0],
             [0, 0, 1],
             [1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]])
        actual_subset = self.DFR.pick_fingerprints(example_pool, number)
        assert np.array_equal(actual_subset, expected_subset)

    def test_pick_fingerprints_big_number_indigo(self):
        # Assert that picking given amount of fingerprints from an actual
        # fingerprint pool works when it is bigger than the pool itself
        example_pool = self.DFR.Indigo_fingerprints
        number = 100000
        actual_subset = self.DFR.pick_fingerprints(example_pool, number)
        assert len(actual_subset) == number

    @pytest.mark.xfail(raises=ValueError)
    def test_generate_fingerprints_for_dataset_valueerror(self):
        """ValueError is the sum of Indigo, RDKit, PIKAChU and CDK proportions is not 1"""
        self.DFR.generate_fingerprints_for_dataset(
            size=100,
            indigo_proportion=0.15,
            rdkit_proportion=0.3,
            pikachu_proportion=0.25,
            cdk_proportion=0.55,
            aug_proportion=0.5,
        )

    def test_generate_fingerprints_for_dataset(self):
        # Difficult to test this
        # --> Assert that the total number of fingerprints fits
        # --> Assert that number of depiction fingerprints fits
        fingerprints = self.DFR.generate_fingerprints_for_dataset(
            size=100,
            indigo_proportion=0.15,
            rdkit_proportion=0.3,
            # pikachu_proportion=0.25, # default value
            cdk_proportion=0.3,
            aug_proportion=0.5,
        )

        assert len(fingerprints) == 100
        num_aug_fps = len([fp for fp in fingerprints
                           if len(fp) != 1])
        # 49 is the expected outcome here as 50 fingerprints cannot be evenly
        # distributed while keeping the proportions. The function rounds the
        # indices. Nothing to worry about.
        assert num_aug_fps == 49

    def test_distribute_elements_evenly(self):
        # Assert that the elements from elements_to_be_distributed are
        # distributed evenly over the other given elements
        elements_to_be_distributed = ["A", "B", "C", "D"]
        fixed_elements_1 = [1, 2, 3]
        fixed_elements_2 = [4, 5, 6]
        expected_output = [[1, "A"], [2, "B"], [3], [4, "C"], [5, "D"], [6]]
        actual_output = self.DFR.distribute_elements_evenly(
            elements_to_be_distributed, fixed_elements_1, fixed_elements_2
        )
        assert actual_output == expected_output

    def test_correct_amount_of_FP_to_pick(self):
        # Test that the amount of fingerprints is adapted when more fingerprints
        # are supposed to be picked than there are fingerprints
        test_n = 23
        test_fingerprints = ["A"] * 5
        picked_fingerprints , corrected_n = self.DFR.correct_amount_of_FP_to_pick(
            test_fingerprints,
            test_n
        )
        assert list(picked_fingerprints) == ["A"] * 20
        assert corrected_n == 3

class TestRandomDepictorConstruction:

    def test_default_depicter(self):
        """RandomDepictor default values should match that of the default RandomDepictorConfig"""
        depicter =  RandomDepictor()
        config = RandomDepictorConfig()
        assert depicter.seed == config.seed
        assert depicter.hand_drawn == config.hand_drawn

    def test_init_param_override(self):
        """Values passed to init should override defaults"""
        config = RandomDepictorConfig()
        assert config.seed != 21
        assert not config.hand_drawn
        depicter =  RandomDepictor(seed=21, hand_drawn=True)
        assert depicter.seed == 21
        assert depicter.hand_drawn

    def test_config_override(self):
        """Config passed to init should override defaults"""
        config = RandomDepictorConfig(seed=21, hand_drawn=True)
        depicter =  RandomDepictor(config=config)
        assert depicter.seed == 21
        assert depicter.hand_drawn

    @pytest.mark.xfail(raises=ValueError) 
    def test_invalid_style(self):
        """Invalid style passed to config should raise exception"""
        _ = RandomDepictorConfig(styles=["pio", "cdk"])
    
    def test_empty_style_list(self):
        """Empty style list passed to config should raise exception"""
        with pytest.raises(ValueError) as excinfo:
            _ = RandomDepictorConfig(styles=[])
        assert 'Empty list' in str(excinfo.value)

    def test_omega_config_rdc(self):
        """Can create RandomDepictorConfig from yaml"""
        s = """
        # RandomDepictorConfig:
            seed: 21
            augment: False
            styles:
                - cdk
        """
        dict_config = OmegaConf.create(s)
        rdc = RandomDepictorConfig.from_config(dict_config)
        assert rdc.seed == 21
        assert not rdc.hand_drawn
        assert not rdc.augment
        assert len(rdc.styles) == 1
        assert 'cdk' in rdc.styles

    def test_omega_config_rd(self, tmp_path):
        """Can create RandomDepictor from yaml"""
        s = """
        RandomDepictorConfig:
            seed: 21
            augment: False
            styles:
                - cdk
                - indigo
        """
        temp_config_file = tmp_path / "omg.yaml"
        temp_config_file.write_text(s)
        rd = RandomDepictor.from_config(config_file=temp_config_file)
        assert rd.seed == 21
        assert not rd.hand_drawn


class TestRandomDepictor:
    depictor = RandomDepictor()

    def test_depict_and_resize_indigo(self):
        # Assert that an image is returned with different types
        # of input SMILES str
        test_smiles = ['c1ccccc1',
                       '[Otto]C1=C([XYZ123])C([R1])=C([Y])C([X1])=C1[R]',
                       'c1ccccc1[R1]']
        for smiles in test_smiles:
            im = self.depictor.indigo_depict(smiles)
            assert type(im) == np.ndarray

    def test_depict_and_resize_rdkit(self):
        # Assert that an image is returned with different types
        # of input SMILES str
        test_smiles = ['c1ccccc1',
                       '[Otto]C1=C([XYZ123])C([R1])=C([Y])C([X])=C1[R]']
        for smiles in test_smiles:
            im = self.depictor.rdkit_depict(smiles)
            assert type(im) == np.ndarray

    def test_cdk_depict(self):
        # Assert that an image is returned with different types
        # of input SMILES str
        test_smiles = ['c1ccccc1',
                       '[Otto]C1=C([XYZ123])C([R1])=C([Y])C([X])=C1[R]']
        for smiles in test_smiles:
            im = self.depictor.cdk_depict(smiles)
            assert type(im) == np.ndarray

    def test_depict_and_resize_pikachu(self):
        # Assert that an image is returned with different types
        # of input SMILES str
        test_smiles = ['c1ccccc1',
                       '[R1]C1=C([X23])C([R])=C([Z])C([X])=C1[R]']
        for smiles in test_smiles:
            im = self.depictor.pikachu_depict(smiles)
            assert type(im) == np.ndarray
            
    def test_random_depiction_with_coordinates(self):
        smiles = "CCC"
        with RandomDepictor() as depictor:
            for index in range(20):
                if index < 10:
                    depiction, cx_smiles = depictor.random_depiction_with_coordinates(smiles)
                    
                else:
                    depiction, cx_smiles = depictor.random_depiction_with_coordinates(smiles,
                                                                                      augment=True)
                assert type(depiction) == np.ndarray
                assert cx_smiles[:3] == smiles
                
    def test_get_depiction_functions_normal(self):
        # For a molecule without isotopes or R groups, all toolkits can be used
        observed = self.depictor.get_depiction_functions('c1ccccc1C(O)=O')
        expected = [
            self.depictor.rdkit_depict,
            self.depictor.indigo_depict,
            self.depictor.cdk_depict,
            self.depictor.pikachu_depict,
        ]
        # symmetric_difference
        difference = set(observed) ^ set(expected)
        assert not difference

    def test_get_depiction_functions_isotopes(self):
        # PIKAChU can't handle isotopes
        observed = self.depictor.get_depiction_functions("[13CH3]N1C=NC2=C1C(=O)N(C(=O)N2C)C")
        expected = [
            self.depictor.rdkit_depict,
            self.depictor.indigo_depict,
            self.depictor.cdk_depict,
        ]
        difference = set(observed) ^ set(expected)
        assert not difference

    def test_get_depiction_functions_R(self):
        # RDKit depicts "R" without indices as '*' (which is not desired)
        observed = self.depictor.get_depiction_functions("[R]N1C=NC2=C1C(=O)N(C(=O)N2C)C")
        expected = [
            self.depictor.indigo_depict,
            self.depictor.cdk_depict,
            self.depictor.pikachu_depict,
        ]
        difference = set(observed) ^ set(expected)
        assert not difference

    def test_get_depiction_functions_X(self):
        # RDKit and Indigo don't depict "X"
        observed = self.depictor.get_depiction_functions("[X]N1C=NC2=C1C(=O)N(C(=O)N2C)C")
        expected = [
            self.depictor.cdk_depict,
            self.depictor.pikachu_depict,
        ]
        difference = set(observed) ^ set(expected)
        assert not difference

    def test_smiles_to_mol_str(self):
        # Compare generated mol file str with reference string
        mol_str = self.depictor._smiles_to_mol_block("CC")
        mol_str_lines = mol_str.split('\n')
        with open('Tests/test.mol', 'r') as ref_mol_file:
            ref_lines = ref_mol_file.readlines()
        for line_index in range(len(ref_lines)):
            ref_line = ref_lines[line_index][:-1]
            test_line = mol_str_lines[line_index]
            # Ignore the line that contains the timestamp
            if "CDK" not in ref_line:
                assert ref_line == test_line

    def test_random_depiction(self):
        # Test random_depiction function and by doing this,
        # get_depiction_functions()
        smiles_list = [
            "[R]C1=C([R])C([R])=C([R])C([R])=C1[R]",
            "[*]C1=C([Y])C([R])=C([R])C([R])=C1[R]",
            "[R0]C1=C([R12])C([R1])=C([R3])C([R12])=C1[R]",
            "[X]C1=C([Y])C([Z])=C([R3])C([R12])=C1[R]",
            "[Otto]C1=C([Z])C([R1])=C([Y])C([X1])=C1[R]"
            "CC(=[O+]C)CC"
        ]
        for smiles in smiles_list:
            for _ in range(5):
                im = self.depictor.random_depiction(smiles)
                assert type(im) == np.ndarray

    def test_has_r_group(self):
        # Test samples SMILES
        assert self.depictor.has_r_group("[R]CC[Br]COC")
        assert self.depictor.has_r_group("c1ccccc1([X])")
        assert self.depictor.has_r_group("c1ccccc1([Y])")
        assert self.depictor.has_r_group("c1ccccc1([Z])")
        assert not self.depictor.has_r_group("[Cl]CC[Br]COC")


class TestRandomMarkushStructureCreator:
    """
    Unit tests for methods of RandomMarkushStructureCreator
    """
    depictor = RandomDepictor()
    markush_creator = RandomMarkushStructureCreator()

    def test_generate_markush_structure_dataset(self,):
        smiles_list = ["CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                       "C1CCCCC1N[C@](C)(F)C(=O)O",
                       "C1CCCCC1"]
        r_group_smiles_list = self.markush_creator.generate_markush_structure_dataset(smiles_list)
        for smiles in r_group_smiles_list:
            # Assert that every generated markush structure actually contains an R group
            assert self.depictor.has_r_group(smiles)
            # Assert that every generated markush structure can be depicted with RanDepict
            for _ in range(5):
                depiction = self.depictor.random_depiction(smiles)
                assert type(depiction) == np.ndarray

    def test_insert_R_group_var_contains_R(self):
        # Assert that an R group has been inserted
        input_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        output_smiles = self.markush_creator.insert_R_group_var(input_smiles, 3)
        assert self.depictor.has_r_group(output_smiles)
        
    def test_insert_R_group_var_can_be_depicted(self):
        input_smiles = "ClCN1C=NC2=C1C(=[O])N(C(=O)N2C)CCl"
        output_smiles = self.markush_creator.insert_R_group_var(input_smiles, 3)
        # Assert that the output SMILES str leads is depicted by RanDepict
        for _ in range(10):
            depiction = self.depictor.random_depiction(output_smiles)
            assert type(depiction) == np.ndarray

    def test_add_explicite_hydrogen_to_smiles(self):
        # Assert that hydrogen atoms are added
        input_smiles = "CCC"
        expected_output = "C([H])([H])([H])C([H])([H])C([H])([H])[H]"
        observed_output = self.markush_creator.add_explicite_hydrogen_to_smiles(input_smiles)
        assert expected_output == observed_output
        
    def test_remove_explicite_hydrogen_to_smiles(self):
        # Assert that hydrogen atoms are removed
        input_smiles = "C([H])([H])([H])C([H])([H])C([H])([H])[H]"
        expected_output = "CCC"
        observed_output = self.markush_creator.remove_explicite_hydrogen_from_smiles(input_smiles)
        assert expected_output == observed_output
    
    def test_get_valid_replacement_positions_simple_chain(self):
        # Simple example case
        observed = self.markush_creator.get_valid_replacement_positions("CCCCCC")
        expected = list(range(6))
        assert observed == expected
        
    def test_get_valid_replacement_positions_isotope(self):
        # Isotopes like 13C are not valid replacement positions (avoid eg. "13R")
        observed = self.markush_creator.get_valid_replacement_positions("CCC[13C]CC")
        expected = [0, 1, 2, 8, 9]
        assert observed == expected
    
    def test_get_valid_replacement_positions_charge(self):
        # charged atoms are not valid replacement positions (avoid eg. "[R+]")
        observed = self.markush_creator.get_valid_replacement_positions("CCC[C+]CC")
        expected = [0, 1, 2, 7, 8]
        assert observed == expected
    
    def test_get_valid_replacement_positions_with_hydrogen(self):
        # Simple example case
        observed = self.markush_creator.get_valid_replacement_positions("([H])([H])([H])CO([H])")
        expected = [2, 7, 12, 15, 19]
        assert observed == expected

    def test_get_valid_replacement_positions_ring(self):
        # Assert that ring syntax in SMILES remains intact
        observed = self.markush_creator.get_valid_replacement_positions("C1CCCCC1")
        expected = [2, 3, 4, 5]
        assert observed == expected

    def test_get_valid_replacement_positions_caffeine(self):
        # More complex example
        observed = self.markush_creator.get_valid_replacement_positions("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
        expected = [0, 3, 11, 18, 25, 27]
        assert observed == expected
        
    def test_get_valid_replacement_positions_alanine(self):
        # Example with stereo
        smiles = "C([H])([H])([H])[C@@]([H])(C(=O)O[H])N([H])[H]"
        observed = self.markush_creator.get_valid_replacement_positions(smiles)
        expected = [0, 3, 8, 13, 23, 27, 34, 40, 44]
        assert observed == expected
        
