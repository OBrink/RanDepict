from RanDepict import DepictionFeatureRanges
from rdkit import DataStructs


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
        assert len(fingerprint_schemes) == 4
        cdk_scheme, rdkit_scheme, indigo_scheme, _ = fingerprint_schemes
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

    def test_pick_fingerprints(self):
        # Assert that a diverse subset is picked
        example_input = [[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        expected_subset = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        actual_subset = self.DFR.pick_fingerprints(example_input, 3)
        assert actual_subset == expected_subset

    def test_generate_fingerprints_for_dataset(self):
        # Difficult to test this
        # --> Assert that the number of fingerprints fits the given proportions
        fingerprints = self.DFR.generate_fingerprints_for_dataset(
            size=100,
            indigo_proportion=0.15,
            rdkit_proportion=0.3,
            cdk_proportion=0.55,
            aug_proportion=0.5,
        )
        indigo_fps, rdkit_fps, cdk_fps, aug_fps = fingerprints
        assert len(indigo_fps) == 15
        assert len(rdkit_fps) == 30
        assert len(cdk_fps) == 55
        assert len(aug_fps) == 50

    def test_distribute_elements_evenly(self):
        # Assert that the elements from elements_to_be_distributed are
        # distributed evenly over the other given elements
        elements_to_be_distributed = ["A", "B", "C", "D"]
        fixed_elements_1 = [1, 2, 3]
        fixed_elements_2 = [4, 5, 6]
        expected_output = [(1, "A"), (2, "B"), (3), (4, "C"), (5, "D"), (6)]
        actual_output = self.DFR.distribute_elements_evenly(
            elements_to_be_distributed, fixed_elements_1, fixed_elements_2
        )
        assert actual_output == expected_output