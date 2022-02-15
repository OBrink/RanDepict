from numpy import asscalar
from RanDepict import RandomDepictor, DepictionFeatureRanges


class TestDepictionFeatureRanges:
    DFR = DepictionFeatureRanges()
    
    def test_random_choice(self):
        chosen_element = self.DFR.random_choice(range(4), log_attribute='thickness')
        # Assert that random_choice picks an element from the given iterable
        assert chosen_element in range(4)
        # Assert that random_choice saves the range of values it picks from
        assert getattr(self.DFR, 'thickness_range') == range(4)
    
    
    def test_generate_fingerprint_schemes(self):
        # This cannot really be tested so we simply assert that it generates
        # every scheme when running generate_fingerprint_schemes()
        fingerprint_schemes = self.DFR.generate_fingerprint_schemes()
        assert len(fingerprint_schemes) == 4
        cdk_scheme, rdkit_scheme, indigo_scheme, aug_scheme = fingerprint_schemes
        assert 'indigo' in list(indigo_scheme.keys())[0]
        assert 'rdkit' in list(rdkit_scheme.keys())[0]
        assert 'cdk' in list(cdk_scheme.keys())[0]
        
    
    def test_generate_fingerprint_scheme(self):
        # Assert that generate_fingerprint_scheme generates takes a map of keywords and ranges
        # and returns a map of keyword and every fingerprint position to a condition
        example_ID_range_map = {'thickness': [0, 1, 2, 3], 'kekulized': [True, False]}
        expected_result = {
            'thickness': [{'position': 0, 'one_if': 0}, 
                        {'position': 1, 'one_if': 1}, 
                        {'position': 2, 'one_if': 2}, 
                        {'position': 3, 'one_if': 3}], 
            'kekulized': [{'position': 4, 'one_if': True}]}
        actual_result = self.DFR.generate_fingerprint_scheme(example_ID_range_map)
        assert actual_result == expected_result
    
    
    def test_get_FP_building_blocks(self):
        # Example scheme
        scheme = {'thickness': [{'position': 0, 'one_if': 0}, 
                                {'position': 1, 'one_if': 1}, 
                                {'position': 2, 'one_if': 2}, 
                                {'position': 3, 'one_if': 3}],    
                    'kekulized': [{'position': 4, 'one_if': True}]}
        # Resulting building blocks
        building_blocks = [[[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]], [[1], [0]]]
        assert self.DFR.get_FP_building_blocks(scheme) == building_blocks


    def test_distribute_elements_evenly(self):
        elements_to_be_distributed = ["A", "B", "C", "D"]
        elements_1 = [1, 2, 3]
        elements_2 = [4, 5, 6]
        expected_result = [(1, "A"), (2, "B"), (3), (4, "C"), (5, "D"), (6)]
        actual_result = self.DFR.distribute_elements_evenly(elements_to_be_distributed,
                                            elements_1,
                                            elements_2)
        assert expected_result == actual_result
        
    def test_split_into_n_sublists(self):
        # Assert that an iterable is split into even lists
        actual_result = self.DFR.split_into_n_sublists(range(10), 2)
        expected_result = [[0,1,2,3,4], [5,6,7,8,9]]
        assert actual_result == expected_result
        # The same for an even split if the length of the iterable does not allow a clean split
        actual_result = self.DFR.split_into_n_sublists(range(9), 2)
        expected_result = [[0,1,2,3,4], [5,6,7,8]]
        assert actual_result == expected_result
        
    
    def test_get_number_of_possible_fingerprints(self):
        pass