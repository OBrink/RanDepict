from RanDepict import RandomDepictor, DepictionFeatureRanges

def test_get_FP_building_blocks():
    DFR = DepictionFeatureRanges()
    # Example scheme
    scheme = {'thickness': [{'position': 0, 'one_if': 0}, 
                            {'position': 1, 'one_if': 1}, 
                            {'position': 2, 'one_if': 2}, 
                            {'position': 3, 'one_if': 3}],    
                'kekulized': [{'position': 4, 'one_if': True}]}
    # Resulting building blocks
    building_blocks = [[[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]], [[1], [0]]]
    assert DFR.get_FP_building_blocks(scheme) == building_blocks


def test_distribute_elements_evenly():
    DFR = DepictionFeatureRanges()
    elements_to_be_distributed = ["A", "B", "C", "D"]
    elements_1 = [1, 2, 3]
    elements_2 = [4, 5, 6]
    expected_result = [(1, "A"), (2, "B"), (3), (4, "C"), (5, "D"), (6)]
    actual_result = DFR.distribute_elements_evenly(elements_to_be_distributed,
                                         elements_1,
                                         elements_2)
    assert expected_result == actual_result

