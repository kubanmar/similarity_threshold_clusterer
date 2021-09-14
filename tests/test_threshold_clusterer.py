import pytest
from threshold_clusterer import ThresholdClusterer

@pytest.fixture
def clusterer():
    return ThresholdClusterer()

def test_get_neighbors_by_threshold(clusterer):
    matrix = [[1, 0.5, 0.3], [0.5, 1, 0.8], [0.3, 0.8, 1]]
    ids = [1,2,3]
    neighbors = [clusterer.get_neighbors_by_threshold(matrix, ids, id_) for id_ in ids]
    assert neighbors == [[], [3], [2]], "wrong neighbors found"

def test_purge_orphans(clusterer):
    matrix = [[1, 0.5, 0.3], [0.5, 1, 0.8], [0.3, 0.8, 1]]
    ids = [1,4,3]
    clusterer._similarity_matrix = matrix
    clusterer._matrix_row_identifier = ids
    clusterer.purge_orphans()
    assert clusterer._similarity_matrix == [[1, 0.8], [0.8, 1]], "Wrong matrix entries removed"
    assert clusterer._matrix_row_identifier == [4, 3], "Wrong matrix identifiers removed"

def test_get_neighbor_dict(clusterer):
    matrix = [[1, 0.5, 0.3], [0.5, 1, 0.8], [0.3, 0.8, 1]]
    ids = [1,4,3]
    clusterer._similarity_matrix = matrix
    clusterer._matrix_row_identifier = ids
    assert  clusterer.get_neighbor_dict() == {4 : [3], 3 : [4]}, "Wrong neighbor dict constructed"

def test_indices_list_from_items(clusterer):
    test_list = [1, [2,3,4]]
    assert clusterer.indices_list_from_items(test_list) == [1,2,3,4], "failed to concatenate list"

def test_get_cluster_overlap(clusterer):
    assert clusterer.get_cluster_overlap([1, [2,3,4]],[2, [3,4,5]]) == 3, "Wrong cluster overlap"

def test_score_cluster(clusterer):
    matrix = [[1, 0.5, 0.3], [0.5, 1, 0.8], [0.3, 0.8, 1]]
    ids = [1,4,3]
    clusterer._similarity_matrix = matrix
    clusterer._matrix_row_identifier = ids
    clusterer.clusters = [4, [3]]
    assert clusterer.score_cluster(4, [3]) == 0.8, "Wrong cluster score"

def test_purge_clustered_entries(clusterer):
    matrix = [[1, 0.5, 0.3], [0.5, 1, 0.8], [0.3, 0.8, 1]]
    ids = [1,4,3]
    clusterer._similarity_matrix = matrix
    clusterer._matrix_row_identifier = ids
    clusterer.clusters = [[4, [3]]]
    clusterer.purge_clustered_entries()
    assert clusterer._similarity_matrix == [[1]], "Wrong matrix entries removed"
    assert clusterer._matrix_row_identifier == [1], "Wrong matrix identifiers removed"

def test_get_largest_cluster(clusterer):
    test_dict = {1:[2,3], 4:[5,6,7]}
    assert clusterer.get_largest_cluster(test_dict) == [4, [5,6,7]], "Largest cluster not found"

def test_fit(clusterer):
    matrix = [[1, 0.5, 0.3], [0.5, 1, 0.8], [0.3, 0.8, 1]]
    clusterer.fit(matrix)
    assert clusterer.clusters == [[1, [2]]], "Wrong clusters found"

def test_labels_(clusterer):
    clusterer.clusters = [[1, [2]]]
    clusterer._initial_matrix_size = 3
    assert clusterer.labels_ == [-1,0,0], "wrong cluster labels returned"