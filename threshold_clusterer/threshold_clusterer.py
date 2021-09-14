import numpy as np

class ThresholdClusterer:
    
    def __init__(self, threshold = 0.8) -> None:
        self.threshold = threshold
        self.clusters = []
        self._similarity_matrix = None
        self._matrix_row_identifier = None
        self._initial_matrix_size = None
        
    def get_neighbors_by_threshold(self, matrix: list, 
                                         identifier_list: list, 
                                         reference_identifier: int) -> list: 
        """
        Get list of neighbors from a similarity matrix for a reference identifier.
        
        **Arguments**
        
            matrix: *list*[*list*] of shape N x N 
                Input matrix
                
            identifier_list: *list*[*int*] of length N
                List of identifiers that relate the matrix entries to the matrix ids in the input matrix
                
            reference_identifer: *int*
                Identifier that corresponds to one entry of the input matrix
        
        **Returns**
            
            neigbours_list: *list*[*int*]
                List of matrix row identifiers that are more similar to the reference than the threshold
        """
        reference_matrix_index = list(identifier_list).index(reference_identifier)
        neighbors_below_threshold = filter(lambda x: x[1] >= self.threshold, zip(identifier_list, matrix[reference_matrix_index]))
        return [x[0] for x in neighbors_below_threshold if x[0] != reference_identifier]
    
    def get_cluster_overlap(self, cluster1: list, cluster2: list) -> list: 
        """
        Get number of common entries in two clusters.
        
        **Arguments**
            
            cluster1, cluster2: *list*[*Any*, *list*[*Any*]]
                clusters for which the overlapping entries are checked
                
        **Returns**
        
            n_common_entries: *int*
                number of entries that are found in both clusters
        """
        ids1 = self.indices_list_from_items(cluster1) #
        ids2 = self.indices_list_from_items(cluster2)
        return len(set(ids1).intersection(ids2))
    
    def get_neighbor_dict(self) -> dict: 
        """
        Get a dictionary that contains all entries of the similarty matrix that have neighbors that are more similar than the threshold.
        
        **Returns**
        
            neighbors_dict: *dict*
                Dictionary of type {<reference_identifier> : [<neighbor1_identifier>, ...], ...} of matrix index identifiers.
        """
        neighbors_below_threshold = [self.get_neighbors_by_threshold(self._similarity_matrix, self._matrix_row_identifier, id_) for id_ in self._matrix_row_identifier]
        return dict(filter(lambda x: len(x[1]) > 0, zip(self._matrix_row_identifier, neighbors_below_threshold)))
    
    def purge_clustered_entries(self) -> None: 
        """
        Remove clustered entries from the similarity matrix and identifier list.
        """
        clustered_entries = np.concatenate([self.indices_list_from_items(cluster) for cluster in self.clusters])
        self._purge_by_list(clustered_entries)

    def purge_orphans(self) -> None: 
        """
        Remove orphans from the similarity matrix and identifier list.
        """
        neighbors_above_threshold = [self.get_neighbors_by_threshold(self._similarity_matrix, self._matrix_row_identifier, identifier) for identifier in self._matrix_row_identifier]
        orphans = list(dict(filter(lambda x: len(x[1]) == 0, zip(self._matrix_row_identifier, neighbors_above_threshold))).keys())
        self._purge_by_list(orphans)
        
    def _purge_by_list(self, list_) -> None: 
        list_ = [item for item in list_ if item in self._matrix_row_identifier]
        indices_of_identifiers = [self._matrix_row_identifier.index(id_) for id_ in list_]
        for index in sorted(indices_of_identifiers, reverse=True):
            self._similarity_matrix.pop(index)
            self._matrix_row_identifier.pop(index)
        for matrix_index in range(len(self._similarity_matrix)):
            for index in sorted(indices_of_identifiers, reverse=True):
                self._similarity_matrix[matrix_index].pop(index)
        return None
        
    def get_largest_cluster(self, neighbors_dict: dict) -> list:
        """
        Get largest, i.e. with the highest number of neighbors, cluster from the dictionary of neighbors.
        
        **Arguments**
            
            neighbors_dict: *dict*
                A dictionary as returned by `ThresholdClusterer().get_neighbor_dict()`
                
        **Returns**
        
            largest_cluster: *list*[*Any*, *list*[*Any*]]
                The largest cluster from the neighbors dict.
                        
        """
        return list(max(neighbors_dict.items(), key = lambda x: len(x[1])))

    def indices_list_from_items(self, items: list) -> list: 
        """
        Transfrom a list of type [*Any*, [*Any*,...]] to a flat list.
        
        **Arguments**
            
            items: *list*[*Any*, *list*[*Any*]]
                list to be transformed
        
        **Returns**
        
            flat_list: *list*[Any]
                list items in a flat list
        """
        return [items[0]] + list(items[1]) 
    
    def score_cluster(self, cluster_center: int, cluster_members: list) -> float: 
        """
        Calculate the mean similarities of cluster members to the cluster center.
        
        **Arguments**
        
            cluster_center: *int*
                Matrix row identifier of the cluster center
                
            cluster_members: *list*[*int*]
                Matrix row identifiers of the cluster members
        
        **Returns**
        
            score: *float*
                Mean similarity of cluster members to the cluster center
        
        """
        cluster_center_index = self._matrix_row_identifier.index(cluster_center)
        cluster_member_indices = [self._matrix_row_identifier.index(id_) for id_ in cluster_members]
        values = [self._similarity_matrix[cluster_center_index][index] for index in cluster_member_indices]
        return float(np.mean(values))
    
    def fit(self, X, verbose = True) -> object:
        """
        Fit clusters from similarity matrix. Cluster labels can be accessed by the `labels_` property.
        Orphans are labelled as *-1*.
        
        **Arguments**
        
            X: *list*[*list*]
                Input similarity matrix
                
        **Keyword arguments**
        
            verbose: *bool*
                Write fitting progress in terms of remaining entries to STDOUT.
                
                Default value is `True`.
                
        **Returns**
            
            self: *ThresholdClusterer*
                The return value of this function is the object itself.
        """
        self._similarity_matrix = list(X)
        self._initial_matrix_size = len(X)
        self._matrix_row_identifier = list(range(self._initial_matrix_size))
        self.purge_orphans()
        neighbors_dict = self.get_neighbor_dict()
        if verbose:
            print(len(neighbors_dict), end = '\r')
        while True: # Runtime can not be estimated
            largest_cluster = max(neighbors_dict.items(), key = lambda x: len(x[1]))
            overlaps_with_largest = [cluster for cluster in neighbors_dict.items() if self.get_cluster_overlap(largest_cluster, cluster) > 0] 
            competitors = list(filter(lambda x: len(x[1]) == len(largest_cluster[1]), overlaps_with_largest))
            best = list(max(competitors, key = lambda x: self.score_cluster(*x)))
            self.clusters.append(best)
            self.purge_clustered_entries()
            neighbors_dict = self.get_neighbor_dict()
            print('                 ', end = '\r')
            print(len(neighbors_dict), end = '\r')
            if len(neighbors_dict) == 0:
                break
        return self
    
    @property
    def labels_(self) -> list:
        labels = []
        if len(self.clusters) == 0:
            raise AttributeError('Clusters are not defined. Execute the fit() method first.')
        for identifier in list(range(self._initial_matrix_size)):
            found = False
            for idx, cluster in enumerate(self.clusters):
                if identifier in self.indices_list_from_items(cluster):
                    labels.append(idx)
                    found = True
                    break
            if not found:
                labels.append(-1)
        return labels