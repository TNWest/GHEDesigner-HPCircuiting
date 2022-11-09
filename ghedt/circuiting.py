from os.path import isfile
from math import ceil
from math import sqrt
from math import floor

import numpy as np
import pandas as pd
from xlsxwriter.utility import xl_rowcol_to_cell
from scipy.sparse import csgraph
from scipy.sparse import csr_matrix

# from ghedt.RowWise.Shape import Shapes # needed for future updates


class PipeNetwork:
    def __init__(self,
        **kwargs
    ):

        self.assign_and_validate_arguments(kwargs)

        # Load Cost and Pipe Size Data
        self.pipe_size_data = pd.read_csv(self.pipe_size_data_filename)
        self.cost_data = pd.read_csv(self.cost_data_filename)

        # Determine existing paths and distances
        self.borehole_connection_distances, self.borehole_connection_paths = \
                                            self.path_calculation(self.coords, self.coords)


    def assign_and_validate_arguments(self, kwargs: dict):

        # Ensure that all necessary arguments have been given.
        assert "coords" in kwargs
        assert "boreholes_per_circuit" in kwargs
        assert "circuits_per_vault" in kwargs
        assert "root_location" in kwargs
        assert "cost_data_filename" in kwargs
        assert "pipe_size_data_filename" in kwargs

        # Check that all inputs are the correct datatype, and assign them to a property.

        assert isinstance(kwargs["coords"], list)
        assert all(isinstance(val, float) for c in kwargs["coords"] for val in c)
        self.coords = np.array(kwargs["coords"], dtype=np.double)

        assert isinstance(kwargs["boreholes_per_circuit"], int)
        self.boreholes_per_circuit = kwargs["boreholes_per_circuit"]

        assert isinstance(kwargs["circuits_per_vault"], int)
        self.circuits_per_vault = kwargs["circuits_per_vault"]

        assert isinstance(kwargs["root_location"], list)
        assert all(isinstance(val, float) for val in kwargs["root_location"])
        self.root_location = kwargs["root_location"]

        if "max_iter" in kwargs:
            assert isinstance(kwargs["max_iter"], int)
            self.max_iter = kwargs["max_iter"]
        else:
            self.max_iter = 10

        if "clustering_seed" in kwargs:
            assert isinstance(kwargs["clustering_seed"], int)
            self.clustering_seed = kwargs["clustering_seed"]
        else:
            self.clustering_seed = None

        assert isinstance(kwargs["cost_data_filename"], str)
        assert isfile(kwargs["cost_data_filename"])
        self.cost_data_filename = kwargs["cost_data_filename"]

        assert isinstance(kwargs["pipe_size_data_filename"], str)
        assert isfile(kwargs["pipe_size_data_filename"])
        self.pipe_size_data_filename = kwargs["pipe_size_data_filename"]

    def build_excell_reference_string(self,
                            row_indices: np.array, column_indices: np.array, sheet_name: str):
        reference_string_array = ["("]
        number_of_references = len(row_indices)
        for reference in range(number_of_references):
            reference_string = xl_rowcol_to_cell(row_indices[reference], column_indices[reference])
            if reference < number_of_references - 1:
                reference_string_array.append("".join(["\'", sheet_name,"\'!" ,reference_string + ","]))
            else:
                reference_string_array.append("".join(["\'", sheet_name,"\'!", reference_string + ""]))
        reference_string_array.append(")")
        return "".join(reference_string_array)

    def save_to_excel_file(self, filename="circuiting.xlsx"):

        desired_figure_width = 1000
        axis_offset = 5
        x_max = ceil((int(np.max(self.coords[:, 0])) + axis_offset) / 10.0) * 10
        x_min = floor((int(np.min(self.coords[:, 0])) - axis_offset) / 10.0) * 10
        y_max = ceil((int(np.max(self.coords[:, 1])) + axis_offset) / 10.0) * 10
        y_min = floor((int(np.min(self.coords[:, 1])) - axis_offset) / 10.0) * 10
        x_axis_size = x_max - x_min
        y_axis_size = y_max - y_min
        yx_ratio = y_axis_size / x_axis_size

        writer = pd.ExcelWriter(filename, engine='xlsxwriter')
        workbook = writer.book

        # Sheet 1 Contains Vault Assignments
        sheet_1_name = "Vault Assignments"
        sheet_1_list = []
        sheet_1_list.append([self.root_location[0], self.root_location[1], -1])
        sheet_1_list.extend([[self.vault_cluster_centroids[i][0], self.vault_cluster_centroids[i][1], i] for i
                                in range(self.number_of_vaults)])
        sheet_1_list.extend([[self.coords[i][0],self.coords[i][1],self.vault_clusters[i]]
                        for i in range(self.NBH)])
        sheet_1_frame = pd.DataFrame(sheet_1_list, columns=["b_x","b_y","Vault Index"])
        sheet_1_frame.to_excel(writer, sheet_name=sheet_1_name)

        # Creating Plot for sheet 1
        worksheet_1 = writer.sheets[sheet_1_name]
        s1_chart_1 = workbook.add_chart({'type': 'scatter'})
        for vault in range(self.number_of_vaults):
            list_of_rows = np.where(self.vault_clusters == vault)[0] + 4
            col_x = sheet_1_frame.columns.get_loc('b_x') + 1
            col_y = sheet_1_frame.columns.get_loc('b_y') + 1
            category_references = self.build_excell_reference_string(list_of_rows,
                                            col_x * np.ones(len(list_of_rows), dtype=np.int64), sheet_1_name)
            value_references = self.build_excell_reference_string(list_of_rows,
                                            col_y * np.ones(len(list_of_rows), dtype=np.int64), sheet_1_name)
            s1_chart_1.add_series({
                'name': "".join(["Vault: ", str(vault)]),
                'categories': category_references,
                'values': value_references,
                'marker': {'type': 'automatic', 'size': 3},
            })
        s1_chart_1.set_x_axis({'name': 'x (m)', 'max': x_max, 'min': x_min})
        s1_chart_1.set_y_axis({'name': 'y (m)', 'max': y_max, 'min': y_min})
        s1_chart_1.set_plotarea({
            'x': desired_figure_width,
            'y': desired_figure_width * yx_ratio
        })
        # s1_chart_1.set_size({'width': 1080, 'width': 810})
        s1_chart_1.set_legend({'position': 'bottom'})
        worksheet_1.insert_chart(3, 8, s1_chart_1)

        # The following sheets contain circuit assignments and the circuit topology
        for vault_number in range(self.number_of_vaults):
            sheet_list = []
            vault_circuit_inds = np.append([-1, -1], self.circuit_clusters[vault_number], axis=0)
            number_of_coords = len(vault_circuit_inds)
            column_list = ["b_x", "b_y", "circuit labels"]
            column_list.extend(np.arange(0, number_of_coords).tolist())
            vault_inds_to_global_inds = np.where(self.vault_clusters == vault_number)[0]
            for coord_number in range(number_of_coords):
                sheet_list_row = []
                if coord_number == 0:
                    sheet_list_row.extend(self.root_location)
                    sheet_list_row.append(-1)
                elif coord_number == 1:
                    sheet_list_row.extend(self.vault_cluster_centroids[vault_number])
                    sheet_list_row.append(-1)
                else:
                    sheet_list_row.extend(self.coords[vault_inds_to_global_inds[coord_number - 2]])
                    sheet_list_row.append(vault_circuit_inds[coord_number])
                sheet_list_row.extend(self.circuit_topologies[vault_number][coord_number])
                sheet_list.append(sheet_list_row)
            sheet_frame = pd.DataFrame(sheet_list, columns=column_list)
            sheet_frame.to_excel(writer, sheet_name="Vault_{}_CrctAsnmnts_a_Tplgy".format(vault_number))

        # Sheet 2 contains the trenching information
        sheet_2_list = []
        column_list = ["Trench Length","Point 1 Type", "Point 2 Type", "Point 1 x", "Point 1 y",
                       "Point 2 x", "Point 2 y"]
        number_of_trenches = len(self.trench_lengths)
        for trench in range(number_of_trenches):
            sheet_2_list.append([self.trench_lengths[trench], self.trench_point_types[trench][0],
                                 self.trench_point_types[trench][1]])
            # Get point locations
            sheet_2_list[trench].extend(self.get_point(self.trench_point_types[trench][0],
                                        self.trench_point_indices[trench][0]))
            sheet_2_list[trench].extend(self.get_point(self.trench_point_types[trench][1],
                                        self.trench_point_indices[trench][1]))
        sheet_2_frame = pd.DataFrame(sheet_2_list, columns=column_list)
        sheet_2_frame.to_excel(writer, sheet_name="Trench_Network")
        writer.close()


    def get_point(self, point_type, point_index):
        if point_type == 1:
            return self.root_location
        elif point_type == 2:
            return self.vault_cluster_centroids[point_index]
        elif point_type == 3:
            pass # pass-through points not implemented as of yet
        elif point_type == 4:
            return self.coords[point_index]
        else:
            raise ValueError('Point Type {} is not implemented.'.format(point_type))
        return point


    def define_pipe_network_topology(self):
        # Determine network topology:
        #   In this form, each circuit is stored separately in this list.
        #   Each circuit's network is stored as a binary 2d array defining the topology
        #   (1 -> connection, 2 -> no connection). This does not define the actual pipe system
        #   (i.e. the actual routing of pipes between boreholes, it only defines which boreholes should be connected).
        # This calls the define_trench_network method as the current pipe network topology design approach depends
        # on the trench network.

        # Determine the maximum number of boreholes per vault.
        self.NBH = len(self.coords)
        self.minimum_number_of_circuits = int(ceil(float(self.NBH)/float(self.boreholes_per_circuit)))
        self.number_of_vaults = int(ceil(float(self.minimum_number_of_circuits) / float(self.circuits_per_vault)))
        self.boreholes_per_vault = int(ceil(float(self.NBH) / float(self.number_of_vaults)))

        # Group boreholes into vaults.
        self.vault_clusters, self.vault_cluster_centroids = self.k_mean_clustering(
                                                            np.arange(0, self.NBH), self.boreholes_per_vault,
                                                            max_iter=self.max_iter,
                                                            clustering_seed=self.clustering_seed)

        # Group boreholes into circuits
        circuit_clusters = []
        circuit_cluster_centers = []
        self.actual_number_of_circuits = 0
        vault_indices_array = []
        for vault_number in range(self.number_of_vaults):
            # Find indices in coordinates that are a part of this vault
            vault_indices = np.where(self.vault_clusters == vault_number)[0]
            vault_indices_array.append(vault_indices)

            circuit_cluster, circuit_centroids = self.k_mean_clustering(
                                                vault_indices, self.boreholes_per_circuit,
                                                max_iter=self.max_iter,
                                                clustering_seed=self.clustering_seed)

            self.actual_number_of_circuits += len(circuit_centroids)
            circuit_clusters.append(circuit_cluster)
            circuit_cluster_centers.append(circuit_centroids)

        # This is a 'loose' definition of the pipe network topology as the pipe network topology could be determined
        # from this and the trench network; however, a more precise definition would be more useful.
        self.circuit_clusters = circuit_clusters
        self.circuit_cluster_centers = circuit_cluster_centers
        self.vault_indices_array = vault_indices_array

        # Now that the circuit and vault assignments have been determined for each borehole, the trench network
        # can be determined.
        completion_result = self.define_trench_network()

        # Since the trench network is now defined, the individual circuit topologies can be found.
        # It should be noted that the first two 'boreholes' in each circuit topology array are the root and the vault.
        circuit_topologies = [] # temporary list for circuit topologies
        for vault_number in range(self.number_of_vaults):
            number_of_circuits = len(circuit_cluster_centers[vault_number])
            number_of_points_in_vault = len(self.vault_indices_array[vault_number]) + 2
            circuit_topologies.append([])

            # Connect root to vault
            root_row = [1 if point == 1 else 0 for point in range(number_of_points_in_vault)]
            circuit_topologies[vault_number].append(root_row)

            # Connect vault to circuits
            vault_row = [1 if point == 0 or (self.vault_indices_array[vault_number][point - 2]
                              in self.vault_circuit_connection[vault_number] and point != 1) else 0
                              for point in range(number_of_points_in_vault)]
            circuit_topologies[vault_number].append(vault_row)
            circuit_topologies[vault_number].extend([[0 for p2 in range(number_of_points_in_vault)]
                                                     for p1 in range(number_of_points_in_vault - 2)])
            for circuit_number in range(number_of_circuits):
                
                circuit_inds = np.where(circuit_clusters[vault_number] == circuit_number)[0]
                number_of_points_in_circuit = len(circuit_inds) + 2


                for p1 in range(number_of_points_in_circuit - 2):
                    circuit_topologies[vault_number].append([0])
                    p1_vault_index = circuit_inds[p1]
                    if p1_vault_index in self.vault_circuit_connection[vault_number]:
                        circuit_topologies[vault_number][p1_vault_index + 2][1] = 1

                    for p2 in range(number_of_points_in_circuit - 2):
                        p2_vault_index = circuit_inds[p2]
                        connection = self.circuit_trenches[vault_number][circuit_number][p1][p2]
                        if connection > 0:
                            circuit_topologies[vault_number] \
                            [p1_vault_index + 2][p2_vault_index + 2] = 1

        self.circuit_topologies = circuit_topologies
        return 0

    def minimum_spanning_tree(self, coord_inds):
        # Getting Distance Between Each Point and Every Other Point
        dist = self.borehole_connection_distances[coord_inds][:, coord_inds]

        # Making Dense Graph
        dense_graph = dist

        sparse_graph = csr_matrix(dense_graph)

        MST = csgraph.minimum_spanning_tree(sparse_graph, overwrite=True)
        MST_dense = csgraph.csgraph_to_dense(MST)

        return MST_dense


    def define_trench_network(self):
        # Determine Trench Connections: Each Connection is defined with four values.
        # 1st Value: Determines which list point 1 is in (i.e. 1 -> root, 2 -> vault, 3 -> null point,
        # and 4 -> borehole.
        # 2nd Value: Determines the kind of point that point 2 is.
        # 3rd Value: index to find point 1 in its respective list.
        # 4th Value: index to find point 2 in its respective list.

        # Trench Connection Storage
        trench_point_types = []
        trench_point_indices = []
        trench_lengths = []
        circuit_trenches = []
        vault_circuit_connection = []

        # Create Trench Network Vault-by-vault and circuit-by-circuit
        for vault_number in range(self.number_of_vaults):

            # Get Boreholes that are a part of this vault
            vault_indices = self.vault_indices_array[vault_number]
            vault_coords = self.coords[vault_indices]
            vault_circuit_clusters = self.circuit_clusters[vault_number]
            vault_location = self.vault_cluster_centroids[vault_number]

            # Place trench between root and current vault
            trench_point_types.append((1,2))
            trench_point_indices.append((0, vault_number))
            dist, path = self.path_calculation([self.root_location], [vault_location])
            trench_lengths.append(dist[0][0])
            circuit_trenches.append([])
            vault_circuit_connection.append([])

            for circuit_number in range(max(self.circuit_clusters[vault_number]) + 1):

                # Get Boreholes that are a part of this circuit
                circuit_inds = np.where(circuit_number == vault_circuit_clusters)[0]
                circuit_coords = vault_coords[circuit_inds]

                # Place a trench between the current vault and the closest borehole of the current circuit
                dist, connection_paths = self.path_calculation(circuit_coords, [vault_location])
                closest_borehole = np.where(dist == dist.min())[0][0]
                vault_node_connection = vault_indices[circuit_inds[closest_borehole]]
                vault_circuit_connection[vault_number].append(vault_node_connection)
                trench_point_types.append((2,4))
                trench_point_indices.append((vault_number, vault_node_connection))
                trench_lengths.append(dist[0][0])

                # Place trenches to form the minimum spanning tree of the current circuit
                circuit_trench_network = self.minimum_spanning_tree(circuit_inds) # returns 2d numpy array
                circuit_trenches[vault_number].append(circuit_trench_network)
                number_of_points = len(circuit_trench_network) # square array
                for p1 in range(number_of_points):
                    for p2 in range(number_of_points):
                        #if p2 <= p1: # All connections are shown above the diagonal
                            #continue
                        #else:
                        connection = circuit_trench_network[p1][p2]
                        if connection != 0: # A valid connection will have a non-zero distance
                            trench_point_types.append((4, 4)) # Currently all boreholes are directly connected
                            trench_point_indices.append((circuit_inds[p1], circuit_inds[p2]))
                            trench_lengths.append(connection)
                        else:
                            continue


        # Convert lists to numpy arrays and store them
        self.trench_point_types = np.array(trench_point_types, dtype=np.int32)
        self.trench_point_indices = np.array(trench_point_indices, dtype=np.int32)
        self.trench_lengths = trench_lengths
        self.circuit_trenches = circuit_trenches
        self.vault_circuit_connection = vault_circuit_connection

        return 0


    def estimate_ghe_cost(self):
        pass


    def path_calculation(self, point_array1: np.array, point_array2: np.array):

        number_of_p1s = len(point_array1)
        number_of_p2s = len(point_array2)

        # Determine the shortest path between each point between both arrays.
        point_paths = [ [[] for j in range(number_of_p2s)] for i in range(number_of_p1s)]
        path_distances = np.zeros((number_of_p1s, number_of_p2s), dtype=np.double)

        for i in range(number_of_p1s):
            p1 = point_array1[i]
            for j in range(number_of_p2s):
                p2 = point_array2[j]
                path_distances[i][j] = self.euler_distance(p1, p2)
                point_paths[i][j].extend([p1, p2])
        # point_paths = np.array(point_paths)
        return path_distances, point_paths


    def euler_distance(self, p1: np.array, p2: np.array):
        return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


    def compute_cluster_means(self, coords: np.array, cluster_labels: np.array, max_label: int,
                              cluster_count: int, n_coords: int):
        means = np.zeros((max_label + 1, 2), dtype=np.double)

        for i in range(n_coords):
            current_coord = coords[i]
            c_label = cluster_labels[i]
            means[c_label] += current_coord
        for j in range(max_label + 1):
            means[j] /= cluster_count[j]
        return means


    def distance_subset_sums(self, coord_indices: int, cluster_assignments: np.array, number_of_clusters: int):
        distance_sums = np.zeros((len(coord_indices), number_of_clusters), dtype=np.double)
        for local_coord_index in range(len(coord_indices)):
            coord_index = coord_indices[local_coord_index]
            for cluster in range(number_of_clusters):
                distance_sum = 0
                for borehole in range(len(cluster_assignments)):
                    if cluster_assignments[borehole] == cluster:
                        distance_sum += self.borehole_connection_distances[coord_index, borehole]
                distance_sums[local_coord_index, cluster] = distance_sum
        return distance_sums


    def k_mean_clustering(self, coord_inds: np.array, points_per_cluster, max_iter: int=10, clustering_seed: int=None):

        if clustering_seed is not None:
            np.random.seed(clustering_seed)

        coords = self.coords[coord_inds]
        n_coords = len(coords)
        borehole_clusters = np.zeros(n_coords, dtype=np.int32)
        boreholes_included = np.zeros(n_coords, dtype=np.bool)
        num_of_clusters = int(ceil(float(n_coords) / points_per_cluster))

        # centroid initilization
        cluster_centers = self.k_mean_initilization(coord_inds, n_coords, num_of_clusters)

        # Order Points Based on the benefit of moving them to another cluster
        dist, paths = self.path_calculation(coords, cluster_centers)

        # Assign point to their initial cluster.
        cluster_not_full = np.ones(num_of_clusters, dtype=np.bool)
        cluster_count = np.zeros(num_of_clusters, dtype=np.int32)
        while True:
            # Sort points based on the differences between their maximum and minimum distances (determining
            # which points are the most important to be assigned a good centroid).
            min_dists = np.amin(dist, axis=1, where=(cluster_not_full), initial=float('inf'))
            max_dists = np.amax(dist, axis=1, where=cluster_not_full, initial=float('-inf'))
            point_rankings = min_dists - max_dists
            coord_transfer_order = point_rankings.argsort()

            for coord_to_transfer in coord_transfer_order:

                if boreholes_included[coord_to_transfer]:
                    continue

                closest_centroid = np.where(
                    dist[coord_to_transfer] == dist[coord_to_transfer].min(
                        where=cluster_not_full, initial=float('inf')))[0]
                borehole_clusters[coord_to_transfer] = closest_centroid
                boreholes_included[coord_to_transfer] = True
                cluster_count[closest_centroid] += 1

                if cluster_count[closest_centroid] >= points_per_cluster:
                    cluster_not_full[closest_centroid] = False
                    break
            if np.all(cluster_not_full == False) or np.all(boreholes_included == True):
                break

        # Adjust borehole cluster assignments until the distance in each cluster is minimized as much as possible
        iteration = 0
        current_dists = np.zeros(n_coords, dtype=float)
        while max_iter > iteration:

            # Alternative Exit Condition
            swap_made = False

            # Compute Cluster Means
            c_means = self.compute_cluster_means(coords, borehole_clusters, num_of_clusters - 1, cluster_count, n_coords)

            # Get Distance Between Each Datapoint and The Cluster Means
            # dist = self.distance_subset_sums(coord_inds, borehole_clusters, num_of_clusters)
            dist, paths = self.path_calculation(coords, c_means)

            # Sort Data Based on the Difference Between the Distance of Their Current Assignment vs. Their Optimal One
            i = 0
            for dist_row in dist:
                current_dists[i] = dist_row[borehole_clusters[i]]
                i += 1
            min_dists = np.amin(dist, axis=1)
            point_rankings = current_dists - min_dists
            coord_swap_order = point_rankings.argsort()

            # Iterate through the coordinates in the order of the possible gain
            element_swapped = np.zeros(len(coord_swap_order), dtype=bool)
            for coord_to_swap in coord_swap_order:
                if element_swapped[coord_to_swap]:
                    continue
                current_cluster = borehole_clusters[coord_to_swap]
                preferred_clusters = np.argsort(dist[coord_to_swap])
                element_moved = False
                dist1 = dist[coord_to_swap][current_cluster]
                if cluster_count[preferred_clusters[0]] < points_per_cluster:
                    element_moved = True
                    borehole_clusters[coord_to_swap] = preferred_clusters[0]
                    element_swapped[coord_to_swap] = True
                    cluster_count[preferred_clusters[0]] += 1
                    swap_made = True
                    continue
                # for preferred_cluster in preferred_clusters:
                for other_coord_to_swap in coord_swap_order:
                    if element_moved:
                        break
                    if other_coord_to_swap == coord_to_swap:
                        continue
                    other_cluster = borehole_clusters[other_coord_to_swap]
                    gain2 = dist[other_coord_to_swap][other_cluster] - dist[other_coord_to_swap][current_cluster]
                    gain1 = dist1 - dist[coord_to_swap][other_cluster]
                    if gain1 + gain2 > 0:
                        borehole_clusters[coord_to_swap] = other_cluster
                        borehole_clusters[other_coord_to_swap] = current_cluster
                        element_swapped[coord_to_swap] = True
                        element_swapped[other_coord_to_swap] = True
                        element_moved = True
                        swap_made = True
            iteration += 1
            if not swap_made:
                break
        print("Iterations for cluster: ", iteration)
        cluster_means = self.compute_cluster_means(coords, borehole_clusters, num_of_clusters - 1, cluster_count, n_coords)
        return borehole_clusters, cluster_means


    def k_mean_initilization(self, coord_inds: np.array, n_coords: int, number_of_clusters: int):

        coords = self.coords[coord_inds]
        centroids = np.zeros((number_of_clusters, 2), dtype=np.double)
        available_indices = np.arange(0, n_coords)

        # Randomly select the first centroid
        first_centroid_ind = coord_inds[np.random.choice(available_indices)]
        centroids[0] = self.coords[first_centroid_ind]
        centroid_indices = [first_centroid_ind]

        for centroid_number in range(1, number_of_clusters):

            # distance_indices = np.array([[coord_index, centroid_index] for centroid_index
            #                              in centroid_indices for coord_index in coord_inds])
            # borehole_distances = self.borehole_connection_distances[distance_indices[:,0], distance_indices[:,1]]
            borehole_distances = self.borehole_connection_distances[coord_inds[:, np.newaxis], centroid_indices]
            
            # Get the distances between each borehole and the closest centroid
            if len(np.shape(borehole_distances)) > 1:
                borehole_distances = borehole_distances.min(axis=1)

            # Select Another Centroid
            #centroid_ind = np.random.choice(available_indices, p=dist)
            centroid_ind = coord_inds[np.argmax(borehole_distances)]
            centroid_indices.append(centroid_ind)
            centroids[centroid_number] = self.coords[centroid_ind]
            

        return centroids




