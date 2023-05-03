from os.path import isfile
from math import ceil
from math import sqrt
from math import floor
from math import sqrt

import numpy as np
import pandas as pd
# import ghedt.search_routines as srs
from xlsxwriter.utility import xl_rowcol_to_cell
from scipy.sparse import csgraph
from scipy.sparse import csr_matrix
from ghedt.RowWise.Shape import Shapes
import pandapipes as pp
import csv

# from ghedt.RowWise.Shape import Shapes # needed for future updates


class PipeNetwork:
    def __init__(self,
        **kwargs
    ):

        self.assign_and_validate_arguments(kwargs)

        # Load Cost and Pipe Size Data
        self.pipe_size_data = pd.read_csv(self.pipe_size_data_filename)
        self.pipe_size_data_dict = dict(list(self.pipe_size_data.values))
        self.cost_data = pd.read_csv(self.cost_data_filename)

        # Convert the given property boundary and nogo zones into Shape objects
        self.property_boundary_shape = Shapes(self.property_boundary)
        nogo_zone_array = []
        for nogo_zone_coord in self.nogo_zone_coords:
            nogo_zone_array.append(Shapes(nogo_zone_coord))
        self.nogo_zones = nogo_zone_array

        self.trenches = {}

        # Create a pipe routing grid of points
        self.create_pipe_routing_grid()

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
        # assert "ghe" in kwargs
        assert "max_ghe_pressure_drop" in kwargs
        assert "property_boundary" in kwargs
        assert "nogo_zones" in kwargs
        assert "design_height" in kwargs


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

        assert isinstance(kwargs["property_boundary"], list)
        assert all(isinstance(val, list) for val in kwargs["property_boundary"])
        assert all(isinstance(val, float) for point in kwargs["property_boundary"] for val in point)
        self.property_boundary = kwargs["property_boundary"]

        assert isinstance(kwargs["design_height"], float)
        self.design_height = kwargs["design_height"]

        if "nogo_zones" in kwargs:
            assert isinstance(kwargs["nogo_zones"], list)
            assert all(isinstance(val, list) for val in kwargs["nogo_zones"])
            assert all(isinstance(val_2, list) for val in kwargs["nogo_zones"]
                       for val_2 in val)
            assert all(isinstance(val_3, float) for val in kwargs["nogo_zones"]
                       for val_2 in val for val_3 in val_2)
            self.nogo_zone_coords = kwargs["nogo_zones"]
        else:
            self.nogo_zone_coords = []

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

        if "routing_grid_fidelity" in kwargs:
            assert isinstance(kwargs["routing_grid_fidelity"], float)
            self.routing_grid_fidelity = kwargs["routing_grid_fidelity"]
        else:
            self.routing_grid_fidelity = 1.0 # in m

        assert isinstance(kwargs["cost_data_filename"], str)
        assert isfile(kwargs["cost_data_filename"])
        self.cost_data_filename = kwargs["cost_data_filename"]
        with open(self.cost_data_filename, "r") as cost_input:
            csv_reader = csv.reader(cost_input)
            cost_input_array = []
            for line in csv_reader:
                cost_input_array.append(line[1])
            self.fixed_borehole_cost = float(cost_input_array[0])
            self.drilling_cost = float(cost_input_array[1])
            self.trenching_cost = float(cost_input_array[2])


        assert isinstance(kwargs["pipe_size_data_filename"], str)
        assert isfile(kwargs["pipe_size_data_filename"])
        self.pipe_size_data_filename = kwargs["pipe_size_data_filename"]

        #assert isinstance(kwargs["ghe"], srs.Bisection1D) or \
               #isinstance(kwargs["ghe"], srs.RowWiseModifiedBisectionSearch)
        #self.ghe = kwargs["ghe"]

        assert isinstance(kwargs["max_ghe_pressure_drop"], float) # in ft
        self.max_ghe_pressure_drop = kwargs["max_ghe_pressure_drop"]

        if "return_type" in kwargs:
            self.return_type = kwargs["return_type"]
        else:
            self.return_type = "direct"

    def create_pipe_routing_grid(self):

        # Create Inital grid
        x_min = self.property_boundary_shape.minx
        x_max = self.property_boundary_shape.maxx
        y_min = self.property_boundary_shape.miny
        y_max = self.property_boundary_shape.maxy

        minimum_step_size = self.routing_grid_fidelity

        number_of_x_points = int((x_max - x_min) / minimum_step_size)
        number_of_y_points = int((y_max - y_min) / minimum_step_size)

        x_vals = np.linspace(x_min, x_max, num=number_of_x_points, dtype=np.double)
        y_vals = np.linspace(y_min, y_max, num=number_of_y_points, dtype=np.double)

        number_of_points_in_grid = number_of_y_points * number_of_x_points
        x_step = x_vals[1] - x_vals[0]
        y_step = y_vals[1] - y_vals[0]

        # Create array of point locations
        pipe_routing_grid_coords = []
        for x_val in x_vals:
            for y_val in y_vals:
                pipe_routing_grid_coords.append([x_val, y_val])

        # Determine which points need to be trimmed from the grid
        for i in range(number_of_points_in_grid - 1, -1, -1):
            current_coord = pipe_routing_grid_coords[i]
            should_point_be_trimmed = False
            if not self.property_boundary_shape.pointintersect(current_coord):
                should_point_be_trimmed = True
            if not should_point_be_trimmed:
                for nogo_zone in self.nogo_zones:
                    if nogo_zone.pointintersect(current_coord):
                        should_point_be_trimmed = True
            if should_point_be_trimmed:
                pipe_routing_grid_coords.pop(i)
        number_of_points_in_grid = len(pipe_routing_grid_coords)
        self.pipe_routing_grid_points = pipe_routing_grid_coords

        # Form dense connection tree based on the distances between each point (ensuring not to include connections
        # that intersect the property boundary or one of the nogo zones).
        dense_grid_network = []
        for i in range(number_of_points_in_grid):
            current_row_of_connections = []
            for j in range(number_of_points_in_grid):
                # if j == 389:
                    # print("here")
                if i == j:
                    current_row_of_connections.append(0.0)
                    continue
                point_1 = pipe_routing_grid_coords[i]
                point_2 = pipe_routing_grid_coords[j]
                distance = self.euler_distance(point_1,
                                               point_2)
                should_connection_be_included = True
                property_intersections = self.property_boundary_shape \
                    .lineintersect([point_1[0], point_1[1],
                                    point_2[0], point_2[1]])
                if len(property_intersections) > 0:
                    should_connection_be_included = False
                if should_connection_be_included:
                    for nogo_zone in self.nogo_zones:
                        nogo_intersections = nogo_zone.lineintersect([point_1[0],
                                                        point_1[1], point_2[0], point_2[1]])
                        if len(nogo_intersections) > 0:
                            should_connection_be_included = False
                if should_connection_be_included:
                    current_row_of_connections.append(distance)
                else:
                    current_row_of_connections.append(0)
            dense_grid_network.append(current_row_of_connections)
        self.pipe_routing_grid_connections = np.array(dense_grid_network, dtype=np.float)

        # Determine the shortest paths between each node in the network.
        self.pipe_routing_grid_distances, self.pipe_routing_grid_path_predecessors = \
            csgraph.shortest_path(self.pipe_routing_grid_connections, directed=False, return_predecessors=True)

        # Create dictionary to record which routing points where actually used (for pipe sizing purposes).
        self.pipe_routing_grid_points_used = {}
        for grid_point in range(len(self.pipe_routing_grid_points)):
            self.pipe_routing_grid_points_used[grid_point] = False

    def build_excel_reference_string(self,
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
            category_references = self.build_excel_reference_string(list_of_rows,
                                            col_x * np.ones(len(list_of_rows), dtype=np.int64), sheet_1_name)
            value_references = self.build_excel_reference_string(list_of_rows,
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

        # The following sheets 1.# contain circuit assignments and the circuit topology
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
            sheet_name = "Vault_{}_CrctAsnmnts_a_Tplgy".format(vault_number)
            sheet_frame.to_excel(writer, sheet_name=sheet_name)

            # Creating Plot for sheet 1.#
            worksheet = writer.sheets[sheet_name]
            s_chart_1 = workbook.add_chart({'type': 'scatter'})
            max_circuit_label = np.max(vault_circuit_inds)
            for circuit_number in range(max_circuit_label):
                list_of_rows = np.where(vault_circuit_inds == circuit_number)[0] + 1
                col_x = sheet_frame.columns.get_loc('b_x') + 1
                col_y = sheet_frame.columns.get_loc('b_y') + 1
                category_references = self.build_excel_reference_string(list_of_rows,
                                                    col_x * np.ones(len(list_of_rows),
                                                    dtype=np.int64), sheet_name)
                value_references = self.build_excel_reference_string(list_of_rows,
                                                    col_y * np.ones(len(list_of_rows),
                                                    dtype=np.int64), sheet_name)
                s_chart_1.add_series({
                    'name': "".join(["Circuit: ", str(circuit_number)]),
                    'categories': category_references,
                    'values': value_references,
                    'marker': {'type': 'automatic', 'size': 3},
                })
            s_chart_1.set_x_axis({'name': 'x (m)', 'max': x_max, 'min': x_min})
            s_chart_1.set_y_axis({'name': 'y (m)', 'max': y_max, 'min': y_min})
            s_chart_1.set_plotarea({
                'x': desired_figure_width,
                'y': desired_figure_width * yx_ratio
            })
            # s1_chart_1.set_size({'width': 1080, 'width': 810})
            s_chart_1.set_legend({'position': 'bottom'})
            worksheet.insert_chart(3, 8, s_chart_1)

        # Sheet 3 contains the pipe routing grid
        sheet_3_list = []
        sheet_3_name = "Pipe_Routing_Grid"
        column_list = ["Point x","Point y"]
        number_of_points = len(self.pipe_routing_grid_points)
        for grid_point in self.pipe_routing_grid_points:
            sheet_3_list_row = [grid_point[0], grid_point[1]]
            sheet_3_list.append(sheet_3_list_row)
        sheet_3_frame = pd.DataFrame(sheet_3_list, columns=column_list)
        sheet_3_frame.to_excel(writer, sheet_name=sheet_3_name)

        # Creating Plot for sheet 3
        worksheet_3 = writer.sheets[sheet_3_name]
        s3_chart_1 = workbook.add_chart({'type': 'scatter'})
        col_x = sheet_3_frame.columns.get_loc('Point x') + 1
        category_references = "".join([sheet_3_name, "!", xl_rowcol_to_cell(1, col_x),
                                       ":", xl_rowcol_to_cell(number_of_points, col_x)])
        value_references = "".join([sheet_3_name, "!", xl_rowcol_to_cell(1, col_x + 1),
                                       ":", xl_rowcol_to_cell(number_of_points, col_x + 1)])
        s3_chart_1.add_series({
            'name': "".join(["Grid Points"]),
            'categories': category_references,
            'values': value_references,
            'marker': {'type': 'automatic', 'size': 3},
        })
        s3_chart_1.set_x_axis({'name': 'x (m)', 'max': x_max, 'min': x_min})
        s3_chart_1.set_y_axis({'name': 'y (m)', 'max': y_max, 'min': y_min})
        s3_chart_1.set_plotarea({
            'x': desired_figure_width,
            'y': desired_figure_width * yx_ratio
        })
        # s1_chart_1.set_size({'width': 1080, 'width': 810})
        s3_chart_1.set_legend({'none': True})
        worksheet_3.insert_chart(3, 10, s3_chart_1)




        # Sheet 2 contains the trenching information
        sheet_2_list = []
        sheet_2_name = "Trench_Network"
        column_list = ["Trench Length", "Point 1 Type", "Point 2 Type", "Point 1 Index", "Point 2 Index",
                       "Point 1 x", "Point 1 y",
                       "Point 2 x", "Point 2 y"]
        number_of_trenches = len(self.trenches)
        trench_dict = self.trenches
        for trench in trench_dict:
            local_trench_dict = trench_dict[trench]
            sheet_2_list_row = [local_trench_dict["length"],
                                local_trench_dict["connection_types"][0],
                                local_trench_dict["connection_types"][1],
                                local_trench_dict["connection_indices"][0],
                                local_trench_dict["connection_indices"][1],
                                local_trench_dict["connection_locations"][0][0],
                                local_trench_dict["connection_locations"][0][1],
                                local_trench_dict["connection_locations"][1][0],
                                local_trench_dict["connection_locations"][1][1]]
            sheet_2_list.append(sheet_2_list_row)
        sheet_2_frame = pd.DataFrame(sheet_2_list, columns=column_list)
        sheet_2_frame.to_excel(writer, sheet_name=sheet_2_name)

        '''# Creating Plot for sheet 2
        worksheet_2 = writer.sheets[sheet_2_name]
        s2_chart_1 = workbook.add_chart({'type': 'scatter', 'subtype': 'straight'})
        for trench_number in range(len(trench_dict)):
            col_x = sheet_2_frame.columns.get_loc('Point 1 x') + 1
            row_list = np.array([trench_number + 1, trench_number + 1], dtype=np.int64)
            column_x_list = np.array([col_x, col_x + 2], dtype=np.int64)
            column_y_list = np.array([col_x + 1, col_x + 3], dtype=np.int64)

            category_references = self.build_excel_reference_string(row_list,
                                                                    column_x_list,
                                                                    sheet_2_name)
            value_references = self.build_excel_reference_string(row_list,
                                                                 column_y_list,
                                                                 sheet_2_name)
            s2_chart_1.add_series({
                'name': "".join(["Trench: ", str(trench_number)]),
                'categories': category_references,
                'values': value_references,
                # 'marker': {'type': 'automatic', 'size': 3},
            })
        s2_chart_1.set_x_axis({'name': 'x (m)', 'max': x_max, 'min': x_min})
        s2_chart_1.set_y_axis({'name': 'y (m)', 'max': y_max, 'min': y_min})
        s2_chart_1.set_plotarea({
            'x': desired_figure_width,
            'y': desired_figure_width * yx_ratio
        })
        # s1_chart_1.set_size({'width': 1080, 'width': 810})
        s2_chart_1.set_legend({'none': True})
        worksheet_2.insert_chart(3, 10, s2_chart_1)'''

        # Sheet 2b contains the trench plot.
        sheet_2b_list = []
        sheet_2b_name = "Trench_Network_Plot"
        column_list = ["Borehole x values", "Borehole y values", "Trench Point x values", "Trench Point y values"]

        # Add the borehole locations
        for borehole_location in self.coords:
            sheet_2b_list.append([(borehole_location[0]), (borehole_location[1])])

        # Add the trench point locations with blank lines between each trench
        sheet_2b_list_current_index = 0
        for trench_key in trench_dict:
            trench = trench_dict[trench_key]
            if len(sheet_2b_list) > sheet_2b_list_current_index:
                sheet_2b_list[sheet_2b_list_current_index].extend([
                    (trench["connection_locations"][0][0]),
                    (trench["connection_locations"][0][1])
                ])
                sheet_2b_list_current_index += 1
            else:
                sheet_2b_list.append([
                    "",
                    "",
                    (trench["connection_locations"][0][0]),
                    (trench["connection_locations"][0][1])
                ])
                sheet_2b_list_current_index += 1
            if len(sheet_2b_list) > sheet_2b_list_current_index:
                sheet_2b_list[sheet_2b_list_current_index].extend([
                    (trench["connection_locations"][1][0]),
                    (trench["connection_locations"][1][1])
                ])
                sheet_2b_list_current_index += 1
            else:
                sheet_2b_list.append([
                    "",
                    "",
                    (trench["connection_locations"][1][0]),
                    (trench["connection_locations"][1][1])
                ])
                sheet_2b_list_current_index += 1
            if len(sheet_2b_list) > sheet_2b_list_current_index:
                sheet_2b_list[sheet_2b_list_current_index].extend([
                    "",
                    ""
                ])
                sheet_2b_list_current_index += 1
            else:
                sheet_2b_list.append([
                    "",
                    "",
                    "",
                    ""
                ])
                sheet_2b_list_current_index += 1
        sheet_2b_frame = pd.DataFrame(sheet_2b_list, columns=column_list)
        sheet_2b_frame.to_excel(writer, sheet_name=sheet_2b_name)

        # Creating Plot for sheet 2b
        worksheet_2b = writer.sheets[sheet_2b_name]
        s2b_chart_1 = workbook.add_chart({'type': 'scatter', 'subtype': 'straight'})

        # The first series represents the trenches.
        col_x = sheet_2b_frame.columns.get_loc('Trench Point x values') + 1
        # row_list = np.arange(1, len(sheet_2b_list), dtype=np.int64)
        # column_x_list = np.ones(len(row_list), dtype=np.int64) * col_x
        # column_y_list = np.ones(len(row_list), dtype=np.int64) * (col_x + 1)

        category_references = ''.join([sheet_2b_name, '!$D2:$D$', str(len(sheet_2b_list))])
        value_references = ''.join([sheet_2b_name, '!$E2:$E$', str(len(sheet_2b_list))])
        s2b_chart_1.add_series({
            'name': "".join("Trenches"),
            'categories': category_references,
            'values': value_references,
            'line': {
                'color': 'black',
                'width': 1
            }
            # 'marker': {'type': 'automatic', 'size': 3},
        })

        # Add Second Series for borehole locations
        col_x = sheet_2b_frame.columns.get_loc('Borehole x values') + 1
        row_list = np.arange(1, len(self.coords), dtype=np.int64)
        column_x_list = np.ones(len(row_list), dtype=np.int64) * col_x
        column_y_list = np.ones(len(row_list), dtype=np.int64) * (col_x + 1)

        category_references = ''.join([sheet_2b_name, '!$B2:$B$', str(len(sheet_2b_list))])
        value_references = ''.join([sheet_2b_name, '!$C2:$C$', str(len(sheet_2b_list))])

        s2b_chart_1.add_series({
            'name': "".join("Boreholes"),
            'categories': category_references,
            'values': value_references,
            'marker': {
                'type': 'circle',
                'size': 3,
                'border': {'color': 'blue'},
                'fill': {'color': 'blue'}
            },
            'line': {'none': True}
        })


        # Finish Chart Creation
        s2b_chart_1.set_x_axis({'name': 'x (m)', 'max': x_max, 'min': x_min})
        s2b_chart_1.set_y_axis({'name': 'y (m)', 'max': y_max, 'min': y_min})
        s2b_chart_1.set_plotarea({
            'x': desired_figure_width,
            'y': desired_figure_width * yx_ratio
        })
        # s1_chart_1.set_size({'width': 1080, 'width': 810})
        s2b_chart_1.set_legend({'none': True})
        worksheet_2b.insert_chart(3, 10, s2b_chart_1)


        # The next sheets will represent the pipe information for each circuit.
        circuit_number = 0
        for net in self.pipe_networks:
            # Sheet 4 contains the pipe junction information
            sheet_4_name = "Pipe_Junction_Info_" + str(circuit_number)
            sheet_4_frame = net.res_junction
            sheet_4_frame.to_excel(writer, sheet_name=sheet_4_name)

            # Sheet 5 contains the pipe flow information
            sheet_5_name = "Pipe_Flow_Info_" + str(circuit_number)
            sheet_5_frame = net.res_pipe
            sheet_5_frame.to_excel(writer, sheet_name=sheet_5_name)

            # Sheet 6 contains the pipe size information
            sheet_6_name = "Pipe_Size_Info_" + str(circuit_number)
            sheet_6_frame = net.pipe
            sheet_6_frame.to_excel(writer, sheet_name=sheet_6_name)
            circuit_number += 1



        writer.close()

    def get_point(self, point_type, point_index):
        if point_type == 1:
            return self.root_location
        elif point_type == 2:
            return self.vault_cluster_centroids[point_index]
        elif point_type == 3:
            return self.pipe_routing_grid_points[point_index]
        elif point_type == 4:
            return self.coords[point_index]
        else:
            raise ValueError('Point Type {} is not implemented.'.format(point_type))


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
        dist, path_predecessors = self.path_calculation(
            self.coords[coord_inds], self.coords[coord_inds])

        sparse_graph = csr_matrix(dist)

        MST = csgraph.minimum_spanning_tree(sparse_graph, overwrite=True)
        MST_dense = csgraph.csgraph_to_dense(MST)

        return MST_dense, path_predecessors

    def add_trench(self, point_1_type, point_1_ind,
                   point_2_type, point_2_ind):

        trench_key = "{}_{}_{}_{}".format(point_1_type, point_1_ind,
                                          point_2_type, point_2_ind)
        if trench_key in self.trenches:
            return 1

        trench_length = self.euler_distance(self.get_point(point_1_type, point_1_ind),
                                              self.get_point(point_2_type, point_2_ind))
        self.trenches[trench_key] = {
            "length" : trench_length,
            "connection_types" : (point_1_type, point_2_type),
            "connection_indices" : (point_1_ind, point_2_ind),
            "connection_locations" : (self.get_point(point_1_type, point_1_ind),
                                              self.get_point(point_2_type, point_2_ind))
        }
        return 0



    def define_trench_network(self):
        # Determine Trench Connections: Each Connection is defined with four values.
        # 1st Value: Determines which list point 1 is in (i.e. 1 -> root, 2 -> vault, 3 -> null point,
        # and 4 -> borehole.
        # 2nd Value: Determines the kind of point that point 2 is.
        # 3rd Value: index to find point 1 in its respective list.
        # 4th Value: index to find point 2 in its respective list.

        # Trench Connection Storage
        # trench_point_types = []
        # trench_point_indices = []
        # trench_lengths = []
        circuit_trenches = []
        vault_circuit_connection = []
        circuit_topology_information = []

        # Create Trench Network Vault-by-vault and circuit-by-circuit
        for vault_number in range(self.number_of_vaults):

            # Get Boreholes that are a part of this vault
            vault_indices = self.vault_indices_array[vault_number]
            vault_coords = self.coords[vault_indices]
            vault_circuit_clusters = self.circuit_clusters[vault_number]
            vault_location = self.vault_cluster_centroids[vault_number]

            # Place trench between root and current vault
            dist, path = self.path_calculation([self.root_location], [vault_location])
            self.add_trench(1, 0, 3, path[0][0][0])
            if len(path[0][0]) > 1:
                for i in range(len(path[0][0]) - 1):
                    self.add_trench(3, path[0][0][i], 3,
                                    path[0][0][i + 1])
            self.add_trench(3, path[0][0][-1], 2,
                            vault_number)
            circuit_trenches.append([])
            vault_circuit_connection.append([])
            circuit_topology_information_by_vault = []

            for circuit_number in range(max(self.circuit_clusters[vault_number]) + 1):

                circuit_graph_connections = [] # encoding scheme: (other connection point, weight of connection)
                # [This is a list of tuples]
                circuit_point_types = [] # This is a list where each element corresponds to the type of point for
                number_of_connections = []
                boreholes_included = []
                borehole_indices = []

                # Get Boreholes that are a part of this circuit
                circuit_inds = np.where(circuit_number == vault_circuit_clusters)[0]
                circuit_coords = vault_coords[circuit_inds]

                # Place a trench between the current vault and the closest borehole of the current circuit
                dist, path = self.path_calculation(circuit_coords, [vault_location])

                circuit_point_types.append(2)
                circuit_graph_connections.append([])
                number_of_connections.append(0)

                closest_borehole = np.where(dist == dist.min())[0][0]
                vault_node_connection = vault_indices[circuit_inds[closest_borehole]]
                self.add_trench(4, vault_node_connection, 3, path[closest_borehole][0][0])


                # circuit_graph_connections[0].append(1, )
                circuit_point_types.append(3)
                circuit_graph_connections.append([])
                number_of_connections.append(0)

                if len(path[closest_borehole][0]) > 1:
                    for i in range(len(path[closest_borehole][0]) - 1):
                        self.add_trench(3, path[closest_borehole][0][i], 3,
                                        path[closest_borehole][0][i + 1])

                        circuit_point_types.append(3)
                        circuit_graph_connections.append([])
                        number_of_connections.append(0)

                        trench_length = self.euler_distance(self.get_point(3, path[closest_borehole][0][i]),
                                                            self.get_point(3, path[closest_borehole][0][i + 1]))

                        circuit_graph_connections[i + 1].append([i + 2,
                                                        trench_length])
                        number_of_connections[i+1] += 1
                        number_of_connections[i+2] += 1

                self.add_trench(3, path[closest_borehole][0][-1], 2, vault_number)

                trench_length = self.euler_distance(self.get_point(2, vault_number),
                                                    self.get_point(3, path[closest_borehole][0][-1]))
                circuit_graph_connections[0].append([len(path[closest_borehole][0]),
                                                        trench_length])
                number_of_connections[0] += 1
                number_of_connections[len(path[closest_borehole][0])] += 1

                circuit_point_types.append(4)
                circuit_graph_connections.append([])
                number_of_connections.append(0)
                boreholes_included.append(vault_node_connection)
                borehole_indices.append(len(circuit_graph_connections) - 1)

                trench_length = self.euler_distance(self.get_point(4, vault_node_connection),
                                                    self.get_point(3, path[closest_borehole][0][0]))
                circuit_graph_connections[-1].append([1,
                                                    trench_length])
                number_of_connections[-1] += 1
                number_of_connections[1] += 1


                vault_circuit_connection[vault_number].append(vault_node_connection)

                # Place trenches to form the minimum spanning tree of the current circuit
                circuit_trench_network, path = self.minimum_spanning_tree(vault_indices[circuit_inds])
                circuit_trenches[vault_number].append(circuit_trench_network)
                number_of_points = len(circuit_trench_network) # square array
                connections_included = {}
                for p1 in range(number_of_points):
                    connections_included[p1] = {}
                    for p2 in range(number_of_points):
                        connections_included[p1][p2] = False
                for p1 in range(number_of_points):
                    for p2 in range(number_of_points):
                        if connections_included[p1][p2] or connections_included[p2][p1]:
                            continue
                        connection = circuit_trench_network[p1][p2]
                        if connection != 0: # A valid connection will have a non-zero distance
                            connections_included[p1][p2] = True
                            self.add_trench(4, vault_indices[circuit_inds[p1]], 3, path[p1][p2][0])

                            if not vault_indices[circuit_inds[p1]] in boreholes_included:
                                circuit_point_types.append(4)
                                circuit_graph_connections.append([])
                                number_of_connections.append(0)
                                boreholes_included.append(vault_indices[circuit_inds[p1]])
                                borehole_indices.append(len(circuit_graph_connections) - 1)

                            circuit_point_types.append(3)
                            circuit_graph_connections.append([])
                            number_of_connections.append(0)

                            trench_length = self.euler_distance(self.get_point(4, vault_indices[circuit_inds[p1]]),
                                                                self.get_point(3, path[p1][p2][0]))
                            circuit_graph_connections[-1].append([borehole_indices[boreholes_included \
                                                                 .index(vault_indices[circuit_inds[p1]])],
                                                                 trench_length])
                            number_of_connections[-1] += 1
                            number_of_connections[borehole_indices[boreholes_included \
                                                                 .index(vault_indices[circuit_inds[p1]])]] += 1

                            if len(path[p1][p2]) > 1:
                                for i in range(len(path[p1][p2]) - 1):
                                    self.add_trench(3, path[p1][p2][i], 3,
                                                    path[p1][p2][i + 1])

                                    circuit_point_types.append(3)
                                    circuit_graph_connections.append([])
                                    number_of_connections.append(0)

                                    trench_length = self.euler_distance(self.get_point(3, path[p1][p2][i]),
                                                                        self.get_point(3, path[p1][p2][
                                                                            i + 1]))

                                    circuit_graph_connections[-2].append([len(circuit_graph_connections)-1,
                                                                            trench_length])
                                    number_of_connections[-2] += 1
                                    number_of_connections[-1] += 1
                            borehole_added = False
                            if not vault_indices[circuit_inds[p2]] in boreholes_included:
                                borehole_added = True
                                circuit_point_types.append(4)
                                circuit_graph_connections.append([])
                                number_of_connections.append(0)
                                boreholes_included.append(vault_indices[circuit_inds[p2]])
                                borehole_indices.append(len(circuit_graph_connections) - 1)

                            self.add_trench(3, path[p1][p2][-1], 4, vault_indices[circuit_inds[p2]])

                            trench_length = self.euler_distance(self.get_point(4, vault_indices[circuit_inds[p2]]),
                                                                self.get_point(3, path[p1][p2][0]))

                            last_relay_index = None
                            if borehole_added:
                                last_relay_index = -2
                            else:
                                last_relay_index = -1
                            circuit_graph_connections[last_relay_index].append([borehole_indices[boreholes_included \
                                                                 .index(vault_indices[circuit_inds[p2]])],
                                                                  trench_length])
                            number_of_connections[last_relay_index] += 1
                            number_of_connections[borehole_indices[boreholes_included \
                                .index(vault_indices[circuit_inds[p2]])]] += 1
                circuit_topology_information_by_vault.append([circuit_point_types,
                                                              number_of_connections, circuit_graph_connections])



            circuit_topology_information.append(circuit_topology_information_by_vault)


        # Convert lists to numpy arrays and store them
        # self.trench_point_types = np.array(trench_point_types, dtype=np.int32)
        # self.trench_point_indices = np.array(trench_point_indices, dtype=np.int32)
        # self.trench_lengths = trench_lengths
        self.circuit_trenches = circuit_trenches
        self.vault_circuit_connection = vault_circuit_connection
        self.circuit_topology_information = circuit_topology_information

        return 0

    def path_calculation(self, point_array1: np.array, point_array2: np.array, pathfinding=True):

        number_of_p1s = len(point_array1)
        number_of_p2s = len(point_array2)

        # Determine the shortest path between each point between both arrays.
        point_paths = [ [[] for j in range(number_of_p2s)] for i in range(number_of_p1s)]
        path_distances = np.zeros((number_of_p1s, number_of_p2s), dtype=np.double)



        if not pathfinding:
            for i in range(number_of_p1s):
                for j in range(number_of_p2s):
                    dist = self.euler_distance(point_array1[i], point_array2[j])
                    path_distances[i][j] = dist
        else:
            # For each given point, determine the closest point to it on the pipe routing grid.
            closest_point_1 = np.zeros(number_of_p1s, dtype=np.int32)
            closest_point_2 = np.zeros(number_of_p2s, dtype=np.int32)

            for j in range(len(point_array1)):
                point = point_array1[j]
                minimum_distance = float('inf')
                closest_grid_point_ind = None
                for i in range(len(self.pipe_routing_grid_points)):
                    grid_location = self.pipe_routing_grid_points[i]
                    distance = self.euler_distance(point, grid_location)
                    if distance < minimum_distance:
                        minimum_distance = distance
                        closest_grid_point_ind = i
                closest_point_1[j] = closest_grid_point_ind

            for j in range(len(point_array2)):
                point = point_array2[j]
                minimum_distance = float('inf')
                closest_grid_point_ind = None
                for i in range(len(self.pipe_routing_grid_points)):
                    grid_location = self.pipe_routing_grid_points[i]
                    distance = self.euler_distance(point, grid_location)
                    if distance < minimum_distance:
                        minimum_distance = distance
                        closest_grid_point_ind = i
                closest_point_2[j] = closest_grid_point_ind

            # For each combination of points, trace a path between the two selected points
            # on the pipe routing grid.
            grid_predecessors = self.pipe_routing_grid_path_predecessors
            grid_distances = self.pipe_routing_grid_distances
            for i in range(number_of_p1s):
                closest_grid_point_ind_1 = closest_point_1[i]
                for j in range(number_of_p2s):
                    closest_grid_point_ind_2 = closest_point_2[j]
                    goal_node = closest_grid_point_ind_2
                    current_node = closest_grid_point_ind_1
                    path_distances[i][j] = grid_distances\
                        [closest_grid_point_ind_1][closest_grid_point_ind_2]
                    while current_node != goal_node:
                        point_paths[i][j].append(current_node)
                        self.pipe_routing_grid_points_used[current_node] = True
                        if current_node == -9999:
                            print('Halt!!!')
                        current_node = grid_predecessors[goal_node, current_node]
                    point_paths[i][j].append(goal_node)

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

    '''def cluster_distances(self, coords, borehole_clusters, number_of_clusters, cluster_count):
        number_of_points = len(coords)
        cluster_distances = np.zeros((number_of_points, number_of_clusters))
        coordinates_by_cluster = [[] for i in range(number_of_clusters)]
        for i in range(number_of_clusters):
            for j in range(number_of_points):
                if borehole_clusters[j] == i:
                    coordinates_by_cluster[i].append(coords[j])
        for i in range(number_of_points):
            for j in range(number_of_clusters):
                distance_sums = np.sum([self.euler_distance(coords[i],
                                                cluster_coord) for
                                        cluster_coord in coordinates_by_cluster[j]])
                number_in_cluster = None
                if j == borehole_clusters[i]:
                    number_in_cluster = cluster_count[j] - 1
                else:
                    number_in_cluster = cluster_count[j]
                cluster_distances[i][j] = distance_sums / float(number_in_cluster)
        return cluster_distances'''

    def k_mean_clustering(self, coord_inds: np.array, points_per_cluster, max_iter: int=100, clustering_seed: int=None):

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
                        where=cluster_not_full, initial=float('inf')))[0][0]
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
            # dist = self.cluster_distances(coords, borehole_clusters, num_of_clusters, cluster_count)
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
                else:
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

    def generate_borehole_network(self, fluid_temperature, design_height, burial_depth, pp_network,
                                  nominal_pressure=1.0, pipe_diameter=.5):

        # creating borehole junctions
        entry_junction = pp.create_junction(pp_network, pn_bar=nominal_pressure, height_m=-burial_depth,
                                            tfluid_k=fluid_temperature)
        bottom_junction = pp.create_junction(pp_network, pn_bar=nominal_pressure,
                                             height_m=-burial_depth - design_height,
                                             tfluid_k=fluid_temperature)
        exit_junction = pp.create_junction(pp_network, pn_bar=nominal_pressure, height_m=-burial_depth,
                                           tfluid_k=fluid_temperature)

        # creating borehole pipes
        entry_pipe, exit_pipe = pp.create_pipes_from_parameters(pp_network, [entry_junction, bottom_junction],
                                                                [bottom_junction, exit_junction],
                                                                length_km=design_height / 1000,
                                                                diameter_m=pipe_diameter)

        return [entry_junction, bottom_junction, exit_junction], [entry_pipe, exit_pipe]


    def size_pipe_network(self):

        # Constants and Conversion Factors
        accel_of_gravity = 32.2 # in ft/s^2
        standard_temperature = 293.15 # in K
        slugft3_to_kgm3 = 515.379
        psf_per_bar = 2088.5434273039364
        standard_pressure = 1.0

        available_pipe_sizes = self.pipe_size_data["Inner Diameter (m)"]

        # Beginning Diameter
        initial_diameter = np.max(available_pipe_sizes)

        net = pp.create_empty_network(fluid="water")  # Create an empty pipe network.
        fluid_density = net["fluid"].get_density(standard_temperature) / slugft3_to_kgm3

        # Convert the given pressure drops (in ft) to bar
        max_ghe_pressure_drop_bar = self.max_ghe_pressure_drop *\
                                         (fluid_density * accel_of_gravity) / psf_per_bar
        print("Maximum Pressure Drop in Bar: ", max_ghe_pressure_drop_bar)

        # Getting Design Information
        #burial_depth = self.ghe.GFunction.D_values[0]
        #design_height = self.ghe.averageHeight()
        #borehole_pipe_diameter = self.ghe.pipe.r_in * 2
        burial_depth = 2
        design_height = self.design_height
        flowrate_per_borehole = .2
        borehole_pipe_diameter = 0.0215396
        # flowrate_per_borehole = self.ghe.m_flow_borehole

        circuit_pipe_networks = []

        circuit_topology_information = self.circuit_topology_information
        for vault_number in range(len(self.circuit_topology_information)):
            for circuit_number in range(len(self.circuit_topology_information[vault_number])):
                circuit_point_types, number_of_connections, circuit_graph_connections =\
                    self.circuit_topology_information[vault_number][circuit_number]

                net = pp.create_empty_network(fluid="water")  # Create an empty pipe network.

                ghe_ingress = pp.create_junction(net, height_m=-burial_depth,
                                                 tfluid_k=standard_temperature, pn_bar=standard_pressure)
                ghe_egress = pp.create_junction(net, height_m=-burial_depth,
                                                tfluid_k=standard_temperature, pn_bar=standard_pressure)

                # Create Junctions
                supply_nodes = []
                exit_nodes = []
                reverse_nodes = []
                borehole_pipes = []
                past_first_borehole = False
                for circuit_point in circuit_point_types:
                    # if circuit_point == 1:
                    if circuit_point == 2:
                        supply_nodes.append(ghe_ingress)
                        exit_nodes.append(ghe_egress)
                        reverse_nodes.append(None)
                    elif circuit_point == 3:
                        if past_first_borehole:
                            reverse_nodes.append(pp.create_junction(net, height_m=-burial_depth,
                                                 tfluid_k=standard_temperature, pn_bar=standard_pressure))
                        else:
                            reverse_nodes.append(None)
                        supply_nodes.append(pp.create_junction(net, height_m=-burial_depth,
                                                 tfluid_k=standard_temperature, pn_bar=standard_pressure))
                        exit_nodes.append(pp.create_junction(net, height_m=-burial_depth,
                                                 tfluid_k=standard_temperature, pn_bar=standard_pressure))
                    elif circuit_point == 4:
                        if not past_first_borehole:
                            past_first_borehole = True
                        b_junctions, b_pipes = self.generate_borehole_network(standard_temperature, design_height, burial_depth, net,
                                                       pipe_diameter=borehole_pipe_diameter)
                        borehole_pipes.extend(b_pipes)
                        reverse_nodes.append(b_junctions[-1])
                        supply_nodes.append(b_junctions[0])
                        exit_nodes.append(pp.create_junction(net, height_m=-burial_depth,
                                                 tfluid_k=standard_temperature, pn_bar=standard_pressure))

                # Create Pipes
                past_first_borehole = False
                header_pipes = []
                number_of_boreholes = 0
                for i in range(len(circuit_point_types)):
                    point_type = circuit_point_types[i]
                    if point_type == 2:
                        for trench in circuit_graph_connections[i]:
                            other_node = trench[0]
                            pipe_length = trench[1]
                            header_pipes.extend([
                            pp.create_pipe_from_parameters(net, supply_nodes[i], supply_nodes[other_node],
                                                           length_km=pipe_length / 1000,
                                                           diameter_m=initial_diameter),
                            pp.create_pipe_from_parameters(net, exit_nodes[i], exit_nodes[other_node],
                                                           length_km=pipe_length / 1000,
                                                           diameter_m=initial_diameter)])
                    elif point_type == 3:
                        for trench in circuit_graph_connections[i]:
                            other_node = trench[0]
                            pipe_length = trench[1]
                            if past_first_borehole and reverse_nodes[other_node] is not None:
                                header_pipes.append(
                                pp.create_pipe_from_parameters(net, reverse_nodes[i], reverse_nodes[other_node],
                                                               length_km=pipe_length / 1000,
                                                               diameter_m=initial_diameter))
                            header_pipes.extend([
                            pp.create_pipe_from_parameters(net, supply_nodes[i], supply_nodes[other_node],
                                                           length_km=pipe_length / 1000,
                                                           diameter_m=initial_diameter),
                            pp.create_pipe_from_parameters(net, exit_nodes[i], exit_nodes[other_node],
                                                           length_km=pipe_length / 1000,
                                                           diameter_m=initial_diameter)])
                    elif point_type == 4:
                        number_of_boreholes += 1
                        for trench in circuit_graph_connections[i]:
                            other_node = trench[0]
                            pipe_length = trench[1]
                            if not past_first_borehole:
                                past_first_borehole = True
                            header_pipes.extend([
                            pp.create_pipe_from_parameters(net, supply_nodes[i], supply_nodes[other_node],
                                                           length_km=pipe_length / 1000,
                                                           diameter_m=initial_diameter),
                            pp.create_pipe_from_parameters(net, exit_nodes[i], exit_nodes[other_node],
                                                           length_km=pipe_length / 1000,
                                                           diameter_m=initial_diameter)])
                            if reverse_nodes[other_node] is not None:
                                header_pipes.append(
                                    pp.create_pipe_from_parameters(net, reverse_nodes[i], reverse_nodes[other_node],
                                                               length_km=pipe_length / 1000,
                                                               diameter_m=initial_diameter))
                        if number_of_connections[i] == 1:
                            header_pipes.append(
                            pp.create_pipe_from_parameters(net, exit_nodes[i], reverse_nodes[i],
                                                           length_km=1 / 1000,
                                                           diameter_m=initial_diameter))
                sink = pp.create_sink(net, ghe_egress, flowrate_per_borehole * number_of_boreholes)
                pressure_grid = pp.create_ext_grid(net, junction=ghe_ingress, p_bar=max_ghe_pressure_drop_bar)
                pp.plotting.simple_plot(net, plot_sinks=True)

                # The current pipe sizing works as follows:
                # 1. The pipe sizes for the boreholes is predetermined from the thermal analysis *completed
                # 2. All of the header pipes are set to the maximum available pipe diameter. *completed
                # 3. A binary search is performed over the pipe diameters where all the header pipes
                # are set to the same diameter each step. The goal is to find the minimum diameter possible
                # for which the total pressure drop across the system is still below the given pressure drop.
                # 4. Next the pipe diameters are refined by decreasing the pipe pair with the highest flowrate
                # for each step until the pressure drop prevents any further decreases.

                # First ensure that the constraints can be met with the maximum pipe size.
                pp.pipeflow(net)
                if min(net.res_junction['p_bar']) < 0:
                    raise ValueError('Given pressure constraints cannot be met with given borehole pipe'
                                     'size and maximum header pipe size.')

                # 3.
                selected_pipe_size = 0
                previous_lower_pipe_size = -1
                previous_upper_pipe_size = -1
                lower_pipe_size = 0
                upper_pipe_size = len(available_pipe_sizes) - 1
                maximum_iter = 100
                iter = 0
                while ((lower_pipe_size != previous_lower_pipe_size) or \
                        (upper_pipe_size != previous_upper_pipe_size)) and iter < maximum_iter:
                    for pipe in header_pipes:
                        net.pipe['diameter_m'][pipe] = available_pipe_sizes[selected_pipe_size]
                    pp.pipeflow(net)
                    if min(net.res_junction['p_bar']) < 0:
                        previous_lower_pipe_size = lower_pipe_size
                        lower_pipe_size = selected_pipe_size
                    else:
                        previous_upper_pipe_size = upper_pipe_size
                        upper_pipe_size = selected_pipe_size
                    selected_pipe_size = int(floor(0.5 * (upper_pipe_size + lower_pipe_size)))
                    iter += 1
                selected_pipe_size = upper_pipe_size

                # Update the pipe sizes to the selected one.
                for pipe in header_pipes:
                    net.pipe['diameter_m'][pipe] = available_pipe_sizes[selected_pipe_size]
                pp.pipeflow(net)

                # Calculate Flow Imbalance
                borehole_pipe_flowrates = []
                for borehole_pipe in borehole_pipes:
                    borehole_pipe_ind = borehole_pipe
                    borehole_pipe_flowrates.append(abs(net.res_pipe['mdot_to_kg_per_s'][borehole_pipe_ind]))
                print("Maximum Flow Imbalance of: ", str(np.max(borehole_pipe_flowrates)
                                                         - np.min(borehole_pipe_flowrates)))
                print("Maximum Borehole Flowrate of: ", str(np.max(borehole_pipe_flowrates)))
                print("Minimum Borehole Flowrate of: ", str(np.min(borehole_pipe_flowrates)))

                circuit_pipe_networks.append(net)


        '''
        # 4.

        # Generate list tracking the sizes of each pipe
        header_pipe_sizes = np.array([selected_pipe_size for pipe_pair in header_pipes])
        pipes_available_to_change = np.array([True for pipe_pair in header_pipes])

        iter = 0
        while (pipes_available_to_change.any() == True) and iter < maximum_iter:
            pipe_flowrates = net.res_pipe['mdot_to_kg_per_s'].abs()
            pipes_sorted_by_flowrate = np.flip(np.argsort(pipe_flowrates))
            move_to_next_loop = False
            for pipe_index in pipes_sorted_by_flowrate:
                if pipes_available_to_change[pipe_index]:
                    pipe_pair = header_pipes[pipe_index]
                    new_pipe_size = header_pipe_sizes[pipe_index] - 1
                    net.pipe['diameter_m'][pipe_pair[0]] = available_pipe_sizes[new_pipe_size]
                    net.pipe['diameter_m'][pipe_pair[1]] = available_pipe_sizes[new_pipe_size]
                    pp.pipeflow(net)
                    if min(net.res_junction['p_bar']) < 0:
                        pipes_available_to_change[pipe_index] = False
                        net.pipe['diameter_m'][pipe_pair[0]] = available_pipe_sizes[new_pipe_size + 1]
                        net.pipe['diameter_m'][pipe_pair[1]] = available_pipe_sizes[new_pipe_size + 1]
                        pp.pipeflow(net)
                    else:
                        move_to_next_loop = True
                if move_to_next_loop:
                    break
        '''




        # Store pipeflow net
        self.pipe_networks = circuit_pipe_networks


    def estimate_ghe_cost(self):

        trench_lengths = [self.trenches[trench_key]["length"] for trench_key in self.trenches]
        pipe_size_data = self.pipe_size_data_dict
        pipe_network_info = self.pipe_networks
        depth_tiered_cost = [[1000, self.drilling_cost]]
        trenching_cost = self.trenching_cost

        cost_descriptions, pipe_costs =\
            NetworkCost(self.coords, trench_lengths, pipe_network_info, pipe_size_data,
                        design_height=self.design_height, unit_trench_cost=trenching_cost,
                        depth_tiered_cost=depth_tiered_cost, moving_cost=self.fixed_borehole_cost)
        self.cost_descriptions = cost_descriptions
        self.pipe_costs = pipe_costs

def NetworkCost(borehole_coordinates, trench_lengths, pipe_network_info,
                pipe_size_data
                , unit_trench_cost=70, design_height=120
                , depth_tiered_cost=[[50, 100], [150, 120], [300, 140]]
                , fuse_cost=3.5, moving_cost=.5, pipe_laying_cost=10):

    total_cost = 0
    cost_descriptions = ["Total Cost"]

    trenching_cost = TrenchCost(trench_lengths, unit_trench_cost)
    cost_descriptions.append("Trenching Cost")
    total_cost += trenching_cost

    drilling_cost = DrillCost(borehole_coordinates, design_height, depth_tiered_cost, fuse_cost, moving_cost)
    cost_descriptions.append("Drilling Cost")
    total_cost += drilling_cost

    piping_cost = PipeCost(pipe_size_data, pipe_network_info)
    cost_descriptions.append("Piping Cost")
    total_cost += piping_cost

    return (cost_descriptions, (total_cost, trenching_cost, drilling_cost, piping_cost))

def TrenchCost(trenching_network, unit_trench_cost):
    total_trenching = np.sum(trenching_network)
    return total_trenching * unit_trench_cost

def DrillCost(borehole_coordinates, design_height, depth_tiered_cost, fuse_cost, moving_cost):
    nbh = len(borehole_coordinates)
    # drill_cost = nbh * (fuse_cost + moving_cost)
    drill_cost = 0.0
    height_left = design_height
    cost_per_borehole = 0
    for i in range(len(depth_tiered_cost)):
        delta_height = None
        if i == 0:
            delta_height = depth_tiered_cost[i][0]
        else:
            delta_height = depth_tiered_cost[i][0] - depth_tiered_cost[i - 1][0]
        if delta_height > height_left:
            cost_per_borehole += height_left * depth_tiered_cost[i][1]
            break
        else:
            cost_per_borehole += delta_height * depth_tiered_cost[i][1]
            height_left -= delta_height
    drill_cost += cost_per_borehole * nbh
    return drill_cost

def PipeCost(pipe_size_data, pipe_network_info):
    total_piping_cost = 0
    pipe_material_cost, pipe_length = PipeMaterialCost(pipe_size_data, pipe_network_info)
    total_piping_cost += pipe_material_cost
    # total_piping_cost += PipeInstallationCost(pipe_laying_cost, pipe_length)
    return total_piping_cost

def PipeMaterialCost(pipe_size_data, pipe_network_info):
    cost = 0
    total_length = 0
    pipe_size_dict = dict(pipe_size_data)
    diameter_to_size = np.vectorize(pipe_size_dict.__getitem__)
    for pn in pipe_network_info:
        diameters = pn.pipe.loc[:, 'diameter_m'].to_numpy()
        lengths = pn.pipe.loc[:, "length_km"].to_numpy() * 1000
        pipe_costs_per_m = diameter_to_size(diameters)
        cost += np.sum(lengths * pipe_costs_per_m)
        total_length += np.sum(lengths)
    return cost, total_length

def PipeInstallationCost(pipe_laying_cost, pipe_length):
    return pipe_laying_cost * pipe_length


