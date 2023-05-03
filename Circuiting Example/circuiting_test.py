import time

from ghedt.circuiting import PipeNetwork
import csv

def read_points_from_file(filename, first_line=False):
    input_points = []
    with open(filename, "r", newline="") as inputFile:
        cR = csv.reader(inputFile, delimiter=",")
        # first_line = False
        for line in cR:
            if first_line:
                first_line = False
                continue
            input_points.append([float(line[0]), float(line[1])])
    return input_points

def main():
    # Example Inputs
    coordinate_file = "Input Files\\Borefield_Coordinates.csv"
    property_boundary_file = "Input Files\\Example_Property.csv"
    nogo_zone_file = "Input Files\\Example_Nogo_Zone.csv"
    result_file = "OUtput Files\\PipeNetworkClass_Example.xlsx"
    cost_data_filename = "Input Files\\Cost_Factors.csv"
    pipe_size_data_filename = "Input Files\\Pipe_Size_Data.csv"
    boreholes_per_circuit = 10
    circuits_per_vault = 10
    root_location = [80.0, 65.0]
    max_iter = 1000
    routing_grid_fidelity = 5.0

    # Read in borehole locations from file
    input_points = read_points_from_file(coordinate_file, first_line=True)
    property_boundary = read_points_from_file(property_boundary_file, first_line=False)
    nogo_zone = []
    nogo_zone = [read_points_from_file(nogo_zone_file)] # Needs to be a list of lists

    # Instantiate PipeNetwork object
    pipe_network_definition = {
        "coords": input_points, # Borehole locations
        "boreholes_per_circuit": boreholes_per_circuit, # Maximum number of boreholes per circuit
        "circuits_per_vault": circuits_per_vault, # Maximum number of circuits per vault
        "root_location": root_location, # [x, y] location of the entrance/exit to the GHE
        "cost_data_filename": cost_data_filename, # Name of csv file containing cost inputs
        "pipe_size_data_filename": pipe_size_data_filename, # Name of csv file with available pipe diameters and
                                                            # their respective prices.
        "max_iter": max_iter, # Maximum number of iterations for the grouping algorithm.
        "nogo_zones": nogo_zone, # List of polygons defining 'nogo zones'
        "property_boundary": property_boundary, # Polygon defining the property
        "routing_grid_fidelity": routing_grid_fidelity, # The distance between the points in the routing grid
        "max_ghe_pressure_drop": 30.0, # ft. # Maximum allowable pressure drop in ft. of water
        "design_height": 134.584 # The height of the GHE
    }

    # Define the pipe-network topology (the trenching topology is also defined in this process)
    tic = time.time()
    pipe_network = PipeNetwork(**pipe_network_definition)
    pipe_network.define_pipe_network_topology()
    toc = time.time()
    print("Network Topology Comp Time (s): ", toc - tic)

    # Size the pipe network.
    tic = time.time()
    pipe_network.size_pipe_network()
    toc = time.time()
    print("Pipe Sizing Comp Time (s) ", toc - tic)

    # Output Pipe Network to file.
    pipe_network.save_to_excel_file(filename=result_file)

    # Estimate and print the estimated GHE cost.
    tic = time.time()
    pipe_network.estimate_ghe_cost()
    toc = time.time()
    print("GHE Cost Comp Time (s) ", toc - tic)
    cost_vals = pipe_network.pipe_costs
    cost_descriptions = pipe_network.cost_descriptions
    for cost_index in range(len(cost_vals)):
        print(cost_descriptions[cost_index] + " " + str(cost_vals[cost_index]))


if __name__ == '__main__':
    main()