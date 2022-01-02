# Jack C. Cook
# Thursday, October 28, 2021

# Purpose: show how to design a constrained rectangular field.

import ghedt as dt
import ghedt.peak_load_analysis_tool as plat
import ghedt.pygfunction as gt
import pandas as pd
from time import time as clock


def main():
    # Borehole dimensions
    # -------------------
    H = 96.  # Borehole length (m)
    D = 2.  # Borehole buried depth (m)
    r_b = 0.075  # Borehole radius]
    # B = 5.  # Borehole spacing (m)

    # Pipe dimensions
    # ---------------
    r_out = 26.67 / 1000. / 2.  # Pipe outer radius (m)
    r_in = 21.6 / 1000. / 2.  # Pipe inner radius (m)
    s = 32.3 / 1000.  # Inner-tube to inner-tube Shank spacing (m)
    epsilon = 1.0e-6  # Pipe roughness (m)

    # Pipe positions
    # --------------
    # Single U-tube [(x_in, y_in), (x_out, y_out)]
    pos = plat.media.Pipe.place_pipes(s, r_out, 1)
    # Single U-tube BHE object
    bhe_object = plat.borehole_heat_exchangers.SingleUTube

    # Thermal conductivities
    # ----------------------
    k_p = 0.4  # Pipe thermal conductivity (W/m.K)
    k_s = 2.0  # Ground thermal conductivity (W/m.K)
    k_g = 1.0  # Grout thermal conductivity (W/m.K)

    # Volumetric heat capacities
    # --------------------------
    rhoCp_p = 1542. * 1000.  # Pipe volumetric heat capacity (J/K.m3)
    rhoCp_s = 2343.493 * 1000.  # Soil volumetric heat capacity (J/K.m3)
    rhoCp_g = 3901. * 1000.  # Grout volumetric heat capacity (J/K.m3)

    # Thermal properties
    # ------------------
    # Pipe
    pipe = plat.media.Pipe(pos, r_in, r_out, s, epsilon, k_p, rhoCp_p)
    # Soil
    ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
    soil = plat.media.Soil(k_s, rhoCp_s, ugt)
    # Grout
    grout = plat.media.ThermalProperty(k_g, rhoCp_g)

    # Inputs related to fluid
    # -----------------------
    # Fluid properties
    mixer = 'MEG'  # Ethylene glycol mixed with water
    percent = 0.  # Percentage of ethylene glycol added in
    fluid = gt.media.Fluid(mixer=mixer, percent=percent)

    # Fluid properties
    V_flow_borehole = 0.2  # System volumetric flow rate (L/s)

    # Define a borehole
    borehole = gt.boreholes.Borehole(H, D, r_b, x=0., y=0.)

    # Simulation parameters
    # --------------------------------
    # Simulation start month and end month
    start_month = 1
    n_years = 20
    end_month = n_years * 12
    # Maximum and minimum allowable fluid temperatures
    max_EFT_allowable = 35  # degrees Celsius
    min_EFT_allowable = 5  # degrees Celsius
    # Maximum and minimum allowable heights
    max_Height = 135.  # in meters
    min_Height = 60  # in meters
    sim_params = plat.media.SimulationParameters(
        start_month, end_month, max_EFT_allowable, min_EFT_allowable,
        max_Height, min_Height)

    # Process loads from file
    # -----------------------
    # read in the csv file and convert the loads to a list of length 8760
    hourly_extraction: dict = \
        pd.read_csv('../../Atlanta_Office_Building_Loads.csv').to_dict('list')
    # Take only the first column in the dictionary
    hourly_extraction_ground_loads: list = \
        hourly_extraction[list(hourly_extraction.keys())[0]]

    # --------------------------------------------------------------------------

    # Rectangular design constraints are the land and range of B-spacing
    length = 85.  # m
    width = 36.5  # m
    B_min = 3.  # m
    B_max = 10.  # m

    # Perform field selection using bisection search between a 1x1 and 32x32
    coordinates_domain = \
        dt.domains.rectangular(length, width, B_min, B_max, disp=True)

    output_folder = 'Rectangle_Domain'
    dt.domains.visualize_domain(coordinates_domain, output_folder)

    tic = clock()
    bisection_search = dt.search_routines.Bisection1D(
        coordinates_domain, V_flow_borehole, borehole, bhe_object,
        fluid, pipe, grout, soil, sim_params, hourly_extraction_ground_loads,
        disp=False)
    toc = clock()
    print('Time to perform bisection search: {0:.2f} seconds'.format(toc - tic))

    nbh = len(bisection_search.selected_coordinates)
    print('Number of boreholes: {}'.format(nbh))

    print('Borehole spacing: {0:.2f}'.format(bisection_search.ghe.GFunction.B))

    # Perform sizing in between the min and max bounds
    tic = clock()
    ghe = bisection_search.ghe
    ghe.compute_g_functions()

    ghe.size(method='hybrid')
    toc = clock()
    print('Time to compute g-functions and size: {0:.2f} '
          'seconds'.format(toc - tic))
    print('Sized height of boreholes: {0:.2f} m'.format(ghe.bhe.b.H))
    print('Total drilling depth: {0:.2f} m'.format(ghe.bhe.b.H * nbh))

    # Define the polygonal available zone
    perimeter = [[0., 0.], [85., 0.], [85., 80.], [0., 80.]]
    # Define the length in the x and y for the no-go zone
    l_x_building = 50
    l_y_building = 33.3
    origin_x, origin_y = (15, 36.5)
    # Make a rectangular perimeter given side lengths and the origin
    no_go = dt.utilities.make_rectangle_perimeter(
        l_x_building, l_y_building, origin=(origin_x, origin_y)
    )

    # Create and save a figure of the selected rectangular field with the go and
    # no-go zones highlighted
    coordinates = bisection_search.selected_coordinates
    fig, ax = dt.gfunction.GFunction.visualize_area_and_constraints(
        perimeter, coordinates, no_go=no_go)
    fig.savefig('base_case.png', bbox_inches='tight', pad_inches=0.1)

    # Create and save a figure of the land description with no boreholes
    fig, ax = dt.gfunction.GFunction.visualize_area_and_constraints(
        perimeter, [], no_go=no_go)
    fig.savefig('land_description.png', bbox_inches='tight', pad_inches=0.1)

    # Export the fields that were simulated during the selection process in the
    # order which they were calculated
    folder = 'Calculated_Temperature_Fields/'
    dt.utilities.create_if_not(folder)
    count = 0
    for key in bisection_search.calculated_temperatures:
        _coordinates = coordinates_domain[key]
        fig, ax = dt.gfunction.GFunction.visualize_area_and_constraints(
            perimeter, _coordinates, no_go=no_go)
        name = str(count).zfill(2)
        fig.savefig(folder + name + '.png', bbox_inches='tight', pad_inches=0.1)
        count += 1


if __name__ == '__main__':
    main()