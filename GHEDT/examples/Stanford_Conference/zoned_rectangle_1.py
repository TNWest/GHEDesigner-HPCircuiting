# Jack C. Cook
# Friday, September 17, 2021

import GHEDT.PLAT as PLAT
import GHEDT.PLAT.pygfunction as gt
import matplotlib.pyplot as plt
import pandas as pd
import gFunctionDatabase as gfdb
import GHEDT


def main():
    # --------------------------------------------------------------------------

    # Borehole dimensions
    # -------------------
    H = 100.  # Borehole length (m)
    D = 2.  # Borehole buried depth (m)
    r_b = 150. / 1000. / 2.  # Borehole radius
    B = 3.65  # m

    # Pipe dimensions
    # ---------------
    r_out = 26.67 / 1000. / 2.  # Pipe outer radius (m)
    r_in = 21.6 / 1000. / 2.  # Pipe inner radius (m)
    s = 32.3 / 1000.  # Inner-tube to inner-tube Shank spacing (m)
    epsilon = 1.0e-6  # Pipe roughness (m)

    # Pipe positions
    # --------------
    # Single U-tube [(x_in, y_in), (x_out, y_out)]
    pos = PLAT.media.Pipe.place_pipes(s, r_out, 1)
    # Single U-tube BHE object
    bhe_object = PLAT.borehole_heat_exchangers.SingleUTube

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
    pipe = PLAT.media.Pipe(pos, r_in, r_out, s, epsilon, k_p, rhoCp_p)
    # Soil
    ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
    soil = PLAT.media.Soil(k_s, rhoCp_s, ugt)
    # Grout
    grout = PLAT.media.ThermalProperty(k_g, rhoCp_g)

    # Number in the x and y
    # ---------------------
    N = 11
    M = 19
    configuration = 'zoned'
    nbh = N * M

    # GFunction
    # ---------
    # Access the database for specified configuration
    z = gfdb.Management.retrieval.Retrieve(configuration)
    # There is just one value returned in the unimodal domain for rectangles
    z_unimodal = z.retrieve(N, M)
    key = list(z_unimodal.keys())[-5]
    print('The key value: {}'.format(key))
    r_data = z_unimodal[key]
    key_split = key.split('_')
    Ni, Mi = key_split[len(key_split)-2:len(key_split)]
    Ni = int(Ni)
    Mi = int(Mi)

    # Configure the database data for input to the goethermal GFunction object
    geothermal_g_input = gfdb.Management. \
        application.GFunction.configure_database_file_for_usage(r_data)

    # Initialize the GFunction object
    GFunction = gfdb.Management.application.GFunction(**geothermal_g_input)

    # Inputs related to fluid
    # -----------------------
    V_flow_system = 41.8  # System volumetric flow rate (L/s)
    mixer = 'MEG'  # Ethylene glycol mixed with water
    percent = 0.  # Percentage of ethylene glycol added in
    # Fluid properties
    fluid = gt.media.Fluid(mixer=mixer, percent=percent)

    # Define a borehole
    borehole = gt.boreholes.Borehole(H, D, r_b, x=0., y=0.)

    # Simulation start month and end month
    # ------------------------------------
    start_month = 1
    n_years = 20
    end_month = n_years * 12
    # Initial ground temperature
    ugt = 18.3  # undisturbed ground temperature in Celsius
    # Maximum and minimum allowable fluid temperatures
    max_EFT_allowable = 35  # degrees Celsius
    min_EFT_allowable = 5  # degrees Celsius
    # Maximum and minimum allowable heights
    max_Height = 280  # in meters
    min_Height = 50  # in meters
    sim_params = PLAT.media.SimulationParameters(
        start_month, end_month, max_EFT_allowable, min_EFT_allowable,
        max_Height, min_Height)

    # Process loads from file
    # -----------------------
    # read in the csv file and convert the loads to a list of length 8760
    hourly_extraction: dict = \
        pd.read_csv('Atlanta_Office_Building_Loads.csv').to_dict('list')
    # Take only the first column in the dictionary
    hourly_extraction_ground_loads: list = \
        hourly_extraction[list(hourly_extraction.keys())[0]]

    # --------------------------------------------------------------------------

    # Initialize Hourly GLHE object
    GHE = GHEDT.ground_heat_exchangers.GHE(
        V_flow_system, B, bhe_object, fluid, borehole, pipe, grout, soil,
        GFunction, sim_params, hourly_extraction_ground_loads)

    GHE.size(method='hybrid')

    print('The sized height: {}'.format(GHE.bhe.b.H))

    print('The sized height: {}'.format(GHE.bhe.b.H))
    print('Number of boreholes: {}'.format(len(GFunction.bore_locations)))
    print('Total depth: {}'.format(GHE.bhe.b.H * len(GFunction.bore_locations)))

    # Plot go and no-go zone with corrected borefield
    # -----------------------------------------------
    zr = gfdb.coordinates.ZonedRectangle(M, N, B, Nix=Mi, Niy=Ni)
    zr.gen_z_rect()
    coordinates = list(zr.coordinates.values())[0]

    perimeter = [[0., 0.], [70.104, 0.], [70.104, 80.772], [0., 80.772]]
    no_go = [[9.997, 36.51], [9.997, 69.79], [59.92, 69.79], [59.92, 36.51]]

    fig, ax = GFunction.visualize_area_and_constraints(perimeter,
                                                       coordinates,
                                                       no_go=no_go)

    fig.savefig('zoned_rectangle_case_1.png')

    # Plot long time step g-functions
    # -------------------------------
    fig, ax = GFunction.visualize_g_functions()

    B_over_H = B / GHE.bhe.b.H
    # interpolate for the Long time step g-function
    g_function, rb_value, D_value, H_eq = \
        GFunction.g_function_interpolation(B_over_H)
    # correct the long time step for borehole radius
    g_function_corrected = \
        GFunction.borehole_radius_correction(g_function,
                                             rb_value,
                                             GHE.bhe.b.r_b)
    ax.plot(GFunction.log_time, g_function_corrected, '--')

    fig.savefig('zoned_rectangle_case_1_gFunctions.png')


if __name__ == '__main__':
    main()

