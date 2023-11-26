This notebook directory contains notebooks used to create simulated templates
and fit them using any combination of template models and a GP.

Settings are always adjustable via the second cell of the notebooks. 
All other aspects of the fitting code. The plotting code may require some 
adjustments. Contact me if there is any confusion.

Set Up:
1. Install Packages

2. Reset Main Directory Location
	a. Go to utils/ed_fcts.py
	b. Find the "load_data_dir" function
	c. Change main_data_dir to the directory containing the repo

Steps to Using Code:

1. Infer simulation normalizations from the data
    Source: 1_poissonian_fit_temps.ipynb
    
    Main Inputs to Change:
        sim_name - String identifying name of simulation directory
        ._temp_list - Templates used to to fit the data

    Output: 
        Posterior samples 
        SVI results object

    Note: ._temp_list split into "rig", "hyb", and "var", but all that matters
    is the union of these lists.

2. Generate pseudodata from mean normalizations of previous fit
    Source: sim_map_maker.ipynb

    Main Inputs to Change:
        sim_name - Specifies fit to data

    Output:
        Numpy arrays containing pseudodata

3. Fit pseudodata using settings of choice
    Source: 1_poissonian_fit_all.ipynb

    Main Inputs to Change:
        See 2nd cell

    Output:
        Posterior samples
        SVI results object
        (GP Samples if GP is used in the fit)

4. Basic Plots
    Source: plotter_gp.ipynb

    Note: This plotter file works only for the "GP -> Blg" fit. It requires changes
    for "No GP" or "GP -> Blg + NFW" 1_poissonian_fit_temps.