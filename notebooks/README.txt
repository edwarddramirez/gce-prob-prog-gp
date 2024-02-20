This notebook directory contains notebooks used to create simulated templates
and fit them using any combination of template models and a GP.

Settings are always adjustable via the second cell of the notebooks. 
All other aspects of the fitting code. The plotting code may require some 
adjustments. Contact me if there is any confusion.

Set Up:
0. Install Packages

1. Reset Main Directory Location
	a. Go to utils/ed_fcts.py
	b. Find the "load_data_dir" function
	c. Change main_data_dir to the directory containing the repo

Steps to Using Code:

0. Infer realistic simulation normalizations from the data
    Source: 0-temps_fit_to_real_data.ipynb
    
    Main Inputs to Change:
        sim_name - String identifying name of simulation directory
        ._temp_list - Templates used to to fit the data

    Output: 
        Posterior samples 
        SVI results object

    Note: ._temp_list split into "rig", "hyb", and "var", but all that matters
    is the union of these lists.

1. Generate pseudodata from mean normalizations of previous fit
    Source: 1-sim_map_maker.ipynb

    Main Inputs to Change:
        sim_name - Specifies fit to data

    Output:
        Numpy arrays containing pseudodata

2. Fit pseudodata using desired model (GP + Bkgd Templates)
    Source: 2-gp_fit.ipynb

    Main Inputs to Change:
        See INPUT CELLs (One for GP fit, One for Template Extraction)

    Output:
        Posterior samples
        SVI results object
        (GP Samples if GP is used in the fit)

3. Load Fit Data
    Source: 3-load_fit_data.ipynb

    Loads data obtained from previous fit and regenerates some plots from previous notebook