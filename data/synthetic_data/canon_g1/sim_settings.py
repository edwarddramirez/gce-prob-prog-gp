# INPUT CELL

# name of the default simulation
sim_name = 'canon_g1'

# ----------------------------------- 
# fit settings (BEWARE: NO COMMAS!)
rig_temp_list = ['iso', 'psc', 'bub']
hyb_temp_list = ['pib', 'ics', 'blg']
var_temp_list = ['nfw']
is_gp = False
nfw_gamma = 1.0

# -----------------------------------
# data settings
data_file = 'fermi_data'

# ----------------------------------- 
# energy bin and SVI fit settings
ebin = 10
str_ebin = '10'

guide = 'mvn'
str_guide = 'mvn'

n_steps = 100000
str_n_steps = '100000'

lr = 0.0002
str_lr = '0.0002'

num_particles = 8
str_num_particles = '8'

svi_seed = 4242
str_svi_seed = '4242'