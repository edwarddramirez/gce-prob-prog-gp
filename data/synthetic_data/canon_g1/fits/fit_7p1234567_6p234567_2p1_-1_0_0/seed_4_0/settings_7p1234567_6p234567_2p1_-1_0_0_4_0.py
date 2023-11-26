# Important Model Settings
rig_temp_list = ['iso', 'psc', 'bub']
hyb_temp_list = ['pib', 'ics']
var_temp_list = ['nfw']
is_gp = True
gp_deriv = True
data_file = 'canon_g1'
rig_temp_sim = ['iso', 'psc', 'bub']
hyb_temp_sim = ['pib', 'ics', 'blg']
var_temp_sim = ['nfw']
is_custom_blg = False
custom_blg_id = 0
sim_seed = 4; str_sim_seed = '4'
Nu = 100
u_option = 'fixed'
u_grid_type = 'healpix_bins'
u_weights = 'data'
Np = 100
p_option = 'match_u'
Nsub = 500

# Rest of model parameters set to default values
ebin = 10
is_float64 = True
debug_nans = False
no_ps_mask = False
p_grid_type = 'healpix_bins'
p_weights = None
gp_kernel = 'ExpSquared'
gp_params = ['float', 'float']
gp_scale_option = 'Linear'
monotonicity_hyperparameter = 0.005
nfw_gamma = 1.0

# SVI Parameters 
ebin = 10
str_ebin = str(ebin)
guide = 'mvn'
str_guide = guide
n_steps = 10000
str_n_steps = str(n_steps)
lr = 0.002
str_lr = '0.002'
num_particles = 8
str_num_particles = str(num_particles)
svi_seed = 0
str_svi_seed = '0'