"""Poissonian models for healpix with energy binning."""

# Yitian block
import os
import sys
sys.path.append("..")

import numpy as np
import healpy as hp

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import jax.scipy.optimize as optimize
from jax.scipy.stats.multivariate_normal import logpdf
from jax.example_libraries import stax

import optax
import numpyro
import numpyro.distributions as dist
from numpyro import optim
from numpyro.infer import SVI, Trace_ELBO, autoguide, TraceMeanField_ELBO
from numpyro.infer.elbo import Trace_ELBO_2
from numpyro.infer.reparam import NeuTraReparam
from numpyro.infer import MCMC, NUTS
from numpyro.contrib.tfp.mcmc import ReplicaExchangeMC
from tensorflow_probability.substrates import jax as tfp

from utils import create_mask as cm
from utils.sph_harm import Ylm
from utils.map_utils import to_nside

from templates.rigid_templates import EbinTemplate, Template, BulgeTemplates
from templates.variable_templates import NFWTemplate, LorimerDiskTemplate
from likelihoods.pll_jax import log_like_poisson

# Gaussian Process Block
import tinygp  # for Gaussian process regression
from tinygp import GaussianProcess, kernels, transforms
from utils import ed_fcts as ef
from jax.scipy.special import logit, expit
import tqdm

# minipyro functions
from jax import jit
from collections import namedtuple

# functions for fitting template models to GP models
from numpyro.infer import Predictive
from jax.random import poisson
# better to load GPU from the main script that runs the fit
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# SVIRunResult cannot be saved if inside the class
SVIRunResult = namedtuple("SVIRunResult", ["params", "state", "losses", 
                                        "min_params", "min_state", "min_loss",
                                        "recorded_steps", "recorded_params", "recorded_states"])

class EbinPoissonModel:
    """
    Energy binned model for poisson fits.

    Parameters
    ----------
    nside : 512 or lower powers of 2.
        HEALPix NSIDE parameter.
    data_class : str
        Data class.
    temp_class : str
        Template class.
    mask_class : str
        Mask class.
    mask_roi_r_outer : float
        Outer radius of the region of interest mask, in degrees.
    mask_roi_b : float
        Latitude boundary of the region of interest mask, in degrees.
    dif_names : list of str, can be empty
        List of diffuse model names.
    blg_names : list of str, can be empty
        List of bulge model names.
    nfw_gamma : {'vary', float}
        NFW gamma parameter, can be either 'vary' (default) or a float number.
    disk_option : {'vary', 'fixed', 'none'}
        Option for the disk model.
    l_max : int
        Maximum multipole moment for the harmonic expansion, default is -1 (turned off).

    # Ed's parameters
    rig_temp_list : list of str
        List of rigid template names.
    hyb_temp_list : list of str
        List of hybrid template names.
    var_temp_list : list of str
        List of variable template names.
    is_gp : bool
        Whether to use Gaussian process.
    data_file : str
        Name of the data file.
    is_float64 : bool
        Whether to use float64 for the model.
    debug_nans : bool
        Whether to debug nans.
    no_ps_mask : bool
        Whether to use the point source mask.
    Nu : int
        Number of inducing points.
    u_option : {'float', 'fixed'}
        Option for inducing points.
    u_grid_type : {'healpix_bins', 'sunflower', 'square', 'hex'}
        Type of grid for inducing points.
    u_weights : None or array
        Weights for inducing points, if u_grid_type = healpix_bins
    Np : int
        Number of derivative points.
    p_option : {'float', 'fixed'}
        Option for derivative points.
    p_grid_type : {'healpix_bins', 'healpix'}
        Type of grid for derivative points.
    p_weights : None or array
        Weights for derivative points.
    Nsub : int
        Number of subsamples.
    gp_kernel : {'ExpSquared', 'RationalQuadratic', 'Matern32', 'Matern52'}
        Kernel for the Gaussian process.
    gp_params : list of float
        Parameters for the Gaussian process.
    gp_scale_option : {'Linear', 'Cholesky'}
        Option for the Gaussian process scale.
    gp_deriv : bool
        Whether to use derivatives in the Gaussian process.
    monotonicity_hyperparameter : float
        Hyperparameter for the monotonicity constraint.
    """

    def __init__(
        self,
        nside = 128,
        ps_cat = '3fgl',
        data_class = 'fwhm000-0512-bestpsf-nopsc',
        temp_class = 'ultracleanveto-bestpsf',
        mask_class = 'fwhm000-0512-bestpsf-mask',
        mask_roi_r_outer = 20.,
        mask_roi_b = 2.,
        dif_names = ['modelo'],
        blg_names = ['mcdermott2022'],
        nfw_gamma = 'vary',
        disk_option = 'none',
        l_max = -1,

        # Ed's parameters
        ebin = 10,
        rng_key = jax.random.PRNGKey(42),
        rig_temp_list = ['iso', 'psc', 'bub'], # 'iso', 'psc', 'bub'
        hyb_temp_list = ['pib', 'ics'], # pib, ics, blg
        var_temp_list = ['nfw', 'dsk'], # nfw, dsk
        is_gp = True,
        gp_deriv = True,
        data_file = 'fermi_data', # fermi_data, sim
        rig_temp_sim = None,
        hyb_temp_sim = None,
        var_temp_sim = None,
        is_custom_blg = False,
        custom_blg_id = None,
        sim_seed = 1,
        is_float64 = False,
        debug_nans = False,
        no_ps_mask = False,
        Nu = 12*12,
        u_option = 'fixed', # 'float' or 'fixed'
        u_grid_type = 'healpix_bins',
        u_weights = None,
        Np = 12*12,
        p_option = 'match_u', # 'float' or 'fixed'
        p_grid_type = 'healpix_bins',
        p_weights = None,
        Nsub = None,
        gp_kernel = 'ExpSquared',
        gp_params = [10., 'float'],
        gp_scale_option = 'Linear', # 'Linear' or 'Cholesky'
        monotonicity_hyperparameter = 10.,
    ):
        
        self.nside = nside
        self.ps_cat = ps_cat
        self.data_class = data_class
        self.temp_class = temp_class
        self.mask_class = mask_class
        self.mask_roi_r_outer = mask_roi_r_outer
        self.mask_roi_b = mask_roi_b
        self.dif_names = dif_names
        self.blg_names = blg_names
        self.nfw_gamma = nfw_gamma
        self.disk_option = disk_option
        self.l_max = l_max

        # Ed's parameters
        self.ebin = 10
        self.rng_key = rng_key
        self.rig_temp_list = rig_temp_list
        self.hyb_temp_list = hyb_temp_list
        self.var_temp_list = var_temp_list
        self.is_custom_blg = is_custom_blg
        self.custom_blg_id = custom_blg_id
        self.is_gp = is_gp
        self.gp_deriv = gp_deriv
        self.data_file = data_file 
        self.rig_temp_sim = rig_temp_sim
        self.hyb_temp_sim = hyb_temp_sim
        self.var_temp_sim = var_temp_sim
        self.sim_seed = sim_seed
        self.is_float64 = is_float64
        self.debug_nans = debug_nans
        self.no_ps_mask = no_ps_mask
        self.Nu = Nu
        self.u_option = u_option
        self.u_grid_type = u_grid_type
        self.u_weights = u_weights
        self.Np = Np
        self.p_option = p_option
        self.p_grid_type = p_grid_type
        self.p_weights = p_weights
        self.Nsub = Nsub
        self.gp_kernel = gp_kernel
        self.gp_params = gp_params
        self.gp_scale_option = gp_scale_option
        self.monotonicity_hyperparameter = monotonicity_hyperparameter
        
        # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
        jax.config.update("jax_enable_x64", is_float64)
        jax.config.update("jax_debug_nans", debug_nans)

        if self.nside > 128:
            ebin_data_dir = '../data/fermi_data_573w/ebin'
            if not os.path.isdir(ebin_data_dir):
                print('NSIDE > 128 requires ebin_512 dataset.')
        else:
            ebin_data_dir = '../data/fermi_data_573w/ebin_128'
        default_data_dir = '../data/fermi_data_573w/fermi_data_256'
        
        # ======== Energy Binning ==========
        self.energy_bins = jnp.logspace(0.2,200,40) # Yitian's energy binning (GeV)
        self.eval = self.energy_bins[ebin]

        #========== Data ==========
        if data_file == 'fermi_data':
            self.counts = jnp.array(
                to_nside(
                    np.load(f'{ebin_data_dir}/counts-{self.data_class}.npy'),
                    self.nside,
                    mode='sum',
                ),
                # dtype = jnp.int32
            )
        elif data_file == 'custom':
            print('Manually Load Data File in Notebook!')
        else: 
            # sim_data_dir = '../data/synthetic_data/' + data_file + '/'
            sim_data_dir = ef.load_data_dir(data_file) # [alternative]
            self.sim_data_dir = sim_data_dir
            if self.rig_temp_sim is None:
                self.rig_temp_sim = self.rig_temp_list
            if self.hyb_temp_sim is None:
                self.hyb_temp_sim = self.hyb_temp_list
                if is_gp:
                    hyb_temp_sim.append('blg')
            if self.var_temp_sim is None:
                self.var_temp_sim = self.var_temp_list
            temp_names_sim = rig_temp_sim + hyb_temp_sim + var_temp_sim
            self.temp_names_sim = temp_names_sim

            # sim_file_name = ef.make_sim_file_name(temp_names_sim, sim_seed, is_custom_blg, custom_blg_id)
            # self.counts = jnp.array(np.load(sim_data_dir + sim_file_name), dtype = jnp.float32)
            sim_file_name = ef.make_pseudodata_file(temp_names_sim, sim_data_dir, create_dir = False, return_name=True, sim_seed=sim_seed, 
                                                       is_custom_blg=is_custom_blg, custom_blg_id=custom_blg_id)
            self.counts = jnp.array(np.load(sim_file_name), dtype = jnp.float32)


        self.exposure = to_nside(
            np.load(f'{ebin_data_dir}/exposure-{self.data_class}.npy'),
            self.nside,
            mode='sum',
        )
        
        #========== Mask ==========
        # NOTE: added some comments on old mask and added new ps mask recommended by Yitian
        # shape (40, 196608): depends on energy bins

        if not self.no_ps_mask:
            # old mask array
            # mask_ps_arr = to_nside(np.load('data/fermi_data_573w/ebin_128/mask-fwhm000-0512-bestpsf-mask.npy'), 
            #                             nside) > 0'
            
            # TODO: Masked array masks too many of the simulated point sources. Need Yitian to modify mask for simulations or add more point sources
            mask_ps_arr_no_eng = np.load('../data/mask_3fgl_0p8deg.npy') # shape (196608,): no dependence on energy bins ; already Boolean so no inequality
            self.mask_ps_arr = np.tile(mask_ps_arr_no_eng, (40, 1)) # shape (40, 196608): depends on energy bins
            self.mask_roi_arr = np.asarray([
                cm.make_mask_total(
                    nside=self.nside,
                    band_mask=True,
                    band_mask_range=self.mask_roi_b,
                    mask_ring=True,
                    inner=0,
                    outer=self.mask_roi_r_outer,
                    custom_mask=mask_ps_at_eng
                )
                for mask_ps_at_eng in self.mask_ps_arr
            ])
        else:  # no ps mask
            self.mask_roi_arr = np.tile(
                cm.make_mask_total(
                    nside=self.nside,
                    band_mask=True,
                    band_mask_range=self.mask_roi_b,
                    mask_ring=True,
                    inner=0,
                    outer=self.mask_roi_r_outer
                )
                , (40, 1))
        
        self.normalization_mask = np.asarray(
            cm.make_mask_total(
                nside=self.nside,
                band_mask=True,
                band_mask_range=2,
                mask_ring=True,
                inner=0,
                outer=25,
            )
        )

        # ======== initialize svi results ==========
        self.svi_results = None

        # ======== load templates ==========
        self.temp_list = self.rig_temp_list + self.hyb_temp_list + self.var_temp_list
        self.temps = {}
        self.load_templates(self.temp_list, self.blg_names, self.dif_names) 

    def load_templates(self, temp_list, blg_names, dif_names, ie = 10):
        # note we load temps before we load the model
        # this allows us to load all temps once and continue if we are fitting to other data
        #========== Rigid templates ==========

        if self.nside > 128:
            ebin_data_dir = '../data/fermi_data_573w/ebin'
            if not os.path.isdir(ebin_data_dir):
                print('NSIDE > 128 requires ebin_512 dataset.')
        else:
            ebin_data_dir = '../data/fermi_data_573w/ebin_128'
        
        default_data_dir = '../data/fermi_data_573w/fermi_data_256'

        if 'iso' in temp_list:
            self.temps['iso'] = EbinTemplate(
                self.exposure.copy(),
                norm_mask=self.normalization_mask,
            )
        if 'psc' in temp_list:
            self.temps['psc'] = EbinTemplate(
                to_nside(np.load(f'{ebin_data_dir}/psc-bestpsf-3fgl.npy'), self.nside),
                norm_mask=self.normalization_mask,
            )
        if 'bub' in temp_list:
            self.temps['bub'] = Template(
                to_nside(np.load(f'{default_data_dir}/template_bub.npy'), self.nside),
                norm_mask=self.normalization_mask
            )
        if 'dsk' in temp_list:
            self.temps['dsk'] = Template(
                to_nside(np.load(f'{default_data_dir}/template_dsk_z1p0.npy'), self.nside),
                norm_mask=self.normalization_mask
            )
        # old NFW template (not used)
        if 'nfw' in temp_list:
            self.temps['nfw'] = Template(
                to_nside(np.load(f'{default_data_dir}/template_nfw_g1p0.npy'), self.nside),
                norm_mask=self.normalization_mask
            )
        if 'nfw' in temp_list:
            self.nfw_temp = NFWTemplate(nside=self.nside)
            self.nfw_temp.set_mask(self.mask_roi_arr[ie])
            if self.nfw_gamma != 'vary':
                gamma = self.nfw_gamma
                self.nfw_temp_fixed = self.nfw_temp.get_NFW2_template(gamma=gamma)[~self.mask_roi_arr[ie]]

        if self.l_max >= 0:
            npix = hp.nside2npix(self.nside)
            theta_ary, phi_ary = hp.pix2ang(self.nside, np.arange(npix))
            Ylm_list = [
                [np.real(Ylm(l, m, theta_ary, phi_ary)) for m in range(-l + 1, l + 1)]
                for l in range(1, self.l_max + 1)
            ]
            self.Ylm_temps = np.array([item for sublist in Ylm_list for item in sublist])
        
        #========== Hybrid (rigid) templates ==========
        self.n_dif_temps = len(self.dif_names)
        if 'pib' in temp_list:
            self.pib_temps = [
                EbinTemplate(
                    to_nside(np.load(f'{ebin_data_dir}/{dif_name}pibrem-{self.temp_class}.npy'), self.nside),
                    norm_mask=self.normalization_mask,
                )
                for dif_name in dif_names
            ]
        if 'ics' in temp_list:
            self.ics_temps = [
                EbinTemplate(
                    to_nside(np.load(f'{ebin_data_dir}/{dif_name}ics-{self.temp_class}.npy'), self.nside),
                    norm_mask=self.normalization_mask,
                )
                for dif_name in dif_names
            ]
        if ('blg' in temp_list) or (self.u_weights == 'blg'):
            self.n_blg_temps = len(self.blg_names)
            self.blg_temps = [
                Template(
                    BulgeTemplates(template_name=blg_name, nside_out=self.nside)(),
                    norm_mask=self.normalization_mask,
                )
                for blg_name in blg_names
            ]

         #========== Variable templates ==========
        if 'nfw' in temp_list:
            self.nfw_temp = NFWTemplate(nside=self.nside)
        if 'dsk' in temp_list:
            self.dsk_temp = LorimerDiskTemplate(nside=self.nside)
        
        #========== sample expand keys ==========
        self.samples_expand_keys = {}
        if 'pib' in temp_list:
            self.samples_expand_keys['pib'] = [f'theta_pib_{n}' for n in self.dif_names]
        if 'ics' in temp_list:
            self.samples_expand_keys['ics'] = [f'theta_ics_{n}' for n in self.dif_names]
        if 'blg' in temp_list:
            self.samples_expand_keys['blg'] = [f'theta_blg_{n}' for n in self.blg_names]

    def load_kernel(self, before_fit = True, params = None):
        # generate kernel parameters
        if before_fit == True:
            scale = self.gp_params[0]
            amp = self.gp_params[1]
            
            if scale == 'float':
                if self.gp_scale_option == 'Linear':
                    scale = numpyro.param('scale', 8. * jnp.ones(()), constraint=dist.constraints.positive)
                elif self.gp_scale_option == 'Cholesky':
                    if scale != 'float':
                        return NotImplementedError
                    scale = numpyro.param('scale', jnp.array([jnp.log(8.),jnp.log(8.),0.]))
                    scale_diag = jnp.exp(scale[:2])
                    scale_off = scale[2:]
            if amp == 'float':
                amp = numpyro.param('amp', jnp.ones(()), constraint=dist.constraints.positive)

            # load kernel
            if self.gp_kernel == 'ExpSquared': # most numerically stable
                unit_kernel = kernels.ExpSquared()
            elif self.gp_kernel == 'RationalQuadratic': # tested
                alpha = self.gp_params[-1]
                if alpha == 'float':
                    alpha = numpyro.param('alpha', jnp.ones(()), constraint=dist.constraints.positive)
                unit_kernel = kernels.RationalQuadratic(alpha = alpha, distance = kernels.distance.L2Distance())
            elif self.gp_kernel == 'Matern32': # untested
                unit_kernel = kernels.Matern32(distance = kernels.distance.L2Distance())
            elif self.gp_kernel == 'Matern52': # untested
                unit_kernel = kernels.Matern52(distance = kernels.distance.L2Distance())

            # load scale transformations
            if self.gp_scale_option == 'Linear':
                scale = jnp.ones(()) / scale # scale is a scale factor that rescales the spatial coords; not the same as the kernel lengthscale
                base_kernel = amp**2. * transforms.Linear(scale, unit_kernel)
            elif self.gp_scale_option == 'Cholesky':
                base_kernel = amp**2. * transforms.Cholesky.from_parameters(scale_diag, scale_off, unit_kernel)

        else:
            # generate kernel parameters
            scale = self.gp_params[0]
            amp = self.gp_params[1]
            
            if scale == 'float':
                scale = params['scale']
            if amp == 'float':
                amp = params['amp']

            # load kernel
            if self.gp_kernel == 'ExpSquared': # most numerically stable
                unit_kernel = kernels.ExpSquared()
            elif self.gp_kernel == 'RationalQuadratic': # tested
                alpha = self.gp_params[2]
                if alpha == 'float':
                    alpha = params['alpha']
                unit_kernel = kernels.RationalQuadratic(alpha = alpha, distance = kernels.distance.L2Distance())
            elif self.gp_kernel == 'Matern32': # untested
                unit_kernel = kernels.Matern32(distance = kernels.distance.L2Distance())
            elif self.gp_kernel == 'Matern52': # untested
                unit_kernel = kernels.Matern52(distance = kernels.distance.L2Distance())

            if self.gp_scale_option == 'Linear':
                scale = jnp.ones(()) / scale # scale is a scale factor that rescales the spatial coords; not the same as the kernel lengthscale
                base_kernel = amp**2. * transforms.Linear(scale, unit_kernel)
            elif self.gp_scale_option == 'Cholesky':
                scale_diag = jnp.exp(scale[:2])
                scale_off = scale[2:]
                base_kernel = amp**2. * transforms.Cholesky.from_parameters(scale_diag, scale_off, unit_kernel)

        return base_kernel
        
    def load_inducing_points(self, before_fit = True, params = None):
        # requires: config_model to set up inducing points
        if self.u_option == 'None':
            xu_f = None
        elif self.u_option == 'float':
            if before_fit == True:
                # NOTE: May be able to limit parameters with plate
                logit_ru, logit_angu = self.initial_u
            
                # generate induced points
                lru = numpyro.param(
                    "lru", logit_ru
                )

                lau = numpyro.param(
                    "lau", logit_angu
                )

                ru = 20. * expit(lru)
                angu = 2. * jnp.pi * expit(lau)

                xu = ru * jnp.cos(angu)
                yu = ru * jnp.sin(angu)

                xu_f = jnp.vstack([xu.T,yu.T]).T
            else:
                lru = params["lru"]
                lau = params["lau"]

                ru = 20. * expit(lru)
                angu = 2. * jnp.pi * expit(lau)

                xu = ru * jnp.cos(angu)
                yu = ru * jnp.sin(angu)

                xu_f = jnp.vstack([xu.T,yu.T]).T
        else:
            xu_f = self.xu_f

        return xu_f

    #========== Configure Model Pre-Numpyro ==========
    def config_model(self, ebin=10):
        
        if ebin == 'all':
            raise NotImplementedError
        else:
            ie = int(ebin)
            
        if 'nfw' in self.temp_list:
            self.nfw_temp = NFWTemplate(nside=self.nside)
            self.nfw_temp.set_mask(self.mask_roi_arr[ie])
            if self.nfw_gamma != 'vary':
                gamma = self.nfw_gamma
                self.nfw_temp_fixed = self.nfw_temp.get_NFW2_template(gamma=gamma)[~self.mask_roi_arr[ie]]
        
        # generate useful quantities from class
        mask = self.mask_roi_arr[ie]
        data = self.counts[ie][~mask]
        nside = self.nside
        Nu = self.Nu
        Np = self.Np
        Nsub = self.Nsub

        if self.is_gp:
            # generate coordinates of healpix bins
            self.x = ef.get_x_from_mask(mask,nside)

            # generate inducing points
            if self.u_option == 'None':
                self.xu_f = None
            elif self.u_option == 'float':
                ru = 20. * jnp.sqrt(jax.random.uniform(jax.random.PRNGKey(55), (Nu,)))
                angu = 2. * jnp.pi * jax.random.uniform(jax.random.PRNGKey(56), (Nu,))

                logit_ru = logit(ru / 20.)
                logit_angu = logit(angu / (2. * jnp.pi))

                self.initial_u = [logit_ru, logit_angu]
            elif self.u_option == 'fixed':
                if self.u_grid_type == 'healpix_bins':
                    if self.u_weights == 'blg': # weight probability of inducing point loc by data
                        if self.is_custom_blg: # probably better placed where blg_temps is defined, whatever for now
                            sim_data_dir = self.sim_data_dir
                            cblg_temp = jnp.asarray(np.load(sim_data_dir + 'custom_blg_' + str(self.custom_blg_id) + '.npy', allow_pickle=True))
                            self.u_weights = cblg_temp[~mask] # zero outside mask, single energy bin
                        else:
                            blg_temps_at_bin = jnp.asarray([blg_temp.at_bin(ie, mask=mask) for blg_temp in self.blg_temps])
                            self.u_weights = jnp.mean(blg_temps_at_bin, axis=0)
                    elif self.u_weights == 'data': # weight probability of inducing point loc by data
                        self.u_weights = data
                    elif self.u_weights == 'uniform': # weight probability of inducing point loc by uniform
                        self.u_weights = None
                    self.xu_f = ef.get_u_from_mask(Nu, mask, grid_type=self.u_grid_type, weights=self.u_weights, nside=nside)
                
                # uniform grids on the circle; made using the ef helper functions (see helper functions for references)
                # made using ../scratch/grid.ipynb in gce-prob-prog-ed-v0.2 main directory
                elif self.u_grid_type == 'square':
                    # Square grid inside circle
                    # The number of grid points inside the circle (Nc) depends discretely on the number of grid points in the square (N)
                    # So, we load the Nc vs Nsq data and instead of calculating it, we just look it up
                    # and use it to define Nu and then convert Nu to Nsq to form the grid.
                    N, Nc = np.load('../utils/Ncirc_vs_Nsq.npy', allow_pickle = True)
                    if self.Nu not in Nc:
                        raise ValueError('Nu must be in Nc. Please see utils/Ncirc_vs_Nsq.txt for valid values of Nu.')
                    else:
                        sN = int(np.sqrt(N[Nc == self.Nu][0]))
                        _, arr_xy = ef.square_mesh(-20, 20, -20, 20, sN, sN)
                        arr_r = np.sqrt(arr_xy[:,0]**2 + arr_xy[:,1]**2)
                        arr_xy = arr_xy[arr_r <= 20]
                        self.xu_f = jnp.array(arr_xy)
                elif self.u_grid_type == 'hex':
                    # Hexagonal grid inside circle
                    # The number of grid points inside the circle (Nhex) depends nontrivially on the spacing parameter d.
                    # So, we load the Nhex vs d data and instead of calculating it, we just look it up
                    # and use it to define Nu and then convert Nu to d to form the grid.
                    N_arr, d_arr = np.load('../utils/Nhex_vs_d.npy', allow_pickle = True)
                    if self.Nu not in N_arr:
                        raise ValueError('Nu must be in Nhex. Please see utils/Nhex_vs_d.txt for valid values of Nu.')
                    else:
                        d = d_arr[N_arr == self.Nu][0]
                        arr_xy = ef.hex_grid(20., d)
                        self.xu_f = jnp.array(arr_xy)
                elif self.u_grid_type == 'sunflower':
                    # Generate grid using sunflower pattern with optional regularization parameter alpha to 
                    # allow points to uniformly fill the boundary of the circle
                    self.xu_f = jnp.array(20. * ef.sunflower(Nu, 0)) # multiply by radius to extend to 20 degree circle

            # generate derivative points
            if self.gp_deriv:
                if self.p_option == 'float':
                    raise NotImplementedError
                elif self.p_option == 'fixed':
                    self.xp_f = ef.get_u_from_mask(Np, mask, grid_type=self.p_grid_type, weights=self.p_weights, nside=nside)
                elif self.p_option == 'match_u':
                    if Np != Nu:
                        raise ValueError('Np must equal Nu if p_option is match_u')
                    self.xp_f = self.xu_f

# ====== Numpyro Implementation of Model ======
    def model(self, ebin=10, gp_rng_key = jax.random.PRNGKey(4242)):
        
        if ebin == 'all':
            raise NotImplementedError
        else:
            ie = int(ebin)
            
        mask = self.mask_roi_arr[ie]
        data = self.counts[ie][~mask]

        # ----------------------------------------------------

        #===== rigid templates processing =====
        # all templates should be already normalized
        if 'iso' in self.temp_list:
            S_iso = numpyro.sample('S_iso', dist.Uniform(1e-3, 50))
        if 'psc' in self.temp_list:
            S_psc = numpyro.sample('S_psc', dist.Uniform(1e-3, 50))
        if 'bub' in self.temp_list:
            S_bub = numpyro.sample('S_bub', dist.Uniform(1e-3, 50))
        
        if self.n_dif_temps > 0:
            if 'pib' in self.temp_list:
                S_pib = numpyro.sample('S_pib', dist.Uniform(1e-3, 100))
            if 'ics' in self.temp_list:
                S_ics = numpyro.sample('S_ics', dist.Uniform(1e-3, 100))
            if self.n_dif_temps > 1:
                if 'pib' in self.temp_list:
                    theta_pib = numpyro.sample("theta_pib", dist.Dirichlet(jnp.ones((self.n_dif_temps,)) / self.n_dif_temps))
                if 'ics' in self.temp_list:
                    theta_ics = numpyro.sample("theta_ics", dist.Dirichlet(jnp.ones((self.n_dif_temps,)) / self.n_dif_temps))

        # ===== variable templates processing =====
        if 'blg' in self.temp_list:
            if self.n_blg_temps == 1:
                S_blg = numpyro.sample('S_blg', dist.Uniform(1e-3, 10))
            else:
                theta_blg = numpyro.sample("theta_blg", dist.Dirichlet(jnp.ones((self.n_blg_temps,)) / self.n_blg_temps))

        if 'nfw' in self.temp_list:
            S_nfw = numpyro.sample('S_nfw', dist.Uniform(1e-3, 50))
            if self.nfw_gamma == 'vary':
                gamma = numpyro.sample("gamma", dist.Uniform(0.2, 2))
#             else:
#                 gamma = self.nfw_gamma

        if 'dsk' in self.temp_list:
            if self.disk_option in ['vary', 'fixed']:
                S_dsk = numpyro.sample('S_dsk', dist.Uniform(1e-3, 5))
                if self.disk_option == 'vary':
                    zs = numpyro.sample("zs", dist.Uniform(0.1, 2.5))
                    C  = numpyro.sample("C",  dist.Uniform(0.05, 15.))

        #===== Gaussian Process Initialization =====
        if self.is_gp:
            x = self.x
            xu_f = self.load_inducing_points(before_fit = True)
            
            # generate derivative points and augment data
            if self.gp_deriv:
                if self.p_option == 'match_u':
                    xp_f = self.xu_f
                    x2 = jnp.concatenate([xp_f,xp_f])
                    d2x = jnp.concatenate([jnp.ones(self.Np),jnp.zeros(self.Np)])
                    d2y = jnp.concatenate([jnp.zeros(self.Np),jnp.ones(self.Np)])
                    xp_aug = jnp.vstack([x2.T,d2x.T,d2y.T]).T
                else:
                    xp_f = self.xp_f
                    x3 = jnp.concatenate([xp_f,xp_f,xp_f])
                    d3x = jnp.concatenate([jnp.zeros(self.Np),jnp.ones(self.Np),jnp.zeros(self.Np)])
                    d3y = jnp.concatenate([jnp.zeros(self.Np),jnp.zeros(self.Np),jnp.ones(self.Np)])
                    xp_aug = jnp.vstack([x3.T,d3x.T,d3y.T]).T

            # load kernel parameters
            base_kernel = self.load_kernel(before_fit = True)
            
            # load derivative kernel or base kernel
            if self.u_option == 'None':
                gp = GaussianProcess(base_kernel, x, diag=1e-3) # p(x)
                log_rate = numpyro.sample("log_rate", gp.numpyro_dist())
            else:
                if self.gp_deriv:
                    kernel = ef.DerivativeKernel(base_kernel)

                    # augment the induced points
                    dxu = jnp.zeros(self.Nu)
                    dyu = jnp.zeros(self.Nu)
                    xu_aug = jnp.vstack([xu_f.T,dxu.T,dyu.T]).T

                    gp_u = GaussianProcess(kernel, xu_aug, diag=1e-3) # p(u)
                else:
                    kernel = base_kernel
                    gp_u = GaussianProcess(kernel, xu_f, diag=1e-3) # p(u)
                
                log_rate_u = numpyro.sample("log_rate_u", gp_u.numpyro_dist())

        # defining the likelihood with subsampling
        with numpyro.plate('data', size=len(data), dim=-1, subsample_size=self.Nsub) as ind:
            # initialize array of counts
            if self.Nsub == None:
                mu = jnp.zeros(len(data))
            else:
                mu = jnp.zeros(self.Nsub)
            
            #===== rigid templates =====
            # all templates should be already normalized
            if 'iso' in self.temp_list:
                mu += S_iso * jnp.asarray(self.temps['iso'].at_bin(ie, mask=mask))[ind]
            if 'psc' in self.temp_list:
                mu += S_psc * jnp.asarray(self.temps['psc'].at_bin(ie, mask=mask))[ind]
            if 'bub' in self.temp_list:
                mu += S_bub * jnp.asarray(self.temps['bub'].at_bin(ie, mask=mask))[ind]
                
            #===== hybrid templates =====
            # all templates should be already normalized
            if self.n_dif_temps > 0:
                if self.n_dif_temps == 1:
                    if 'pib' in self.temp_list:
                        mu += S_pib * self.pib_temps[0].at_bin(ie, mask=mask)[ind]
                    if 'ics' in self.temp_list:
                        mu += S_ics * self.ics_temps[0].at_bin(ie, mask=mask)[ind]
                else:
                    if 'pib' in self.temp_list:
                        pib_temps_at_bin = jnp.asarray([pib_temp.at_bin(ie, mask=mask) for pib_temp in self.pib_temps])
                        mu += S_pib * jnp.dot(theta_pib, pib_temps_at_bin)[ind]
                    if 'ics' in self.temp_list:
                        ics_temps_at_bin = jnp.asarray([ics_temp.at_bin(ie, mask=mask) for ics_temp in self.ics_temps])
                        mu += S_ics * jnp.dot(theta_ics, ics_temps_at_bin)[ind]
                
                if 'blg' in self.temp_list:
                    if self.n_blg_temps == 1:
                        mu += S_blg * self.blg_temps[0].at_bin(ie, mask=mask)[ind]
                    else:
                        blg_temps_at_bin = jnp.asarray([blg_temp.at_bin(ie, mask=mask) for blg_temp in self.blg_temps])
                        mu += S_blg * jnp.dot(theta_blg, blg_temps_at_bin)[ind]
                
            #===== variable templates =====
            if 'nfw' in self.temp_list:
                if self.nfw_gamma == 'vary':
                    mu += S_nfw * self.nfw_temp.get_NFW2_template(gamma=gamma)[~mask][ind]
                else:
                    mu += S_nfw * self.nfw_temp_fixed[ind]
            
            if 'dsk' in self.temp_list:
                if self.disk_option in ['vary', 'fixed']:
                    if self.disk_option == 'vary':
                        temp_dsk = self.dsk_temp.get_template(zs=zs, C=C)[~mask]
                    else:
                        temp_dsk = self.temps['dsk'].at_bin(ie, mask=mask)
                    mu += S_dsk * temp_dsk[ind]

            #===== Gaussian Process =====
            if self.is_gp:
                x_sub = x[ind] # load angular coordinates of bins
                if self.u_option == 'None':
                    # if self.Nsub != None:
                    #     raise ValueError('Subsampling is not supported for Vanilla GP')
                    # elif self.gp_deriv:
                    #     raise NotImplementedError
                    # else:
                    mu += jnp.exp(log_rate)[ind]
                
                else:
                    if self.gp_deriv: # augment bins for derivative GP
                        # augment data
                        dx = jnp.zeros(self.Nsub)
                        dy = jnp.zeros(self.Nsub)
                        x_sub = jnp.vstack([x_sub.T,dx.T,dy.T]).T

                    sample_keys = jax.random.split(gp_rng_key, 3)
                    key, key_x, key_xp = sample_keys

                    _, gp_x = gp_u.condition(log_rate_u, x_sub, diag=1e-2) # p(x|u)
                    log_rate = gp_x.sample(key_x)
                    mu += jnp.exp(log_rate)

                    if self.gp_deriv:
                        # NOTE: Strictly speaking should condition on gp_x or gp_x_u, but gp_u more numerically stable
                        # _, gp_cond = gp_x.condition(log_rate, xp_aug, diag = 1e-3)
                        _, gp_cond = gp_u.condition(log_rate_u, xp_aug, diag=1e-3) # condition GP on log_rate and sample derivatives at x_aug[Nx:]
                        log_rate_xp_aug = gp_cond.sample(key_xp) # log_rate_deriv
                        
                        if self.p_option == 'match_u':
                            log_rate_xp = log_rate_u
                            log_rate_xp_px = log_rate_xp_aug[:self.Np]
                            log_rate_xp_py = log_rate_xp_aug[self.Np:]
                        else:
                            Np2 = self.Np * 2
                            log_rate_xp = log_rate_xp_aug[:self.Np]
                            log_rate_xp_px = log_rate_xp_aug[self.Np:Np2]
                            log_rate_xp_py = log_rate_xp_aug[Np2:]
                        
                        rate_xp = jnp.exp(log_rate_xp) 
                        rate_xp_px = rate_xp * log_rate_xp_px # rate_deriv_x
                        rate_xp_py = rate_xp * log_rate_xp_py # rate_deriv_y

                        x_mag = jnp.sqrt(jnp.sum(xp_f**2., axis=1))
                        x_hat = xp_f / x_mag[:,None]
                        rate_p = rate_xp_px * x_hat[:,0] + rate_xp_py * x_hat[:,1]
                        numpyro.factor("constraint", -self.monotonicity_hyperparameter * jnp.sum(jnp.where(rate_p > 0, rate_p, 0.))) 

            # ===== Poisson likelihood =====
            numpyro.factor('log_likelihood', log_like_poisson(mu, data[ind])) # L(x|u)

    #========== SVI ==========
    # NOTE: added option to optim argument to allow for different optimizers
    def fit_SVI(
        self, rng_key=jax.random.PRNGKey(42),
        guide='iaf', optimizer=None, num_flows=3, hidden_dims=[64, 64],
        n_steps=5000, lr=0.006, num_particles=8, progress_bar = True,
        **model_static_kwargs,
    ):
        if guide == 'mvn':
            self.guide = autoguide.AutoMultivariateNormal(self.model)
        elif guide == 'iaf':
            self.guide = autoguide.AutoIAFNormal(
                self.model,
                num_flows=num_flows,
                hidden_dims=hidden_dims,
                nonlinearity=stax.Tanh
            )
        elif guide == 'iaf_mixture':
            num_base_mixture = 8
            class AutoIAFMixture(autoguide.AutoIAFNormal):
                def get_base_dist(self):
                    C = num_base_mixture
                    mixture = dist.MixtureSameFamily(
                        dist.Categorical(probs=jnp.ones(C) / C),
                        dist.Normal(jnp.arange(float(C)), 1.)
                    )
                    return mixture.expand([self.latent_dim]).to_event()
            self.guide = AutoIAFMixture(
                self.model,
                num_flows=num_flows,
                hidden_dims=hidden_dims,
                nonlinearity=stax.Tanh
            )
        else:
            raise NotImplementedError

        if optimizer == None:
            optimizer = optim.optax_to_numpyro(
                optax.chain(
                    optax.clip(1.),
                    optax.adam(lr),
                )
            )

        svi = SVI(
            self.model, self.guide, optimizer,
            Trace_ELBO(num_particles=num_particles),
            **model_static_kwargs,
        )
        self.svi_results = svi.run(rng_key, n_steps, progress_bar=progress_bar)
        self.svi_model_static_kwargs = model_static_kwargs
        
        return self.svi_results

    def get_svi_samples(self, rng_key=jax.random.PRNGKey(42), num_samples=50000, expand_samples=True):
        
        rng_key, key = jax.random.split(rng_key)
        self.svi_samples = self.guide.sample_posterior(
            rng_key=rng_key,
            params=self.svi_results.params,
            sample_shape=(num_samples,)
        )
        
        if expand_samples:
            self.svi_samples = self.expand_samples(self.svi_samples)
            
        return self.svi_samples

    # NOTE: Can probably achieve this using Predictive
    def get_gp_samples(self, num_samples=1000, custom_mask = None):
        nside = self.nside
        Nu = self.Nu
        svi_results = self.svi_results
        params = svi_results.params
        samples = self.svi_samples

        if custom_mask is None:
            mask = self.mask_roi_arr[10]
        else:
            mask = custom_mask
        x_p = ef.get_x_from_mask(mask,nside) # predicted x given sampled u
        
        if self.is_gp:
            if self.u_option == 'None':
                print('Samples are already produced by the guide itself.')
                raise NotImplementedError
            xu_f = self.load_inducing_points(before_fit = False, params = params)

            # load kernel with best-fit GP parameters
            base_kernel = self.load_kernel(before_fit = False, params = params)
            gp_u = GaussianProcess(base_kernel, xu_f, diag=1e-3)

            samples_u = samples['log_rate_u']

            # NOTE: Cannot load GP as input, so need to define sampling functions after
            # defining the GP

            # sub: 'Subset'
            # int: 'interpolate'
            @jax.jit
            def create_gp_sample_(n,x,samples,key):
                log_rate_sub = samples.at[n].get()
                _, gp_int = gp_u.condition(log_rate_sub, x, diag=1e-3)
                log_rate_int = gp_int.sample(jax.random.PRNGKey(key))
                return log_rate_int
            
            def create_samples_int_(num_samples,x,samples,keys_ind):
                for n in tqdm.tqdm(range(num_samples)):
                    key = keys_ind[n]
                    if n == 0:
                        samples_int = create_gp_sample_(n,x,samples,key)
                    else:
                        samples_int = np.vstack((samples_int, create_gp_sample_(n,x,samples,key)))
                return samples_int

            keys_ind = np.random.randint(low=0,high=1000000,size=num_samples + 1)
            self.gp_samples = create_samples_int_(num_samples,x_p,samples_u,keys_ind)
            return self.gp_samples
        else:
            raise ValueError('GP is not enabled.')

    #========== Custom SVI Implementation (Debug) ==========
    # NOTE: added option to optim argument to allow for different optimizers
    def cfit_SVI(
        self, rng_key=jax.random.PRNGKey(42),
        guide='iaf', optimizer=None, num_flows=3, hidden_dims=[64, 64],
        n_steps=5000, lr=0.006, num_particles=8, progress_bar = True,
        early_stop = np.inf, record_states = True,
        **model_static_kwargs,
    ):
        self.guide_name = guide
        if guide == 'mvn':
            self.guide = autoguide.AutoMultivariateNormal(self.model)
        elif guide == 'iaf':
            self.guide = autoguide.AutoIAFNormal(
                self.model,
                num_flows=num_flows,
                hidden_dims=hidden_dims,
                nonlinearity=stax.Tanh
            )
        elif guide == 'iaf_mixture':
            num_base_mixture = 8
            class AutoIAFMixture(autoguide.AutoIAFNormal):
                def get_base_dist(self):
                    C = num_base_mixture
                    mixture = dist.MixtureSameFamily(
                        dist.Categorical(probs=jnp.ones(C) / C),
                        dist.Normal(jnp.arange(float(C)), 1.)
                    )
                    return mixture.expand([self.latent_dim]).to_event()
            self.guide = AutoIAFMixture(
                self.model,
                num_flows=num_flows,
                hidden_dims=hidden_dims,
                nonlinearity=stax.Tanh
            )
        else:
            raise NotImplementedError
        
        # initialize svi using new patched code TODO: Update the rest of the pipeline (especially gp2temp/step2 to account for this change)
        svi, svi_state = self.init_svi(rng_key = rng_key, optimizer = optimizer, lr = lr, num_particles = num_particles)

        self.svi_results = self.svi_loop(svi, svi_state, progress_bar = progress_bar, num_steps = n_steps, rng_key = rng_key, early_stop = early_stop, 
                                         record_states = record_states, **model_static_kwargs)
        self.svi_model_static_kwargs = model_static_kwargs
        
        return self.svi_results
    
    # patch (02/16/2024)
    ### Fixes issue where loaded svi_results does not produce accurate samples
    ### using Predictive or guide.sample_posterior
    ### For issue: see v0.2/.../1d_gaussian_example/step_1_test.ipynb
    ###                v0.2/.../1d_gaussian_example/step_1_test_load.ipynb
    ### Basically, loading the guide and giving it the appropriate params does not give the accurate posterior
    ### You need to initialize the SVI again (it appears you don't need to run it again or load it at the best fit state)
    def init_svi(self, rng_key = jax.random.PRNGKey(0), optimizer = None, lr = 0.006, num_particles = 16, ebin = 10):
        if optimizer == None:
            optimizer = optim.optax_to_numpyro(
                optax.chain(
                    optax.clip(1.),
                    optax.adam(lr), # replaced adam for vanilla GP fit
                )
            )
        
        svi = SVI(
            self.model, self.guide, optimizer,
            Trace_ELBO(num_particles=num_particles),
        )

        # define initial svi state
        rng_key, key, gp_key = jax.random.split(rng_key, 3)
        svi_state = svi.init(key, ebin, gp_key)
        return svi, svi_state
    
    def svi_loop(self, svi, svi_state, progress_bar = True, num_steps = 1000, rng_key = jax.random.PRNGKey(0), early_stop = np.inf,
                   ebin = 10, record_states = True):
        # update function
        def body_fn(svi_state, _):
            gp_rng_key = jax.random.split(svi_state.rng_key)[-1]
            svi_state, loss = svi.update(svi_state, ebin, gp_rng_key)
            return svi_state, loss
        
        # initialize lists of losses
        recorded_steps = []
        recorded_params = []
        recorded_states = []
        losses = [] 
        min_loss = np.inf
        
        # training loop
        if progress_bar:
            with tqdm.trange(1, num_steps + 1) as t:
                batch = max(num_steps // 200, 1)
                for i in t:
                    # with jax.checking_leaks():
                    svi_state, loss = jit(body_fn)(svi_state, None)
                    losses.append(loss) 
                    
                    rec_batch = num_steps // 400 # needs to be much lower for van GP
                    if record_states == True:
                        if i % rec_batch == 0:
                            recorded_steps.append(i)
                            recorded_params.append(svi.get_params(svi_state))
                            recorded_states.append(svi_state)

                    if loss < min_loss:
                        min_loss = loss
                        min_svi_state = svi_state
                        min_step = i
                        
                    if abs(min_step - i) > early_stop:
                        print('Stopped Early at Step ' + str(i))
                        break

                    if i % batch == 0:
                        avg_loss = sum(losses[i - batch :]) / batch
                        t.set_postfix_str(
                            "init loss: {:.4f}, min loss {:.4f}, avg loss [{}-{}]: {:.4f}".format(
                                losses[0], min_loss, i - batch + 1, i, avg_loss
                            ),
                            refresh=False,
                        )
        else:
            for i in tqdm.tqdm(range(num_steps)):
                svi_state, loss = jit(body_fn)(svi_state, None)
                losses.append(loss)

                if loss < min_loss:
                    min_loss = loss
                    min_svi_state = svi_state
                    min_step = i
                        
                if abs(min_step - i) > early_stop:
                    print('Stopped Early at Step ' + str(i))
                    break

                rec_batch = num_steps // 100
                if i % rec_batch == 0:
                    recorded_steps.append(i)
                    recorded_params.append(svi.get_params(svi_state))
                    recorded_states.append(svi_state)
                    
        losses = jnp.stack(losses) 

        # Report the final values of the variational parameters
        # in the guide after training.
        params = svi.get_params(svi_state)
        min_params = svi.get_params(min_svi_state)
        
        return SVIRunResult(params, svi_state, losses,
                            min_params, min_svi_state, min_loss,
                            recorded_steps, recorded_params, recorded_states)

    def recorded_log_likelihoods_1(self, rng_key=jax.random.PRNGKey(10), num_samples=16, svi_results = None):
        # development: v0.2/nfw_recovery/1d_gaussian_example/step_2_sidd_debug.ipynb

        ie = self.ebin
        guide = self.guide
        if svi_results is None:
            svi_results = self.svi_results
        mask = self.mask_roi_arr[ie]
        data = self.counts[ie][~mask]

        # calculate the log_likelihood in a vectorizable way
        # will be jitted later
        def body_fn(params, rng_key):
            # split keys for processes
            rng_key, pred_key, gp_key  = jax.random.split(rng_key, 3)

            # load template posterior
            temp_pred = Predictive(self.guide, num_samples = num_samples, params = params)
            temp_samples = temp_pred(pred_key, ie)

            mu = jnp.zeros(len(data))
            
            #===== rigid templates =====
            # all templates should be already normalized
            if 'iso' in self.temp_list:
                mu += temp_samples['S_iso'][:,None] * jnp.asarray(self.temps['iso'].at_bin(ie, mask=mask))[None,:]
            if 'psc' in self.temp_list:
                mu += temp_samples['S_psc'][:,None] * jnp.asarray(self.temps['psc'].at_bin(ie, mask=mask))[None,:]
            if 'bub' in self.temp_list:
                mu += temp_samples['S_bub'][:,None] * jnp.asarray(self.temps['bub'].at_bin(ie, mask=mask))[None,:]
                
            #===== hybrid templates =====
            # all templates should be already normalized
            if self.n_dif_temps > 0:
                if self.n_dif_temps == 1:
                    if 'pib' in self.temp_list:
                        mu += temp_samples['S_pib'][:,None] * self.pib_temps[0].at_bin(ie, mask=mask)[None,:]
                    if 'ics' in self.temp_list:
                        mu += temp_samples['S_ics'][:,None] * self.ics_temps[0].at_bin(ie, mask=mask)[None,:]
                else:
                    if 'pib' in self.temp_list:
                        pib_temps_at_bin = jnp.asarray([pib_temp.at_bin(ie, mask=mask) for pib_temp in self.pib_temps])
                        mu += temp_samples['S_pib'][:,None] * jnp.dot(temp_samples['theta_pib'], pib_temps_at_bin)[None,:]
                    if 'ics' in self.temp_list:
                        ics_temps_at_bin = jnp.asarray([ics_temp.at_bin(ie, mask=mask) for ics_temp in self.ics_temps])
                        mu += temp_samples['S_ics'][:,None] * jnp.dot(temp_samples['theta_ics'], ics_temps_at_bin)[None,:]
                
                if 'blg' in self.temp_list:
                    if self.n_blg_temps == 1:
                        mu += temp_samples['S_blg'][:,None] * self.blg_temps[0].at_bin(ie, mask=mask)[None,:]
                    else:
                        blg_temps_at_bin = jnp.asarray([blg_temp.at_bin(ie, mask=mask) for blg_temp in self.blg_temps])
                        mu += temp_samples['S_blg'][:,None] * jnp.dot(temp_samples['theta_blg'], blg_temps_at_bin)[None,:]
                
            #===== variable templates =====
            if 'nfw' in self.temp_list:
                if self.nfw_gamma == 'vary':
                    get_NFW2_template_vec = jax.vmap(self.nfw_temp.get_NFW2_template) # shape is correct
                    mu += temp_samples['S_nfw'][:,None] * get_NFW2_template_vec(gamma=temp_samples['gamma'])[:,~mask]
                else:
                    mu += temp_samples['S_nfw'][:,None] * self.nfw_temp_fixed[None,:]
            
            if 'dsk' in self.temp_list:
                if self.disk_option in ['vary', 'fixed']:
                    if self.disk_option == 'vary':
                        temp_dsk = self.dsk_temp.get_template(zs=zs, C=C)[~mask]
                    else:
                        temp_dsk = self.temps['dsk'].at_bin(ie, mask=mask)
                    mu += temp_samples['S_dsk'][:,None] * temp_dsk[None,:]

            if self.is_gp:
                if self.u_option == 'None':
                    mu += jnp.exp(temp_samples['log_rate'])
                else:
                    mu += jnp.exp(self.cget_gp_samples_vec(gp_key, num_samples, svi_results))

            # load function values and compute log likelihood
            # print(log_like_poisson(rate, y).shape)
            ll = jnp.sum(log_like_poisson(mu, data)) / num_samples

            return ll, rng_key

        # load list of log_likelihoods, rng_key, and num_samples use to estime the log_likelihood
        ll_list = []
        for n in tqdm.tqdm(range(len(svi_results.recorded_steps))):
            params = svi_results.recorded_params[n]
            ll, rng_key = jit(body_fn)(params, rng_key)
            ll_list.append(ll)

        return ll_list
    
    def recorded_log_likelihoods(self, rng_key=jax.random.PRNGKey(10), num_samples=16):
        # development: v0.2/nfw_recovery/1d_gaussian_example/step_2_sidd_debug.ipynb

        ie = self.ebin
        pred = self.pred
        guide = self.guide
        svi_results = self.svi_results
        params = self.params
        x = self.x
        mask = self.mask_roi_arr[ie]
        u_option = self.u_option

        # generate data in a vectorizable way
        # key is to vectorize the operation with respect to a vector of rng_keys
        def generate_data(gp_rng_key):
            if u_option == 'None':
                gp_key, poiss_key = jax.random.split(gp_rng_key, 2)
                lam = jnp.exp(pred(gp_key, ie)['log_rate']).T
                y = poisson(poiss_key, lam)
                y = jnp.squeeze(y, axis = -1) # shape mismatch otherwise (shape_ll ~ (shape_x, shape_x))
            else:
                xu_f = self.load_inducing_points(before_fit = False, params = params)

                # generate kernel parameters
                base_kernel = self.load_kernel(before_fit = False, params = params)
                gp_u = GaussianProcess(base_kernel, xu_f, diag=1e-3)

                gp_u_key, gp_key, poiss_key = jax.random.split(gp_rng_key, 3)
                log_rate_u = pred(gp_u_key, ie)['log_rate_u'].T
                log_rate_u = jnp.squeeze(log_rate_u, axis = -1) # shape mismatch otherwise (shape_ll ~ (shape_x, shape_x))
                _, gp_x = gp_u.condition(log_rate_u, x, diag=1e-3) # p(x|u)
                rate = jnp.exp(gp_x.sample(gp_key))
                y = poisson(poiss_key, rate)
            return y

        generate_data_vec = jax.vmap(generate_data)

        # calculate the log_likelihood in a vectorizable way
        # will be jitted later
        def body_fn(params, rng_key):
            # split keys for processes
            rng_key, pred_key, gp_key  = jax.random.split(rng_key, 3)

            # load template posterior
            temp_pred = Predictive(self.guide, num_samples = num_samples, params = params)
            temp_samples = temp_pred(pred_key, ie)

            mu = jnp.zeros(len(x))
            
            #===== rigid templates =====
            # all templates should be already normalized
            if 'iso' in self.gp_temp_list:
                mu += temp_samples['S_iso'][:,None] * jnp.asarray(self.temps['iso'].at_bin(ie, mask=mask))[None,:]
            if 'psc' in self.gp_temp_list:
                mu += temp_samples['S_psc'][:,None] * jnp.asarray(self.temps['psc'].at_bin(ie, mask=mask))[None,:]
            if 'bub' in self.gp_temp_list:
                mu += temp_samples['S_bub'][:,None] * jnp.asarray(self.temps['bub'].at_bin(ie, mask=mask))[None,:]
                
            #===== hybrid templates =====
            # all templates should be already normalized
            if self.n_dif_temps > 0:
                if self.n_dif_temps == 1:
                    if 'pib' in self.gp_temp_list:
                        mu += temp_samples['S_pib'][:,None] * self.pib_temps[0].at_bin(ie, mask=mask)[None,:]
                    if 'ics' in self.gp_temp_list:
                        mu += temp_samples['S_ics'][:,None] * self.ics_temps[0].at_bin(ie, mask=mask)[None,:]
                else:
                    if 'pib' in self.gp_temp_list:
                        pib_temps_at_bin = jnp.asarray([pib_temp.at_bin(ie, mask=mask) for pib_temp in self.pib_temps])
                        mu += temp_samples['S_pib'][:,None] * jnp.dot(temp_samples['theta_pib'], pib_temps_at_bin)[None,:]
                    if 'ics' in self.gp_temp_list:
                        ics_temps_at_bin = jnp.asarray([ics_temp.at_bin(ie, mask=mask) for ics_temp in self.ics_temps])
                        mu += temp_samples['S_ics'][:,None] * jnp.dot(temp_samples['theta_ics'], ics_temps_at_bin)[None,:]
                
                if 'blg' in self.gp_temp_list:
                    if self.n_blg_temps == 1:
                        mu += temp_samples['S_blg'][:,None] * self.blg_temps[0].at_bin(ie, mask=mask)[None,:]
                    else:
                        blg_temps_at_bin = jnp.asarray([blg_temp.at_bin(ie, mask=mask) for blg_temp in self.blg_temps])
                        mu += temp_samples['S_blg'][:,None] * jnp.dot(temp_samples['theta_blg'], blg_temps_at_bin)[None,:]
                
            #===== variable templates =====
            if 'nfw' in self.gp_temp_list:
                if self.nfw_gamma == 'vary':
                    get_NFW2_template_vec = jax.vmap(self.nfw_temp.get_NFW2_template) # shape is correct
                    mu += temp_samples['S_nfw'][:,None] * get_NFW2_template_vec(gamma=temp_samples['gamma'])[:,~mask]
                else:
                    mu += temp_samples['S_nfw'][:,None] * self.nfw_temp_fixed[None,:]
            
            if 'dsk' in self.gp_temp_list:
                if self.disk_option in ['vary', 'fixed']:
                    if self.disk_option == 'vary':
                        temp_dsk = self.dsk_temp.get_template(zs=zs, C=C)[~mask]
                    else:
                        temp_dsk = self.temps['dsk'].at_bin(ie, mask=mask)
                    mu += temp_samples['S_dsk'][:,None] * temp_dsk[None,:]

            # generate keys for gp samples
            gp_rng_key_vec = jax.random.split(gp_key, num_samples)

            # compute log-likelihood w.r.t. random dataset
            y = generate_data_vec(gp_rng_key_vec)

            # load function values and compute log likelihood
            # print(log_like_poisson(rate, y).shape)
            ll = jnp.sum(log_like_poisson(mu, y)) / num_samples

            return ll, rng_key

        # load list of log_likelihoods, rng_key, and num_samples use to estime the log_likelihood
        ll_list = []
        for n in tqdm.tqdm(range(len(svi_results.recorded_steps))):
            params = svi_results.recorded_params[n]
            ll, rng_key = jit(body_fn)(params, rng_key)
            ll_list.append(ll)

        return ll_list
    
    def recorded_quants(self, rng_key=jax.random.PRNGKey(10), num_samples=10000, normalized = False, 
                        true_temp_params = None, svi_results = None):
        # development: v0.2/nfw_recovery/1d_gaussian_example/step_2_sidd_debug.ipynb

        ie = self.ebin
        if svi_results is None:
            svi_results = self.svi_results

        # generate samples for jitting
        def sample_gen(params, rng_key):
            # split keys for processes
            rng_key, pred_key, gp_key  = jax.random.split(rng_key, 3)

            # load template posterior
            temp_pred = Predictive(self.guide, num_samples = num_samples, params = params)
            temp_samples = temp_pred(pred_key, ie)

            return temp_samples

        # load list of log_likelihoods, rng_key, and num_samples use to estime the log_likelihood
        temp_q_list = []
        for n in tqdm.tqdm(range(len(svi_results.recorded_steps))):
            rng_key, key = jax.random.split(rng_key)
            params = svi_results.recorded_params[n]
            temp_samples = jit(sample_gen)(params, key)
            temp_qs = {}
            for k in list(temp_samples.keys()):
                if k in ['_auto_latent', 'log_rate_u', 'log_rate']:
                    pass
                else:
                    if normalized == False:
                        temp_qs[k] = np.quantile(temp_samples[k], q = [0.05, 0.5, 0.95], axis = 0)
                    else:
                        if true_temp_params is None:
                            raise ValueError('true_temp_params must be specified if normalized == True')
                        else:
                            # note, using numpy since jax numpy is failing
                            temp_qs[k] = np.quantile( (temp_samples[k] - true_temp_params[k]) / (true_temp_params[k]), q = [0.05, 0.5, 0.95], axis = 0)
            temp_q_list.append(temp_qs)

        return temp_q_list
    
    def cget_svi_samples(self, rng_key=jax.random.PRNGKey(42), num_samples=50000, expand_samples=True, 
                         min_loss = False):
        # NOTE: Using sample_posterior twice failed previously
        # If you get tracedarray conversion errors, consider generating samples
        # using NumPyro Predictive fct (see gce-prob-prog-ed-v0.2/custom_svi_run/1d_svgp_bootval.ipynb)
        if min_loss == False:
            params = self.svi_results.params
        else:
            params = self.svi_results.min_params
        
        rng_key, key = jax.random.split(rng_key)
        self.svi_samples = self.guide.sample_posterior(
            rng_key=rng_key,
            params=params,
            sample_shape=(num_samples,)
        )
        if expand_samples:
            self.svi_samples = self.expand_samples(self.svi_samples)

        return self.svi_samples
    
    def expand_samples(self, samples):
        new_samples = {}
        for k in samples.keys():
            if k in self.samples_expand_keys:
                for i in range(samples[k].shape[-1]):
                    new_samples[self.samples_expand_keys[k][i]] = samples[k][...,i]
            elif k in ['auto_shared_latent']:
                pass
            else:
                new_samples[k] = samples[k]
        return new_samples
        
    # NOTE: Can probably achieve this using Predictive
    def cget_gp_samples(self, svi_results, samples, num_samples=1000, custom_mask = None, min_loss = False):
        nside = self.nside
        Nu = self.Nu

        if custom_mask is None:
            mask = self.mask_roi_arr[10]
        else:
            mask = custom_mask
        x_p = ef.get_x_from_mask(mask,nside) # predicted x given sampled u
        
        if min_loss:
            params = svi_results.min_params
        else:
            params = svi_results.params

        if self.is_gp:
            if self.u_option == 'None':
                print('Samples are already produced by the guide itself.')
                raise NotImplementedError
            xu_f = self.load_inducing_points(before_fit = False, params = params)
                
            # load kernel with best-fit GP parameters
            base_kernel = self.load_kernel(before_fit = False, params = params)
            gp_u = GaussianProcess(base_kernel, xu_f, diag=1e-3)

            samples_u = samples['log_rate_u']

            # NOTE: Cannot load GP as input, so need to define sampling functions after
            # defining the GP

            # sub: 'Subset'
            # int: 'interpolate'
            @jax.jit
            def create_gp_sample_(n,x,samples,key):
                log_rate_sub = samples.at[n].get()
                _, gp_int = gp_u.condition(log_rate_sub, x, diag=1e-3)
                log_rate_int = gp_int.sample(jax.random.PRNGKey(key))
                return log_rate_int
            
            def create_samples_int_(num_samples,x,samples,keys_ind):
                for n in tqdm.tqdm(range(num_samples)):
                    key = keys_ind[n]
                    if n == 0:
                        samples_int = create_gp_sample_(n,x,samples,key)
                    else:
                        samples_int = np.vstack((samples_int, create_gp_sample_(n,x,samples,key)))
                return samples_int

            keys_ind = np.random.randint(low=0,high=1000000,size=num_samples + 1)
            self.gp_samples = create_samples_int_(num_samples,x_p,samples_u,keys_ind)
            return self.gp_samples
        else:
            raise ValueError('GP is not enabled.')

    # replacement to cget_gp_samples
    # TODO: Super redundant again, generate_data appears in 3 different functions
    # Would want to clean this code up and remove all the redundancies
    # For now, just pressing alone
    def cget_gp_samples_vec(self, rng_key, num_samples, svi_results = None):
        # development: v0.2/nfw_recovery/1d_gaussian_example/step_2_sidd_debug.ipynb
        # TODO: NOTE: You just made a much faster (vectorized) GP sampler, so you can use this 
        # to generate many gp samples fast from the inducing point GP (if necessary)
        # NOTE: would be more efficient if you use Predictive to only load the GP samples
        ## but, assume it is not the bottleneck in the pipeline
        # NOTE: If you want to use these samples to study correlations, will need to sample
        # templates together with the GP

        ie = self.ebin
        guide = self.guide
        if svi_results is None:
            svi_results = self.svi_results
        pred = self.pred
        params = svi_results.params
        x = self.x
        mask = self.mask_roi_arr[ie]
        u_option = self.u_option

        # generate keys for gp samples
        gp_rng_key_vec = jax.random.split(rng_key, num_samples)

        # generate data in a vectorizable way
        # key is to vectorize the operation with respect to a vector of rng_keys
        def generate_data(gp_rng_key):
            if u_option == 'None':
                gp_key, poiss_key = jax.random.split(gp_rng_key, 2)
                lam = jnp.exp(pred(gp_key, ie)['log_rate']).T
                y = poisson(poiss_key, lam)
                y = jnp.squeeze(y, axis = -1) # shape mismatch otherwise (shape_ll ~ (shape_x, shape_x))
            else:
                xu_f = self.load_inducing_points(before_fit = False, params = params)

                # generate kernel parameters with best-fit GP parameters
                base_kernel = self.load_kernel(before_fit = False, params = params)
                gp_u = GaussianProcess(base_kernel, xu_f, diag=1e-3)

                gp_u_key, gp_key, poiss_key = jax.random.split(gp_rng_key, 3)
                log_rate_u = pred(gp_u_key, ie)['log_rate_u'].T
                log_rate_u = jnp.squeeze(log_rate_u, axis = -1) # shape mismatch otherwise (shape_ll ~ (shape_x, shape_x))
                _, gp_x = gp_u.condition(log_rate_u, x, diag=1e-3) # p(x|u)
                log_rate = gp_x.sample(gp_key)
            return log_rate

        generate_data_vec = jax.vmap(generate_data)

        def body_fn():
            gp_rng_key_vec = jax.random.split(rng_key, num_samples)
            log_rate = generate_data_vec(gp_rng_key_vec)
            return log_rate
        
        return jit(body_fn)()
        
    def predictive(self, guide, num_samples = 1, params = None):
        # generate predictive distribution of guide from a fit
        pred = Predictive(guide, num_samples = num_samples, params = params)
        self.pred = pred

    # smarter solution would be to just use one fit_SVI function
    # for both fits; just need to be careful about how the guides and models
    # are defined (might break backwards-compatibility)
    def config_temp_model(self, gp_temp_list = None, params = None, guide = 'mvn', ebin = 10):

        if ebin == 'all':
            raise NotImplementedError
        else:
            ie = int(ebin)
    
        # load best-fit parameters specifying the guide from a previous fit
        if self.svi_results is None:
            if params is None:
                raise ValueError('SVI must be run first or guide params must be provided.')
            self.params = params
            
            if guide == 'mvn':
                self.gp_guide = autoguide.AutoMultivariateNormal(self.model)
            else:
                raise NotImplementedError

        else:
            self.params = self.svi_results.params
            self.gp_guide = self.guide

        # swap roles for fit function
        self.gp_model = self.model
        self.model = self.temp_model

        # token svi initialization (doesn't need to be accurate, just needs to be initialized) 
        # necessary for accurate predictive distribution
        # see patch from 02/16/2024
        rng_key = jax.random.PRNGKey(0)
        rng_key, key, gp_key = jax.random.split(rng_key, 3)
        optim = numpyro.optim.Adam(0.01)
        svi = SVI(self.gp_model, self.gp_guide, optim, Trace_ELBO(20))
        svi_state = svi.init(key, ebin, gp_key)

        # generate predictive distribution
        self.predictive(self.gp_guide, num_samples = 1, params = self.params)

        # load names of templates to fit to gp
        if gp_temp_list is None and self.temp_names_sim is not None:
            gp_temp_list = [temp for temp in self.temp_names_sim if temp not in self.temp_list]
        else:
            raise ValueError('gp_temp_list must be specified.')
        self.gp_temp_list = gp_temp_list

        # initialize templates that gp models (so are not included in the first model)
        self.load_templates(self.gp_temp_list, self.blg_names, self.dif_names)

    def temp_model(self, ebin = 10, gp_rng_key = jax.random.PRNGKey(424242)):
        # NOTE, need to perform config_model even if didn't perform GP fit
        # in the same notebook
    
        # check for gp
        if self.is_gp == False:
            raise ValueError('GP must be enabled.')

        if ebin == 'all':
            raise NotImplementedError
        else:
            ie = int(ebin)
            
        x = self.x
        mask = self.mask_roi_arr[ie]
        params = self.params

        # ----------------------------------------------------

        #===== rigid templates processing =====
        # all templates should be already normalized
        if 'iso' in self.gp_temp_list:
            S_iso = numpyro.sample('S_iso', dist.Uniform(1e-3, 50.))
        if 'psc' in self.gp_temp_list:
            S_psc = numpyro.sample('S_psc', dist.Uniform(1e-3, 50))
        if 'bub' in self.gp_temp_list:
            S_bub = numpyro.sample('S_bub', dist.Uniform(1e-3, 50))
        
        if self.n_dif_temps > 0:
            if 'pib' in self.gp_temp_list:
                S_pib = numpyro.sample('S_pib', dist.Uniform(1e-3, 100))
            if 'ics' in self.gp_temp_list:
                S_ics = numpyro.sample('S_ics', dist.Uniform(1e-3, 100))
            if self.n_dif_temps > 1:
                if 'pib' in self.gp_temp_list:
                    theta_pib = numpyro.sample("theta_pib", dist.Dirichlet(jnp.ones((self.n_dif_temps,)) / self.n_dif_temps))
                if 'ics' in self.gp_temp_list:
                    theta_ics = numpyro.sample("theta_ics", dist.Dirichlet(jnp.ones((self.n_dif_temps,)) / self.n_dif_temps))

        # ===== variable templates processing =====
        if 'blg' in self.gp_temp_list:
            if self.n_blg_temps == 1:
                S_blg = numpyro.sample('S_blg', dist.Uniform(1e-3, 10.))
            else:
                theta_blg = numpyro.sample("theta_blg", dist.Dirichlet(jnp.ones((self.n_blg_temps,)) / self.n_blg_temps))

        if 'nfw' in self.gp_temp_list:
            S_nfw = numpyro.sample('S_nfw', dist.Uniform(1e-3, 5.))
            if self.nfw_gamma == 'vary':
                gamma = numpyro.sample("gamma", dist.Uniform(0.2, 2))
#             else:
#                 gamma = self.nfw_gamma

        if 'dsk' in self.gp_temp_list:
            if self.disk_option in ['vary', 'fixed']:
                S_dsk = numpyro.sample('S_dsk', dist.Uniform(1e-3, 5))
                if self.disk_option == 'vary':
                    zs = numpyro.sample("zs", dist.Uniform(0.1, 2.5))
                    C  = numpyro.sample("C",  dist.Uniform(0.05, 15.))
                
        if self.u_option != 'None':
            xu_f = self.load_inducing_points(before_fit = False, params = params)

            # generate kernel parameters using best-fit GP parameters
            base_kernel = self.load_kernel(before_fit = False, params = params)
            gp_u = GaussianProcess(base_kernel, xu_f, diag=1e-3)

        # subsampling is pointless here ; assume you will not be using subsampling
        with numpyro.plate('x', size=len(x), dim=-1, subsample_size=self.Nsub) as ind:
            
            if self.Nsub == None:
                mu = jnp.zeros(len(x))
            else:
                mu = jnp.zeros(self.Nsub)

            x_sub = self.x[ind] # load angular coordinates of bins
            
            #===== rigid templates =====
            # all templates should be already normalized
            if 'iso' in self.gp_temp_list:
                mu += S_iso * jnp.asarray(self.temps['iso'].at_bin(ie, mask=mask))[ind]
            if 'psc' in self.gp_temp_list:
                mu += S_psc * jnp.asarray(self.temps['psc'].at_bin(ie, mask=mask))[ind]
            if 'bub' in self.gp_temp_list:
                mu += S_bub * jnp.asarray(self.temps['bub'].at_bin(ie, mask=mask))[ind]
                
            #===== hybrid templates =====
            # all templates should be already normalized
            if self.n_dif_temps > 0:
                if self.n_dif_temps == 1:
                    if 'pib' in self.gp_temp_list:
                        mu += S_pib * self.pib_temps[0].at_bin(ie, mask=mask)[ind]
                    if 'ics' in self.gp_temp_list:
                        mu += S_ics * self.ics_temps[0].at_bin(ie, mask=mask)[ind]
                else:
                    if 'pib' in self.gp_temp_list:
                        pib_temps_at_bin = jnp.asarray([pib_temp.at_bin(ie, mask=mask) for pib_temp in self.pib_temps])
                        mu += S_pib * jnp.dot(theta_pib, pib_temps_at_bin)[ind]
                    if 'ics' in self.gp_temp_list:
                        ics_temps_at_bin = jnp.asarray([ics_temp.at_bin(ie, mask=mask) for ics_temp in self.ics_temps])
                        mu += S_ics * jnp.dot(theta_ics, ics_temps_at_bin)[ind]
                
                if 'blg' in self.gp_temp_list:
                    if self.n_blg_temps == 1:
                        mu += S_blg * self.blg_temps[0].at_bin(ie, mask=mask)[ind]
                    else:
                        blg_temps_at_bin = jnp.asarray([blg_temp.at_bin(ie, mask=mask) for blg_temp in self.blg_temps])
                        mu += S_blg * jnp.dot(theta_blg, blg_temps_at_bin)[ind]
                
            #===== variable templates =====
            if 'nfw' in self.gp_temp_list:
                if self.nfw_gamma == 'vary':
                    mu += S_nfw * self.nfw_temp.get_NFW2_template(gamma=gamma)[~mask][ind]
                else:
                    mu += S_nfw * self.nfw_temp_fixed[ind]
            
            if 'dsk' in self.gp_temp_list:
                if self.disk_option in ['vary', 'fixed']:
                    if self.disk_option == 'vary':
                        temp_dsk = self.dsk_temp.get_template(zs=zs, C=C)[~mask]
                    else:
                        temp_dsk = self.temps['dsk'].at_bin(ie, mask=mask)
                    mu += S_dsk * temp_dsk[ind]

            if self.u_option != 'None':
                # ===== Generate Data from GP Model =====
                rng_key, gp_u_key, gp_key, poiss_key = jax.random.split(gp_rng_key, 4)
                with numpyro.handlers.block():    # prevents g1 sites from being detected
                    log_rate_u = self.pred(gp_u_key, ie)['log_rate_u'].T
                    log_rate_u = jnp.squeeze(log_rate_u, axis = -1) # shape mismatch otherwise (shape_ll ~ (shape_x, shape_x))
                    _, gp_x = gp_u.condition(log_rate_u, x_sub, diag=1e-3) # p(x|u)
                    rate = jnp.exp(gp_x.sample(gp_key))
                    y = numpyro.deterministic('y', poisson(poiss_key, rate))
            else:
                rng_key, gp_key, poiss_key = jax.random.split(gp_rng_key, 3)
                with numpyro.handlers.block():    # prevents g1 sites from being detected
                    lam = numpyro.deterministic('lam', jnp.exp(self.pred(gp_key, ie)['log_rate']).T )
                    # jax.debug.print('lam = {}', lam[ind].shape)
                    y = numpyro.deterministic('y', poisson(poiss_key, lam[ind]))
                    y = jnp.squeeze(y, axis = -1) # shape mismatch otherwise (shape_ll ~ (shape_x, shape_x))
            
            # load function values and compute log likelihood
            numpyro.factor("log_likelihood", log_like_poisson(mu, y)) 

    # ======== Step 2 (Yitian ; Rate Version) =========
    def temp_model_2(self, ebin = 10, gp_rng_key = jax.random.PRNGKey(424242)):
        # NOTE, need to perform config_model even if didn't perform GP fit
        # in the same notebook
    
        # check for gp
        if self.is_gp == False:
            raise ValueError('GP must be enabled.')

        if ebin == 'all':
            raise NotImplementedError
        else:
            ie = int(ebin)
            
        x = self.x
        mask = self.mask_roi_arr[ie]

        # ----------------------------------------------------

        #===== rigid templates processing =====
        # all templates should be already normalized
        if 'iso' in self.gp_temp_list:
            S_iso = numpyro.sample('S_iso', dist.Uniform(1e-3, 50.))
        if 'psc' in self.gp_temp_list:
            S_psc = numpyro.sample('S_psc', dist.Uniform(1e-3, 50))
        if 'bub' in self.gp_temp_list:
            S_bub = numpyro.sample('S_bub', dist.Uniform(1e-3, 50))
        
        if self.n_dif_temps > 0:
            if 'pib' in self.gp_temp_list:
                S_pib = numpyro.sample('S_pib', dist.Uniform(1e-3, 100))
            if 'ics' in self.gp_temp_list:
                S_ics = numpyro.sample('S_ics', dist.Uniform(1e-3, 100))
            if self.n_dif_temps > 1:
                if 'pib' in self.gp_temp_list:
                    theta_pib = numpyro.sample("theta_pib", dist.Dirichlet(jnp.ones((self.n_dif_temps,)) / self.n_dif_temps))
                if 'ics' in self.gp_temp_list:
                    theta_ics = numpyro.sample("theta_ics", dist.Dirichlet(jnp.ones((self.n_dif_temps,)) / self.n_dif_temps))

        # ===== variable templates processing =====
        if 'blg' in self.gp_temp_list:
            if self.n_blg_temps == 1:
                S_blg = numpyro.sample('S_blg', dist.Uniform(1e-3, 10.))
            else:
                theta_blg = numpyro.sample("theta_blg", dist.Dirichlet(jnp.ones((self.n_blg_temps,)) / self.n_blg_temps))

        if 'nfw' in self.gp_temp_list:
            S_nfw = numpyro.sample('S_nfw', dist.Uniform(1e-3, 5.))
            if self.nfw_gamma == 'vary':
                gamma = numpyro.sample("gamma", dist.Uniform(0.2, 2))
#             else:
#                 gamma = self.nfw_gamma

        if 'dsk' in self.gp_temp_list:
            if self.disk_option in ['vary', 'fixed']:
                S_dsk = numpyro.sample('S_dsk', dist.Uniform(1e-3, 5))
                if self.disk_option == 'vary':
                    zs = numpyro.sample("zs", dist.Uniform(0.1, 2.5))
                    C  = numpyro.sample("C",  dist.Uniform(0.05, 15.))

        mvn = dist.MultivariateNormal(loc = self.loc, covariance_matrix = self.inv_cov) # NOTE: cov -> inv_cov (this works much better ; see notion)
        
        # # subsampling is pointless here ; assume you will not be using subsampling
        # with numpyro.plate('x', size=len(x), dim=-1, subsample_size=self.Nsub) as ind:
            
        #     if self.Nsub == None:
        #         mu = jnp.zeros(len(x))
        #     else:
        #         mu = jnp.zeros(self.Nsub)

        #     x_sub = self.x[ind] # load angular coordinates of bins
            
        #     #===== rigid templates =====
        #     # all templates should be already normalized
        #     if 'iso' in self.gp_temp_list:
        #         mu += S_iso * jnp.asarray(self.temps['iso'].at_bin(ie, mask=mask))[ind]
        #     if 'psc' in self.gp_temp_list:
        #         mu += S_psc * jnp.asarray(self.temps['psc'].at_bin(ie, mask=mask))[ind]
        #     if 'bub' in self.gp_temp_list:
        #         mu += S_bub * jnp.asarray(self.temps['bub'].at_bin(ie, mask=mask))[ind]
                
        #     #===== hybrid templates =====
        #     # all templates should be already normalized
        #     if self.n_dif_temps > 0:
        #         if self.n_dif_temps == 1:
        #             if 'pib' in self.gp_temp_list:
        #                 mu += S_pib * self.pib_temps[0].at_bin(ie, mask=mask)[ind]
        #             if 'ics' in self.gp_temp_list:
        #                 mu += S_ics * self.ics_temps[0].at_bin(ie, mask=mask)[ind]
        #         else:
        #             if 'pib' in self.gp_temp_list:
        #                 pib_temps_at_bin = jnp.asarray([pib_temp.at_bin(ie, mask=mask) for pib_temp in self.pib_temps])
        #                 mu += S_pib * jnp.dot(theta_pib, pib_temps_at_bin)[ind]
        #             if 'ics' in self.gp_temp_list:
        #                 ics_temps_at_bin = jnp.asarray([ics_temp.at_bin(ie, mask=mask) for ics_temp in self.ics_temps])
        #                 mu += S_ics * jnp.dot(theta_ics, ics_temps_at_bin)[ind]
                
        #         if 'blg' in self.gp_temp_list:
        #             if self.n_blg_temps == 1:
        #                 mu += S_blg * self.blg_temps[0].at_bin(ie, mask=mask)[ind]
        #             else:
        #                 blg_temps_at_bin = jnp.asarray([blg_temp.at_bin(ie, mask=mask) for blg_temp in self.blg_temps])
        #                 mu += S_blg * jnp.dot(theta_blg, blg_temps_at_bin)[ind]
                
        #     #===== variable templates =====
        #     if 'nfw' in self.gp_temp_list:
        #         if self.nfw_gamma == 'vary':
        #             mu += S_nfw * self.nfw_temp.get_NFW2_template(gamma=gamma)[~mask][ind]
        #         else:
        #             mu += S_nfw * self.nfw_temp_fixed[ind]
            
        #     if 'dsk' in self.gp_temp_list:
        #         if self.disk_option in ['vary', 'fixed']:
        #             if self.disk_option == 'vary':
        #                 temp_dsk = self.dsk_temp.get_template(zs=zs, C=C)[~mask]
        #             else:
        #                 temp_dsk = self.temps['dsk'].at_bin(ie, mask=mask)
        #             mu += S_dsk * temp_dsk[ind]

        #     log_rate = jnp.log(mu)

        #     # ===== Compute Log Likelihood of Guide Model =====
        #     numpyro.factor("log_likelihood", mvn.log_prob(log_rate)) 
        
        # subsampling is pointless here ; assume you will not be using subsampling
        
        if self.Nsub == None:
            mu = jnp.zeros(len(x))
        else:
            mu = jnp.zeros(self.Nsub)

        x_sub = self.x # load angular coordinates of bins
        
        #===== rigid templates =====
        # all templates should be already normalized
        if 'iso' in self.gp_temp_list:
            mu += S_iso * jnp.asarray(self.temps['iso'].at_bin(ie, mask=mask))
        if 'psc' in self.gp_temp_list:
            mu += S_psc * jnp.asarray(self.temps['psc'].at_bin(ie, mask=mask))
        if 'bub' in self.gp_temp_list:
            mu += S_bub * jnp.asarray(self.temps['bub'].at_bin(ie, mask=mask))
            
        #===== hybrid templates =====
        # all templates should be already normalized
        if self.n_dif_temps > 0:
            if self.n_dif_temps == 1:
                if 'pib' in self.gp_temp_list:
                    mu += S_pib * self.pib_temps[0].at_bin(ie, mask=mask)
                if 'ics' in self.gp_temp_list:
                    mu += S_ics * self.ics_temps[0].at_bin(ie, mask=mask)
            else:
                if 'pib' in self.gp_temp_list:
                    pib_temps_at_bin = jnp.asarray([pib_temp.at_bin(ie, mask=mask) for pib_temp in self.pib_temps])
                    mu += S_pib * jnp.dot(theta_pib, pib_temps_at_bin)
                if 'ics' in self.gp_temp_list:
                    ics_temps_at_bin = jnp.asarray([ics_temp.at_bin(ie, mask=mask) for ics_temp in self.ics_temps])
                    mu += S_ics * jnp.dot(theta_ics, ics_temps_at_bin)
            
            if 'blg' in self.gp_temp_list:
                if self.n_blg_temps == 1:
                    mu += S_blg * self.blg_temps[0].at_bin(ie, mask=mask)
                else:
                    blg_temps_at_bin = jnp.asarray([blg_temp.at_bin(ie, mask=mask) for blg_temp in self.blg_temps])
                    mu += S_blg * jnp.dot(theta_blg, blg_temps_at_bin)
            
        #===== variable templates =====
        if 'nfw' in self.gp_temp_list:
            if self.nfw_gamma == 'vary':
                mu += S_nfw * self.nfw_temp.get_NFW2_template(gamma=gamma)[~mask]
            else:
                mu += S_nfw * self.nfw_temp_fixed
        
        if 'dsk' in self.gp_temp_list:
            if self.disk_option in ['vary', 'fixed']:
                if self.disk_option == 'vary':
                    temp_dsk = self.dsk_temp.get_template(zs=zs, C=C)[~mask]
                else:
                    temp_dsk = self.temps['dsk'].at_bin(ie, mask=mask)
                mu += S_dsk * temp_dsk

        log_rate = jnp.log(mu)

        # ===== Compute Log Likelihood of Guide Model =====
        numpyro.factor("log_likelihood", mvn.log_prob(log_rate) - jnp.sum(log_rate)) # NOTE: Not sure if this factor should be here
        # numpyro.factor("log_likelihood", logpdf(log_rate, self.loc, self.cov)) 
        
    def config_temp_model_2(self, gp_temp_list = None, params = None, guide = 'mvn', ebin = 10):
        '''
        Note, hastily built. Need to clean up and make more general. For example,
        only works for ExpSquared kernel. The indexing assumed there are 4 
        template parameters that are modelled by the guide.
        '''
        if ebin == 'all':
            raise NotImplementedError
        else:
            ie = int(ebin)
    
        # load best-fit parameters specifying the guide from a previous fit
        if self.svi_results is None:
            if params is None:
                raise ValueError('SVI must be run first or guide params must be provided.')
            self.params = params
            
            if guide == 'mvn':
                self.gp_guide = autoguide.AutoMultivariateNormal(self.model)
            else:
                raise NotImplementedError

        else:
            self.params = self.svi_results.params
            params = self.params
            self.gp_guide = self.guide

        # swap roles for fit function
        self.gp_model = self.model
        self.model = self.temp_model_2
        
        # load names of templates to fit to gp
        if gp_temp_list is None and self.temp_names_sim is not None:
            gp_temp_list = [temp for temp in self.temp_names_sim if temp not in self.temp_list]
        else:
            raise ValueError('gp_temp_list must be specified.')
        self.gp_temp_list = gp_temp_list

        if 'nfw' in self.temp_list:
            Nt = len(self.temp_list) + 1
        else:
            Nt = len(self.temp_list) 

        if self.u_option == 'None':
            m = self.params['auto_loc'][Nt:]
            L = self.params['auto_scale_tril'][Nt:,Nt:]
            S = L @ L.T
            self.loc = m ; self.cov = S
        else:
            m = self.params['auto_loc'][Nt:]
            L = self.params['auto_scale_tril'][Nt:,Nt:]
            S = L @ L.T

            # generate kernel parameters with best-fit GP parameters
            base_kernel = self.load_kernel(before_fit = False, params = params)

            # want to compute the blocks of the covariance matrix
            x_aug = jnp.concatenate([self.x, self.xu_f])
    
            gp = GaussianProcess(base_kernel, x_aug, diag=1e-3)
            Kaa = gp.covariance
            Nx = self.x.shape[0]
            Kxx = Kaa[:Nx, :Nx]
            Kuu = Kaa[Nx:, Nx:]
            Kxu = Kaa[:Nx, Nx:]

            # define the A matrix
            # print(Kxu.shape, jnp.linalg.inv(Kuu).shape, Kxx.shape, S.shape, m.shape)
            A = Kxu @ jnp.linalg.inv(Kuu)

            # define the mean
            loc = A @ m

            # define the covariance
            cov = Kxx + A @ (S - Kuu) @ A.T  
            self.loc = loc ; self.cov = cov
        self.inv_cov = jnp.linalg.inv(self.cov)

        # initialize templates that gp models (so are not included in the first model)
        self.load_templates(self.gp_temp_list, self.blg_names, self.dif_names)

    def cfit_SVI_2(
        self, rng_key=jax.random.PRNGKey(42),
        guide='iaf', optimizer=None, num_flows=3, hidden_dims=[64, 64],
        n_steps=5000, lr=0.006, num_particles=8, progress_bar = True,
        early_stop = np.inf,
        **model_static_kwargs,
    ):
        self.guide_name = guide
        if guide == 'mvn':
            self.guide = autoguide.AutoMultivariateNormal(self.model)
        elif guide == 'iaf':
            self.guide = autoguide.AutoIAFNormal(
                self.model,
                num_flows=num_flows,
                hidden_dims=hidden_dims,
                nonlinearity=stax.Tanh
            )
        elif guide == 'iaf_mixture':
            num_base_mixture = 8
            class AutoIAFMixture(autoguide.AutoIAFNormal):
                def get_base_dist(self):
                    C = num_base_mixture
                    mixture = dist.MixtureSameFamily(
                        dist.Categorical(probs=jnp.ones(C) / C),
                        dist.Normal(jnp.arange(float(C)), 1.)
                    )
                    return mixture.expand([self.latent_dim]).to_event()
            self.guide = AutoIAFMixture(
                self.model,
                num_flows=num_flows,
                hidden_dims=hidden_dims,
                nonlinearity=stax.Tanh
            )
        else:
            raise NotImplementedError

        if optimizer == None:
            optimizer = optim.optax_to_numpyro(
                optax.chain(
                    optax.clip(1.),
                    optax.adam(lr),
                )
            )



        svi = SVI(
            self.model, self.guide, optimizer,
            Trace_ELBO_2(num_particles=num_particles),
        )
        self.svi_results = self.svi_loop(svi, progress_bar = progress_bar, num_steps = n_steps, rng_key = rng_key, early_stop = early_stop)
        self.svi_model_static_kwargs = model_static_kwargs
        
        return self.svi_results

    #========== NeuTra ==========
    def get_neutra_model(self):
        """Get model reparameterized via neural transport."""
        neutra = NeuTraReparam(self.guide, self.svi_results.params)
        model = lambda x: self.model(**self.svi_model_static_kwargs)
        self.model_neutra = neutra.reparam(model)
        
    
    #========== NUTS ==========
    def run_nuts(self, num_chains=4, num_warmup=500, num_samples=5000, step_size=0.1,
                 rng_key=jax.random.PRNGKey(0), use_neutra=True, **model_static_kwargs):
        
        if use_neutra:
            self.get_neutra_model()
            model = self.model_neutra
        else:
            model = self.model
        
        kernel = NUTS(model, max_tree_depth=4, dense_mass=False, step_size=step_size)
        self.nuts_mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, chain_method='vectorized')
        if use_neutra:
            self.nuts_mcmc.run(rng_key, None)
        else:
            self.nuts_mcmc.run(rng_key, **model_static_kwargs)
        
        return self.nuts_mcmc
    
    
    #========== PTHMC ==========
    def run_parallel_tempering_hmc(self, num_samples=5000, step_size_base=5e-2, num_leapfrog_steps=3, num_adaptation_steps=600, rng_key=jax.random.PRNGKey(0), use_neutra=True):
        
        # Geometric temperatures decay
        inverse_temperatures = 0.5 ** jnp.arange(4.)

        # If everything was Normal, step_size should be ~ sqrt(temperature).
        step_size = step_size_base / jnp.sqrt(inverse_temperatures)[..., None]

        def make_kernel_fn(target_log_prob_fn):

            hmc = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size, num_leapfrog_steps=num_leapfrog_steps)

            adapted_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=hmc,
            num_adaptation_steps=num_adaptation_steps)

            return adapted_kernel
        
        if use_neutra:
            self.get_neutra_model()
            model = self.model_neutra
        else:
            model = lambda x: self.model(**self.svi_model_static_kwargs)
        
        kernel = ReplicaExchangeMC(model, inverse_temperatures=inverse_temperatures, make_kernel_fn=make_kernel_fn)
        self.pt_mcmc = MCMC(kernel, num_warmup=num_adaptation_steps, num_samples=num_samples, num_chains=1, chain_method='vectorized')
        self.pt_mcmc.run(rng_key, None)
        
        return self.pt_mcmc
    
    
    #========== MAP ==========
    def fit_MAP(
        self, rng_key=jax.random.PRNGKey(42),
        lr=0.1, n_steps=10000, num_particles=8,
        **model_static_kwargs,
    ):
        guide = autoguide.AutoDelta(self.model)
        optimizer = optim.optax_to_numpyro(optax.chain(optax.clip(1.), optax.adamw(lr)))
        svi = SVI(
            self.model, guide, optimizer,
            loss=Trace_ELBO(num_particles=num_particles),
            **model_static_kwargs,
        )
        svi_results = svi.run(rng_key, n_steps)
        self.MAP_estimates = guide.median(svi_results.params)
        
        return svi_results
    
    # helper function to check shapes of sites in the traced model
    # previously used to debug shape issues in LL calculation
    def print_trace_shapes(self):
        with numpyro.handlers.seed(rng_seed=1):
            trace = numpyro.handlers.trace(self.model).get_trace(ebin = 10, gp_rng_key = jax.random.PRNGKey(32))
        print(numpyro.util.format_shapes(trace))