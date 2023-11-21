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
from jax.example_libraries import stax

import optax
import numpyro
import numpyro.distributions as dist
from numpyro import optim
from numpyro.infer import SVI, Trace_ELBO, autoguide, TraceMeanField_ELBO
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
from tqdm import tqdm

# better to load GPU from the main script that runs the fit
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
        Nsub = 1000,
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
        self.ebin = 10,
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
        else: 
            sim_data_dir = '../data/synthetic_data/' + data_file + '/'
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

        #========== Rigid templates ==========
        self.temp_list = self.rig_temp_list + self.hyb_temp_list + self.var_temp_list
        self.temps = {}

        if 'iso' in self.temp_list:
            self.temps['iso'] = EbinTemplate(
                self.exposure.copy(),
                norm_mask=self.normalization_mask,
            )
        if 'psc' in self.temp_list:
            self.temps['psc'] = EbinTemplate(
                to_nside(np.load(f'{ebin_data_dir}/psc-bestpsf-3fgl.npy'), self.nside),
                norm_mask=self.normalization_mask,
            )
        if 'bub' in self.temp_list:
            self.temps['bub'] = Template(
                to_nside(np.load(f'{default_data_dir}/template_bub.npy'), self.nside),
                norm_mask=self.normalization_mask
            )
        if 'dsk' in self.temp_list:
            self.temps['dsk'] = Template(
                to_nside(np.load(f'{default_data_dir}/template_dsk_z1p0.npy'), self.nside),
                norm_mask=self.normalization_mask
            )
        if 'nfw' in self.temp_list:
            self.temps['nfw'] = Template(
                to_nside(np.load(f'{default_data_dir}/template_nfw_g1p0.npy'), self.nside),
                norm_mask=self.normalization_mask
            )

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
        if 'pib' in self.temp_list:
            self.pib_temps = [
                EbinTemplate(
                    to_nside(np.load(f'{ebin_data_dir}/{dif_name}pibrem-{self.temp_class}.npy'), self.nside),
                    norm_mask=self.normalization_mask,
                )
                for dif_name in dif_names
            ]
        if 'ics' in self.temp_list:
            self.ics_temps = [
                EbinTemplate(
                    to_nside(np.load(f'{ebin_data_dir}/{dif_name}ics-{self.temp_class}.npy'), self.nside),
                    norm_mask=self.normalization_mask,
                )
                for dif_name in dif_names
            ]
        if ('blg' in self.temp_list) or (self.u_weights == 'blg'):
            self.n_blg_temps = len(self.blg_names)
            self.blg_temps = [
                Template(
                    BulgeTemplates(template_name=blg_name, nside_out=self.nside)(),
                    norm_mask=self.normalization_mask,
                )
                for blg_name in blg_names
            ]

         #========== Variable templates ==========
        if 'nfw' in self.temp_list:
            self.nfw_temp = NFWTemplate(nside=self.nside)
        if 'dsk' in self.temp_list:
            self.dsk_temp = LorimerDiskTemplate(nside=self.nside)
        
        #========== sample expand keys ==========
        self.samples_expand_keys = {}
        if 'pib' in self.temp_list:
            self.samples_expand_keys['pib'] = [f'theta_pib_{n}' for n in self.dif_names]
        if 'ics' in self.temp_list:
            self.samples_expand_keys['ics'] = [f'theta_ics_{n}' for n in self.dif_names]
        if 'blg' in self.temp_list:
            self.samples_expand_keys['blg'] = [f'theta_blg_{n}' for n in self.blg_names]

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
            if self.u_option == 'float':
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
    def model(self, ebin=10):
        
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
            S_nfw = numpyro.sample('S_nfw', dist.Uniform(1e-3, 5000))
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
            if self.u_option == 'float':
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
                xu_f = self.xu_f
            
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

            # generate kernel parameters
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
                alpha = self.gp_params[2]
                if alpha == 'float':
                    alpha = numpyro.param('alpha', jnp.ones(()), constraint=dist.constraints.positive)
                unit_kernel = kernels.RationalQuadratic(alpha = alpha, distance = kernels.distance.L2Distance())
            elif self.gp_kernel == 'Matern32': # untested
                unit_kernel = kernels.Matern32()
            elif self.gp_kernel == 'Matern52': # untested
                unit_kernel = kernels.Matern52()

            # load scale transformations
            if self.gp_scale_option == 'Linear':
                scale = jnp.ones(()) / scale # scale is a scale factor that rescales the spatial coords; not the same as the kernel lengthscale
                base_kernel = amp**2. * transforms.Linear(scale, unit_kernel)
            elif self.gp_scale_option == 'Cholesky':
                base_kernel = amp**2. * transforms.Cholesky.from_parameters(scale_diag, scale_off, unit_kernel)
            
            # load derivative kernel or base kernel
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
                x_sub = self.x[ind] # load angular coordinates of bins

                if self.gp_deriv: # augment bins for derivative GP
                    # augment data
                    dx = jnp.zeros(self.Nsub)
                    dy = jnp.zeros(self.Nsub)
                    x_sub = jnp.vstack([x_sub.T,dx.T,dy.T]).T

                sample_keys = jax.random.split(jax.random.PRNGKey(
                    np.random.randint(0, 1000000)), 3)
                key, key_x, key_xp = sample_keys

                _, gp_x = gp_u.condition(log_rate_u, x_sub, diag=1e-3) # p(x|u)
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
            # numpyro.sample("obs", dist.Poisson(mu), obs=data[ind])  # alternative choice works for SVI
    
            # #===== deterministic =====
            # if ('nfw' in self.temp_list) and ('blg' in self.temp_list):
            #     numpyro.deterministic('f_blg', S_blg / (S_blg + S_nfw))

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
        # NOTE: Example below
# # ===========   NEW OPTIMIZER  ===========
#         schedule = optax.piecewise_constant_schedule(
#             init_value=lr,
#             boundaries_and_scales={
#                 # int(1000): 0.05,
#                 # int(1000): 0.01,
#                 int(4000): 0.001,
#             } # new schedule
#         )
#         optimizer = optim.optax_to_numpyro(
#             optax.chain(
#                 optax.clip(1.),
#                 optax.adamw(learning_rate=schedule), 
#             )
#         )
        # schedule = optax.warmup_exponential_decay_schedule(
        #     init_value=0.001,
        #     peak_value=0.03,
        #     warmup_steps=500,
        #     transition_steps=2500,
        #     decay_rate=1./jnp.exp(1.),
        #     transition_begin=5000,
        # )
        # optimizer = optim.optax_to_numpyro(
        #     optax.chain(
        #         optax.clip(1.),
        #         optax.adamw(learning_rate=schedule), 
        #     )
        # )

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
        
    def get_gp_samples(self, num_samples=1000, custom_mask = None):
        nside = self.nside
        Nu = self.Nu
        svi_results = self.svi_results
        samples = self.svi_samples

        if custom_mask is None:
            mask = self.mask_roi_arr[10]
        else:
            mask = custom_mask
        x_p = ef.get_x_from_mask(mask,nside) # predicted x given sampled u
        
        if self.is_gp:
            if self.u_option == 'float':
                lru = svi_results.params["lru"]
                lau = svi_results.params["lau"]

                ru = 20. * expit(lru)
                angu = 2. * jnp.pi * expit(lau)

                xu = ru * jnp.cos(angu)
                yu = ru * jnp.sin(angu)

                xu_f = jnp.vstack([xu.T,yu.T]).T
            else:
                xu_f = self.xu_f
            # generate kernel parameters
            scale = self.gp_params[0]
            amp = self.gp_params[1]
            
            if scale == 'float':
                scale = svi_results.params['scale']
            if amp == 'float':
                amp = svi_results.params['amp']

            # load kernel
            if self.gp_kernel == 'ExpSquared': # most numerically stable
                unit_kernel = kernels.ExpSquared()
            elif self.gp_kernel == 'RationalQuadratic': # tested
                alpha = self.gp_params[2]
                if alpha == 'float':
                    alpha = svi_results.params['alpha']
                unit_kernel = kernels.RationalQuadratic(alpha = alpha, distance = kernels.distance.L2Distance())
            elif self.gp_kernel == 'Matern32': # untested
                unit_kernel = kernels.Matern32()
            elif self.gp_kernel == 'Matern52': # untested
                unit_kernel = kernels.Matern52()

            if self.gp_scale_option == 'Linear':
                scale = jnp.ones(()) / scale # scale is a scale factor that rescales the spatial coords; not the same as the kernel lengthscale
                base_kernel = amp**2. * transforms.Linear(scale, unit_kernel)
            elif self.gp_scale_option == 'Cholesky':
                scale_diag = jnp.exp(scale[:2])
                scale_off = scale[2:]
                base_kernel = amp**2. * transforms.Cholesky.from_parameters(scale_diag, scale_off, unit_kernel)

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
                for n in tqdm(range(num_samples)):
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
        
        