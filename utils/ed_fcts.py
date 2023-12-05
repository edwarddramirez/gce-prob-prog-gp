import tinygp  # for Gaussian process regression
from tinygp import GaussianProcess, kernels, transforms
from functools import partial

import os
import sys

import numpy as np
import healpy as hp

import jax
import jax.numpy as jnp
from jax.scipy.special import logit, expit

from templates.rigid_templates import EbinTemplate, Template, BulgeTemplates
import corner as corner
from tqdm import tqdm

from utils.cart import to_cart

import importlib # for importing modules with python code

# NOTE: This is a very incomplete list of functions. You have done much work previously. For example,
# 1D RFFs, 1D Derivative Kernels, JAX interpolation, etc. are not included here.

# ======== Grid Makers ========
def initialize_skygrid(mask=None, nside=128):
    if mask is None:
        mask = np.zeros(hp.nside2npix(nside)).astype(bool)
    pix = np.arange(hp.nside2npix(nside))[~mask]
    theta_pix, phi_pix = hp.pix2ang(nside, pix)

    phi_pix[phi_pix>np.pi] = phi_pix[phi_pix>np.pi]-2*np.pi
    phi_pix = -phi_pix
    theta_pix = theta_pix-np.pi/2 
    theta_pix = -theta_pix

    return phi_pix, theta_pix

def get_x_from_mask(mask, nside=128):
    phi, theta = initialize_skygrid(nside=nside, mask=mask)
    x = jnp.vstack([phi, theta]).T * 180 / np.pi
    return x

def get_u_from_mask(Nu, mask, grid_type='healpix_bins', weights=None, nside=128):
    data_indices = np.where(~mask)[0]
    data_indices = np.asarray(data_indices)

    if grid_type=='healpix_bins':
        if weights is None:
            sample = jax.random.choice(jax.random.PRNGKey(34), data_indices, shape=(Nu,), p=weights, replace=False) 
        else:
            wght_map = np.zeros(hp.nside2npix(nside))
            wght_map[~mask] = weights
            wght_map = wght_map[data_indices]
            wght_map = wght_map / np.sum(wght_map)

            sample = jax.random.choice(jax.random.PRNGKey(35), data_indices, shape=(Nu,), p=wght_map, replace=False) 
        
        phi_pix, theta_pix = initialize_skygrid(nside=nside, mask=None)
        phi_sample = phi_pix[sample]
        theta_sample = theta_pix[sample]
        x_sample = jnp.vstack([phi_sample, theta_sample]).T * 180 / np.pi
        xu_f = x_sample

        return jnp.asarray(xu_f)
    
    elif type == 'regular':
        if jnp.sqrt(Nu) % 1 == 0:
            sNu = jnp.sqrt(self.Nu).astype(int)
        else:
            raise ValueError('Nu must be a perfect square')
        t = jnp.linspace(0.,1.,sNu)
        t1 = jnp.concatenate([t for i in range(sNu)])
        t2 = jnp.array(jnp.concatenate([t[n] * jnp.ones(sNu) for n in range(sNu)]))
        xu = -20. + 40. * t1
        yu = -20. + 40. * t2
        xu_f = jnp.vstack([xu.T,yu.T]).T

        return jnp.asarray(xu_f)

# ======== Derivative Gaussian Process ========
# 2D Derivative Kernel
class DerivativeKernel(kernels.Kernel):
    """
    A kernel that takes the derivative of another kernel with respect to one of its
    arguments. The argument to be differentiated with respect to is specified by the
    boolean flags d1x, d1y, d2x, d2y. For example, if d1x is True, then the kernel
    will be differentiated with respect to the first argument's x-coordinate.

    This kernel is useful for computing the derivatives of a Gaussian process with
    respect to its inputs.

    Parameters
    ----------
    kernel : Kernel
        The kernel to differentiate.
    d1x : bool (0,1)
        Whether to differentiate with respect to the first argument's x-coordinate.
    d1y : bool (0,1)
        Whether to differentiate with respect to the first argument's y-coordinate.
    d2x : bool (0,1)
        Whether to differentiate with respect to the second argument's x-coordinate.
    d2y : bool (0,1)
        Whether to differentiate with respect to the second argument's y-coordinate.

    Returns
    -------
    DerivativeKernel
        A kernel that computes the derivative of the given kernel with respect to
        the specified arguments.
    """
    def __init__(self, kernel):
        self.kernel = kernel
    def evaluate(self, X1, X2):
        """
        Evaluate the kernel matrix and its derivatives.

        Parameters
        ----------
        X1 : array-like, shape=(n_samples, 2)
            The first set of samples.
        X2 : array-like, shape=(n_samples, 2)
            The second set of samples.
        
        Returns
        -------
        array-like, shape=(n_samples, n_samples)
            The kernel matrix.
        """

        x1, y1, d1x, d1y = X1
        x2, y2, d2x, d2y = X2

        # Evaluate the kernel matrix and all of its relevant derivatives
        K = self.kernel.evaluate(jnp.array([x1,y1]), jnp.array([x2,y2]))
        # For stationary kernels, these are related just by a minus sign, but we'll
        # evaluate them both separately for generality's sake
        dK_dx2 = jax.grad(lambda x2_: self.kernel.evaluate(jnp.array([x1,y1]),jnp.array([x2_,y2])))(x2)
        dK_dx1 = jax.grad(lambda x1_: self.kernel.evaluate(jnp.array([x1_,y1]),jnp.array([x2,y2])))(x1)
        d2K_dx1dx2 = (jax.grad( lambda x2_: 
            ( jax.grad( lambda x1_: self.kernel.evaluate(jnp.array([x1_,y1]),jnp.array([x2_,y2]))) )(x1) 
            ) )(x2) 
        
        dK_dy2 = jax.grad(lambda y2_: self.kernel.evaluate(jnp.array([x1,y1]),jnp.array([x2,y2_])))(y2)
        dK_dy1 = jax.grad(lambda y1_: self.kernel.evaluate(jnp.array([x1,y1_]),jnp.array([x2,y2])))(y1)
        d2K_dy1dy2 = (jax.grad( lambda y2_: 
            ( jax.grad( lambda y1_: self.kernel.evaluate(jnp.array([x1,y1_]),jnp.array([x2,y2_]))) )(y1) 
            ) )(y2) 
        
        d2K_dx1dy2 = (jax.grad( lambda y2_: 
            ( jax.grad( lambda x1_: self.kernel.evaluate(jnp.array([x1_,y1]),jnp.array([x2,y2_]))) )(x1) 
            ) )(y2) 
        d2K_dy1dx2 = (jax.grad( lambda x2_: 
            ( jax.grad( lambda y1_: self.kernel.evaluate(jnp.array([x1,y1_]),jnp.array([x2_,y2]))) )(y1) 
            ) )(x2) 
        return jnp.where(d1x, 
                         jnp.where(d2x, d2K_dx1dx2, jnp.where(d2y, d2K_dx1dy2, dK_dx1)),
                         jnp.where(d1y, jnp.where(d2x, d2K_dy1dx2, jnp.where(d2y, d2K_dy1dy2, dK_dy1)), 
                                   jnp.where(d2x, dK_dx2, jnp.where(d2y, dK_dy2, K)))
                                   )
    
# 2D Polar Kernel (with Derivative Kernel)
def cartesian_to_polar_(x,y):
    r = jnp.sqrt(x**2. + y**2.)
    theta = jnp.arctan2(y,x)
    return jnp.array([r,theta]).T

def Wendland_C2(t, tau, c):
    # tau >= 4
    T = t/c
    return (1. + tau * T) * jnp.power(1.-T, tau)

class AngularDistance(kernels.stationary.Distance):
    def distance(self, theta1, theta2):
        # return jnp.arccos(jnp.cos(theta1 - theta2))
        theta_diff = jnp.abs(theta1 - theta2)
        return jnp.where(theta_diff < jnp.pi, theta_diff, jnp.abs(2. * jnp.pi - theta_diff) ) 
        
class GeodesicKernel(kernels.Kernel):
    def __init__(self, tau):
        self.tau = tau
        self.distance = AngularDistance()
    
    def evaluate(self, theta1, theta2):
        angular_distance = self.distance.distance(theta1, theta2)
        return Wendland_C2(angular_distance, self.tau, jnp.pi)    
    
class PolarKernel(kernels.Kernel):
    def __init__(self, kernel_r, kernel_ang, sigma, alpha1, alpha2):
        self.kernel_r = kernel_r
        self.kernel_ang = kernel_ang
        
        self.sigma = sigma
        self.alpha1 = alpha1
        self.alpha2 = alpha2
    
    def evaluate(self,X1,X2):
        rho1, theta1 = X1
        rho2, theta2 = X2
        
        return (self.sigma * 
                (1. + self.alpha1 * self.kernel_r.evaluate(rho1,rho2)) * 
                (1. + self.alpha2 * self.kernel_ang.evaluate(theta1,theta2)))

class DerivativePolarKernel(kernels.Kernel):
    def __init__(self, kernel_r, kernel_ang, sigma, alpha1, alpha2):
        self.kernel_r = kernel_r
        self.kernel_ang = kernel_ang
        
        self.sigma = sigma
        self.alpha1 = alpha1
        self.alpha2 = alpha2
    
    def evaluate(self,X1,X2):
        rho1, theta1, d1 = X1
        rho2, theta2, d2 = X2
        
        # Differentiate the kernel function: the first derivative wrt x1
        Kp = jax.grad(self.kernel_r.evaluate, argnums=0)
        # ... and the second derivative
        Kpp = jax.grad(Kp, argnums=1)
        # Evaluate the kernel matrix and all of its relevant derivatives
        K = self.kernel_r.evaluate(rho1, rho2)
        d2K_dx1dx2 = Kpp(rho1, rho2)
        # For stationary kernels, these are related just by a minus sign, but we'll
        # evaluate them both separately for generality's sake
        dK_dx2 = jax.grad(self.kernel_r.evaluate, argnums=1)(rho1, rho2)
        dK_dx1 = Kp(rho1, rho2)
        
        return ( self.sigma * (
                jnp.where(
                    d1, jnp.where(d2, self.alpha1 * d2K_dx1dx2, self.alpha1 * dK_dx1), 
                    jnp.where(d2, self.alpha1 * dK_dx2, 1. + self.alpha1 * K)
            ) )
                * (1. + self.alpha2 * self.kernel_ang.evaluate(theta1,theta2) )
               )
    
# ====== Plotting Utilities ======
# given name of variable and value, print a table of the quantities
def print_table(name, value):
    print(r'\begin{table}[htb]')
    print(r'\centering')
    print(r'\begin{tabular}{|c|c|}')
    print(r'\hline')
    print(r'Quantity & Value \\')
    print(r'\hline')
    print(r'{} & {:.3f} \\'.format(name, value))
    print(r'\hline')
    print(r'\end{tabular}')
    print(r'\end{table}')

# given names and values, print a table of the quantities
def print_table_multi(names, values):
    # add curly brackets to subscripted text
    for i in range(len(names)):
        if '_' in names[i]:
            names[i] = names[i].replace('_', '_{')
            names[i] = names[i] + '}' 

    print(r'\begin{table}[htb]')
    print(r'\centering')
    print(r'\begin{tabular}{|c|c|}')
    print(r'\hline')
    print(r'Quantity & Value \\')
    print(r'\hline')
    for i in range(len(names)):
        print(r'{} & {:.3f} \\'.format(names[i], values[i]))
    print(r'\hline')
    print(r'\end{tabular}')
    print(r'\end{table}')

pi = np.pi
dtor = pi / 180.0

from healpy import projaxes as PA
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from healpy import pixelfunc

def gnomview_ed(
    map=None,
    fig=None,
    rot=None,
    coord=None,
    unit="",
    xsize=200,
    ysize=None,
    reso=1.5,
    title="Gnomonic view",
    nest=False,
    remove_dip=False,
    remove_mono=False,
    gal_cut=0,
    min=None,
    max=None,
    flip="astro",
    format="%.3g",
    cbar=True,
    cmap=None,
    badcolor="gray",
    bgcolor="white",
    norm=None,
    hold=False,
    sub=None,
    reuse_axes=False,
    margins=None,
    notext=False,
    return_projected_map=False,
    no_plot=False,
    alpha=None,
    cbar_title = None,
):
    """Plot a healpix map (given as an array) in Gnomonic projection.

    Parameters
    ----------
    map : array-like
      The map to project, supports masked maps, see the `ma` function.
      If None, use a blank map, useful for
      overplotting.
    fig : None or int, optional
      A figure number. Default: None= create a new figure
    rot : scalar or sequence, optional
      Describe the rotation to apply.
      In the form (lon, lat, psi) (unit: degrees) : the point at
      longitude *lon* and latitude *lat* will be at the center. An additional rotation
      of angle *psi* around this direction is applied.
    coord : sequence of character, optional
      Either one of 'G', 'E' or 'C' to describe the coordinate
      system of the map, or a sequence of 2 of these to rotate
      the map from the first to the second coordinate system.
    unit : str, optional
      A text describing the unit of the data. Default: ''
    xsize : int, optional
      The size of the image. Default: 200
    ysize : None or int, optional
      The size of the image. Default: None= xsize
    reso : float, optional
      Resolution (in arcmin). Default: 1.5 arcmin
    title : str, optional
      The title of the plot. Default: 'Gnomonic view'
    nest : bool, optional
      If True, ordering scheme is NESTED. Default: False (RING)
    min : float, scalar, optional
      The minimum range value
    max : float, scalar, optional
      The maximum range value
    flip : {'astro', 'geo'}, optional
      Defines the convention of projection : 'astro' (default, east towards left, west towards right)
      or 'geo' (east towards roght, west towards left)
    remove_dip : bool, optional
      If :const:`True`, remove the dipole+monopole
    remove_mono : bool, optional
      If :const:`True`, remove the monopole
    gal_cut : float, scalar, optional
      Symmetric galactic cut for the dipole/monopole fit.
      Removes points in latitude range [-gal_cut, +gal_cut]
    format : str, optional
      The format of the scale label. Default: '%g'
    cmap : a color map
       The colormap to use (see matplotlib.cm)
    badcolor : str
      Color to use to plot bad values
    bgcolor : str
      Color to use for background
    hold : bool, optional
      If True, replace the current Axes by a GnomonicAxes.
      use this if you want to have multiple maps on the same
      figure. Default: False
    sub : int or sequence, optional
      Use only a zone of the current figure (same syntax as subplot).
      Default: None
    reuse_axes : bool, optional
      If True, reuse the current Axes (should be a GnomonicAxes). This is
      useful if you want to overplot with a partially transparent colormap,
      such as for plotting a line integral convolution. Default: False
    margins : None or sequence, optional
      Either None, or a sequence (left,bottom,right,top)
      giving the margins on left,bottom,right and top
      of the axes. Values are relative to figure (0-1).
      Default: None
    notext: bool, optional
      If True: do not add resolution info text. Default=False
    return_projected_map : bool, optional
      if True returns the projected map in a 2d numpy array
    no_plot : bool, optional
      if True no figure will be created
    alpha : float, array-like or None
      An array containing the alpha channel, supports masked maps, see the `ma` function.
      If None, no transparency will be applied.
      See an example usage of the alpha channel transparency in the documentation under
      "Other tutorials"

    See Also
    --------
    mollview, cartview, orthview, azeqview
    """
    import pylab

    if map is None:
        map = np.zeros(12) + np.inf
        cbar = False

    # Ensure that the nside is valid
    nside = pixelfunc.get_nside(map)
    pixelfunc.check_nside(nside, nest=nest)

    if not (hold or sub or reuse_axes):
        f = pylab.figure(fig, figsize=(5.8, 6.4))
        if not margins:
            margins = (0.075, 0.05, 0.075, 0.05)
        extent = (0.0, 0.0, 1.0, 1.0)
    elif hold:
        f = pylab.gcf()
        left, bottom, right, top = np.array(pylab.gca().get_position()).ravel()
        if not margins:
            margins = (0.0, 0.0, 0.0, 0.0)
        extent = (left, bottom, right - left, top - bottom)
        f.delaxes(pylab.gca())
    elif reuse_axes:
        f = pylab.gcf()
    else:  # using subplot syntax
        f = pylab.gcf()
        if hasattr(sub, "__len__"):
            nrows, ncols, idx = sub
        else:
            nrows, ncols, idx = sub // 100, (sub % 100) // 10, (sub % 10)
        if idx < 1 or idx > ncols * nrows:
            raise ValueError("Wrong values for sub: %d, %d, %d" % (nrows, ncols, idx))
        c, r = (idx - 1) % ncols, (idx - 1) // ncols
        if not margins:
            margins = (0.01, 0.0, 0.0, 0.02)
        extent = (
            c * 1.0 / ncols,
            1.0 - (r + 1) * 1.0 / nrows,
            1.0 / ncols,
            1.0 / nrows,
        )
    if not reuse_axes:
        extent = (
            extent[0] + margins[0],
            extent[1] + margins[1],
            extent[2] - margins[2] - margins[0],
            extent[3] - margins[3] - margins[1],
        )
    # f=pylab.figure(fig,figsize=(5.5,6))

    # Starting to draw : turn interactive off
    wasinteractive = pylab.isinteractive()
    pylab.ioff()
    try:
        map = pixelfunc.ma_to_array(map)
        if reuse_axes:
            ax = f.gca()
        else:
            ax = PA.HpxGnomonicAxes(
                f, extent, coord=coord, rot=rot, format=format, flipconv=flip
            )
            f.add_axes(ax)
        if remove_dip:
            map = pixelfunc.remove_dipole(map, gal_cut=gal_cut, nest=nest, copy=True)
        elif remove_mono:
            map = pixelfunc.remove_monopole(map, gal_cut=gal_cut, nest=nest, copy=True)
        img = ax.projmap(
            map,
            nest=nest,
            coord=coord,
            vmin=min,
            vmax=max,
            xsize=xsize,
            ysize=ysize,
            reso=reso,
            cmap=cmap,
            norm=norm,
            badcolor=badcolor,
            bgcolor=bgcolor,
            alpha=alpha,
        )

        if cbar:
            im = ax.get_images()[0]
            b = im.norm.inverse(np.linspace(0, 1, im.cmap.N + 1))
            v = np.linspace(im.norm.vmin, im.norm.vmax, im.cmap.N)
            mappable = plt.cm.ScalarMappable(
                norm=matplotlib.colors.Normalize(vmin=im.norm.vmin, vmax=im.norm.vmax),
                cmap=cmap,
            )
            if matplotlib.__version__ >= "0.91.0":
                cb = f.colorbar(
                    mappable,
                    ax=ax,
                    orientation="horizontal",
                    shrink=0.5,
                    aspect=25,
                    ticks=PA.BoundaryLocator(),
                    pad=0.15,
                    fraction=0.1,
                    boundaries=b,
                    values=v,
                    format=format,
                )
            else:
                cb = f.colorbar(
                    mappable,
                    orientation="horizontal",
                    shrink=0.5,
                    aspect=25,
                    ticks=PA.BoundaryLocator(),
                    pad=0.15,
                    fraction=0.1,
                    boundaries=b,
                    values=v,
                    format=format,
                )
            if cbar_title is not None:
              cb.ax.set_title(cbar_title, y = -1.4, pad = -14)
            # set title below the colorbar
            # cb.ax.title.set_position([0.5, -5])
            cb.solids.set_rasterized(True)
        ax.set_title(title)
        ax.set_xlabel('l [deg]')
        # add tick marks
        # ax.set_xticks(np.arange(-180, 180, 3), minor=True)
        x_grid_points = np.arange(-20,20+5,5) 
        for x_grid in x_grid_points:
          ax.axvline(x_grid * np.pi/180, color='white', linestyle='-', lw = 0.5,alpha = 0.5)
        ax.set_ylabel('b [deg]')
        y_grid_points = np.arange(-20,20+5,5) 
        for y_grid in y_grid_points:
          ax.axhline(y_grid * np.pi/180, color='white', linestyle='-', lw = 0.5, alpha = 0.5)
        ax.scatter(0,0, marker = '+', color = 'white', s = 100)
        # ax.set_yticks(np.arange(-90, 90, 3), minor=True)
        if not notext:
            ax.text(
                -0.07,
                0.02,
                "%g '/pix,   %dx%d pix"
                % (
                    ax.proj.arrayinfo["reso"],
                    ax.proj.arrayinfo["xsize"],
                    ax.proj.arrayinfo["ysize"],
                ),
                fontsize=12,
                verticalalignment="bottom",
                transform=ax.transAxes,
                rotation=90,
            )
            ax.text(
                -0.07,
                0.6,
                ax.proj.coordsysstr,
                fontsize=14,
                fontweight="bold",
                rotation=90,
                transform=ax.transAxes,
            )
            lon, lat = np.around(ax.proj.get_center(lonlat=True), ax._coordprec)
            ax.text(
                0.5,
                -0.03,
                "(%g,%g)" % (lon, lat),
                verticalalignment="center",
                horizontalalignment="center",
                transform=ax.transAxes,
            )
        # add xlabel and ylabel using ax.text
        ax.text(0.5, -0.1, 'l [5 deg]', transform=ax.transAxes, horizontalalignment='center')
        ax.text(-0.1, 0.5, 'b [5 deg]', transform=ax.transAxes, verticalalignment='center', rotation=90)
        if cbar:
            cb.ax.text(
                1.05,
                0.30,
                unit,
                fontsize=14,
                fontweight="bold",
                transform=cb.ax.transAxes,
                ha="left",
                va="center",
            )
        f.sca(ax)
    finally:
        pylab.draw()
        if wasinteractive:
            pylab.ion()
            # pylab.show()
        if no_plot:
            pylab.close(f)
            f.clf()
            ax.cla()
    if return_projected_map:
        return img
    
def gnomview_plot(map,title=None,cmap=None,cbar_title=r'$\log_{10}(\lambda)$',alpha=np.ones(hp.nside2npix(128))):
    gnomview_ed(
    map,
    fig = 1,
    rot=[0, 0],
    coord = 'G',
    xsize=2500,
    ysize=2500,
    reso=1,
    title=title, 
    hold=True,
    cmap=cmap,
    notext=True,
    cbar_title=cbar_title,
    alpha=alpha,)

def load_inducing_points(self,svi_results):
    if self.is_gp is False:
        print('Not a GP model')
        return None
    
    u_option = self.u_option
    if u_option == 'fixed':
        xu_f = self.xu_f
    elif u_option == 'float':
        lru = svi_results.params["lru"]
        lau = svi_results.params["lau"]

        ru = 20. * expit(lru)
        angu = 2. * jnp.pi * expit(lau)

        xu = ru * jnp.cos(angu)
        yu = ru * jnp.sin(angu)

        xu_f = jnp.vstack([xu.T,yu.T]).T
    return xu_f

def plot_inducing_points(xu_f, x_p, mu):
    # make a scatter plot of the induced points colored by mu
    plt.figure(figsize=(8,6))
    # plt.scatter(xu, yu, c=np.exp(mu)[:Nu], s=50, cmap='viridis')
    Nu = xu_f.shape[0]
    plt.scatter(
        x_p[:, 0],
        x_p[:, 1],
        c='k',
        s=0.1,
        marker='o',
    )
    plt.scatter(xu_f[:,0], xu_f[:,1], c=np.exp(mu[:Nu]), s=10, cmap='viridis')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Inducing Points Colored by GP Mean')

def convert_masked_array_to_hp_array(masked_array, mask, log_option = False, nside = 128):
    '''
    Plot a masked array on a healpix map in log10 scale. 
    That is, the array's indices correspond to applying 
    array[~mask] on the "full" array.
    ''' 
    full_array = np.zeros(hp.nside2npix(nside))
    full_array[~mask] = masked_array
    
    if log_option:
        hp_map = hp.ma(np.log10(full_array))
    else:
        hp_map = hp.ma(full_array)
    hp_map.mask = mask
    return hp_map

def make_corner_plots(samples, with_mean_vlines = False, sim_vlines = False, temp_dict = None, with_log_rate_u = False, print_latex_means = False):
    '''
    Make corner plots from samples dictionary

    Parameters
    ----------
    samples : dict
        Dictionary of samples from the numpyro model
    with_mean_vlines : bool, optional
        Whether to include vertical lines for the mean of each parameter
    with_log_rate_u : bool, optional
        Removes log_rate_u parameter samples if necessary
    '''
    names = list(samples.keys())
    if with_log_rate_u:
        names.remove('log_rate_u')
        if names == []:
            print('No parameters to plot')
            return None

    template_sample_array = np.zeros((len(names), len(samples[names[0]])))
    for i in range(len(names)):
        name = names[i]
        template_sample_array[i] = samples[name]

    fig = corner.corner(template_sample_array.T, labels=names, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12})

    if with_mean_vlines:
        N_var = len(names)
        axes = np.array(fig.axes).reshape((N_var, N_var))
        for i in range(len(names)):
            ax = axes[i,i]
            name = names[i]
            ax.axvline(np.mean(samples[name]), color='red', linestyle='--')

    if sim_vlines:
        N_var = len(names)
        axes = np.array(fig.axes).reshape((N_var, N_var))

        for i in range(len(names)):
            name = names[i]
            ax = axes[i,i]
            if name not in list(temp_dict.keys()):
                continue
            else:
                ax.axvline(temp_dict[name], color='red', linestyle='--')

    if print_latex_means:
        print('LaTeX Table of Parameter Means')
        print('=========================')
        print('Generate Table with OverLeaf using Output Below:')
        print('')
        print_table_multi(names, np.mean(template_sample_array, axis = 1))
        

# def make_sim_file_name(temp_names, sim_key, is_custom_blg=False, custom_blg_id=None):
#     sim_file_name = 'sim_'

#     # NOTE: This does not contain all possible temp_names (Ylm)
#     all_temp_names = ['iso', 'psc', 'bub', 'pib', 'ics', 'blg', 'nfw', 'dsk']

#     if set(temp_names) == set(all_temp_names):
#         sim_file_name += 'all_'
#     else:
#         # order temp_names as in all_temp_names (so input doesn't affect file name order)
#         ordered_temp_names = [name for name in all_temp_names if name in temp_names]
#         for name in ordered_temp_names:
#             if name == 'blg' and is_custom_blg:
#                 sim_file_name += 'cblg' + str(custom_blg_id) + '_'
#             else:
#                 sim_file_name += name + '_'

#     sim_file_name += str(sim_key) + '.npy'

#     return sim_file_name

def make_sim_file_name(temp_names, is_custom_blg=False, custom_blg_id=None):
    sim_file_name = 'sim_'

    # NOTE: This does not contain all possible temp_names (Ylm)
    all_temp_names = ['iso', 'psc', 'bub', 'pib', 'ics', 'blg', 'nfw', 'dsk']

    if set(temp_names) == set(all_temp_names):
        sim_file_name += 'all_'
    else:
        # order temp_names as in all_temp_names (so input doesn't affect file name order)
        ordered_temp_names = [name for name in all_temp_names if name in temp_names]
        for name in ordered_temp_names:
            if name == 'blg' and is_custom_blg:
                sim_file_name += 'cblg' + str(custom_blg_id) + '_'
            else:
                sim_file_name += name + '_'
        # remove last underscore and add slash
        sim_file_name = sim_file_name[:-1] + '/'
    return sim_file_name 

def make_pseudodata_file(temp_names, data_dir, create_dir = False, return_name = True, sim_seed = None, is_custom_blg=False, custom_blg_id=None):
    pseudodata_dir = data_dir + 'pseudodata/'
    sim_dir = pseudodata_dir + make_sim_file_name(temp_names, is_custom_blg=is_custom_blg, custom_blg_id=custom_blg_id)
    if create_dir:
        os.system("mkdir -p "+sim_dir)

    if return_name:
        return sim_dir + 'pois_draw_' + str(sim_seed) + '.npy'

def generate_temp_sample_maps(samples, ebinmodel, gp_samples = None, custom_num = None, nfw_gamma = 'vary'):
    # NOTE: Does not create samples for disk or Ylm

    ie = 10
    nside = ebinmodel.nside
    mask_p = ebinmodel.mask_roi_arr[ie]

    all_temp_names = ['iso', 'psc', 'bub', 'pib', 'ics', 'blg', 'nfw', 'dsk']
    param_names = list(samples.keys())
    names = [params_names.replace('S_', '') for params_names in param_names]

    temp_sample_dict = {}
    num_samples = samples[param_names[0]].shape[0]

    if custom_num is not None:
        num_samples = custom_num

    if gp_samples is not None:
        if custom_num == None:
            num_samples = np.min([samples[param_names[0]].shape[0], gp_samples.shape[0]])
        temp_sample_dict['gp'] = jnp.exp(gp_samples)[:num_samples]
        
        names.remove('log_rate_u')
        if names == []:
            print('No templates to sum')
            return temp_sample_dict

    ordered_temp_names = [name for name in all_temp_names if name in names]
    for name in ordered_temp_names:
        print(name)
        if name in ['iso', 'psc', 'bub']: # rigid templates
            S = samples['S_' + name][:num_samples]
            buf_S = S[:,np.newaxis] # add axis corresponding to spatial info
            norm_temp = jnp.asarray(ebinmodel.temps[name].at_bin(ie, mask=mask_p))
            buf_temp = norm_temp[np.newaxis,:] # add axis corresponding to sample info
            temp_sample_dict[name] = buf_S * buf_temp
        
        elif name == 'pib':
            S_pib = samples['S_' + name]
            pib_temps_at_bin = jnp.asarray([pib_temp.at_bin(ie, mask=mask_p) for pib_temp in ebinmodel.pib_temps])
            temp_sample_dict[name] = np.array([S_pib[i] * pib_temps_at_bin[0] for i in tqdm(range(num_samples))])
        elif name == 'ics':
            S_ics = samples['S_' + name]
            ics_temps_at_bin = jnp.asarray([ics_temp.at_bin(ie, mask=mask_p) for ics_temp in ebinmodel.ics_temps])
            temp_sample_dict[name] = np.array([S_ics[i] * ics_temps_at_bin[0] for i in tqdm(range(num_samples))])
        elif name == 'blg':
            S_blg = samples['S_' + name]
            blg_temps_at_bin = jnp.asarray([blg_temp.at_bin(ie, mask=mask_p) for blg_temp in ebinmodel.blg_temps])
            temp_sample_dict[name] = np.array([S_blg[i] * blg_temps_at_bin[0] for i in tqdm(range(num_samples))])
        
        elif name == 'nfw':
            S_nfw = samples['S_' + name]
            if nfw_gamma == 'vary':
                gamma = samples['gamma']
                temp_sample_dict[name] = np.array([S_nfw[i] * jnp.asarray(ebinmodel.nfw_temp.get_NFW2_template(gamma=gamma[i]))[~mask_p] for i in tqdm(range(num_samples))])
            else:
                gamma = nfw_gamma
                temp_sample_dict[name] = np.array([S_nfw[i] * jnp.asarray(ebinmodel.nfw_temp.get_NFW2_template(gamma=gamma))[~mask_p] for i in tqdm(range(num_samples))])

    return temp_sample_dict

def tot_log_counts_hist(temp_sample_dict, temp_sim_dict = None, temp_sim_names = None, bins = np.linspace(1,5,100), alpha = 0.75, histtype = 'step', gp_model_nfw = False):
    fig = plt.figure(figsize=(12, 6), dpi= 120)
    ax = fig.add_subplot(111)

    all_temp_names = ['iso', 'psc', 'bub', 'pib', 'ics', 'blg', 'gp', 'nfw', 'dsk']
    ccodes = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C5', 'C6', 'C7']
    names = list(temp_sample_dict.keys())

    ordered_names = [name for name in all_temp_names if name in names]
    for k in range(len(ordered_names)):
        name = ordered_names[k]
        idx = all_temp_names.index(name)
        ccode = ccodes[idx]
        # if name == 'gp':
        #     temp_sum = jnp.exp(temp_sample_dict[name]).sum(axis = 1) # sum over spatial bins
        #     ax.hist(np.log10(temp_sum), bins = bins, alpha = 0.75, label = name, density = True, histtype = 'step', color = ccode)
        # else:
        temp_sum = temp_sample_dict[name].sum(axis = 1) # sum over spatial bins
        ax.hist(np.log10(temp_sum), bins = bins, alpha = alpha, label = name, density = True, histtype = histtype, color = ccode)

    if temp_sim_dict is not None:
        names_sim = temp_sim_names # this piece is provided by the "settings" file since we only save a dictionary with all the fit parameters
        ordered_names_sim = [name for name in all_temp_names if name in names_sim]
        if gp_model_nfw: # TODO: Update so it utilizes ID or difference between templates and sim templates
            for k in range(len(ordered_names_sim)):
                name = ordered_names_sim[k]
                idx = all_temp_names.index(name)
                ccode = ccodes[idx]
                if ordered_names_sim[k] == 'nfw':
                    continue
                elif ordered_names_sim[k] == 'blg':
                    temp_sum_sim = temp_sim_dict['blg'].sum(axis = 0) + temp_sim_dict['nfw'].sum(axis = 0)
                    ax.axvline(np.log10(temp_sum_sim), linestyle='--', c = ccode)
                else:
                    temp_sum_sim = temp_sim_dict[name].sum(axis = 0)
                    ax.axvline(np.log10(temp_sum_sim), linestyle='--', c = ccode)
        else:
            for k in range(len(ordered_names_sim)):
                name = ordered_names_sim[k]
                idx = all_temp_names.index(name)
                ccode = ccodes[idx]
                temp_sum_sim = temp_sim_dict[name].sum(axis = 0)
                ax.axvline(np.log10(temp_sum_sim), linestyle='--', c = ccode)

    ax.legend()
    ax.set_xlabel(r'$\log_{10}(\mathrm{counts})$')
    ax.set_ylabel(r'$\mathrm{density}$')
    ax.set_title(r'$\mathrm{Counts\ in\ Unmasked\ Region}$')

def diff_counts_hist(temp_sample_dict, temp_sim_dict = None, temp_sim_names = None, bins = np.linspace(-4000,4000,100)):
    fig = plt.figure(figsize=(12, 6), dpi= 120)
    ax = fig.add_subplot(111)

    all_temp_names = ['iso', 'psc', 'bub', 'pib', 'ics', 'blg', 'gp', 'nfw', 'dsk']
    ccodes = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C5', 'C6', 'C7']
    names = list(temp_sample_dict.keys())

    ordered_names = [name for name in all_temp_names if name in names]
    for k in range(len(ordered_names)):
        name = ordered_names[k]
        idx = all_temp_names.index(name)
        ccode = ccodes[idx]
        # if name == 'gp':
        #     temp_sum = jnp.exp(temp_sample_dict[name]).sum(axis = 1) # sum over spatial bins
        #     ax.hist(np.log10(temp_sum), bins = bins, alpha = 0.75, label = name, density = True, histtype = 'step', color = ccode)
        # else:
        if name == 'gp':
            temp_sum = temp_sample_dict[name].sum(axis = 1) # sum over spatial bins
            temp_sum_sim = temp_sim_dict['blg'].sum(axis = 0)
        else:
            temp_sum = temp_sample_dict[name].sum(axis = 1) # sum over spatial bins
            temp_sum_sim = temp_sim_dict[name].sum(axis = 0)
        ax.hist((temp_sum - temp_sum_sim) / temp_sum_sim, bins = bins, alpha = 0.75, label = name, density = True, histtype = 'step', color = ccode)
        ax.axvline(0, color = 'k', ls = '--')

    ax.legend()
    ax.set_xlabel(r'$\Delta N_{\mathrm{tot, rel}}$')
    ax.set_ylabel(r'$\mathrm{density}$')
    ax.set_title(r'$\mathrm{Counts\ in\ Unmasked\ Region}$')

def tot_counts_hist_all_temps(temp_sample_dict, temp_sim_dict, temp_sim_names = None, bins = np.linspace(-2500,2500,20)):
    fig, ax1 = plt.subplots(figsize=(8, 6), dpi= 120)

    names = list(temp_sample_dict.keys())
    names_sim = temp_sim_names # this piece is provided by the "settings" file since we only save a dictionary with all the fit parameters

    tot_sum = 0.
    for name in names:
        tot_sum += temp_sample_dict[name].sum(axis = 1) # sum over spatial bins

    tot_sum_sim = 0
    for name in names_sim:
        tot_sum_sim += temp_sim_dict[name].sum(axis = 0)

    ax1.hist((tot_sum - tot_sum_sim), bins = bins, alpha = 0.5, label = r'$N_{\rm true} = $' + str(tot_sum_sim), density = True, color='blue')

    
    ax1.set_xlabel(r'$\Delta N_{\rm tot}$')
    ax1.set_ylabel(r'$\mathrm{density}$')

    ax2 = ax1.twiny()

    bins_rel = bins / tot_sum_sim

    ax2.hist((tot_sum - tot_sum_sim) / tot_sum_sim, bins = bins_rel, alpha = 0.5, density = True, color='blue')
    ax2.set_xlabel(r'$\Delta N_{\rm tot, rel}$')

    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.show()

def rebin(a, shape): # rebin an image to lower resolution
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def reduce_cart_res(cart_map, res_scale = 1): # return a map of lower resolution (cart_coords will account for this)
    return rebin(cart_map, (cart_map.shape[0]//res_scale, cart_map.shape[1]//res_scale))

def healpix_to_cart(map_data, mask_roi, n_pixels=80, nside = 128, map_size = 40):
    map_pred = np.zeros(hp.nside2npix(nside))
    map_pred[~mask_roi] = map_data
    map_data_cart = to_cart(map_pred, n_pixels = n_pixels, pixelsize = map_size/n_pixels, frame = 'Galactic')
    return map_data_cart

def multi_healpix_to_cart(map_data_samples, mask_roi, n_pixels=80, nside = 128):
    num_samples = map_data_samples.shape[0]
    map_data_samples_cart = np.zeros((num_samples, n_pixels, n_pixels))
    for i in tqdm(range(num_samples)):
        map_data_samples_cart[i] = healpix_to_cart(map_data_samples[i], mask_roi, n_pixels = n_pixels, nside = nside)
    return map_data_samples_cart

def cart_coords(n_pixels=80, res_scale = 1, map_size = 40):
    x_low = -map_size/2 ; x_high = map_size/2
    Nx1, Nx2 = [n_pixels, n_pixels]
    x1_plt = jnp.linspace(x_low, x_high, int(Nx1 / res_scale) + 1)
    x2_plt = jnp.linspace(x_low, x_high, int(Nx2 / res_scale) + 1)
    x1_c = 0.5 * (x1_plt[1:] + x1_plt[:-1]) ; x2_c = 0.5 * (x2_plt[1:] + x2_plt[:-1])
    x1, x2 = jnp.meshgrid(x1_c, x2_c)
    x = jnp.stack([x1, x2], axis=-1)
    return Nx1, Nx2, x1_plt, x2_plt, x1_c, x2_c, x

def triple_cart_plot(x1_plt, x2_plt, y_obs, rate, mu_r):
    vmax = np.max([np.max(y_obs), np.max(rate), np.max(mu_r)])
    fig = plt.figure(figsize=(18, 6), dpi= 120)
    ax1 = fig.add_subplot(131)
    plot1 = plt.pcolormesh(x1_plt, x2_plt, y_obs, cmap='viridis',
                norm=mpl.colors.Normalize(vmin=0, vmax=vmax))
    ax1.set_title('Raw Counts')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    ax2 = fig.add_subplot(132)
    plot2 = plt.pcolormesh(x1_plt, x2_plt, rate, cmap='viridis',
                norm=mpl.colors.Normalize(vmin=0, vmax=vmax))
    ax2.set_title('True Count Rate')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    ax3 = fig.add_subplot(133)
    plot3 = plt.pcolormesh(x1_plt, x2_plt, mu_r, cmap='viridis',
                norm=mpl.colors.Normalize(vmin=0, vmax=vmax))
    ax3.set_title('Mean Count Rate')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')

    p0 = ax1.get_position().get_points().flatten()
    p1 = ax2.get_position().get_points().flatten()
    p2 = ax3.get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p0[0], -0.1, p2[2]-p0[0], 0.05])

    plt.colorbar(cax=ax_cbar, orientation='horizontal')
    ax_cbar.set_title('Counts')
    
# def cart_plot_1d(x, x1_plt, x2_plt, q, blg_coord, sim_coord, slice_dir = 'horizontal', slice_val = 2.)
#     fig = plt.figure(figsize=(14, 6), dpi= 120)
#     ax1 = fig.add_subplot(121)
#     ax2 = fig.add_subplot(122)

#     if slice_dir == 'horizontal':
#         y_slice = 2.
#         ny = np.where(np.abs(x2_c - y_slice) < 0.5 * res_scale)[0][1]

#         ax1.plot(x[ny,:,0], q[1][ny,:], c = 'red', label = 'GP prediction')
#         ax1.fill_between(x[ny,:,0], q[0][ny,:], q[2][ny,:], color = 'red', alpha = 0.3)
#         ax1.plot(x[ny,:,0], blg_coord[ny,:], c = 'blue', label = 'True')
#         ax1.errorbar(x[ny,:,0], sim_coord[ny,:], yerr = np.sqrt(sim_coord[ny,:]), fmt = 'o', c = 'k', alpha = 0.5)

#         ax1.set_title('Slice at y = {:.2f} deg'.format(x[ny,0,1]))
#         ax1.set_xlabel('x (deg)')

#         ax2.axhline(y = x[ny,0,1], c = 'k', lw = 1)
        
#     elif slice_dir == 'vertical':
#         x_slice = -2
#         nx = np.where(np.abs(x1_c - x_slice) < 0.5 * res_scale)[0][1]

#         ax.plot(x[:,nx,1], q[1][:,nx], c = 'red', label = 'GP prediction')
#         ax.fill_between(x[:,nx,1], q[0][:,nx], q[2][:,nx], color = 'red', alpha = 0.3)
#         ax.plot(x[:,nx,1], blg_coord[:,nx], c = 'blue', label = 'True')
#         ax.errorbar(x[:,nx,1], sim_coord[:,nx], yerr = np.sqrt(sim_coord[:,nx]), fmt = 'o', c = 'k', alpha = 0.5)
#         ax.set_xlabel('y (deg)')
        


#     ax1.set_ylabel('Counts')
#     ax1.legend(fontsize = 14)
#     ax1.axvline(0, color='k', ls = '--', lw = 0.5)
#     ax1.set_yscale('log')

#     ax2.pcolormesh(x1_plt, x2_plt, blg_coord, cmap='Blues', alpha = 0.5)
#     ax2.pcolormesh(x1_plt, x2_plt, q[1], cmap='Reds', alpha = 0.5)
#     ax2.pcolormesh(x1_plt, x2_plt, sim_coord, cmap='Greys', alpha = 0.2)
#     ax2.set_title('Overlaid Rates')
#     ax2.set_xlabel('x')
#     ax2.set_ylabel('y')

def cart_plot_1d(x, x1_plt, x2_plt, x1_c, x2_c, q, blg_coord, sim_coord=None, slice_dir = 'horizontal', slice_val = 2., res_scale = 1, yscale = 'log'):
    fig = plt.figure(figsize=(14, 6), dpi= 120)

    if slice_dir == 'horizontal':
        ax = fig.add_subplot(121)

        y_slice = 2.
        ny = np.where(np.abs(x2_c - y_slice) < 0.5 * res_scale)[0][1]

        ax.plot(x[ny,:,0], q[1][ny,:], c = 'red', label = 'Prediction')
        ax.fill_between(x[ny,:,0], q[0][ny,:], q[2][ny,:], color = 'red', alpha = 0.3)
        ax.plot(x[ny,:,0], blg_coord[ny,:], c = 'blue', label = 'True')
        if sim_coord is not None:
            ax.errorbar(x[ny,:,0], sim_coord[ny,:], yerr = np.sqrt(sim_coord[ny,:]), fmt = 'o', c = 'k', alpha = 0.5)
        ax.set_xlabel('x (deg)')
        ax.set_ylabel('Counts')
        ax.set_title('Slice at y = {:.2f} deg'.format(x[ny,0,1]))
        ax.legend(fontsize = 14)
        ax.axvline(0, color='k', ls = '--', lw = 0.5)
        ax.set_yscale(yscale)

        ax = fig.add_subplot(122)

        ax.pcolormesh(x1_plt, x2_plt, blg_coord, cmap='Blues', alpha = 0.5)
        ax.pcolormesh(x1_plt, x2_plt, q[1], cmap='Reds', alpha = 0.5)
        if sim_coord is not None:
            ax.pcolormesh(x1_plt, x2_plt, sim_coord, cmap='Greys', alpha = 0.2)
        ax.axhline(y = x[ny,0,1], c = 'k', lw = 1)
        ax.set_title('Overlaid Rates')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    elif slice_dir == 'vertical':
        ax = fig.add_subplot(121)

        x_slice = -2
        nx = np.where(np.abs(x1_c - x_slice) < 0.5 * res_scale)[0][1]

        ax.plot(x[:,nx,1], q[1][:,nx], c = 'red', label = 'Prediction')
        ax.fill_between(x[:,nx,1], q[0][:,nx], q[2][:,nx], color = 'red', alpha = 0.3)
        ax.plot(x[:,nx,1], blg_coord[:,nx], c = 'blue', label = 'True')
        if sim_coord is not None:
            ax.errorbar(x[:,nx,1], sim_coord[:,nx], yerr = np.sqrt(sim_coord[:,nx]), fmt = 'o', c = 'k', alpha = 0.5)
        ax.set_xlabel('y (deg)')
        ax.set_ylabel('Counts')
        ax.set_title('Slice at x = {:.2f} deg'.format(x[0,nx,0]))
        ax.legend(fontsize = 14)
        ax.axvline(0, color='k', ls = '--', lw = 0.5)
        ax.set_yscale(yscale)

        ax = fig.add_subplot(122)

        ax.pcolormesh(x1_plt, x2_plt, blg_coord, cmap='Blues', alpha = 0.5)
        ax.pcolormesh(x1_plt, x2_plt, q[1], cmap='Reds', alpha = 0.5)
        if sim_coord is not None:
            ax.pcolormesh(x1_plt, x2_plt, sim_coord, cmap='Greys', alpha = 0.2)
        ax.axvline(x = x[0,nx,0], c = 'k', lw = 1)
        ax.set_title('Overlaid Rates')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    elif slice_dir == 'diagonal_up':
        # Plot fit curves on a slice of increasing radius
        ax = fig.add_subplot(121)

        Nx = len(x1_c)
        ny = np.arange(0,int(Nx),1)

        r = np.linalg.norm(x[ny,ny,:], axis = -1)
        theta = np.arctan2(x[ny,ny,1], x[ny,ny,0])
        ax.plot(r * np.sign(theta), q[1][ny,ny], c = 'red', label = 'Prediction')
        ax.fill_between(r * np.sign(theta), q[0][ny,ny], q[2][ny,ny], color = 'red', alpha = 0.3)
        ax.plot(r * np.sign(theta), blg_coord[ny,ny], c = 'blue', label = 'True')
        ax.axvline(0, color='k', ls = '--', lw = 0.5)
        if sim_coord is not None:
            ax.errorbar(r * np.sign(theta), sim_coord[ny,ny], yerr = np.sqrt(sim_coord[ny,ny]), fmt = 'o', c = 'k', alpha = 0.5)
        ax.set_xlabel('r (deg)')
        ax.set_ylabel('Counts')
        ax.legend(fontsize = 14)
        ax.set_xlim([-20,20])
        ax.set_yscale(yscale)

        ax = fig.add_subplot(122)

        ax.pcolormesh(x1_plt, x2_plt, blg_coord, cmap='Blues', alpha = 0.5)
        ax.pcolormesh(x1_plt, x2_plt, q[1], cmap='Reds', alpha = 0.5)
        if sim_coord is not None:
            ax.pcolormesh(x1_plt, x2_plt, sim_coord, cmap='Greys', alpha = 0.2)
        ax.plot(x[ny,ny,0], x[ny,ny,1], c = 'k', lw = 1)
        ax.set_title('Overlaid Rates')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    elif slice_dir == 'diagonal_down':
        # Plot fit curves on a slice of increasing radius
        ax = fig.add_subplot(121)

        Nx = len(x1_c)
        ny = np.arange(0,int(Nx),1)

        r = np.linalg.norm(x[ny,Nx-ny-1,:], axis = -1)
        theta = np.arctan2(x[ny,Nx-ny-1,1], x[ny,Nx-ny-1,0])
        ax.plot(r * np.sign(-theta), q[1][ny,Nx-ny-1], c = 'red', label = 'Prediction')
        ax.fill_between(r * np.sign(-theta), q[0][ny,Nx-ny-1], q[2][ny,Nx-ny-1], color = 'red', alpha = 0.3)
        ax.plot(r * np.sign(-theta), blg_coord[ny,Nx-ny-1], c = 'blue', label = 'True')
        if sim_coord is not None:
            ax.errorbar(r * np.sign(-theta), sim_coord[ny,Nx-ny-1], yerr = np.sqrt(sim_coord[ny,Nx-ny-1]), fmt = 'o', c = 'k', alpha = 0.5)
        ax.axvline(0, color='k', ls = '--', lw = 0.5)
        ax.set_xlabel('r (deg)')
        ax.set_ylabel('Counts')
        ax.set_xlim([-20,20])
        ax.legend(fontsize = 14)
        ax.set_yscale(yscale)

        ax = fig.add_subplot(122)

        ax.pcolormesh(x1_plt, x2_plt, blg_coord, cmap='Blues', alpha = 0.5)
        ax.pcolormesh(x1_plt, x2_plt, q[1], cmap='Reds', alpha = 0.5)
        if sim_coord is not None:
            ax.pcolormesh(x1_plt, x2_plt, sim_coord, cmap='Greys', alpha = 0.2)
        ax.plot(x[ny,Nx-ny-1,0], x[ny,Nx-ny-1,1], c = 'k', lw = 1)
        ax.set_title('Overlaid Rates')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

def test_page():
    return "Hopefully this is fine"

def list_files(startpath):
    # from startpath, makes tree view of all files
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

def generate_fit_filename(rig_temp_list, hyb_temp_list, var_temp_list, rig_temp_sim, hyb_temp_sim, var_temp_sim, is_gp, gp_deriv, is_custom_blg, custom_blg_id, mod_id, svi_id, sim_seed, svi_seed, return_module_name = True):
    # convert fit settings to ID string
    name_list = ['blg', 'iso', 'bub', 'pib', 'ics', 'nfw', 'psc', 'dsk']
    sim_list = rig_temp_sim + hyb_temp_sim + var_temp_sim
    # id is index of name_list in sim_list
    sim_id = np.sort([name_list.index(i) + 1 for i in sim_list])
    # combine all ids into single string
    sim_id_str = ''.join(map(str, sim_id))
    # include total number of templates in sim_list in string
    sim_id_str =  str(len(sim_list)) + 'p' + sim_id_str

    temp_list = rig_temp_list + hyb_temp_list + var_temp_list
    # id is index of name_list in temp_list
    temp_id = np.sort([name_list.index(i) + 1 for i in temp_list])
    # combine all ids into single string
    temp_id_str = ''.join(map(str, temp_id))
    # include total number of templates in sim_list in string
    temp_id_str =  str(len(temp_list)) + 'p' + temp_id_str

    # GP ID
    if is_gp is False:
        gp_id = str(0)
    if is_gp is True:
        if gp_deriv is False:
            gp_id = str(1)
        if gp_deriv is True:
            gp_id = str(2)

        gp_list = [temp for temp in sim_list if temp not in temp_list]
        temp_gp_id = np.sort([name_list.index(i) + 1 for i in gp_list])
        temp_gp_id_str = ''.join(map(str, temp_gp_id))
        gp_id = gp_id + 'p' + temp_gp_id_str

    # TODO: Create ID for custom bulge templates
    if is_custom_blg:
        blg_id = str(custom_blg_id)
    else:
        blg_id = str(-1)

    str_mod_id = str(mod_id)
    str_svi_id = str(svi_id)
    # str_sim_seed = str(sim_seed)
    # str_svi_seed = str(svi_seed)

    # create ID string
    # id_str = '_'.join([sim_id_str, temp_id_str, gp_id, blg_id, str_mod_id, str_svi_id, str_sim_seed, str_svi_seed])

    id_str = '_'.join([sim_id_str, temp_id_str, gp_id, blg_id, str_mod_id, str_svi_id])

    str_sim_seed = str(sim_seed)
    str_svi_seed = str(svi_seed)
    seed_str = '_'.join([str_sim_seed, str_svi_seed])

    filename = 'fit_' + id_str + '/' + 'seed_' + seed_str
    if return_module_name is True:
        module_name = 'settings_' + id_str + '_' + seed_str 
        return filename, module_name
    else:
        return filename

def generate_fit_filename_from_ids(sim_id, temp_id, gp_id, blg_id, mod_id, svi_id, sim_seed, svi_seed, return_module_name = True):
    # convert ids to strings
    sim_id_str = str(sim_id)
    temp_id_str = str(temp_id)
    gp_id_str = str(gp_id)
    blg_id_str = str(blg_id)
    mod_id_str = str(mod_id)
    svi_id_str = str(svi_id)
    # sim_seed_str = str(sim_seed)
    # svi_seed_str = str(svi_seed)

    # convert any '.' to 'p' ; necessary for smooth module loading
    sim_id_str = sim_id_str.replace('.', 'p')
    temp_id_str = temp_id_str.replace('.', 'p')
    if temp_id_str == '0p0':
        temp_id_str =  '0p'
    gp_id_str = gp_id_str.replace('.', 'p')
    blg_id_str = blg_id_str.replace('.', 'p')
    mod_id_str = mod_id_str.replace('.', 'p')
    svi_id_str = svi_id_str.replace('.', 'p')
    # sim_seed_str = sim_seed_str.replace('.', 'p')
    # svi_seed_str = svi_seed_str.replace('.', 'p')

    # create ID string
    # id_str = '_'.join([sim_id_str, temp_id_str, gp_id_str, blg_id_str, mod_id_str, svi_id_str, sim_seed_str, svi_seed_str])
    id_str = '_'.join([sim_id_str, temp_id_str, gp_id_str, blg_id_str, mod_id_str, svi_id_str])

    str_sim_seed = str(sim_seed)
    str_svi_seed = str(svi_seed)
    seed_str = '_'.join([str_sim_seed, str_svi_seed])

    filename = 'fit_' + id_str + '/' + 'seed_' + seed_str
    if return_module_name is True:
        module_name = 'settings_' + id_str + '_' + seed_str 
        return filename, module_name
    else:
        return filename

def summary_from_filename(id_str):
    # default list of template names; ID = index
    name_list = ['blg', 'iso', 'bub', 'pib', 'ics', 'nfw', 'psc', 'dsk']

    # print id
    print('ID: ', id_str)

    # extract all the above information from id_str and print it out
    id_list = id_str.split('_')

    if id_list[0] != 'fit':
        print('ID string is not a fit ID')
        return None

    # convert sim_id_str to a list of names now
    sim_id_str_recover = id_list[1]
    sim_id_str_recover = sim_id_str_recover.split('p')[1]
    sim_list = [name_list[int(i)-1] for i in sim_id_str_recover]
    print('Simulated Templates: ', sim_list)

    # repeat for temp_list
    temp_id_str_recover = id_list[2]
    temp_id_str_recover = temp_id_str_recover.split('p')[1]
    temp_list = [name_list[int(i)-1] for i in temp_id_str_recover]
    print('Fit Templates: ', temp_list)

    # repeat for gp_list
    gp_id_str_recover = id_list[3]
    if gp_id_str_recover == '0':
        is_gp = False
        print('GP Template: ', is_gp)
    else:
        gp_id_str_recover = gp_id_str_recover.split('p')[1]
        if gp_id_str_recover[0] == '1':
            gp_deriv = False
            print('GP Template: ', gp_deriv)
        if gp_id_str_recover[0] == '2':
            gp_deriv = True
            print('GP Template: ', gp_deriv)
        gp_list = [name_list[int(i)-1] for i in gp_id_str_recover[:]]
        print('GP is Fitting To: ', gp_list)

    # repeat for blg_id
    blg_id_str_recover = id_list[4]
    if blg_id_str_recover == '0':
        is_custom_blg = False
        print('Custom Bulge: ', is_custom_blg)
    else:
        is_custom_blg = True
        blg_id = int(blg_id_str_recover)
        print('Custom Bulge: ', is_custom_blg)
        print('Custom Bulge ID: ', blg_id)

    # repeat for mod_id
    mod_id_str_recover = id_list[5]
    mod_id = int(mod_id_str_recover)
    print('Model ID: ', str(mod_id) + ' ; See settings.py module for more details')

    # repeat for svi_id
    # svi_id_str_recover = id_list[6]
    # svi_id = int(svi_id_str_recover)
    # print('SVI ID: ', str(svi_id) + ' ; See settings.py module for more details')

    # # repeat for sim_seed
    # sim_seed_str_recover = id_list[7]
    # sim_seed = str(sim_seed_str_recover)
    # print('Pseudodata Seed: ', sim_seed)

    # # repeat for svi_seed
    # svi_seed_str_recover = id_list[8]
    # svi_seed = str(svi_seed_str_recover)
    # print('SVI Fit Seed: ', svi_seed)

    # repeat for svi_id
    svi_id_str_recover = id_list[6].replace('/seed', '')
    svi_id = int(svi_id_str_recover)
    print('SVI ID: ', str(svi_id) + ' ; See settings.py module for more details')

    # repeat for sim_seed
    sim_seed_str_recover = id_list[7]
    sim_seed = str(sim_seed_str_recover)
    print('Pseudodata Seed: ', sim_seed)

    # repeat for svi_seed
    svi_seed_str_recover = id_list[8]
    svi_seed = str(svi_seed_str_recover)
    print('SVI Fit Seed: ', svi_seed)

def load_data_dir(sim_name):
#     main_dir = '/home/edr76/gce-bulge-ed/gce-prob-prog-ed-v0.2/'
    main_dir = '/data/edr76/gce-prob-prog-gp/'
    data_dir = main_dir + 'data/synthetic_data/' + sim_name + '/'
    return data_dir

def poisson_interval(k, alpha=0.32): 
    """ Uses chi2 to get the poisson interval.
    """
    a = alpha
    low, high = (chi2.ppf(a/2, 2*k) / 2, chi2.ppf(1-a/2, 2*k + 2) / 2)
    if k == 0: 
        low = 0.0
    return k - low, high - k

# ======================================================================
# ======================================================================
                # MESH GRID GENERATION FUNCTIONS
    # References:
        # Development Directory: od/projects/pswavelets/gce/notebooks
            # grid_points_per_projected_map.ipynb
# ======================================================================
# ======================================================================

def build_mesh_(x_min, x_max, y_min, y_max, step_x, step_y, return_arrays_for_plotting=False):
    """
    Build 2D mesh given bounds and step sizes.
    
    :param x_min = x-grid lower bound 
    :param x_max = x-grid upper bound 
    :param y_min = y-grid lower bound 
    :param y_max = y-grid upper bound 
    :param step_x = x-grid step size
    :param step_y = y-grid step size
    :param return_arrays_for_plotting = boolean value for plotting
    
    :output 
        return_arrays_for_plotting = False: (x,y) mesh grid
        return_arrays_for_plotting = True: 
            [0] = (x,y) mesh grid
            [1] = array of (x,y) points making up grid
            [2] = array of lower-bound of "x-bins" used to define x-grid points
            [3] = array of lower-bound of "y-bins" used to define y-grid points
    """
    arr_x_plot = np.arange(x_min,x_max+step_x,step_x, dtype = float)
    arr_y_plot = np.arange(y_min,y_max+step_y,step_y, dtype = float)
    arr_x = 0.5 * (arr_x_plot[:-1] + arr_x_plot[1:])
    arr_y = 0.5 * (arr_y_plot[:-1] + arr_y_plot[1:])
    Nx = len(arr_x) ; Ny = len(arr_y)

    mesh_x, mesh_y = np.meshgrid(arr_x,arr_y) # each output array (NxN shaped) contains x or y value at given (i,j)-th position
    mesh_xy = np.stack((mesh_x, mesh_y), axis=-1)
    arr_r = mesh_xy.reshape(Nx*Ny,2) # flatten to 2D array
    
    if return_arrays_for_plotting == False:
        return mesh_xy
    else:
        return [mesh_xy, arr_r, arr_x_plot, arr_y_plot]
    
def square_mesh(x_min, x_max, y_min, y_max, Nx, Ny):
    arr_x = np.linspace(x_min, x_max, Nx)
    arr_y = np.linspace(y_min, y_max, Ny)
    mesh_x, mesh_y = np.meshgrid(arr_x,arr_y)
    mesh_xy = np.stack((mesh_x, mesh_y), axis=-1)
    arr_r = mesh_xy.reshape(Nx*Ny,2)
    return mesh_xy, arr_r

def sunflower(n: int, alpha: float) -> np.ndarray:
    '''
    https://stackoverflow.com/questions/28567166/uniformly-distribute-x-points-inside-a-circle
    '''
    # Number of points respectively on the boundary and inside the cirlce.
    n_exterior = np.round(alpha * np.sqrt(n)).astype(int)
    n_interior = n - n_exterior

    # Ensure there are still some points in the inside...
    if n_interior < 1:
        raise RuntimeError(f"Parameter 'alpha' is too large ({alpha}), all "
                           f"points would end-up on the boundary.")
    # Generate the angles. The factor k_theta corresponds to 2*pi/phi^2.
    k_theta = np.pi * (3 - np.sqrt(5))
    angles = np.linspace(k_theta, k_theta * n, n)

    # Generate the radii.
    r_interior = np.sqrt(np.linspace(0, 1, n_interior))
    r_exterior = np.ones((n_exterior,))
    r = np.concatenate((r_interior, r_exterior))

    # Return Cartesian coordinates from polar ones.
    return r.reshape(n, 1) * np.stack((np.cos(angles), np.sin(angles)), axis=1)

def hex_grid(radius, d_unit):
    '''
    https://www.mathworks.com/matlabcentral/answers/1468201-how-to-create-a-circle-filled-with-equidistant-points-inside-it
    '''
    d = d_unit * radius
    # xall=[]; yall=[]
    dy = np.sqrt(3)/2 * d
    ny = int(np.floor(radius/dy))
    for i in range(-ny, ny+1):
        y = dy*i
        if i % 2==0:
            nx = int(np.floor((np.sqrt(radius**2. - y**2.))/d))
            x = np.arange(-nx, nx + 1)*d
        else:
            nx = int(np.floor((np.sqrt(radius**2. - y**2.)-d/2)/d))
            x = np.arange(-nx-0.5,nx+0.5+1)*d
        
        if i == -ny:
            xall = x
            yall = np.ones(len(x))*y
        else:
            xall = np.concatenate((xall, x))
            yall = np.concatenate((yall, np.ones(len(x))*y))

    xall = np.array(xall)
    yall = np.array(yall)
    xy_all = np.stack((xall, yall), axis=1)
    return xy_all