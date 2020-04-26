# for measuring an initial radial velocity using cross correlation
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
import os
from scipy import interpolate, signal, optimize
from .read_spectrum import read_carpy_fits, Spectrum1D
import utils
import warnings

def measure_order_velocities(wavelength, spectrum, spectrum_err,
                             template_wavelength, template_spectrum,
                             window=51):
    """
    Measure radial velocity of all orders by cross-correlation against a rest-frame template
    
    Can be used to initialize radial velocity array
    """
    
    rvs = np.zeros(wavelength.shape[0]) + np.nan
    
    for k in wavelength.shape[0]:
        ## Normalize spectrum by median
        medcont = signal.median_filter(spectrum[k,:], window)
        norm = spectrum[k,:]/medcont
        norm_err = spectrum_err[k,:]/medcont
        
        ## Run cross correlation
        try:
            rv = cross_correlate(wavelength[k,:], norm, norm_err,
                                 template_wavelength, template_spectrum)
            rvs[k] = rv
        except Exception as e:
            warnings.warn("Cross correlation failed on order {}:".format(k))
            warnings.warn(e)
        
    return rvs

def cross_correlate(wavelength, norm, norm_err,
                    template_wavelength, template_spectrum,
                    vmin=-500, vmax=500, dv=1, full_output=False):
    """
    Brute force cross correlation and find minimum
    """
    
    voff = np.arange(vmin, vmax+dv, dv)
    chi2arr = np.zeros_like(voff)
    for i, v in enumerate(voff):
        shifted_template_spectrum = utils.doppler_shift(template_wavelength, template_spectrum, v)
        chi2arr[i] = np.nansum(((norm - shifted_template_spectrum)/norm_err))
    # Remove nans and infs for interpolation range
    finite = np.isfinite(chi2arr)
    voff, chi2arr = voff[finite], chi2arr[finite]
    vbest = voff[np.argmin(chi2arr)]
    
    chi2func = interpolate.interp1d(voff, chi2arr, fill_value=np.inf)
    if vbest == voff[0]: vbest = voff[1]
    if vbest == voff[-1]: vbest = voff[-2]
    optres = optimize.minimize_scalar(chi2func, bracket=[voff[0],vbest,voff[-1]])
    if not optres.success:
        warnings.warn("RV optimization did not succeed")
    vfit = optres.x
    
    return vfit
