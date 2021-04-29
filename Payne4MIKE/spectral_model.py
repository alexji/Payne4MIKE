# code for a spectral model, i.e. predicting the spectrum of a single star in normalized space.
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
from scipy import interpolate
from scipy import signal
from scipy.stats import norm

#=======================================================================================================================

def leaky_relu(z):
    '''
    This is the activation function used by default in all our neural networks.
    '''
    return z*(z > 0) + 0.01*z*(z < 0)

def sigmoid(z):
    '''
    standard sigmoid
    '''
    return 1./(1 + np.exp(-z))

#--------------------------------------------------------------------------------------------------------------------------

def get_spectrum_from_neural_net(scaled_labels, NN_coeffs):

    '''
    Predict the rest-frame spectrum (normalized) of a single star.
    We input the scaled stellar labels (not in the original unit).
    Each label ranges from -0.5 to 0.5
    '''
    
    # assuming your NN has two hidden layers.
    w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max = NN_coeffs
    inside = np.einsum('ij,j->i', w_array_0, scaled_labels) + b_array_0
    outside = np.einsum('ij,j->i', w_array_1, leaky_relu(inside)) + b_array_1
    spectrum = np.einsum('ij,j->i', w_array_2, leaky_relu(outside)) + b_array_2
    return spectrum

class SpectralModel(object):
    """
    A class that encompasses a Payne spectral model.
    
    The coefficients of the model in order are:
    num_label: stellar labels (this is the trained NN)
    num_order*(coeff_poly+1): number of polynomial coefficients (this is the continuum model)
    num_chunk*2: number of nuisance parameters (this is the RV and vbroad)
    """
    
    def __init__(self,
            NN_coeffs,
            num_stellar_labels,
            x_min, x_max,
            wavelength_payne,
            errors_payne,
            num_order, coeff_poly,
            num_chunk,
            chunk_order_min=None, chunk_order_max=None,
    ):
        self._NN_coeffs = NN_coeffs
        self._num_stellar_labels = num_stellar_labels
        self._x_min = x_min
        self._x_max = x_max
        self._wavelength_payne = wavelength_payne
        self._errors_payne = errors_payne
        self._num_order = num_order
        self._coeff_poly = coeff_poly
        self._num_chunk = num_chunk
        
        if chunk_order_min is None:
            self.chunk_order_min = [0]
        if chunk_order_max is None:
            self.chunk_order_max = [num_chunk-1]
        assert num_chunk == len(self.chunk_order_min)
        assert num_chunk == len(self.chunk_order_max)
        
        
    ### Functions to define in subclasses
    @staticmethod
    def load(fname, num_order, coeff_poly=6, errors_payne=None,
             num_chunk=1, chunk_order_min=None, chunk_order_max=None):
        """
        """
        raise NotImplementedError
    def get_spectrum_from_neural_net(self, scaled_labels):
        """
        Predict the rest-frame spectrum (normalized) of a single star.
        We input the scaled stellar labels (not in the original unit).
        Each label ranges from -0.5 to 0.5
        """
        raise NotImplementedError
    
    ### Functions with default behavior you may want to redefine
    def transform_coefficients(self, popt):
        """
        Transform coefficients into human-readable
        """
        popt_new = popt.copy()
        popt_new[:self.num_stellar_labels] = (popt_new[:self.num_stellar_labels] + 0.5)*(self.x_max-self.x_min) + self.x_min
        popt_new[0] = popt_new[0]*1000.
        for ichunk in range(self.num_chunk):
            irv = -1 - 2*(self.num_chunk - ichunk - 1)
            popt_new[irv] = popt_new[irv]*100.
        return popt_new
    def normalize_stellar_parameter_labels(self, labels):
        """
        Turn physical stellar parameter values into normalized values.
        """
        labels = np.ravel(labels)
        labels[0] = labels[0]/1000.
        new_labels = (labels - self.x_min) / (self.x_max - self.x_min) - 0.5
        assert np.all(np.round(new_labels,3) >= -0.51), (new_labels, labels)
        assert np.all(np.round(new_labels,3) <=  0.51), (new_labels, labels)
        return new_labels
    
    ### The main model evaluation
    def evaluate(self, labels, wavelength, wavelength_normalized=None):
        """
        Evaluate this model at these labels and wavelength
        """
        # Get normalized wavelength for continuum evaluation
        if wavelength_normalized is None:
            wavelength_normalized = self.whitten_wavelength(wavelength)*100.
        
        num_order, num_pixel = wavelength.shape
        spec_predict = np.zeros(num_order*num_pixel)
        errs_predict = np.zeros(num_order*num_pixel)
        
        # make payne models
        _full_spec = self.get_spectrum_from_neural_net(
            scaled_labels = labels[:self.num_stellar_labels]
        )
        
        # allow different RV and broadening for each chunk
        for ichunk in range(self.num_chunk):
            irv = -1 - 2*(self.num_chunk - ichunk - 1)
            ivbroad = -2 - 2*(self.num_chunk - ichunk - 1)

            # Broadening kernel
            win = norm.pdf((np.arange(21)-10.)*(self.wavelength_payne[1]-self.wavelength_payne[0]),\
                           scale=labels[ivbroad]/3e5*5000)
            win = win/np.sum(win)
            
            # vbroad and RV
            full_spec = signal.convolve(_full_spec, win, mode='same')
            full_spec = self.doppler_shift(self.wavelength_payne, full_spec, labels[irv]*100.)
            errors_spec = self.doppler_shift(self.wavelength_payne, self.errors_payne, labels[irv]*100.)
        
            # interpolate into the observed wavelength
            f_flux_spec = interpolate.interp1d(self.wavelength_payne, full_spec)
            f_errs_spec = interpolate.interp1d(self.wavelength_payne, errors_spec)
            
            # loop over all orders
            spec_predict = np.zeros(num_order*num_pixel)
            errs_predict = np.zeros(num_order*num_pixel)
            for k in range(self.chunk_order_min[ichunk], self.chunk_order_max[ichunk]+1):
                scale_poly = 0
                for m in range(self.coeff_poly):
                    scale_poly += (wavelength_normalized[k,:]**m)*labels[self.num_stellar_labels+self.coeff_poly*k+m]
                spec_predict[k*num_pixel:(k+1)*num_pixel] = scale_poly*f_flux_spec(wavelength[k,:])
                errs_predict[k*num_pixel:(k+1)*num_pixel] = scale_poly*f_errs_spec(wavelength[k,:])
            
        return spec_predict, errs_predict
    
    ### Generally useful static methods
    @staticmethod
    def whitten_wavelength(wavelength):
        """
        normalize the wavelength of each order to facilitate the polynomial continuum fit
        """
        wavelength_normalized = np.zeros(wavelength.shape)
        for k in range(wavelength.shape[0]):
            mean_wave = np.mean(wavelength[k,:])
            wavelength_normalized[k,:] = (wavelength[k,:]-mean_wave)/mean_wave
        return wavelength_normalized
    @staticmethod
    def doppler_shift(wavelength, flux, dv):
        """
        dv is in km/s
        positive dv means the object is moving away.
        """
        c = 2.99792458e5 # km/s
        doppler_factor = np.sqrt((1 - dv/c)/(1 + dv/c))
        new_wavelength = wavelength * doppler_factor
        new_flux = np.interp(new_wavelength, wavelength, flux)
        return new_flux
    
    ### Class Properties
    # Properties of the Payne model
    @property
    def NN_coeffs(self):
        return self._NN_coeffs
    @property
    def num_stellar_labels(self):
        return self._num_stellar_labels
    @property
    def x_min(self):
        return self._x_min
    @property
    def x_max(self):
        return self._x_max
    @property
    def wavelength_payne(self):
        return self._wavelength_payne
    @property
    def errors_payne(self):
        return self._errors_payne
    # Nuisance parameters
    @property
    def num_order(self):
        return self._num_order
    @property
    def coeff_poly(self):
        return self._coeff_poly
    @property
    def num_chunk(self):
        return self._num_chunk
    @property
    def num_all_labels(self):
        return self.num_stellar_labels + (1+self.coeff_poly)*self.num_order + 2*self.num_chunk
    
class DefaultPayneModel(SpectralModel):
    @staticmethod
    def load(fname, num_order, coeff_poly=6, errors_payne=None,
             num_chunk=1, chunk_order_min=None, chunk_order_max=None):
        
        tmp = np.load(fname)
        w_array_0 = tmp["w_array_0"]
        w_array_1 = tmp["w_array_1"]
        w_array_2 = tmp["w_array_2"]
        b_array_0 = tmp["b_array_0"]
        b_array_1 = tmp["b_array_1"]
        b_array_2 = tmp["b_array_2"]
        x_min = tmp["x_min"]
        x_max = tmp["x_max"]
        wavelength_payne = tmp["wavelength_payne"]
        NN_coeffs = (w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max)
        
        num_stellar_labels = w_array_0.shape[1]
        
        if errors_payne is None:
            errors_payne = np.zeros_like(wavelength_payne)
        
        return DefaultPayneModel(
            NN_coeffs, num_stellar_labels, x_min, x_max,
            wavelength_payne, errors_payne,
            num_order, coeff_poly, num_chunk,
            chunk_order_min=chunk_order_min,
            chunk_order_max=chunk_order_max
        )

    def get_spectrum_from_neural_net(self, scaled_labels):
        """
        Predict the rest-frame spectrum (normalized) of a single star.
        We input the scaled stellar labels (not in the original unit).
        Each label ranges from -0.5 to 0.5
        """
        w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max = self.NN_coeffs
        inside = np.einsum('ij,j->i', w_array_0, scaled_labels) + b_array_0
        outside = np.einsum('ij,j->i', w_array_1, leaky_relu(inside)) + b_array_1
        spectrum = np.einsum('ij,j->i', w_array_2, leaky_relu(outside)) + b_array_2
        return spectrum
        
