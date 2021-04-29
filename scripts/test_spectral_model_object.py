import numpy as np
import matplotlib.pyplot as plt
import os, sys, time

from Payne4MIKE.spectral_model import DefaultPayneModel
from Payne4MIKE import utils, fitting

from scipy import signal

if __name__=="__main__":
    wavelength, spectrum, spectrum_err = utils.read_in_example()
    num_order, num_pix = wavelength.shape
    #blaze_spectrum = np.array([signal.medfilt(spectrum[i], 51) for i in range(num_order)])
    wavelength_blaze, spectrum_blaze, spectrum_err_blaze = utils.read_in_blaze_spectrum()
    
    dist = np.abs(wavelength[:, np.newaxis] - wavelength_blaze)
    potentialClosest = dist.argmin(axis=1)[:,0]
    wavelength_blaze = wavelength_blaze[potentialClosest,:]
    spectrum_blaze = spectrum_blaze[potentialClosest,:]
    spectrum_err_blaze = spectrum_err_blaze[potentialClosest,:]
    # cull nonsensible values
    spectrum = np.abs(spectrum)
    spectrum_blaze = np.abs(spectrum_blaze)
    # rescale the spectra by its median so it has a more reasonable y-range
    spectrum, spectrum_err = utils.scale_spectrum_by_median(spectrum, spectrum_err)
    spectrum_blaze, spectrum_err_blaze = utils.scale_spectrum_by_median(spectrum_blaze, spectrum_err_blaze)
    # eliminate zero values in the blaze function to avoid dividing with zeros
    # the truncation is quite aggresive, can be improved if needed
    ind_valid = np.min(np.abs(spectrum_blaze), axis=0) != 0
    spectrum_blaze = spectrum_blaze[:,ind_valid]
    spectrum_err_blaze = spectrum_err_blaze[:,ind_valid]
    wavelength_blaze = wavelength_blaze[:,ind_valid]
    # match the wavelength (blaze -> spectrum)
    spectrum_blaze, wavelength_blaze = utils.match_blaze_to_spectrum(wavelength, spectrum, wavelength_blaze, spectrum_blaze)
    # use the blaze to determine telluric region
    smooth_length = 30 # number of pixel in a block that we use to search for telluric features
    threshold = 0.9
    spectrum_err = utils.mask_telluric_region(spectrum_err, spectrum_blaze, smooth_length=30, threshold=0.9)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'../Payne4MIKE/other_data/NN_normalized_spectra_float16.npz')
    model = DefaultPayneModel.load(path, num_order=num_order)
    
    print("starting fit")
    start = time.time()
    RV_array=np.linspace(-2,2.,6)
    out = fitting.fit_global(spectrum, spectrum_err, spectrum_blaze, wavelength,
                             model,
                             RV_array = RV_array, order_choice=[4])
    popt_best, model_spec_best, chi_square = out
    print("Took",time.time()-start)
    x_max, x_min = model.x_max, model.x_min
    popt_best[:4] = (popt_best[:4] + 0.5)*(x_max-x_min) + x_min
    popt_best[0] = popt_best[0]*1000.
    print("[Teff [K], logg, Fe/H, Alpha/Fe] = ",\
          int(popt_best[0]*1.)/1.,\
          int(popt_best[1]*100.)/100.,\
          int(popt_best[2]*100.)/100.,\
          int(popt_best[3]*100.)/100.)
    print("vbroad [km/s] = ", int(popt_best[-2]*10.)/10.)
    print("RV [km/s] = ", int(popt_best[-1]*1000.)/10.)
    print("Chi square = ", chi_square)

