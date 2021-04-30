import numpy as np
import matplotlib.pyplot as plt
import os, sys, time

from Payne4MIKE.spectral_model import DefaultPayneModel
from Payne4MIKE import utils, fitting

from scipy import signal

if __name__=="__main__":
    wavelength, spectrum, spectrum_err = utils.read_in_example()
    wavelength, spectrum, spectrum_err = utils.cut_wavelength(wavelength, spectrum, spectrum_err,
                                                              wavelength_min = 3500, wavelength_max = 7000)
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


    RV_array = np.array([-0.4])


    ########################################
    print("Running with RPA4 relu300")
    path = "/home/aji/train_payne_rpa4/relu300/NN_normalized_spectra_float16.npz"
    model = DefaultPayneModel.load(path, num_order=num_order)
    errors_payne = utils.read_default_model_mask(wavelength_payne=model.wavelength_payne)
    model = DefaultPayneModel.load(path, num_order=num_order, errors_payne=errors_payne)
    
    print("starting fit")
    start = time.time()
    out = fitting.fit_global(spectrum, spectrum_err, spectrum_blaze, wavelength,
                             model,
                             RV_array = RV_array, order_choice=[4])
    popt_best, model_spec_best, chi_square = out
    print("Took",time.time()-start)
    popt_print = model.transform_coefficients(popt_best)
    print("[Teff [K], logg, vt, Fe/H, Alpha/Fe, C/Fe] = ",\
          int(popt_print[0]*1.)/1.,\
          int(popt_print[1]*100.)/100.,\
          int(popt_print[2]*100.)/100.,\
          int(popt_print[3]*100.)/100.,\
          int(popt_print[4]*100.)/100.,\
          int(popt_print[5]*100.)/100.,\
    )
    print("vbroad [km/s] = ", int(popt_print[-2]*10.)/10.)
    print("RV [km/s] = ", int(popt_print[-1]*10.)/10.)
    print("Chi square = ", chi_square)
    
    print("Running with the default Payne4MIKE")
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'../Payne4MIKE/other_data/NN_normalized_spectra_float16.npz')
    model = DefaultPayneModel.load(path, num_order=num_order)
    errors_payne = utils.read_default_model_mask(wavelength_payne=model.wavelength_payne)
    model = DefaultPayneModel.load(path, num_order=num_order, errors_payne=errors_payne)
    
    print("starting fit")
    start = time.time()
    out = fitting.fit_global(spectrum, spectrum_err, spectrum_blaze, wavelength,
                             model,
                             RV_array = RV_array, order_choice=[4])
    popt_best, model_spec_best, chi_square = out
    print("Took",time.time()-start)
    popt_print = model.transform_coefficients(popt_best)
    print("[Teff [K], logg, Fe/H, Alpha/Fe] = ",\
          int(popt_print[0]*1.)/1.,\
          int(popt_print[1]*100.)/100.,\
          int(popt_print[2]*100.)/100.,\
          int(popt_print[3]*100.)/100.)
    print("vbroad [km/s] = ", int(popt_print[-2]*10.)/10.)
    print("RV [km/s] = ", int(popt_print[-1]*10.)/10.)
    print("Chi square = ", chi_square)

