# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 12:40:57 2025

@author: chiliaeva
"""


import numpy as np
from scipy import signal
from scipy.integrate import trapezoid



def find_max_bounds(spectrum, wavelengths, bounds):
    index_start = np.digitize(bounds[0], wavelengths, right=True) # crop the ref spectra, keep the part from wavelengths[0] to wavelengths[-1]
    index_stop = np.digitize(bounds[1], wavelengths, right=True)
    pos = np.argmax(spectrum[index_start:index_stop])
    return index_start + pos
    



def compute_snr(spectrum_tab, wavelengths, x, y, std_bounds, max_interval):
    
    ''' Args : 
    Inputs : 
    spectrum_tab : 3D array containing all the saved spectra (Nx, Ny, Nlambda)
    wavelengths : 1D array wavelengths scale (Nlambda)
    x, y : sptial position of the spectrum
    std_bounds : 2 element sequence, wavelengths between which the std is computed 
    max_interval : 2 element sequence, interval of wavelength where the maximum of fluorescence is expected to be found
    Outputs : 
    snr : scalar 
    '''

    #   wavelengths : wavelengths scale, 1D array
    #   spectrum_tab : matrix containing all the saved spectra, 3D array (Nx, Ny, Nlambda)
    
    spectrum = spectrum_tab[x,y,:]

    # 1) Apply low-pass filter

    fs = 1/(wavelengths[1]-wavelengths[0])
    fc = 0.01 * (1/(wavelengths[1]-wavelengths[0]))
    b, a = signal.butter(4, fc, 'lp', fs=fs, output='ba')
    filtered_spectrum = signal.filtfilt(b, a, spectrum_tab[x,y,:],  padlen=3 * max(len(b), len(a)))
    
    # 2) Spectrum - filtered spectrum 
    diff = spectrum-filtered_spectrum
    
    
    index_start = np.digitize(std_bounds[0], wavelengths, right=True) # crop the ref spectra, keep the part from wavelengths[0] to wavelengths[-1]
    index_stop = np.digitize(std_bounds[1], wavelengths, right=True)
    
    # 3) Compute STD of the noise
    std = np.std(diff[index_start:index_stop])
    
    # 4) Find max of fluorescence 
    pos_max = find_max_bounds(spectrum, wavelengths, max_interval)
    max_value = spectrum[pos_max]
    
    # 5) Select interval such as spectrum(x) >= 0.7 * max 
    selection = np.where(spectrum >= 0.7*max_value, spectrum, 0)
    nb = sum(1 for x in selection if x>0)


    # 6) Integrate the selection
    
    wvlgth_start = np.digitize(max_interval[0], wavelengths, right=True)
    wvlgth_stop = np.digitize(max_interval[1], wavelengths, right=True)
    integral = trapezoid(selection[wvlgth_start : wvlgth_stop], wavelengths[wvlgth_start : wvlgth_stop])  # Integrate y with respect to t (numerically)

    print(wvlgth_start)
    print(wvlgth_stop)
    
    # 7) Compute SNR
    snr = integral/(nb*std)# SNR = 197 for P67B2 = huge !? 
    
    return nb, std, snr, integral

    






























