# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 11:00:18 2025

@author: chiliaeva

"""


import os 

import numpy as np
# import cv2 as cv
from scipy import interpolate

from spas.metadata import read_metadata


#%%

type_reco = 'had_reco'     # 'had_reco' or 'nn_reco'
type_reco_npz = type_reco + '.npz'



root = 'C:/Users/chiliaeva/Documents/Resultats_traitement/'
# root = 'D:/'

root_ref = root + 'ref/'


file_metadata_0 = root + '/wavelengths_metadata.json'


metadata, acquisition_params, spectrometer_params, dmd_params = read_metadata(file_metadata_0)
wavelengths = acquisition_params.wavelengths


#%% DEFINE FIT FUNCTION


def func_fit(x, a1, a2, a3, shift620, shift634, lambd_c, sigma):
    return a1*func620(x-shift620) + a2*func634(x-shift634) + a3*np.exp(-(lambd_c-x)**2/sigma**2)


    
#%% REFERENCE SPECTRA


file_name_ppix620 = 'ref620_3lamda.npy'
file_name_ppix634 = 'ref634_3lamda.npy'
file_name_lambda = 'Lambda.npy'
 

ppix620 = np.load(root_ref + file_name_ppix620)
ppix634 = np.load(root_ref + file_name_ppix634)
lambd = np.load(root_ref + file_name_lambda)
 
 
spectr634 = ppix634[0, :] 
spectr634[0] = 0 # otherwise kernel dies
spectr620 = ppix620[0, :]
spectr620[0] = 0


del ppix620
del ppix634

 
 
 # Normalize the reference spectra
 
spectr620_norm = spectr620/np.amax(spectr620)
spectr620 = spectr620_norm
del spectr620_norm
 
spectr634_norm = spectr634/np.amax(spectr634)
spectr634 = spectr634_norm
del spectr634_norm
 
 
crop_start = np.digitize(wavelengths[0], lambd, right=True) # crop the ref spectra, keep the part from wavelengths[0] to wavelengths[-1]
crop_stop = np.digitize(wavelengths[-1], lambd, right=True)


lambd_crop = lambd[crop_start:crop_stop]
spectr620_crop = spectr620[crop_start:crop_stop]
spectr634_crop = spectr634[crop_start:crop_stop]

lambd = lambd_crop
spectr620 = spectr620_crop
spectr634 = spectr634_crop

del lambd_crop
del spectr620_crop
del spectr634_crop

 
 
# Interpolate the reference spectra 
 
func620 = interpolate.make_interp_spline(lambd, spectr620)  
func634 = interpolate.make_interp_spline(lambd, spectr634)

spectr620_interp = func620(wavelengths) # import wavelengths from metadata
spectr634_interp = func634(wavelengths)


# save in ref folder : 
np.save(root_ref + '_spectr620_interp.npy', spectr620_interp)
np.save(root_ref + '_spectr634_interp.npy', spectr634_interp)


