# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 10:39:39 2025

@author: chiliaeva

Preprocessing program for just 1 measurement (no loop)
"""



import os 

import numpy as np
import cv2 as cv
from scipy import interpolate

from spas.metadata2 import read_metadata


#%%

threshold_ = 4e5 # nb counts/pixel on background for t_i = 1s, for a 16x16 image # threshold for binary masks


type_reco = 'nn_reco'     # 'had_reco' or 'nn_reco'
type_reco_npz = type_reco + '.npz'
if type_reco == 'nn_reco':
    threshold_ = threshold_/4
 
    
root = 'D:/'
root_data = root + 'd/' # all the patient folders were saved in the root_data directory
# folders = os.listdir(root_data)

root_ref = root + 'ref/' # the reference spectra are in the root_ref directory

# "obj_biopsy-10-intern-limit_source_white_LED_f80mm-P2_Walsh_im_16x16_ti_100ms_zoom_x1"

subpath = 'D:/d/P64/obj_biopsy-10-intern-limit_source_white_LED_f80mm-P2_Walsh_im_16x16_ti_100ms_zoom_x1/'
file_cube_white = 'D:/d/P64/obj_biopsy-10-intern-limit_source_white_LED_f80mm-P2_Walsh_im_16x16_ti_100ms_zoom_x1/obj_biopsy-10-intern-limit_source_white_LED_f80mm-P2_Walsh_im_16x16_ti_100ms_zoom_x1_' + type_reco_npz
file_metadata = 'D:/d/P64/obj_biopsy-10-intern-limit_source_white_LED_f80mm-P2_Walsh_im_16x16_ti_100ms_zoom_x1/obj_biopsy-10-intern-limit_source_white_LED_f80mm-P2_Walsh_im_16x16_ti_100ms_zoom_x1_metadata.json'


metadata, acquisition_params, spectrometer_params, dmd_params = read_metadata(file_metadata)
wavelengths = acquisition_params.wavelengths
t_i = spectrometer_params.integration_time_ms


# Read hypercube laser
cubeobj = np.load(file_cube_white)
cubehyper = cubeobj['arr_0']

threshold = threshold_ *t_i*1e-3/(np.shape(cubehyper)[0]*np.shape(cubehyper)[1]/(16**2))  # absolute threshold 

greyscale_img = np.sum(cubehyper, axis=2) # sum the hypercube over all the spectral channels

mask = cv.threshold(greyscale_img, threshold, 1, cv.THRESH_BINARY) # thresholding function

if os.path.isfile(subpath +  type_reco + '_mask.npy') == False :
    np.save(subpath +  type_reco + '_mask.npy', mask[1])
    print("mask saved")
else : 
    print("mask already exists")
    
    
#%%    
    
    


'''
#%% REFERENCE SPECTRA
# TODO: put this part in a separate file


file_name_ppix620 = 'ref620_3lamda.npy'
file_name_ppix634 = 'ref634_3lamda.npy'
file_name_lambda = 'Lambda.npy'
 

ppix620 = np.load(root_ref + file_name_ppix620)
ppix634 = np.load(root_ref + file_name_ppix634)
lambd = np.load(root_ref + file_name_lambda)
 
 
spectr634 = ppix634[0, :] # use 0th line only
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
 
func620 = interpolate.make_interp_spline(lambd, spectr620)  # interp1d is legacy
func634 = interpolate.make_interp_spline(lambd, spectr634)

spectr620_interp = func620(wavelengths) # import wavelengths from metadata
spectr634_interp = func634(wavelengths)


# save in ref folder : 
np.save(root_ref + '_spectr620_interp.npy', spectr620_interp)
np.save(root_ref + '_spectr634_interp.npy', spectr634_interp)



#%% DEFINE FIT FUNCTION


def func_fit(x, a1, a2, a3, shift620, shift634, lambd_c, sigma):
    return a1*func620(x-shift620) + a2*func634(x-shift634) + a3*np.exp(-(lambd_c-x)**2/sigma**2)

'''













































