# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 10:35:59 2025

@author: chiliaeva
"""

import numpy as np

from spas.metadata import read_metadata

import scipy.signal as sg
import scipy.optimize as op
from preprocess_ref_spectra import func_fit


import os
import time



#%% Paths


save_fit_data = True

bin_width = 10

root = 'C:/'

num_patient = 'P61'
num_biopsy = 'B8'

root_data = root + 'd/' + num_patient + '/'
root_ref = root + 'ref/'

# os.mkdir(root + 'fitresults_' + )
root_saveresults = root + 'fitresults/' 

type_reco = 'had_reco'
type_reco_npz = type_reco + '.npz'




# folders = os.listdir(root_data)
# print("folders", folders)


# metadata file to get the wavelengths array :
file_metadata='C:/Users/chiliaeva/Documents/Resultats_traitement/wavelengths_metadata.json'
# file_metadata = 'D:/d/P64/obj_biopsy-10-intern-limit_source_white_LED_f80mm-P2_Walsh_im_16x16_ti_100ms_zoom_x1/obj_biopsy-10-intern-limit_source_white_LED_f80mm-P2_Walsh_im_16x16_ti_100ms_zoom_x1_metadata.json'
metadata, acquisition_params, spectrometer_params, dmd_params = read_metadata(file_metadata)
wavelengths = acquisition_params.wavelengths


file_cube_laser = 'C:/d/P61/obj_biopsy-8-deep-limit_source_Laser_405nm_1.2W_A_0.15_f80mm-P2_Walsh_im_16x16_ti_150ms_zoom_x1/obj_biopsy-8-deep-limit_source_Laser_405nm_1.2W_A_0.15_f80mm-P2_Walsh_im_16x16_ti_150ms_zoom_x1_had_reco.npz'
file_cube_nolight = 'C:/d/P61/obj_biopsy-8-deep-limit_source_No-light_f80mm-P2_Walsh_im_16x16_ti_150ms_zoom_x1/obj_biopsy-8-deep-limit_source_No-light_f80mm-P2_Walsh_im_16x16_ti_150ms_zoom_x1_had_reco.npz'
file_mask = root_data + 'obj_biopsy-8-deep-limit_source_white_LED_f80mm-P2_Walsh_im_16x16_ti_10ms_zoom_x1/' + type_reco + '_mask' + '.npy' 


# file_cube_laser = root_data + 'obj_biopsy-10-intern-limit_source_Laser_405nm_1.2W_A_0.14_f80mm-P2_Walsh_im_16x16_ti_100ms_zoom_x1/obj_biopsy-10-intern-limit_source_Laser_405nm_1.2W_A_0.14_f80mm-P2_Walsh_im_16x16_ti_100ms_zoom_x1_had_reco.npz'
# file_cube_nolight = root_data +'obj_biopsy-10-intern-limit_source_No-light_f80mm-P2_Walsh_im_16x16_ti_100ms_zoom_x1/obj_biopsy-10-intern-limit_source_No-light_f80mm-P2_Walsh_im_16x16_ti_100ms_zoom_x1_had_reco.npz'




#%%


real_spectro_reso = 2 # real resolution of the spectrometer (nm)
theo_spectro_reso = (wavelengths[-1]-wavelengths[0])/len(wavelengths)
kernel_size = int(real_spectro_reso/theo_spectro_reso) + 1 # size of the window for the median filter, must be odd
if kernel_size %2 == 0 : 
    kernel_size = kernel_size + 1


fit_start = 614
fit_stop = 650


band_mask = (wavelengths >= fit_start) & (wavelengths <= fit_stop)
wavelengths = wavelengths[band_mask]
if save_fit_data == True :  
    np.save(root_saveresults + 'wavelengths_mask.npy', wavelengths) 



# Resampling the wavelength scale for the binned spectrum

wvlgth_bin = np.ndarray(wavelengths.size // bin_width, dtype=float)

for i in range(wvlgth_bin.size):
    wvlgth_bin[i] = wavelengths[i*bin_width]
if save_fit_data == True : 
    np.save(root_saveresults + 'wavelengths_mask_bin.npy', wvlgth_bin)     
    



 #%% Import ref spectra


# load interpolated and normalized ref spectra : 
spectr620 = np.load(root_ref + '_spectr620_interp.npy')
spectr634 = np.load(root_ref + '_spectr634_interp.npy')

spectr620 = spectr620[band_mask]
spectr634 = spectr634[band_mask]




#%% FIT


cubeobj = np.load(file_cube_laser)
cubehyper_laser = cubeobj['arr_0']


# Read nolight hypercube 
cubeobj = np.load(file_cube_nolight)
cubehyper_nolight = cubeobj['arr_0']
del cubeobj


# Read mask 
mask = np.load(file_mask)


popt_tab = np.ndarray((cubehyper_laser.shape[0], cubehyper_laser.shape[1], 7), dtype = 'float64')
popt_tab[:] = np.nan       
spectrum_tab = np.ndarray((cubehyper_laser.shape[0], cubehyper_laser.shape[1], np.size(wvlgth_bin)), dtype='float64')
spectrum_tab[:] = np.nan   
 
# Fit for every point of the mask
t0 = time.time()
print('start fit for the entire image', time.time()-t0)


for x_i in range(cubehyper_laser.shape[0]):
    for y_i in range(cubehyper_laser.shape[1]):
        
        if mask[x_i, y_i]!=0:
            
        
            spectr_laser = cubehyper_laser[x_i, y_i, :]
            spectr_nolight = cubehyper_nolight[x_i, y_i, :]
        
            spectr_laser = cubehyper_laser[x_i, y_i, :][band_mask]
            spectr_nolight = cubehyper_nolight[x_i, y_i, :][band_mask]
            
            
            # Binning
            sp_laser_bin = np.ndarray(spectr_laser.size // bin_width, dtype=float)
            sp_nolight_bin = np.ndarray(spectr_laser.size // bin_width, dtype=float)
            
            for i in range(sp_laser_bin.size):
                sp_laser_bin[i] = np.sum(spectr_laser[i*bin_width:(1+i)*bin_width])
                sp_nolight_bin[i] = np.sum(spectr_nolight[i*bin_width:(1+i)*bin_width])
        
            # Remove the no light spectrum
            spectrum = sp_laser_bin - sp_nolight_bin
            spectrum_tab[x_i, y_i, :] = spectrum



            # FIT THE SPECTRUM TO REFERENCE SPECTRA
            
            M = np.abs(np.max(spectrum))
            # p0 = [M/2, M/2, M/8, 0, 0, 585, 10]# initial guess for the fit
            # bounds_inf = [0, 0 ,0 ,-2, -2, 580, 5] 
            # bounds_sup = [M, M, M, 2, 2, 610, 100] 
            
            p0 = [M/2, M/2, M/8, 0, 0, 590, 25]# initial guess for the fit
            bounds_inf = [0, 0 ,0 ,-2, -2, 585, 20] 
            bounds_sup = [M, M, M, 2, 2, 595, 40] 
            
            try : 
                popt, pcov = op.curve_fit(func_fit, wvlgth_bin, spectrum, p0, bounds=(bounds_inf, bounds_sup))
                popt_tab[x_i, y_i, :] = popt
            except RuntimeError:
                pass
        
        
        
print('end fit for image', time.time()-t0)  


if save_fit_data == True:
    os.mkdir(root_saveresults + num_patient + '_' +  type_reco)
    np.save(root_saveresults + num_patient + '_' + type_reco + '/' + num_biopsy + '_' +  type_reco + '_spectrum_tab.npy', spectrum_tab) 
    np.save(root_saveresults + num_patient + '_' + type_reco + '/' + num_biopsy + '_' +  type_reco + '_fit_params.npy', popt_tab) 






























