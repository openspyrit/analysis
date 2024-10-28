# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:27:30 2024

@author: chiliaeva

Postprocessing program from SPIHIM data

Input : 
- reconstructed hypercubes had_reco or nn_reco (.npz files)
- popt_tab obtained with fit_ppix_lipo program
Output :
- abundance maps

"""

import numpy as np
import matplotlib.pyplot as plt
from spas.metadata2 import read_metadata

# @todo : TEMPORARY
from scipy import interpolate
import scipy.signal as sg
###


savefig = False


# Get fit data
root = 'D:/hspc/data/2024a/'
num_patient = 'P64'
num_biopsy = 'B1'
type_reco = 'had_reco'


file_params = root + num_patient + '/' + num_biopsy + '_' + type_reco + '_fit_params.npy'

coef_P620 = np.load(file_params)[:,:,0]
coef_P634 = np.load(file_params)[:,:,1]
coef_lipo = np.load(file_params)[:,:,2]

min_ppix = np.amin([np.nanmin(coef_P620), np.nanmin(coef_P634)])
max_ppix = np.amax([np.nanmax(coef_P620), np.nanmax(coef_P634)])



# metadata file to get the wavelengths array :
file_metadata = 'D:/hspc/data/2024/P60/obj_biopsy-1_anterior-portion_source_Laser_405nm_1.2W_A_0.15_f80mm-P2_Walsh_im_16x16_ti_200ms_zoom_x1/obj_biopsy-1_anterior-portion_source_Laser_405nm_1.2W_A_0.15_f80mm-P2_Walsh_im_16x16_ti_200ms_zoom_x1_metadata.json'
metadata, acquisition_params, spectrometer_params, dmd_params = read_metadata(file_metadata)
wavelengths = acquisition_params.wavelengths


# Get reconstructed data





#%% Abundance maps

plt.figure('coef_P620_a1_map')
plt.clf()
plt.imshow(coef_P620)
plt.clim(min_ppix, max_ppix)
plt.colorbar()
if savefig == True :
    plt.savefig(root + num_patient + '/' + num_biopsy + '_' + type_reco + '_coef_P620_map.png', bbox_inches='tight')




plt.figure('coef_P634_a2_map')
plt.clf()
plt.imshow(coef_P634)
plt.clim(min_ppix, max_ppix)
plt.colorbar()
if savefig == True :
    plt.savefig(root + num_patient + '/' + num_biopsy + '_' + type_reco +'_coef_P634_map.png', bbox_inches='tight')






#######################################################################################################################################################################
################################################################## TEMPORARY ##########################################################################################
#######################################################################################################################################################################

x_i = 9
y_i = 12
real_spectro_reso = 2 # real resolution of the spectrometer (nm)

folder_laser = 'obj_biopsy-1-lateral-portion_source_Laser_405nm_1.2W_A_0.14_f80mm-P2_Walsh_im_64x64_ti_15ms_zoom_x1'
folder_nolight = 'obj_biopsy-1-lateral-portion_source_No-light_f80mm-P2_Walsh_im_64x64_ti_15ms_zoom_x1'


file_cube_laser = root + num_patient + '/' + folder_laser + '/' + folder_laser + '_' + type_reco + '.npz'
file_cube_nolight = root + num_patient + '/' + folder_nolight + '/' + folder_nolight + '_' + type_reco + '.npz'


# Read laser hypercube
cubeobj = np.load(file_cube_laser)
cubehyper_laser = cubeobj['arr_0']


# Read nolight hypercube 
cubeobj = np.load(file_cube_nolight)
cubehyper_nolight = cubeobj['arr_0']
del cubeobj



spectr_laser = cubehyper_laser[x_i, y_i, :]
spectr_nolight = cubehyper_nolight[x_i, y_i, :]


# median filter to smoothen the spectrum

theo_spectro_reso = (wavelengths[-1]-wavelengths[0])/len(wavelengths)

kernel_size = int(real_spectro_reso/theo_spectro_reso) + 1 # size of the window for the median filter, must be odd

if kernel_size %2 == 0 : 
    kernel_size = kernel_size + 1


sp_laser_smth = sg.medfilt(spectr_laser, kernel_size)
sp_nolight_smth = sg.medfilt(spectr_nolight, kernel_size)



# Remove the no light spectrum
spectrum = sp_laser_smth - sp_nolight_smth


 #%% REFERENCE SPECTRA # this part has been moved outside of the loop

folder_path_ref = 'C:/Users/chiliaeva/Documents/data_pilot-warehouse/ref/'
 
file_name_ppix620 = 'ref620_3lamda.npy'
file_name_ppix634 = 'ref634_3lamda.npy'
file_name_lambda = 'Lambda.npy'
 
 
ppix620 = np.load(folder_path_ref + file_name_ppix620)
ppix634 = np.load(folder_path_ref + file_name_ppix634)
lambd = np.load(folder_path_ref + file_name_lambda)
 
 
spectr634 = ppix634[0, :] 
spectr634[0] = 0 # otherwise kernel dies
spectr620 = ppix620[0, :]
spectr620[0] = 0
 
 
 # Normalize the reference spectra
 
spectr620_norm = spectr620/np.amax(spectr620)
spectr620 = spectr620_norm
del spectr620_norm
 
spectr634_norm = spectr634/np.amax(spectr634)
spectr634 = spectr634_norm
del spectr634_norm
 
 
crop_start = np.digitize(wavelengths[0], lambd, right=True)
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


spectr620 = spectr620_interp
spectr634 = spectr634_interp


del spectr620_interp
del spectr634_interp



# Define fit function

spectr620_shift = spectr620
spectr634_shift = spectr634


def func_fit(x, a1, a2, a3, shift620, shift634, lambd_c, sigma):
    return a1*func620(x-shift620) + a2*func634(x-shift634) + a3*np.exp(-(lambd_c-x)**2/sigma**2)











































