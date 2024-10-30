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
from preprocessing import func_fit


savefig = True


# Get fit data
root = 'D:/hspc/fitresults/'
num_patient = 'P69'
num_biopsy = 'B1'
type_reco = 'had_reco'


file_params = root + num_patient + '/' + num_biopsy + '_' + type_reco + '_fit_params.npy'
params_tab = np.load(file_params)

coef_P620 = np.load(file_params)[:,:,0]
coef_P634 = np.load(file_params)[:,:,1]
coef_lipo = np.load(file_params)[:,:,2]

min_ppix = np.amin([np.nanmin(coef_P620), np.nanmin(coef_P634)]) # minimum for Protoporphyrin IX colormap
max_ppix = np.amax([np.nanmax(coef_P620), np.nanmax(coef_P634)])


# metadata file to get the wavelengths array :
file_metadata = 'D:/hspc/data/2024/P60/obj_biopsy-1_anterior-portion_source_Laser_405nm_1.2W_A_0.15_f80mm-P2_Walsh_im_16x16_ti_200ms_zoom_x1/obj_biopsy-1_anterior-portion_source_Laser_405nm_1.2W_A_0.15_f80mm-P2_Walsh_im_16x16_ti_200ms_zoom_x1_metadata.json'
metadata, acquisition_params, spectrometer_params, dmd_params = read_metadata(file_metadata)
wavelengths = acquisition_params.wavelengths



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
    
    


#%% Spectrum, fit and residuals

x_i = 10
y_i = 20


spectrum_tab = np.load(root + num_patient + '/' + num_biopsy + '_' + type_reco + '_spectrum_tab.npy')   # spectrum from data
spectrum = spectrum_tab[x_i, y_i, :]

spectrum_fit = func_fit(wavelengths, *params_tab[x_i, y_i, :])



fig, (ax1, ax2) = plt.subplots(2, 1,  sharex=True
ax1.plot(wavelengths, spectrum,  label='spectrum')
ax1.plot(wavelengths, spectrum_fit, label='fit')
ax1.set_xlabel('wavelength (nm)')
ax2.plot(wavelengths, spectrum-spectrum_fit, label='residuals')
ax2.set_xlabel('wavelength (nm)')
fig.legend()




































