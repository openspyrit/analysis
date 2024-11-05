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
from preprocessing import func_fit

#%%

savefig = False


# Get fit data
root = 'D:/'


root_saveresults = root + 'fitresults/'

# import wavelengths 
wavelengths = np.load(root_saveresults + "wavelengths_mask_606-616.npy")


num_patient = 'P68'
num_biopsy = 'B1'
type_reco = 'had_reco'


# Position of the spectrum to plot :
x = 17
y = 9



#%% Abundance maps


file_params = root_saveresults + num_patient + '/' + num_biopsy + '_' + type_reco + '_fit_params.npy'

params_tab = np.load(file_params)
coef_P620 = np.load(file_params)[:,:,0]
coef_P634 = np.load(file_params)[:,:,1]
coef_lipo = np.load(file_params)[:,:,2]

min_ppix = np.amin([np.nanmin(coef_P620), np.nanmin(coef_P634)]) # minimum for Protoporphyrin IX colormap
max_ppix = np.amax([np.nanmax(coef_P620), np.nanmax(coef_P634)])



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

spectrum_tab = np.load(root + num_patient + '/' + num_biopsy + '_' + type_reco + '_spectrum_tab.npy')   # spectrum from data
spectrum = spectrum_tab[x, y, :]

spectrum_fit = func_fit(wavelengths, *params_tab[x, y, :])


fig, (ax1, ax2) = plt.subplots(2, 1,  sharex=True)
ax1.plot(wavelengths, spectrum,  label='spectrum', linewidth=1)
ax1.plot(wavelengths, spectrum_fit, label='fit', color='orange')
ax1.set_xlabel('wavelength (nm)')
ax2.plot(wavelengths, spectrum-spectrum_fit, label='residuals', color='magenta', linewidth=1)
ax2.set_xlabel('wavelength (nm)')
fig.legend()




































