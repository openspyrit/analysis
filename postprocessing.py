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
import matplotlib.axes as axs
from preprocessing import func_fit
import cv2 as cv


#%%

savefig_maps = True
savefig_spectrum = True
show_spectrum_pos = True


# Get fit data
root = 'D:/'


root_saveresults = root + 'fitresults/'

# import wavelengths 
wavelengths = np.load(root_saveresults + "wavelengths_mask_606-616.npy")


num_patient = 'P63'
num_biopsy = 'B2'
type_reco = 'nn_reco'


# Position of the spectrum to plot :
x = 68
y = 82

mksize = 4


#%% Abundance maps


file_params = root_saveresults + num_patient +  type_reco + '/' + num_biopsy + '_' + type_reco + '_fit_params.npy'

params_tab = np.load(file_params)
coef_P620 = np.load(file_params)[:,:,0]
coef_P634 = np.load(file_params)[:,:,1]
coef_lipo = np.load(file_params)[:,:,2]


if type_reco == 'nn_reco':
    coef_P620 = cv.flip(coef_P620,0)
    coef_P634 = cv.flip(coef_P634,0)
    coef_lipo = cv.flip(coef_lipo, 0)
    
    

min_ppix = np.amin([np.nanmin(coef_P620), np.nanmin(coef_P634)]) # minimum for Protoporphyrin IX colormap
max_ppix = np.amax([np.nanmax(coef_P620), np.nanmax(coef_P634)])



plt.figure('coef_P620_a1_map')
plt.clf()
plt.title('PpIX 620 amplitude')
plt.imshow(coef_P620)
plt.clim(min_ppix, max_ppix)
plt.colorbar()
if show_spectrum_pos == True :
    plt.plot(y,x, "or", markersize = mksize)
if savefig_maps == True :
    plt.savefig(root + 'fig/' + num_patient + num_biopsy + '_' + type_reco + '_x-' + str(x) + '_y-' + str(y) + '_coef_P620_map.png', bbox_inches='tight')




plt.figure('coef_P634_a2_map')
plt.clf()
plt.title('PpIX 634 amplitude')
plt.imshow(coef_P634)
plt.clim(min_ppix, max_ppix)
plt.colorbar()
if show_spectrum_pos == True :
    plt.plot(y,x, "or", markersize = mksize)
if savefig_maps == True :
    plt.savefig(root + 'fig/' + num_patient + num_biopsy + '_' + type_reco + '_x-' + str(x) + '_y-' + str(y) + '_coef_P634_map.png', bbox_inches='tight')
    
    


#%% Spectrum, fit and residuals



spectrum_tab = np.load(root_saveresults + num_patient + type_reco + '/' + num_biopsy + '_' + type_reco + '_spectrum_tab.npy')   # spectrum from data

x_ = x
x = np.shape(spectrum_tab)[0]-x


spectrum = spectrum_tab[x, y, :]

spectrum_fit = func_fit(wavelengths, *params_tab[x, y, :])


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
fig.suptitle('Spectrum at position x=' + str(x_) + ' y=' + str(y))
ax1.plot(wavelengths, spectrum,  label='spectrum', linewidth=1)
ax1.plot(wavelengths, spectrum_fit, label='fit', color='orange')
ax1.set_xlabel('wavelength (nm)')
ax2.plot(wavelengths, spectrum-spectrum_fit, label='residuals', color='magenta', linewidth=1)
ax2.set_xlabel('wavelength (nm)')
fig.legend()
if savefig_spectrum == True :
    fig.savefig(root + 'fig/' + num_patient + num_biopsy + '_' + type_reco + '_x-' + str(x_) + '_y-' + str(y) + 'spectrum.png', bbox_inches='tight')





#%% plot spectrum_tab



plt.figure("Spectrum tab")
plt.clf
plt.imshow(spectrum_tab[:,:,1000])































