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


import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axs
from preprocess_ref_spectra import func_fit
import cv2 as cv
from spas.metadata import read_metadata



#%%

savefig_maps = False
savefig_spectrum = False
show_spectrum_pos = False

type_reco = 'had_reco'


# Get fit data
root = 'C:/'
# root = 'C:/Users/chiliaeva/Documents/Resultats_traitement/'
# root = 'D:/'
root_saveresults = root + 'fitresults_250219/'
root_ref = root + 'ref/'


root_savefig = root_saveresults + 'fig/'
if os.path.exists(root_savefig) == False :
    os.mkdir(root_savefig)


#################### Get the wavelengths_array ###############################
# file_metadata = root + 'wavelengths_metadata.json'
# metadata, acquisition_params, spectrometer_params, dmd_params = read_metadata(file_metadata)
# wavelengths = acquisition_params.wavelengths


############## if wavelengths_mask was saved ##################################
wavelengths = np.load(root_saveresults + "wavelengths_mask.npy")


# wvlgth_bin = np.load(root_saveresults + "wavelengths_mask_606-616.npy")
wvlgth_bin = np.load(root_saveresults + "wavelengths_mask_bin.npy")



num_patient = 'P61_'
num_biopsy = 'B8'



# Position of the spectrum to plot :
x = 6
y = 8
if type_reco == 'nn_reco' :
    x = 4*x
    y = 4*y

mksize = 4



#%% Import interpolated and normalized ref spectra


spectr620 = np.load(root_ref + '_spectr620_interp.npy')
spectr634 = np.load(root_ref + '_spectr634_interp.npy')


########### IMPORT BAND MASK #######################################################
band_mask = np.loadtxt(root_saveresults + "band_mask.npy")
band_mask = list(map(bool,band_mask)) # convert to boolean array


########### IF BAND MASK IS NOT SAVED ##############################################
# fit_start = 614
# fit_stop = 650

# # cut_start = 606
# # cut_stop = 616

# # band_mask = (wavelengths <= cut_start) | (wavelengths >= cut_stop)
# band_mask = (wavelengths >= fit_start) & (wavelengths <= fit_stop) # redundant with main_fit --> SAVE band_mask in main_fit

# wavelengths = wavelengths[band_mask]
#####################################################################################


spectr620 = spectr620[band_mask]
spectr634 = spectr634[band_mask]



bin_width = 10 # redundant with main_fit (do the binning in the preprocessing program ?)
spectr620_bin = np.ndarray(spectr620.size // bin_width, dtype=float)
spectr634_bin = np.ndarray(spectr634.size // bin_width, dtype=float)


for i in range(spectr620_bin.size):
    spectr620_bin[i] = spectr620[i*bin_width]
    spectr634_bin[i] = spectr634[i*bin_width]




'''
# Resampling the wavelength scale for the binned spectrum
bin_width = 10

wvlgth_bin = np.ndarray(wavelengths.size // bin_width, dtype=float)

for i in range(wvlgth_bin.size):
    wvlgth_bin[i] = wavelengths[i*bin_width]
if save_fit_data == True : 
    np.save(root_saveresults + 'wavelengths_mask_bin_' + str(fit_start) + '-' + str(fit_stop) + '.npy', wvlgth_bin)     
'''    
    


#%% Abundance maps


file_params = root_saveresults + num_patient +  type_reco + '/' + num_biopsy + '_' + type_reco + '_fit_params.npy'
params_tab = np.load(file_params)


if type_reco == 'nn_reco':
    params_tab = cv.flip(params_tab, 0)
    
    

#%% 
# Abundance maps 

coef_P620 = np.load(file_params)[:,:,0]
coef_P634 = np.load(file_params)[:,:,1]
coef_lipo = np.load(file_params)[:,:,2]
lambd_c = np.load(file_params)[:,:,5]
sigma = np.load(file_params)[:,:,6]

    


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
    plt.savefig(root_savefig + num_patient + num_biopsy + '_' + type_reco + '_x-' + str(x) + '_y-' + str(y) + '_coef_P620_map.png', bbox_inches='tight')




plt.figure('coef_P634_a2_map')
plt.clf()
plt.title('PpIX 634 amplitude')
plt.imshow(coef_P634)
plt.clim(min_ppix, max_ppix)
plt.colorbar()
if show_spectrum_pos == True :
    plt.plot(y,x, "or", markersize = mksize)
if savefig_maps == True :
    plt.savefig(root_savefig + num_patient + num_biopsy + '_' + type_reco + '_x-' + str(x) + '_y-' + str(y) + '_coef_P634_map.png', bbox_inches='tight')
    
    
    
    
    
    

#%% Spectrum, fit and residuals (with binning)


spectr620_coef = spectr620_bin * coef_P620[x,y]
spectr634_coef = spectr634_bin * coef_P634[x,y]
spectr_lipo_coef = coef_lipo[x,y] * np.exp(-(lambd_c[x,y]-wvlgth_bin)**2/sigma[x,y]**2)    


spectrum_tab = np.load(root_saveresults + num_patient + type_reco + '/' + num_biopsy + '_' + type_reco + '_spectrum_tab.npy')   # spectrum from data
spectrum = spectrum_tab[x, y, :]
spectrum_fit = func_fit(wvlgth_bin, *params_tab[x, y, :])


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
fig.suptitle('Spectrum at position x=' + str(x) + ' y=' + str(y) + ' ' + type_reco)
ax1.plot(wvlgth_bin, spectrum,  label='spectrum', color='black', linewidth=1)
ax1.plot(wvlgth_bin, spectrum_fit, label='fit', color='black', linestyle='dashed', linewidth=1)
ax1.plot(wvlgth_bin, spectr620_coef, label='Pp620', color='red', linewidth=1) # ADDED 24/12/16
ax1.plot(wvlgth_bin, spectr634_coef, label='Pp634', color='blue', linewidth=1) # ADDED 24/12/16
ax1.plot(wvlgth_bin, spectr_lipo_coef, label='lipo', color='darkgreen', linewidth=1) # ADDED 24/12/16
ax1.set_xlabel('wavelength (nm)')
ax1.set_ylabel((0, np.amax([np.amax(spectrum), np.amax(spectrum_fit)])))
ax2.plot(wvlgth_bin, spectrum-spectrum_fit, label='absolute residuals', color='black', linewidth=1)
ax2.set_xlabel('wavelength (nm)')
ax2.set_ylim((-80, 120))
ax3.plot(wvlgth_bin, (spectrum-spectrum_fit)/spectrum_fit, label='relative residuals', color='black', linewidth=1)
ax3.set_xlabel('wavelength (nm)')
# ax3.set_ylim((-80, 120))
fig.legend()
if savefig_spectrum == True :
    fig.savefig(root_savefig + num_patient + num_biopsy + '_' + type_reco + '_x-' + str(x) + '_y-' + str(y) + '-spectrum_+-2nm.png', bbox_inches='tight')




#%% Save 1 averaged spectrum for each measurement : 
    

root_avg_spectra = root_savefig + 'avg_spectra/'
if os.path.exists(root_avg_spectra) == False :
    os.mkdir(root_avg_spectra)


folders = os.listdir(root_saveresults)


for folder in folders : 
    if 'P' in folder : 
        num_patient = folder[0:4]
        type_reco = folder[4:]
        path = os.path.join(root_saveresults, folder)
        files = os.listdir(path)
        for file in files :
            if 'spectrum_tab' in file :
                num_biops = file[0:3]
                subpath = os.path.join(path, file)
                spectrum_tab = np.load(subpath)
                avg_spectrum = np.nanmean(spectrum_tab, (0,1))
                
                plt.figure()
                plt.plot(wvlgth_bin, avg_spectrum)
                plt.grid()
                plt.savefig(root_avg_spectra + num_patient + num_biops + '_' + type_reco + '_spatially_avg_spectrum.png', bbox_inches='tight')
                plt.close()




#%% Save sum of squared residuals 
'''


root_res = root_savefig + 'sum_sq_residuals/'
if os.path.exists(root_res) == False :
    os.mkdir(root_res)



folders = os.listdir(root_saveresults)



for folder in folders : 
    if 'P' in folder :
        num_patient = folder[0:4]
        type_reco = folder[4:]
        path = os.path.join(root_saveresults, folder)
        files = os.listdir(path)
        for file in files :
            if 'fit_params' in file :
                num_biops = file[0:3]
                subpath = os.path.join(path, file)
                spectrum_tab = np.load(subpath)   # spectrum from data
                spectrum_fit = np.ndarray(spectrum_tab.shape, dtype = 'float64')
                spectrum_fit = func_fit(wavelengths, *params_tab[:, :, :])







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


'''


















































