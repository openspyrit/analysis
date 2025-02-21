# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:44:17 2025

@author: chiliaeva


"""


import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.axes as axs
import cv2 as cv
from preprocess_ref_spectra import func620, func634, func_fit
from spas.metadata import read_metadata



legend_on = True
fontsize = 16
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['axes.labelsize'] = fontsize

mpl.rcParams['figure.figsize'] = 12, 6  # figure size in inches

# mpl.rcParams['figure.figsize'] = 12, 12  # figure size in inches



#%%

savefig_spectrum = True

type_reco = 'had_reco'


# Get fit data
# root = 'C:/Users/chiliaeva/Documents/Resultats_traitement/'
root = 'C:/'
# root = 'D:/'
root_saveresults = root + 'fitresults_250219_full-spectra/'
# root_saveresults = root + 'fitresults_250206_narrow_lipo_binning_2/'
root_ref = root + 'ref/'


root_savefig = root_saveresults + 'fig/'
if os.path.exists(root_savefig) == False :
    os.mkdir(root_savefig)



# metadata file to get the wavelengths array :
file_metadata = root + 'wavelengths_metadata.json'
metadata, acquisition_params, spectrometer_params, dmd_params = read_metadata(file_metadata)
wavelengths = acquisition_params.wavelengths


# wvlgth_bin = np.load(root_saveresults + "wavelengths_mask_606-616.npy")
wvlgth_bin = np.load(root_saveresults + "wavelengths_mask_bin.npy")



num_patient = 'P68_'
num_biopsy = 'B6'



# Position of the spectrum to plot :
x = 16
y = 19
if type_reco == 'nn_reco' :
    x = 4*x
    y = 4*y



#%% Import interpolated and normalized ref spectra
# Apply spectral mask and binning 

spectr620 = np.load(root_ref + '_spectr620_interp.npy')
spectr634 = np.load(root_ref + '_spectr634_interp.npy')


# fit_start = 614
# fit_stop = 650

cut_start = 606
cut_stop = 616

band_mask = (wavelengths <= cut_start) | (wavelengths >= cut_stop)
# band_mask = (wavelengths >= fit_start) & (wavelengths <= fit_stop) # redundant with main_fit --> SAVE band_mask in main_fit


wavelengths = wavelengths[band_mask]
spectr620 = spectr620[band_mask]
spectr634 = spectr634[band_mask]



bin_width = 10 # redundant with main_fit (do the binning in the preprocessing program ?)
spectr620_bin = np.ndarray(spectr620.size // bin_width, dtype=float)
spectr634_bin = np.ndarray(spectr634.size // bin_width, dtype=float)


for i in range(spectr620_bin.size):
    spectr620_bin[i] = spectr620[i*bin_width]
    spectr634_bin[i] = spectr634[i*bin_width]



#%% 

file_params = root_saveresults + num_patient +  type_reco + '/' + num_biopsy + '_' + type_reco + '_fit_params.npy'
file_spectra = root_saveresults + num_patient + type_reco + '/' + num_biopsy + '_' + type_reco + '_spectrum_tab.npy'

params_tab = np.load(file_params)
spectrum_tab = np.load(file_spectra)   


if type_reco == 'nn_reco':
    params_tab = cv.flip(params_tab, 0)
    

# Ref spectra multiplied by their respective coefficients
# spectr620_coef = spectr620_bin * params_tab[x,y,0] # params_tab[x,y,0] = coef 620
# spectr634_coef = spectr634_bin * params_tab[x,y,1]


spectr620_coef = func620(wvlgth_bin-params_tab[x,y,3])*params_tab[x,y,0]
spectr634_coef = func634(wvlgth_bin-params_tab[x,y,4])*params_tab[x,y,1]


spectr_lipo_coef = params_tab[x,y,2] * np.exp(-(params_tab[x,y,5]-wvlgth_bin)**2/params_tab[x,y,6]**2)    


spectrum = spectrum_tab[x, y, :] # real spectrum 
spectrum_fit = func_fit(wvlgth_bin, *params_tab[x, y, :]) # fitted spectrum




#%% Plot the spectrum, fit, Pp620, Pp634, lipofuscin at position (x,y) on ax1
# plot absolute residuals on ax2
# plot relative residuals on ax3


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=False)
# fig, ax1 = plt.subplots(1, 1, sharex=False)
fig.suptitle('Spectrum at position x=' + str(x) + ' y=' + str(y) + ' ' + type_reco)
ax1.plot(wvlgth_bin, spectrum,  label='spectrum', color='black')
ax1.plot(wvlgth_bin, spectrum_fit, label='fit', color='black', linestyle='dashed')
ax1.plot(wvlgth_bin, spectr620_coef, label='Pp620', color='red', linestyle='dashed') 
ax1.plot(wvlgth_bin, spectr634_coef, label='Pp634', color='blue', linestyle='dashed') 
ax1.plot(wvlgth_bin, spectr_lipo_coef, label='lipo', color='darkgreen', linestyle='dashed') 
ax1.set_xlabel('wavelength (nm)')
ax1.set_ylabel((0, np.amax([np.amax(spectrum), np.amax(spectrum_fit)])))
ax2.plot(wvlgth_bin, spectrum-spectrum_fit, label='absolute residuals', color='black')
ax2.set_ylim((-80, 120))
ax3.plot(wvlgth_bin, (spectrum-spectrum_fit)/spectrum_fit, label='relative residuals', color='purple')
# ax3.set_ylim((-80, 120))
if legend_on == True :
    fig.legend()
if savefig_spectrum == True :
    fig.savefig(root_savefig + num_patient + num_biopsy + '_' + type_reco + '_x-' + str(x) + '_y-' + str(y) + '-spectrum.png', bbox_inches='tight')
    
    
    
    
#%% Quality criterion  = sum of squared relative residues

rel_res = (spectrum-spectrum_fit)/spectrum_fit
res = np.sum(rel_res**2)




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
                plt.savefig(root_avg_spectra + num_patient + num_biops + '_' + type_reco + '_spatial_avg_spectrum.png', bbox_inches='tight')
                plt.close()



#%%


'''
#############################################################################################################
########################################### TEMPORARY #######################################################
#############################################################################################################
#%% Compare 2 spectra 
# at positions (8,6) and (8,7)

spectrum_pink=spectrum_tab[8,6,:]
spectrum_orange=spectrum_tab[8,7,:]


plt.figure('Comparison')
plt.plot(wvlgth_bin, spectrum_pink, label='(8,6)', color='magenta', linewidth=1)
plt.plot(wvlgth_bin, spectrum_orange, label='(8,7)', color='orange', linewidth=1)
plt.legend()

max_pink = np.amax(spectrum_pink)
max_orange = np.amax(spectrum_orange)

spectrum_pink_norm = spectrum_pink/max_pink
spectrum_orange_norm = spectrum_orange/max_orange



fig_, (ax1_, ax2_) = plt.subplots(2, 1, sharex=True)
ax1_.plot(wvlgth_bin, spectrum_pink_norm, label='(8,6)', color='magenta')
ax1_.plot(wvlgth_bin, spectrum_orange_norm, label='(8,7)', color='orange')
ax1_.set_xlabel('wavelength (nm)')
ax1_.legend()
ax2_.plot(wvlgth_bin, np.abs(spectrum_pink_norm-spectrum_orange_norm), label='|difference|', color='black')
ax2_.legend()

'''
















