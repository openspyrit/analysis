# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:44:17 2025

@author: chiliaeva


"""

import pickle
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.axes as axs
import cv2 
from preprocess_ref_spectra import func620, func634, func_fit
from spas.metadata import read_metadata



legend_on = True
fontsize = 16
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['axes.labelsize'] = fontsize

# mpl.rcParams['figure.figsize'] = 12, 6  # figure size in inches

mpl.rcParams['figure.figsize'] = 12, 12  # figure size in inches



#%%

savefig_spectrum = True
savefig_map = True

type_reco = 'had_reco'


# Get fit data
# root = 'C:/Users/chiliaeva/Documents/Resultats_traitement/'
root = 'C:/'
# root = 'D:/'
root_saveresults = root + 'fitresults_250313_full-spectra_/'
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



num_patient = 'P69_'
num_biopsy = 'B5'


bounds_file = root_saveresults + 'bounds.pickle'

with open(bounds_file, 'rb') as pickle_file:
    bounds = pickle.load(pickle_file)
    


#%% Import interpolated and normalized ref spectra
# Apply spectral mask and binning 

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



#%% 


# Position of the spectrum to plot :
x = 15
y = 19
if type_reco == 'nn_reco' :
    x = 4*x
    y = 4*y


# Spatial binning 
spat_bin = 3 # number of neighbours (in every direction) to take into account in the spatial averaging



file_params = root_saveresults + num_patient +  type_reco + '/' + num_biopsy + '_' + type_reco + '_fit_params.npy'
file_spectra = root_saveresults + num_patient + type_reco + '/' + num_biopsy + '_' + type_reco + '_spectrum_tab.npy'

params_tab = np.load(file_params)
spectrum_tab = np.load(file_spectra)   


if type_reco == 'nn_reco':
    params_tab = cv2.flip(params_tab, 0)
    

# Ref spectra multiplied by their respective coefficients
# spectr620_coef = spectr620_bin * params_tab[x,y,0] # params_tab[x,y,0] = coef 620
# spectr634_coef = spectr634_bin * params_tab[x,y,1]


spectr620_coef = func620(wvlgth_bin-params_tab[x,y,3])*params_tab[x,y,0]
spectr634_coef = func634(wvlgth_bin-params_tab[x,y,4])*params_tab[x,y,1]
spectr_lipo_coef = params_tab[x,y,2] * np.exp(-(params_tab[x,y,5]-wvlgth_bin)**2/params_tab[x,y,6]**2)    


spectrum = spectrum_tab[x, y, :] # real spectrum 

# Spatial averaging of the raw spectrum : 
spectrum = np.nanmean(np.nanmean(spectrum_tab[x-spat_bin:x+spat_bin+1,y-spat_bin:y+spat_bin+1,:], axis=0), axis=0)



# Spatial averaging of the fitting curve and its components : 


spectrum_fit_sum = np.zeros(wvlgth_bin.size, dtype=float)
cpt = 0

for i in range(x-spat_bin, x+spat_bin+1):
    for j in range(y-spat_bin, y+spat_bin+1):
        print("i,j = ", i, j)
        spectrum_temp = func_fit(wvlgth_bin, *params_tab[i, j, :])
        if not np.isnan(spectrum_temp).any() :
            spectrum_fit_sum = spectrum_fit_sum + spectrum_temp # fitted spectrum
            cpt += 1
        

spectrum_fit = spectrum_fit_sum / cpt 





def spatial_avg(array_x, func, spat_bin, x, y):
    # Inputs : 
    # array_x : x scale (such as wvlgth_bin), of size N
    # func : any function
    # spat_bin : width of spatial averaging window
    # x,y : center of the averaging window
    array_sum = np.zeros(array_x.size, dtype=float)
    cpt = 0
    for i in range(x-spat_bin, x+spat_bin+1):
        for j in range(y-spat_bin, y+spat_bin+1):
            array_temp = func(array_x, i, j)
            if not np.isnan(array_temp).any() :
                array_sum = array_sum + array_temp
                cpt += 1
    array_avg = array_sum / cpt
    return array_avg






def ppix620(wvlgth_bin, x, y):
    spectr620_coef = func620(wvlgth_bin-params_tab[x,y,3])*params_tab[x,y,0]
    return spectr620_coef

def ppix634(wvlgth_bin, x, y):
    spectr634_coef = func634(wvlgth_bin-params_tab[x,y,4])*params_tab[x,y,1]
    return spectr634_coef

def lipofuscin(wvlgth_bin, x, y):
    spectr_lipo_coef = params_tab[x,y,2] * np.exp(-(params_tab[x,y,5]-wvlgth_bin)**2/params_tab[x,y,6]**2)   
    return spectr_lipo_coef

    

spectr620_coef = spatial_avg(wvlgth_bin, ppix620, spat_bin, x, y)
spectr634_coef = spatial_avg(wvlgth_bin, ppix634, spat_bin, x, y)
spectr_lipo_coef = spatial_avg(wvlgth_bin, lipofuscin, spat_bin, x, y)






#%% Plot the spectrum, fit, Pp620, Pp634, lipofuscin at position (x,y) on ax1
# plot absolute residuals on ax2
# plot relative residuals on ax3


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=False)
# fig, ax1 = plt.subplots(1, 1, sharex=False)
fig.suptitle('Spectrum at position x=' + str(x) + ' y=' + str(y) + ' ' + 'spatial avg =' + str(spat_bin))
ax1.plot(wvlgth_bin, spectrum,  label='spectrum', color='black')
ax1.plot(wvlgth_bin, spectrum_fit, label='fit', color='black', linestyle='dashed')
ax1.plot(wvlgth_bin, spectr620_coef, label='Pp620', color='red', linestyle='dashed') 
ax1.plot(wvlgth_bin, spectr634_coef, label='Pp634', color='blue', linestyle='dashed') 
ax1.plot(wvlgth_bin, spectr_lipo_coef, label='lipo', color='darkgreen', linestyle='dashed') 
ax1.set_xlabel('wavelength (nm)')
ax1.set_ylabel((0, np.amax([np.amax(spectrum), np.amax(spectrum_fit)])))
ax1.grid()
ax2.plot(wvlgth_bin, spectrum-spectrum_fit, label='absolute residuals', color='black')
ax2.set_ylim((-80, 120))
ax2.grid()
ax3.plot(wvlgth_bin, (spectrum-spectrum_fit)/spectrum_fit, label='relative residuals', color='purple')
ax3.grid()
# ax3.set_ylim((-80, 120))
if legend_on == True :
    fig.legend()
if savefig_spectrum == True :
    fig.savefig(root_savefig + num_patient + num_biopsy + '_' + type_reco + '_x-' + str(x) + '_y-' + str(y) + '_avg-' + str(spat_bin) + '_spectrum.png', bbox_inches='tight')
    
    
    
    
    
#%% Show averaged area on the SNR map : 
'''
IT DOESN'T WORK WIH CV2

# Load SNR map :

root_snr = root + 'snr/'
snr_file = root_snr + num_patient + num_biopsy + '_' + type_reco + '_snr_map.png'

start_point = (x-spat_bin, y-spat_bin)
stop_point = (x+spat_bin+1, y+spat_bin+1)


snr_map = cv2.imread(snr_file)
snr_map = cv2.rectangle(snr_map, start_point, stop_point, (0, 0, 255), 1)
cv2.imshow("SNR map", snr_map)
cv2.waitKey(0)
'''   
    
#%% Show averaged area on the SNR map : 

    

root_snr = root + 'snr/'
snr_file = root_snr + num_patient + num_biopsy + '_' + type_reco + '_snr_tab.npy'

snr_map = np.load(snr_file)



rect_x = x-spat_bin-0.5
rect_y = y-spat_bin-0.5
rect_width = 2*spat_bin+1
rect_height = 2*spat_bin+1


fig, ax = plt.subplots()
ax.imshow(snr_map)
ax.plot(y,x, "or")
rect = mpl.patches.Rectangle((rect_y, rect_x), rect_width, rect_height, linewidth=1, edgecolor='red', facecolor='none')
ax.add_patch(rect)
ax.grid()
if savefig_map:
    fig.savefig(root_savefig + num_patient + num_biopsy + '_' + type_reco + '_x-' + str(x) + '_y-' + str(y) + '_avg-' + str(spat_bin) + '_snr_map.png', bbox_inches='tight')





#%%    
'''    
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









