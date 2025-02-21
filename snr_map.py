# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 18:56:46 2025

@author: chiliaeva

# Signal-to-noise ratio maps

"""

import math
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from snr_functions import compute_snr
from matplotlib_scalebar.scalebar import ScaleBar


len_fov = 5 # size of FOV in mm



#%% Get the files 

savefig_map = False
save_histogram = False

root = 'C:/Users/chiliaeva/Documents/Resultats_traitement/'

type_reco = 'had_reco'
num_patient = 'P61_'
num_biopsy = 'B8_'


root_saveresults = root + 'fitresults_241119_full_spectra/'
file_spectrum_tab = root_saveresults + num_patient + type_reco + '/' + num_biopsy + type_reco + '_' + 'spectrum_tab.npy'
spectrum_tab = np.load(file_spectrum_tab)


root_savefig = root_saveresults + 'fig/snr/'
if os.path.exists(root_savefig) == False :
    os.mkdir(root_savefig)


root_savefig_hist = root_savefig + 'hist/'
if os.path.exists(root_savefig_hist) == False :
    os.mkdir(root_savefig_hist)



file_wvlgth = root_saveresults + 'wavelengths_mask_606-616.npy'
wavelengths = np.load(file_wvlgth)



#%% Spatial loop

std_bounds = [650, 748]
max_interval = [620, 640]

nb_map = np.empty_like(spectrum_tab[:,:,0])
std_map = np.empty_like(spectrum_tab[:,:,0])
snr_map = np.empty_like(spectrum_tab[:,:,0])
integral = np.empty_like(spectrum_tab[:,:,0])


for i in range(spectrum_tab.shape[0]):
    for j in range(spectrum_tab.shape[1]):
        
        nb_map[i, j], std_map[i, j], snr_map[i, j], integral[i,j] =  compute_snr(spectrum_tab, wavelengths, i, j, std_bounds, max_interval)
        
        
    
#%% Plots 
    
scalebar = ScaleBar(len_fov/(np.shape(spectrum_tab)[0]), "mm")
        
plt.figure('STD')
plt.clf()
plt.title('STD map')
plt.imshow(std_map)
plt.colorbar()
ax1 = plt.gca()
ax1.add_artist(scalebar)
if savefig_map == True :
    plt.savefig(root_savefig + num_patient + num_biopsy + type_reco + '_std_map.png',  bbox_inches='tight')
    
    
scalebar = ScaleBar(len_fov/(np.shape(spectrum_tab)[0]), "mm")
    
plt.figure('SNR')
plt.clf()
plt.title('SNR map')
plt.imshow(snr_map)
plt.colorbar()
ax2 = plt.gca()
ax2.add_artist(scalebar)
if savefig_map == True :
    plt.savefig(root_savefig + num_patient + num_biopsy + type_reco + '_snr_map.png',  bbox_inches='tight')
    
    
    
    
    
##############################################################################################################
#%% Statistics 
# Histogram 

bins = 50 


plt.figure('Histogram ')
plt.clf()
plt.title('Number of pixels in the image with a given SNR value')
plt.hist(snr_map.flatten(), bins)
# plt.hist(snr_map, bins, range=(range_min, range_max))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('SNR', fontsize=16)
plt.ylabel('Number of pixels', fontsize=16)
if save_histogram == True :
    plt.savefig(root_savefig_hist + num_patient + num_biopsy + type_reco + '_snr_histogram.png',  bbox_inches='tight')
    
    
    
#%%
# Number of pixels with SNR >= 10 : 

def nb_values_over_n(array, n):
    
    nb = 0    
        
    for i in range(len(array)):
        
        if array[i] >= n :
            nb += 1 
    
    return nb


flat = snr_map.flatten()
nb_non_nan = 0 


for i in range(len(flat)):
    if math.isnan(flat[i]) == False : 
        nb_non_nan += 1



nb_pts_snr_over_ten = nb_values_over_n(flat, 10)    


































 

