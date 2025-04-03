# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 11:32:11 2025

@author: chiliaeva

2025/03/27

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from spas.metadata import read_metadata


#%% Spatial averaging function

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



#%% Spatial averaging of 3D array over 2D dimensions 

            
def spatial_avg_3D(matrix, spat_bin, x, y):
    # Input : 
        # 3D matrix
        # int spat_bin
        # int,int x, y : spatial position
    # Output : 1D array_avg of size matrix.shape[2]
    array_sum = np.zeros(matrix.shape[2], dtype=float)
    cpt = 0
    for i in range(x-spat_bin, x+spat_bin+1):
        for j in range(y-spat_bin, y+spat_bin+1):
            array_temp = matrix[i,j,:]
            if not np.isnan(array_temp).any() :
                array_sum = array_sum + array_temp
                cpt += 1
    array_avg = array_sum / cpt
    return array_avg, cpt





#%% 

type_reco = 'had_reco'

root = 'C:/'
root_saveresults = root + 'fitresults_250313_full-spectra_/'
root_ref = root + 'ref/'


file_metadata = root + 'wavelengths_metadata.json'
metadata, acquisition_params, spectrometer_params, dmd_params = read_metadata(file_metadata)
wavelengths = acquisition_params.wavelengths
wvlgth_bin = np.load(root_saveresults + "wavelengths_mask_bin.npy")


num_patient = 'P69_'
num_biopsy = 'B5'


#%%

# Spatial binning 
spat_bin = 5 # number of neighbours (in every direction) to take into account in the spatial averaging

file_spectra = root_saveresults + num_patient + type_reco + '/' + num_biopsy + '_' + type_reco + '_spectrum_tab.npy'
spectrum_tab = np.load(file_spectra)   



#%%
# spectrum = spectrum_tab[x, y, :]
# Spatial averaging of the raw spectrum : 
# spectrum = np.nanmean(np.nanmean(spectrum_tab[x-spat_bin:x+spat_bin+1,y-spat_bin:y+spat_bin+1,:], axis=0), axis=0)


spectrum_tab_avg = np.empty_like(spectrum_tab)


for i in range(spectrum_tab.shape[0]):
    for j in range(spectrum_tab.shape[1]):
        if not np.isnan(spectrum_tab[i,j,:]).any() :
            spectrum_avg = np.nanmean(np.nanmean(spectrum_tab[i-spat_bin:i+spat_bin+1,j-spat_bin:j+spat_bin+1,:], axis=0), axis=0)
            spectrum_tab_avg[i,j,:] = spectrum_avg
        else :
            spectrum_tab_avg[i,j,:] = np.nan
        
        
        
        
#%% Compare 3 methods of computing the spatial average

x = 5
y = 24


spectrum = spectrum_tab[x,y,:]        
spectrum_avg = spectrum_tab_avg[x,y,:]   
spectrum_avg_fxn = spatial_avg_3D(spectrum_tab, spat_bin, x, y)[0]


        
#%% Plot raw and averaged spectrum at position x, y
        
plt.figure('Raw and averaged spectra')
plt.plot(wvlgth_bin, spectrum,  label='spectrum', color='black')
plt.plot(wvlgth_bin, spectrum_avg,  label='avg spectrum', color='purple')
plt.plot(wvlgth_bin, spectrum_avg_fxn,  label='avg spectrum using function', color='blue')



###############################################################################################################
###############################################################################################################
###############################################################################################################
#%% Plot a slice of the raw and averaged spectrum_tab

nb_slice = 100


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False)
ax1.imshow(spectrum_tab[:,:,nb_slice])
ax1.grid()
ax2.imshow(spectrum_tab_avg[:,:,nb_slice])
ax2.grid()


        
#%% 

file_params = root_saveresults + num_patient +  type_reco + '/' + num_biopsy + '_' + type_reco + '_fit_params.npy'
params_tab = np.load(file_params)


#%% Compute the entire spectrum_tab_avg using the function : 
    
'''
spectrum_tab_avg_fxn = np.empty_like(spectrum_tab)


for i in range(spectrum_tab.shape[0]):
    for j in range(spectrum_tab.shape[1]):
        if not np.isnan(spectrum_tab[i,j,:]).any() :
            spectrum_avg_fxn = spatial_avg_3D(spectrum_tab, spat_bin, i, j)[0]
            spectrum_tab_avg_fxn[i,j,:] = spectrum_avg_fxn
        else :
            spectrum_tab_avg_fxn[i,j,:] = np.nan
        
'''        
        
















 






