# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 11:08:58 2025

@author: chiliaeva

Load Ratio620/634 in npy format
Compute max, min, expectancy, STD
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stat
from subdirs_list import get_next_n_chars


#%% load ratio

save_fig = False

type_reco = 'nn_reco' 
root = 'C:/'

# root_savenpy = root + 'fitresults_250327_full-spectra_spat-bin_0/fig/maps/ratios_npy/'
root_savenpy = root + 'fitresults_250331_nn_reco/fig/maps/ratios_npy/'


num_patient = 'P69_'
num_biopsy = 'B9_'



# file = root_savenpy + num_patient + num_biopsy + type_reco + 'ratio_620_634.npy'
file = root_savenpy + num_patient + num_biopsy + '_' + type_reco + 'ratio_620_634.npy'
ratio = np.load(file)



root_snr = root + 'snr/' + type_reco + '/'
# file_snr = root_snr + num_patient + num_biopsy + type_reco + '_snr_tab.npy'
file_snr = root_snr + num_patient + num_biopsy + type_reco + '_snr_map.npy'
snr_map = np.load(file_snr)


snr_max = np.nanmax(snr_map)
snr_mean = np.nanmean(snr_map)


#%% Plot ratio map


plt.figure('Ratio map')
plt.clf()
plt.imshow(ratio)
plt.colorbar()


plt.figure('Ratio histogram')
plt.clf()
plt.hist(ratio)


ratio_mean = np.nanmean(ratio)


vhigh = snr_max
vlow = 1/snr_max

mask_high = ratio > vhigh
mask_low = ratio < vlow

ratio[mask_high] = np.nan
ratio[mask_low] = np.nan


plt.figure('Ratio map_')
plt.clf()
plt.imshow(ratio)
plt.colorbar()


plt.figure('Ratio histogram_')
plt.clf()
plt.hist(ratio)

ratio_mean_clean = np.nanmean(ratio)


#%%


ratio_flat = ratio.flatten()

ratio = [x for x in ratio_flat if not (math.isnan(x) or x<1e-4 or x>1e4)]


#%% Plot 

plt.figure('Flattened ratio array')
plt.clf()
plt.plot(ratio, 'x')


plt.figure('Histogram')
plt.clf()
plt.hist(ratio, 20)
plt.ylabel('Number of measurements')
plt.xlabel('Value')


#%% Stats

mean = np.mean(ratio)
max_ = np.amax(ratio)
min_ = np.amin(ratio)



#%% First, clean the data : remove very large and very small ratios 

# Option 1 : only keep the values around mean +- 20%




#%% Compute the mean, max and min ratio for all biopsies : 
    
list_files = os.listdir(root_savenpy)  
    
    
mean_ratio_tab = np.empty(np.shape(list_files))
max_ratio_tab = np.empty(np.shape(list_files))
min_ratio_tab = np.empty(np.shape(list_files))


list_measurements = []


for index, file in enumerate(list_files) :
    
    print(index)
    
    num_patient = get_next_n_chars(file, 'P', 3)
    num_biopsy = get_next_n_chars(file, 'B', 2)
    list_measurements.append(num_patient+num_biopsy)
    
    file_path = os.path.join(root_savenpy, file)

    ratio = np.load(file_path)
    ratio_flat = ratio.flatten()
    ratio = [x for x in ratio_flat if not (math.isnan(x) or x<1e-4 or x>1e4)]
    
    if ratio :
        mean_ratio_tab[index] = np.nanmean(ratio)
        max_ratio_tab[index] = np.nanmax(ratio)
        min_ratio_tab[index] = np.nanmin(ratio)
    if not ratio : 
        mean_ratio_tab[index] = np.nan
        max_ratio_tab[index] = np.nan
        min_ratio_tab[index] = np.nan





#%% Compare the ratios for had_reco and nn_reco

mean_ratio_nn = mean_ratio_tab
min_ratio_nn = min_ratio_tab
max_ratio_nn = max_ratio_tab


mean_ratio_had = np.load('C:/ratio_results/mean_ratios_had_reco.npy')
min_ratio_had = np.load('C:/ratio_results/min_ratios_had_reco.npy')
max_ratio_had = np.load('C:/ratio_results/max_ratios_had_reco.npy')


fig, ax = plt.subplots(1, 1, figsize = [16, 8])
ax.tick_params(axis='x', rotation=55, labelsize=8)
ax.plot(list_measurements, mean_ratio_had, 'bx', markersize=10, label='had reco')
ax.plot(list_measurements, mean_ratio_nn, 'rx', markersize=10, label='nn reco')
ax.vlines(list_measurements, mean_ratio_had, mean_ratio_nn, linewidth=2, color='black')
plt.legend()
if save_fig : 
    plt.savefig(root + 'ratio_results/' + 'mean_ratio_nn_had_compare.png', bbox_inches = 'tight')
# plt.ylim(0, 10)
# plt.grid()



#%% 







