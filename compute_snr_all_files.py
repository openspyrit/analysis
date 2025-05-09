# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 12:40:57 2025

@author: chiliaeva
"""


import os
import numpy as np
import matplotlib.pyplot as plt
fontsize = 14



import cv2 as cv
# from scipy import signal
# from scipy.integrate import trapezoid


from subdirs_list import find_subdirs_with_string, find_npy_files, get_next_n_chars 
from snr_functions import compute_snr


#%%

savefig_map = False
savenpy_map = True
savedata = False
savefig_hist = False

type_reco = 'nn_reco'

# root_saveresults = 'C:/fitresults_250327_full-spectra_spat-bin_0/'
root_saveresults = 'C:/fitresults_250331_nn_reco/'


file_wvlgth = root_saveresults + 'wavelengths_mask_bin.npy'
wavelengths = np.load(file_wvlgth)


root_savefig = root_saveresults + 'fig/snr/'
if os.path.exists(root_savefig) == False :
    os.mkdir(root_savefig)


list_results_files = find_subdirs_with_string(root_saveresults, type_reco)
list_spectrum_files = find_npy_files(list_results_files, 'spectrum_tab')


std_bounds = [650, 748]
max_interval = [620, 640]



#%%

mean_snr_tab = np.empty(np.shape(list_spectrum_files))
max_snr_tab = np.empty(np.shape(list_spectrum_files))
mean_std_tab = np.empty(np.shape(list_spectrum_files))

list_measurements = []


for index, file in enumerate(list_spectrum_files) : 
    
    print('file N°', index)
    
    num_patient = get_next_n_chars(file, 'P', 3)
    num_biopsy = get_next_n_chars(file, 'B', 2)
    list_measurements.append(num_patient+num_biopsy)
    
    
    spectrum_tab = np.load(file)

    
    nb_tab = np.empty_like(spectrum_tab[:,:,0])
    std_tab = np.empty_like(spectrum_tab[:,:,0])
    snr_tab = np.empty_like(spectrum_tab[:,:,0])
    integral_tab = np.empty_like(spectrum_tab[:,:,0])


    for i in range(spectrum_tab.shape[0]):
        print (i)
        for j in range(spectrum_tab.shape[1]):
            print(j)
        
            nb_tab[i, j], std_tab[i, j], snr_tab[i, j], integral_tab[i, j] =  compute_snr(spectrum_tab, wavelengths, i, j, std_bounds, max_interval)
        
            
    plt.figure('STD')
    plt.clf()
    plt.imshow(std_tab)
    plt.colorbar()
    if savefig_map == True :
        plt.savefig(root_savefig + num_patient + num_biopsy + type_reco + '_std_map.png',  bbox_inches='tight')
    if savenpy_map == True :
        np.save(root_savefig + num_patient + num_biopsy + type_reco + '_std_map.npy', std_tab, allow_pickle=True)
    
            
    plt.figure('SNR')
    plt.clf()
    plt.imshow(snr_tab)
    plt.colorbar()
    if savefig_map == True :
        plt.savefig(root_savefig + num_patient + num_biopsy + type_reco + '_snr_map.png',  bbox_inches='tight')
    if savenpy_map == True :
        np.save(root_savefig + num_patient + num_biopsy + type_reco + '_snr_map.npy', snr_tab, allow_pickle=True)
    
    
    mean_std = np.nanmean(std_tab)
    mean_std_tab[index] = mean_std
    if savedata == True :
        np.save(root_savefig + num_patient + num_biopsy + type_reco + '_mean_std',  mean_std)
    
    mean_snr = np.nanmean(snr_tab)
    mean_snr_tab[index] = mean_snr
    if savedata == True :
        np.save(root_savefig + num_patient + num_biopsy + type_reco + '_mean_snr',  mean_snr)
    
    max_snr = np.nanmax(snr_tab)
    max_snr_tab[index] = max_snr
        

#%% Save in csv format :

# std_path = root_savefig + type_reco + '_mean_std_tab.csv'

mean_snr_path = root_savefig + type_reco + '_mean_snr_tab.csv'
max_snr_path = root_savefig + type_reco + '_max_snr_tab.csv'


np.savetxt(mean_snr_path, mean_snr_tab, delimiter=',', fmt='%s')
np.savetxt(max_snr_path, max_snr_tab, delimiter=',', fmt='%s')


#######################################################################################
########################################################################################
#%% Statistics 

save_stats = True


fig, ax = plt.subplots(1, 1, figsize = [16, 8])
ax.plot(list_measurements, mean_snr_tab, marker='p', linestyle='')
ax.tick_params(axis='x', rotation=55, labelsize=8)
plt.xlabel("Measurement N°")
plt.ylabel("Mean SNR value")
if save_stats : 
    plt.savefig(root_savefig + type_reco + '_mean_snr_all_measurements', bbox_inches = 'tight')



fig, ax = plt.subplots(1, 1, figsize = [16, 8])
ax.plot(list_measurements, max_snr_tab, marker='p', linestyle='')
ax.tick_params(axis='x', rotation=55, labelsize=8)
plt.xlabel("Measurement N°")
plt.ylabel("Max SNR value")
if save_stats : 
    plt.savefig(root_savefig + type_reco + '_max_snr_all_measurements', bbox_inches = 'tight')



#%% Histograms


def plot_histogram(array, bins, range_min, range_max):
    
    '''
    Arguments : 
        array : 1D array to count and bin
        bins : int, number of bins
        range_min : float
        range_max : float
    '''
    
    plt.figure()
    plt.clf()
    plt.hist(array, bins, range=(range_min, range_max))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('Number of measurements')
    plt.xlabel('')

    
    
# plot_histogram(mean_snr_tab, 100, 0, 20)


#%% 


bins = 100
range_min = 0 
range_max = 20
ticks = np.arange(0, 20, 1)
ftz = 16


plt.figure('Mean SNR histogram')
plt.clf()
plt.hist(mean_snr_tab, bins, range=(range_min, range_max))
plt.xticks(ticks, fontsize=ftz)
plt.yticks(fontsize=ftz)
plt.ylabel('Number of measurements', fontsize=ftz)
plt.xlabel('Mean SNR value', fontsize=ftz)
if savefig_hist == True : 
    plt.savefig(root_savefig + type_reco + '_global_histogram.png',  bbox_inches='tight')
    
    


#%% Count number of measurements with mean SNR >= 10 : 
    
    
def nb_values_over_n(array, n):
    
    nb = 0    
        
    for i in range(len(array)):
        
        if array[i] >= n :
            nb += 1 
    
    return nb
        
        
nb_mean_snr_over_five = nb_values_over_n(mean_snr_tab, 5)    
nb_max_snr_over_five = nb_values_over_n(max_snr_tab, 5)   





























 


























