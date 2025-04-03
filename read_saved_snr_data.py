# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:58:43 2025

@author: chiliaeva

Read saved .npy and .pickle files
"""


import os 
import pickle 
import matplotlib.pyplot as plt
import numpy as np


#%% Get SNR data 

type_reco = 'had_reco'

root = 'C:/'
root_snr = root + 'snr/'


num_patient = 'P68_'
num_biopsy = 'B1_'


# max_snr = np.load(root_snr + num_patient + num_biopsy + type_reco + '_max_snr.npy')
# mean_snr = np.load(root_snr + num_patient + num_biopsy + type_reco + '_mean_snr.npy')
# mean_std = np.load(root_snr + num_patient + num_biopsy + type_reco + '_mean_std.npy')



std_map = np.load(root_snr + num_patient + num_biopsy + type_reco + '_std_tab.npy')
snr_map = np.load(root_snr + num_patient + num_biopsy + type_reco + '_snr_tab.npy')
integral_map = np.load(root_snr + num_patient + num_biopsy + type_reco + '_integral_tab.npy')
integral_width_map = np.load(root_snr + num_patient + num_biopsy + type_reco + '_width_integral.npy')



#%% Compute max and mean


max_snr = np.nanmax(snr_map)
mean_snr = np.nanmean(snr_map)

max_std=np.nanmax(std_map)
mean_std=np.nanmean(std_map)



#%% Plot maps

mksize = 4
show_pos = False

x = 21
y = 18



plt.figure('SNR map')
plt.clf()
plt.imshow(snr_map)
plt.colorbar()
plt.grid()
if show_pos == True :
    plt.plot(y,x, "or", markersize = mksize)


plt.figure('STD map')
plt.clf()
plt.imshow(std_map)
plt.colorbar()
plt.grid()
if show_pos == True :
    plt.plot(y,x, "or", markersize = mksize)











#%% Get fit bounds





















