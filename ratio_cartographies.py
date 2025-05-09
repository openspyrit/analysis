# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 16:04:21 2025

@author: chiliaeva

Ratio 620/634 plot
- standard in log scale : a_620/a_634
- binary


"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import TwoSlopeNorm


#%%

savefig_ratio = True # saves the ratio map in png format
savenpy_ratio = False # saves the .npy file corresponding to the ratio map 

# savefig_spectrum = False

type_reco = 'had_reco'
vmax = 0.5 # max value of ratio in the colormap


# Get fit data
# root = 'D:/'
root = 'C:/'
# root = 'C:/Users/chiliaeva/Documents/Resultats_traitement/'

# root_saveresults = root + 'fitresults_250331_nn_reco/'
# root_saveresults = root + 'fitresults_250331_nn_reco/'
root_saveresults = root + 'fitresults_250327_full-spectra_spat-bin_0/'



root_savefig = root_saveresults + 'fig/'
if os.path.exists(root_savefig) == False :
    os.mkdir(root_savefig)

root_ratios = root_savefig + 'maps/ratios_log_scale_vmax=' + str(vmax) + '_binary/'
if os.path.exists(root_ratios) == False :
    os.mkdir(root_ratios)



num_patient = 'P60_'
num_biopsy = 'B4'



file_params = root_saveresults + num_patient +  type_reco + '/' + num_biopsy + '_' + type_reco + '_fit_params.npy'
params_tab = np.load(file_params)



min_ppix = np.amin([np.nanmin(params_tab[:,:,0]), np.nanmin(params_tab[:,:,1])]) # minimum for Protoporphyrin IX colormap
max_ppix = np.amax([np.nanmax(params_tab[:,:,0]), np.nanmax(params_tab[:,:,1])])



if type_reco == 'nn_reco':
    params_tab = cv2.flip(params_tab, 0)
    
    
    

#%% Log transform the data, then use TwoSlopeNorm


norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
cmap = plt.get_cmap('viridis')

# lim = 1e-4 # values below this limit take this value


#########################################################################################


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
                print('P', num_patient, 'B', num_biops)
                subpath = os.path.join(path, file)
                coef_P620 = np.load(subpath)[:,:,0]
                coef_P634 = np.load(subpath)[:,:,1]
                
                xx = np.linspace(0, coef_P620[0].size, coef_P620[0].size, endpoint = False)
                yy = np.linspace(0, coef_P620[0].size, coef_P620[0].size, endpoint = False)
                X, Y = np.meshgrid(xx, yy)
                
                
                # coef_P620 = np.where(coef_P620 >= lim, coef_P620, lim) # remove values that are too close to zero
                # coef_P634 = np.where(coef_P634 >= lim, coef_P634, lim)
                
                
                ratio = coef_P620/coef_P634
                if savenpy_ratio :
                    np.save(root_ratios + num_patient + num_biops + '_' + type_reco + '_ratio_620_634.npy', ratio)
                ratio = np.where(np.isnan(ratio), 1, ratio)
                log_ratio = np.log10(ratio)
                
                                
                plt.figure()
                plt.imshow(log_ratio, cmap=cmap, norm=norm)
                plt.colorbar()
                plt.grid()
                if savefig_ratio :
                    plt.savefig(root_ratios + num_patient + num_biops + '_' + type_reco + '_ratio_620_634.png', bbox_inches='tight')
                plt.close()
                
                
 
 #%% Binary masks 
 
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
cmap = plt.get_cmap('viridis')
vlim = 0 # threshold value      

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
                print('P', num_patient, 'B', num_biops)
                subpath = os.path.join(path, file)
                coef_P620 = np.load(subpath)[:,:,0]
                coef_P634 = np.load(subpath)[:,:,1]
                
                xx = np.linspace(0, coef_P620[0].size, coef_P620[0].size, endpoint = False)
                yy = np.linspace(0, coef_P620[0].size, coef_P620[0].size, endpoint = False)
                X, Y = np.meshgrid(xx, yy)
                
                
                # coef_P620 = np.where(coef_P620 >= lim, coef_P620, lim) # remove values that are too close to zero
                # coef_P634 = np.where(coef_P634 >= lim, coef_P634, lim)
                
                
                ratio = coef_P620/coef_P634
                if savenpy_ratio :
                    np.save(root_ratios + num_patient + num_biops + '_' + type_reco + '_ratio_620_634.npy', ratio)
                ratio = np.where(np.isnan(ratio), 1, ratio)
                log_ratio = np.log10(ratio)
                
                mask_high = log_ratio > vlim
                mask_low = log_ratio < vlim
                mask_equal = log_ratio = vlim
                
                output = np.zeros((coef_P620.shape[0], coef_P620.shape[1], 3), dtype=np.uint8)
                
                output[mask_low] = [255, 0, 0]  # Blue : 634 is predominant
                output[mask_high] = [0, 255, 255]  # Yellow : 620 is predominant
                output[mask_equal] = [0, 255, 0] # Green : no predominance
                
                cv2.imshow('Thresholded Color Map', output)
                if savefig_ratio :
                    cv2.imwrite(root_ratios + num_patient + num_biops + '_' + type_reco + '_binary_ratio_620_634.png', output)
                cv2.destroyAllWindows()



#%% Compute mean ratios 

folders = os.listdir(root_saveresults)
ratio_mean = []
 

for folder in folders : 
    if 'P' in folder :
        num_patient = folder[0:4]
        type_reco = folder[4:]
        path = os.path.join(root_saveresults, folder)
        files = os.listdir(path)
        for file in files :
            if 'fit_params' in file :
                num_biops = file[0:3]
                print('P', num_patient, 'B', num_biops)
                subpath = os.path.join(path, file)
                coef_P620 = np.load(subpath)[:,:,0]
                coef_P634 = np.load(subpath)[:,:,1]
                
                ratio = coef_P620/coef_P634
                ratio_mean.append(np.nanmean(ratio))
                
                
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   