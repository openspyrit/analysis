# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 12:49:03 2024

@author: chiliaeva
"""


import numpy as np

from spas.metadata2 import read_metadata

import scipy.signal as sg
import scipy.optimize as op
# from scipy import interpolate
# from preprocessing import func620, func634
from preprocessing import func_fit


import os
import time


#%% Paths

root = 'D:/'


root_data = root + 'd/'
root_ref = root + 'ref/'

# os.mkdir(root + 'fitresults_' + )
root_saveresults = root + 'fitresults/'

folders = os.listdir(root_data)
print("folders", folders)


# metadata file to get the wavelengths array :
file_metadata = 'D:/obj_biopsy-1_anterior-portion_source_Laser_405nm_1.2W_A_0.15_f80mm-P2_Walsh_im_16x16_ti_200ms_zoom_x1_metadata.json'
metadata, acquisition_params, spectrometer_params, dmd_params = read_metadata(file_metadata)
wavelengths = acquisition_params.wavelengths



#%% Parameters 

save_fit_data = True

type_reco = 'nn_reco'
type_reco_npz = type_reco + '.npz'




real_spectro_reso = 2 # real resolution of the spectrometer (nm)
theo_spectro_reso = (wavelengths[-1]-wavelengths[0])/len(wavelengths)
kernel_size = int(real_spectro_reso/theo_spectro_reso) + 1 # size of the window for the median filter, must be odd
if kernel_size %2 == 0 : 
    kernel_size = kernel_size + 1




reject_band = [606, 616] # exclude spectral band reject_band from fit (nm). To reject no band, set reject_band[0]=reject_band[1]

band_stop_mask = (wavelengths <= reject_band[0])|(wavelengths >= reject_band[1])
wavelengths = wavelengths[band_stop_mask]
if save_fit_data == True : 
    np.save(root_saveresults + 'wavelengths_mask_' + str(reject_band[0]) + '-' + str(reject_band[1]) + '.npy', wavelengths) 




 #%% Import ref spectra


# load interpolated and normalized ref spectra : 
spectr620 = np.load(root_ref + '_spectr620_interp.npy')
spectr634 = np.load(root_ref + '_spectr634_interp.npy')

spectr620 = spectr620[band_stop_mask]
spectr634 = spectr634[band_stop_mask]




#%% Select the files


list_biopsies = []


for f in folders : 
    path = os.path.join(root_data, f)
    os.mkdir(root_saveresults + f + '_' + type_reco)
    subdirs = os.listdir(path)
    for s in subdirs : 
        nb = int(s[11])
        if nb not in list_biopsies :
            list_biopsies.append(nb)
    for s in subdirs :
        if s[12] != '-' and s[12] != '_' :
            nb_ = int(s[12])
            nb = int(s[11])*10 + nb_
            if nb not in list_biopsies :
                list_biopsies.append(nb)
                
  
          
    print("list of biopsies in", f, ":", list_biopsies)
    for num_biopsy in list_biopsies : 
        print('numero biopsie : ', num_biopsy)
        for s in subdirs :
            if s[11] == str(num_biopsy) :
                subpath = path + '/' + s + '/'
                if "Laser" in s : 
                    file_cube_laser = subpath + s + '_' + type_reco_npz
                elif "No-light" in s :
                    file_cube_nolight = subpath + s + '_' + type_reco_npz
                elif "white" in s : 
                    file_mask = subpath + type_reco + '_mask.npy'
                       
                    
        print("start reading hypercube")            
        # Read laser hypercube
        cubeobj = np.load(file_cube_laser)
        cubehyper_laser = cubeobj['arr_0']
       
        
   
        # Read nolight hypercube 
        cubeobj = np.load(file_cube_nolight)
        cubehyper_nolight = cubeobj['arr_0']
        del cubeobj
        
        # Read mask 
        mask = np.load(file_mask)
        
        
        # @todo : define an array of shape (cube[0], cube[1], nb of param of func_fit)
        popt_tab = np.ndarray((cubehyper_laser.shape[0], cubehyper_laser.shape[1], 7), dtype = 'float64')
        popt_tab[:] = np.nan       
        spectrum_tab = np.ndarray((cubehyper_laser.shape[0], cubehyper_laser.shape[1], np.size(wavelengths)), dtype='float64')
        spectrum_tab[:] = np.nan   
         
        # Fit for every point of the mask
        t0 = time.time()
        print('start fit for the entire image', time.time()-t0)
        
        
        for x_i in range(cubehyper_laser.shape[0]):
            for y_i in range(cubehyper_laser.shape[1]):
                
                if mask[x_i, y_i]!=0:
                    
                
                    spectr_laser = cubehyper_laser[x_i, y_i, :]
                    spectr_nolight = cubehyper_nolight[x_i, y_i, :]
                
        
                    # median filter to smoothen the spectrum
                
                    sp_laser_smth = sg.medfilt(spectr_laser, kernel_size)
                    sp_nolight_smth = sg.medfilt(spectr_nolight, kernel_size)
                
                
                    # Remove the no light spectrum
                    spectrum = sp_laser_smth - sp_nolight_smth
                    spectrum = spectrum[band_stop_mask]
                    
                    spectrum_tab[x_i, y_i, :] = spectrum
        
        
        
                    # FIT THE SPECTRUM TO REFERENCE SPECTRA
                    
                    M = np.abs(np.max(spectrum))
                    p0 = [M/2, M/2, M/8, 0, 0, 585, 10]# initial guess for the fit
                    bounds_inf = [0, 0 ,0 ,-2, -2, 580, 5] 
                    bounds_sup = [M, M, M, 2, 2, 610, 100] 
                    
                    try : 
                        popt, pcov = op.curve_fit(func_fit, wavelengths, spectrum, p0, bounds=(bounds_inf, bounds_sup))
                        popt_tab[x_i, y_i, :] = popt
                    except RuntimeError:
                        pass
                
                
                
        print('end fit for image', time.time()-t0)  
        
   
        if save_fit_data == True:
            np.save(root_saveresults + f + '_' + type_reco + '/B' + str(num_biopsy) + '_' +  type_reco + '_spectrum_tab.npy', spectrum_tab) 
            np.save(root_saveresults + f + '_' + type_reco + '/B' + str(num_biopsy) + '_' +  type_reco + '_fit_params.npy', popt_tab) 


    list_biopsies = []      


    
    














































     





