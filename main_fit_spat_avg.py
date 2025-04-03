# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 12:49:03 2024

@author: chiliaeva
"""

import numpy as np
import pickle

from spas.metadata import read_metadata

import scipy.optimize as op
# from scipy import interpolate
# from preprocessing import func620, func634
from preprocessing import func_fit


import os
import time


#%% Paths


save_fit_data = True

root = 'C:/'

root_data = root + 'd/'
root_ref = root + 'ref/'

# os.mkdir(root + 'fitresults_' + )
root_saveresults = root + 'fitresults_250331_nn_reco/'
if os.path.exists(root_saveresults) == False :
    os.mkdir(root_saveresults)


folders = os.listdir(root_data)
print("folders", folders)


# metadata file to get the wavelengths array :
#file_metadata = 'D:/obj_biopsy-1_anterior-portion_source_Laser_405nm_1.2W_A_0.15_f80mm-P2_Walsh_im_16x16_ti_200ms_zoom_x1_metadata.json'
file_metadata = root + 'wavelengths_metadata.json'
metadata, acquisition_params, spectrometer_params, dmd_params = read_metadata(file_metadata)
wavelengths = acquisition_params.wavelengths



#%% Parameters 

# Fit bounds
bounds = {
"shift_min" : -2,
"shift_max" : 2, # allows the fitting function to shift the Pp ref spectra by -shift_min to +shift_max
"lbd_c_init" : 590, # initial guess for lipofuscin central wavelength (nm)
"lbd_c_min" : 585, # lower bound
"lbd_c_max" : 595, # upper bound
"sigma_init" : 15,
"sigma_min" : 10, 
"sigma_max" : 20}

with open(root_saveresults + 'bounds.pickle', 'wb') as handle:
    pickle.dump(bounds, handle, protocol=pickle.HIGHEST_PROTOCOL)

spat_bin = 0 # number of neighbours (in every direction) to take into account in the spatial averaging

type_reco = 'nn_reco'
type_reco_npz = type_reco + '.npz'


# Perform the fit only on a part of the spectral domain 

############ REJECT A SPECTRAL BAND #############################
reject_band = [606, 616] # exclude spectral band reject_band from fit (nm). To reject no band, set reject_band[0]=reject_band[1]
band_mask = (wavelengths <= reject_band[0])|(wavelengths >= reject_band[1])


########### INCLUDE ONLY THIS BAND ###################
# fit_start = 614
# fit_stop = 650
# band_mask = (wavelengths >= fit_start) & (wavelengths <= fit_stop)


if save_fit_data == True : 
    np.savetxt(root_saveresults + 'band_mask.npy', band_mask, fmt="%5i") 



wavelengths = wavelengths[band_mask]

if save_fit_data == True : 
    np.save(root_saveresults + 'wavelengths_mask.npy', wavelengths) 


# Resampling the wavelength scale for the binned spectrum
bin_width = 10

wvlgth_bin = np.ndarray(wavelengths.size // bin_width, dtype=float)

for i in range(wvlgth_bin.size):
    wvlgth_bin[i] = wavelengths[i*bin_width]
if save_fit_data == True : 
    np.save(root_saveresults + 'wavelengths_mask_bin.npy', wvlgth_bin)     
    
    


 #%% Import ref spectra


# load interpolated and normalized ref spectra : 
spectr620 = np.load(root_ref + '_spectr620_interp.npy')
spectr634 = np.load(root_ref + '_spectr634_interp.npy')

spectr620 = spectr620[band_mask]
spectr634 = spectr634[band_mask]




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
           print('s =', s)
           if (s[11] == str(num_biopsy) and (s[12] == '-' or s[12] == '_') ):
               print('loop1, ', 's[12] :', s[12])
               subpath = path + '/' + s + '/'
               if "Laser" in s : 
                   file_cube_laser = subpath + s + '_' + type_reco_npz
               elif "No-light" in s :
                   file_cube_nolight = subpath + s + '_' + type_reco_npz
               elif "white" in s : 
                   file_mask = subpath + type_reco + '_mask.npy'
           elif s[11:13] == str(num_biopsy) :
               print('loop2, ', 's[12] :', s[12])
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
        spectrum_tab = np.ndarray((cubehyper_laser.shape[0], cubehyper_laser.shape[1], np.size(wvlgth_bin)), dtype='float64')
        spectrum_tab[:] = np.nan   
         
        # Fit for every point of the mask
        t0 = time.time()
        print('start fit for the entire image', time.time()-t0)
        
        
        # Remove background and spectral binning 
        for x_i in range(cubehyper_laser.shape[0]):
            for y_i in range(cubehyper_laser.shape[1]):
                
                if mask[x_i, y_i]!=0:
                    
                
                    spectr_laser = cubehyper_laser[x_i, y_i, :][band_mask]
                    spectr_nolight = cubehyper_nolight[x_i, y_i, :][band_mask]
                    
                    
                    
                    # Binning
                    sp_laser_bin = np.ndarray(spectr_laser.size // bin_width, dtype=float)
                    sp_nolight_bin = np.ndarray(spectr_laser.size // bin_width, dtype=float)
                    
                    for i in range(sp_laser_bin.size):
                        sp_laser_bin[i] = np.sum(spectr_laser[i*bin_width:(1+i)*bin_width])
                        sp_nolight_bin[i] = np.sum(spectr_nolight[i*bin_width:(1+i)*bin_width])
                        
                
                    # Remove the no light spectrum
                    spectrum = sp_laser_bin - sp_nolight_bin
                    spectrum_tab[x_i, y_i, :] = spectrum
                    
    
            
        # Compute spatial average of spectrum_tab
        spectrum_tab_avg = np.empty_like(spectrum_tab)
        

        for i in range(spectrum_tab.shape[0]):
            for j in range(spectrum_tab.shape[1]):
                if not np.isnan(spectrum_tab[i,j,:]).any() :
                    spectrum_avg = np.nanmean(np.nanmean(spectrum_tab[i-spat_bin:i+spat_bin+1,j-spat_bin:j+spat_bin+1,:], axis=0), axis=0)
                    spectrum_tab_avg[i,j,:] = spectrum_avg
                    
                    if not np.isnan(spectrum_avg).any() :
                        
                        # FIT THE SPECTRUM TO REFERENCE SPECTRA
                        M = np.abs(np.max(spectrum_avg))
                        p0 = [M/2, M/2, M/8, 0, 0, bounds["lbd_c_init"], bounds["sigma_init"]] # initial guess for the fit
                        bounds_inf = [0, 0 ,0 , bounds["shift_min"], bounds["shift_min"], bounds["lbd_c_min"], bounds["sigma_min"]] 
                        bounds_sup = [M, M, M, bounds["shift_max"], bounds["shift_max"], bounds["lbd_c_max"], bounds["sigma_max"]] 
                        
                        try : 
                            popt, pcov = op.curve_fit(func_fit, wvlgth_bin, spectrum_avg, p0, bounds=(bounds_inf, bounds_sup))
                            popt_tab[i, j, :] = popt
                        except RuntimeError:
                            pass
                        
                    
                else :
                    spectrum_tab_avg[i,j,:] = np.nan
                
                
                
        print('end fit for image', time.time()-t0)  
        
        if save_fit_data == True:
            np.save(root_saveresults + f + '_' + type_reco + '/B' + str(num_biopsy) + '_' +  type_reco + '_spectrum_tab.npy', spectrum_tab_avg) 
            np.save(root_saveresults + f + '_' + type_reco + '/B' + str(num_biopsy) + '_' +  type_reco + '_fit_params.npy', popt_tab) 

        

    list_biopsies = []      


    
    














































     





