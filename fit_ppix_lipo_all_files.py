# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 12:49:03 2024

@author: chiliaeva
"""


import numpy as np

from spas.metadata2 import read_metadata

import scipy.signal as sg
import scipy.optimize as op
from scipy import interpolate

import os
import time


#%% Parameters


root = 'D:/hspc/data/2024a/' # @todo : temporary, change
folders = os.listdir(root)

 
list_biops = np.arange(1,10,1, dtype=int) # biospies from 1 to 9

type_reco = 'had_reco'
type_reco_npz = type_reco + '.npz'

real_spectro_reso = 2 # real resolution of the spectrometer (nm)


# Save figures ?
save_fit_data = True
# 'plt.savefig(save_fig_path + 'name.png', bbox_inches='tight')'

# metadata file to get the wavelengths array :
file_metadata = 'D:/hspc/data/2024/P60/obj_biopsy-1_anterior-portion_source_Laser_405nm_1.2W_A_0.15_f80mm-P2_Walsh_im_16x16_ti_200ms_zoom_x1/obj_biopsy-1_anterior-portion_source_Laser_405nm_1.2W_A_0.15_f80mm-P2_Walsh_im_16x16_ti_200ms_zoom_x1_metadata.json'
metadata, acquisition_params, spectrometer_params, dmd_params = read_metadata(file_metadata)
wavelengths = acquisition_params.wavelengths




 #%% REFERENCE SPECTRA # this part has been moved outside of the loop
 
##########################################################################################
# @todo : remove this part, import interpolated spectra instead
    
    
folder_path_ref = 'C:/Users/chiliaeva/Documents/data_pilot-warehouse/ref/'
 
file_name_ppix620 = 'ref620_3lamda.npy'
file_name_ppix634 = 'ref634_3lamda.npy'
file_name_lambda = 'Lambda.npy'
 
 
ppix620 = np.load(folder_path_ref + file_name_ppix620)
ppix634 = np.load(folder_path_ref + file_name_ppix634)
lambd = np.load(folder_path_ref + file_name_lambda)
 
 
spectr634 = ppix634[0, :] 
spectr634[0] = 0 # otherwise kernel dies
spectr620 = ppix620[0, :]
spectr620[0] = 0
 
 
 # Normalize the reference spectra
 
spectr620_norm = spectr620/np.amax(spectr620)
spectr620 = spectr620_norm
del spectr620_norm
 
spectr634_norm = spectr634/np.amax(spectr634)
spectr634 = spectr634_norm
del spectr634_norm
 
 
crop_start = np.digitize(wavelengths[0], lambd, right=True) # crop the ref spectra, keep the part from wavelengths[0] to wavelengths[-1]
crop_stop = np.digitize(wavelengths[-1], lambd, right=True)


lambd_crop = lambd[crop_start:crop_stop]
spectr620_crop = spectr620[crop_start:crop_stop]
spectr634_crop = spectr634[crop_start:crop_stop]

lambd = lambd_crop
spectr620 = spectr620_crop
spectr634 = spectr634_crop

del lambd_crop
del spectr620_crop
del spectr634_crop

 
 
# Interpolate the reference spectra 
 
func620 = interpolate.make_interp_spline(lambd, spectr620)  # interp1d is legacy
func634 = interpolate.make_interp_spline(lambd, spectr634)

spectr620_interp = func620(wavelengths) # import wavelengths from metadata
spectr634_interp = func634(wavelengths)


spectr620 = spectr620_interp
spectr634 = spectr634_interp

############################################################################################################






# Define fit function

spectr620_shift = spectr620
spectr634_shift = spectr634


def func_fit(x, a1, a2, a3, shift620, shift634, lambd_c, sigma):
    return a1*func620(x-shift620) + a2*func634(x-shift634) + a3*np.exp(-(lambd_c-x)**2/sigma**2)



#%% Select the files


for f in folders : 
    path = os.path.join(root, f)
    print("numero patient : ", f)
    subdirs = os.listdir(path)
    for num_biopsy in list_biops : 
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

                            

        # Read laser hypercube
        cubeobj = np.load(file_cube_laser)
        cubehyper_laser = cubeobj['arr_0']
        
        
        # Read nolight hypercube 
        cubeobj = np.load(file_cube_nolight)
        cubehyper_nolight = cubeobj['arr_0']
        del cubeobj
        
        # Read mask 
        mask = np.load(file_mask)
        
        
        # @todo : define an array of nine of shape (cube[0], cube[1], nb of param of func_fit)
        popt_tab = np.ndarray((cubehyper_laser.shape[0], cubehyper_laser.shape[1], 7), dtype = 'float64')
        popt_tab[:] = np.nan        
        
         
        # Fit for every point of the mask
        t0 = time.time()
        print('start fit for the entire image', time.time()-t0)
    
        for x_i in range(cubehyper_laser.shape[0]):
            for y_i in range(cubehyper_laser.shape[1]):
                
                if mask[x_i, y_i]!=0:
                    
                    # @todo: add option to remove a band 606-615 nm from the wavelength vector
                    spectr_laser = cubehyper_laser[x_i, y_i, :]
                    spectr_nolight = cubehyper_nolight[x_i, y_i, :]
                
        
                    # median filter to smoothen the spectrum
        
                    theo_spectro_reso = (wavelengths[-1]-wavelengths[0])/len(wavelengths)
        
                    kernel_size = int(real_spectro_reso/theo_spectro_reso) + 1 # size of the window for the median filter, must be odd
        
                    if kernel_size %2 == 0 : 
                        kernel_size = kernel_size + 1
            
            
                    sp_laser_smth = sg.medfilt(spectr_laser, kernel_size)
                    sp_nolight_smth = sg.medfilt(spectr_nolight, kernel_size)
                
                
        
                    # Remove the no light spectrum
                    spectrum = sp_laser_smth - sp_nolight_smth
                    np.save(path + '/B' + str(num_biopsy) + '_' +  type_reco + '_spectrum.npy', spectrum)
        
        
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
                
                else: 
                    popt_tab[x_i, y_i, :] = np.nan
                    
        print('end fit for image', time.time()-t0)  
              
  
   
        if save_fit_data == True:
            np.save(path + '/B' + str(num_biopsy) + '_' +  type_reco + '_fit_params.npy', popt_tab) 
    
    
    


















































     





