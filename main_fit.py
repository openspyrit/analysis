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
from preprocessing import func620, func634, func_fit


import os
import time


#%% Parameters


reject_band = [606, 616] # exclude spectral band reject_band from fit (nm). To reject no band, set reject_band[0]=reject_band[1]


save_fit_data = True
# 'plt.savefig(save_fig_path + 'name.png', bbox_inches='tight')'
save_results_root = 'D:/hspc/fitresults/'

root = 'D:/2024b/' # @todo : temporary, change
folders = os.listdir(root)
print(folders)


type_reco = 'had_reco'
type_reco_npz = type_reco + '.npz'

real_spectro_reso = 2 # real resolution of the spectrometer (nm)



# metadata file to get the wavelengths array :
file_metadata = 'D:/hspc/data/2024/P60/obj_biopsy-1_anterior-portion_source_Laser_405nm_1.2W_A_0.15_f80mm-P2_Walsh_im_16x16_ti_200ms_zoom_x1/obj_biopsy-1_anterior-portion_source_Laser_405nm_1.2W_A_0.15_f80mm-P2_Walsh_im_16x16_ti_200ms_zoom_x1_metadata.json'
metadata, acquisition_params, spectrometer_params, dmd_params = read_metadata(file_metadata)
wavelengths = acquisition_params.wavelengths


band_stop_mask = (wavelengths <= reject_band[0])|(wavelengths >= reject_band[1])
wavelengths = wavelengths[band_stop_mask]
if save_fit_data == True : 
    np.save(save_results_root + 'wavelengths_mask_' + str(reject_band[0]) + '-' + str(reject_band[1]) + '.npy', wavelengths) 


folder_path_ref = 'C:/Users/chiliaeva/Documents/data_pilot-warehouse/ref/'


 #%% import ref spectra


# load interpolated and normalized ref spectra : 
spectr620 = np.load(folder_path_ref + '_spectr620_interp.npy')
spectr634 = np.load(folder_path_ref + '_spectr634_interp.npy')

spectr620 = spectr620[band_stop_mask]
spectr634 = spectr634[band_stop_mask]



#%% Select the files

print("just before folders loop")
for f in folders : 
    print("enter folders loop")
    path = os.path.join(root, f)
    print("numero patient : ", f)
    os.mkdir(save_results_root + f)
    print("patient folder created")
    subdirs = os.listdir(path)
    print("subdirs", subdirs)
    
    cpt = 1
    print("initial cpt : ", 1)
    for s in subdirs : 
        print("enter subdirs loop")
        print("current subdir : ", s)
        if s[11] == str(cpt) :
            print("enter if loop with cpt = :", cpt)
            subpath = path + '/' + s + '/'
            if "Laser" in s :
                file_cube_laser = subpath + s + '_' + type_reco_npz
                print("file_cube_laser = ", file_cube_laser)
            elif "No-light" in s :
                file_cube_nolight = subpath + s + '_' + type_reco_npz
                print("file_cube_nolight = ", file_cube_nolight)
            elif "white" in s : 
                file_mask = subpath + type_reco + '_mask.npy'
                print("file_mask = ", file_mask)
        
    

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
                
                
                # @todo : define an array of nine of shape (cube[0], cube[1], nb of param of func_fit)
                popt_tab = np.ndarray((cubehyper_laser.shape[0], cubehyper_laser.shape[1], 7), dtype = 'float64')
                popt_tab[:] = np.nan        
                
                 
                # Fit for every point of the mask
                t0 = time.time()
                print('start fit for the entire image', time.time()-t0)
                
                spectrum_tab = np.ndarray((cubehyper_laser.shape[0], cubehyper_laser.shape[1], np.size(wavelengths)), dtype='float64')
            
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
                        
                        else: 
                            popt_tab[x_i, y_i, :] = np.nan
                            
                        
                print('end fit for image', time.time()-t0)  
                np.save(save_results_root + f + '/B' + str(cpt) + '_' +  type_reco + '_spectrum_tab.npy', spectrum_tab)     
      
       
                if save_fit_data == True:
                    np.save(save_results_root + f + '/B' + str(cpt) + '_' +  type_reco + '_fit_params.npy', popt_tab) 
        
        
                cpt += 1
                print("cpt incremented to : ", cpt)
    
    
    
    














































     





