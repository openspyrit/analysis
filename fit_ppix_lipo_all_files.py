# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 09:26:53 2024

@author: chiliaeva
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 15:29:39 2024

@author: chiliaeva

P63 B2
0) spectre biopsie - no light
1) filtre médian
2) fit de l'autofluo avec exp décroissante aux extrémités du spectreref
3) soustraire le fit aux données
4) normaliser les spectres de réf et faire le fit avec PpIX 620 et 634 et lipofuscine



"""

import numpy as np

from spas.metadata2 import read_metadata

import scipy.signal as sg
import scipy.optimize as op
from scipy import interpolate

import os
import time


#%% Parameters

 
list_biops = np.arange(1,10,1, dtype=int) # biospies from 1 to 9

type_reco_short = 'had_reco'
type_reco = type_reco_short + '.npz'

real_spectro_reso = 2 # real resolution of the spectrometer (nm)


shift = np.arange(-4, 5, dtype=int) # shift the ref spectra +- 0,1325 nm /unité 

# Save figures ?
save_fit_data = True

# 'plt.savefig(save_fig_path + 'name.png', bbox_inches='tight')'


# crop the reference spectra ? 
crop = True # crop the wavelength scale to only keep the 514-750 nm range


#%% Select the files

root = 'D:/hspc/data/2024/'
folders = os.listdir(root)


for f in folders : 
    path = os.path.join(root, f)
    print("numero patient : ", f)
    subdirs = os.listdir(path)
    for num_biopsy in list_biops : 
        print('numero biopsie : ', num_biopsy)
        for s in subdirs :
            if s[11] == str(num_biopsy) :
                subpath = path + '/' + s + '/'
                # print('current subdir : ', subpath)
                if "Laser" in s : 
                    file_cube_laser = subpath + s + '_' + type_reco
                    file_metadata_laser = subpath + s + '_metadata.json'
                elif "No-light" in s :
                    file_cube_nolight = subpath + s + '_' + type_reco
                    file_metadata_nolight = subpath + s + '_metadata.json'
                    
         
        
        #%% Extract data
        
        
        # Read wavelengths 
        metadata, acquisition_params, spectrometer_params, dmd_params = read_metadata(file_metadata_laser)
        wavelengths = acquisition_params.wavelengths
        
        
        # Read hypercube laser
        cubeobj = np.load(file_cube_laser)
        cubehyper_laser = cubeobj['arr_0']
        
        
        # Read hypercube operating nolight
        cubeobj = np.load(file_cube_nolight)
        cubehyper_nolight = cubeobj['arr_0']
        
        del cubeobj
        
    
        
        ##################################################################################################################
        #%% REFERENCE SPECTRA
        
        # C:/Users/chiliaeva/Documents/data_pilot-warehouse/ref/ref620_3lamda.npy
        
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
        
        
         
        ##################### is this part useful ? #################################
        
        if crop == True : 
        
            crop_start = 672  # 514 nm
            crop_stop =  1822 # 750 nm
            
            lambd_crop = lambd[crop_start:crop_stop]
            spectr620_crop = spectr620[crop_start:crop_stop]
            spectr634_crop = spectr634[crop_start:crop_stop]
            
            lambd = lambd_crop
            spectr620 = spectr620_crop
            spectr634 = spectr634_crop
            
            del lambd_crop
            del spectr620_crop
            del spectr634_crop
        
        ###########################################################################
        
        
        
        # Interpolate the reference spectra 
        
        
        func620 = interpolate.make_interp_spline(lambd, spectr620)  # interp1d is legacy
        func634 = interpolate.make_interp_spline(lambd, spectr634)
        
        spectr620_interp = func620(wavelengths) # import wavelengths from metadata
        spectr634_interp = func634(wavelengths)
        
        
        spectr620 = spectr620_interp
        spectr634 = spectr634_interp
        
        
        del spectr620_interp
        del spectr634_interp
        
        # END REF SPECTRA
        #############################################################################################################################
        
        
        spectr620_shift = spectr620
        spectr634_shift = spectr634
        
        def func_fit(x, a1, a2, a3, lambd_c, sigma):
            return a1*spectr620_shift + a2*spectr634_shift + a3*np.exp(-(lambd_c-x)**2/sigma**2)
        
        
        
        
        
        
        #%%
        # Coefficients of the PpIX620 and PpIX634 states
        
        
        coef_P620 = np.ndarray((cubehyper_laser.shape[0], cubehyper_laser.shape[1]), dtype='float64') # coef of PpIX620 fluorescence state
        coef_P634 = np.ndarray((cubehyper_laser.shape[0], cubehyper_laser.shape[1]), dtype='float64') # coef of PpIX634 fluorescence state
        
        
        res_map = np.ndarray((cubehyper_laser.shape[0], cubehyper_laser.shape[1]), dtype='float64') # squared residues map np.sum((spectrum-func_fit(wavelengths,*popt))**2)
        
        
        std620_map = np.ndarray((cubehyper_laser.shape[0], cubehyper_laser.shape[1]), dtype='float64') # standard deviation of coef_P620
        std634_map = np.ndarray((cubehyper_laser.shape[0], cubehyper_laser.shape[1]), dtype='float64')
        
        
        coef_P620[:] = np.nan
        coef_P634[:] = np.nan
        res_map[:] = np.nan
        std620_map[:] = np.nan
        std634_map[:] = np.nan
        
        
        
        ##############################################################################################################################
        ##########################################################  LOOP #############################################################
        ##############################################################################################################################
        #%% LOOP over all the points of the image
        
        t0 = time.time()
        
        print('start fit for the entire image', time.time()-t0)

        for x_i in range(cubehyper_laser.shape[0]):
            print (x_i)
            for y_i in range(cubehyper_laser.shape[1]):
                
            
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
        
        
                # FIT THE SPECTRUM TO REFERENCE SPECTRA
        
            
                M = np.absolute(np.max(spectrum))
                p0 = [M/2, M/2, M/8, 585, 10]# initial guess for the fit
                bounds_inf = [0,0,0,580,5] 
                bounds_sup = [M,M,M,610,100] 
                
                
                popt, pcov = op.curve_fit(func_fit, wavelengths, spectrum, p0, bounds=(bounds_inf, bounds_sup))
                
                sum_res_sq = np.ndarray((shift.size, shift.size), dtype='float64') 
        
                
                # optimize the position of reference spectra
                for p in shift:
                    for q in shift:
                        spectr620_shift = np.roll(spectr620, p)
                        spectr634_shift = np.roll(spectr634, q)
                        try :
                            popt, pcov = op.curve_fit(func_fit, wavelengths, spectrum, p0, bounds=(bounds_inf, bounds_sup))
                            res = spectrum - func_fit(wavelengths, *popt)
                            sum_res_sq[p,q] = np.sum(res**2)
                        except RuntimeError:
                            pass
                        
                       
                
                
                shift620, shift634 = np.unravel_index(sum_res_sq.argmin(), sum_res_sq.shape) # optimized positions of both ref spectra
        
               
                spectr620_shift = np.roll(spectr620, shift620)        
                spectr634_shift = np.roll(spectr634, shift634)    
                popt, pcov = op.curve_fit(func_fit, wavelengths, spectrum, p0, bounds=(bounds_inf, bounds_sup))        
                
                coef_P620[x_i, y_i] = popt[0] 
                coef_P634[x_i, y_i] = popt[1] 
                std620_map[x_i, y_i] = np.sqrt((pcov[0,0])) 
                std634_map[x_i, y_i] = np.sqrt((pcov[1,1])) 
        
                # Generate y values based on the fitted parameters
                spectrum_fit = func_fit(wavelengths, *popt)
                res = spectrum - spectrum_fit
                res_map[x_i, y_i] = np.sum(res**2)
                
                    
        print('end fit for image', time.time()-t0)  
              
            
        # Save files
        if save_fit_data == True :        
            np.save(path + '/B' + str(num_biopsy) + '_' +  type_reco_short + '_coef_P620.npy', coef_P620)        
            np.save(path + '/B' + str(num_biopsy) + '_' +  type_reco_short + '_coef_P634.npy', coef_P634)    
        
            np.save(path + '/B' + str(num_biopsy) + '_' +  type_reco_short + '_std620_map.npy', std620_map)
            np.save(path + '/B' + str(num_biopsy) + '_' +  type_reco_short + '_std634_map.npy', std634_map)
          
            np.save(path + '/B' + str(num_biopsy) + '_' +  type_reco_short + '_res_map.npy', res_map)      

            
     





