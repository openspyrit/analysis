# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 14:23:22 2024

@author: chiliaeva

Use a thresholding method do determine the binary mask for each measurement.
1 = biopsy in this pixel
0 = no biopsy here
Save the mask

IMPORTANT : use the White Light measurements
"""

import os 

import numpy as np
import matplotlib.pyplot as plt 
import cv2 as cv

from spas.metadata2 import read_metadata

 
list_biops = np.arange(1,10,1, dtype=int) # biospies from 1 to 9


root_0 = 'D:/hspc/data/2024/' # @todo : temporary, remove
root = 'D:/hspc/data/2024a/'


# find t_i in metadata 
# t_i = 
threshold_ = 4e5 # nb counts/pixel on background for t_i = 1s, for a 16x16 image


type_reco = 'had_reco'
type_reco_npz = type_reco + '.npz'


folders = os.listdir(root)

file_metadata = root_0 + 'P60/obj_biopsy-1_anterior-portion_source_white_LED_f80mm-P2_Walsh_im_16x16_ti_10ms_zoom_x1/obj_biopsy-1_anterior-portion_source_white_LED_f80mm-P2_Walsh_im_16x16_ti_10ms_zoom_x1_metadata.json'
         

# Read wavelengths 
metadata, acquisition_params, spectrometer_params, dmd_params = read_metadata(file_metadata)
t_i = spectrometer_params.integration_time_ms
 


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
                if "white" in s : 
                    file_cube_white = subpath + s + '_' + type_reco_npz
         
         
                 
        
        # Read hypercube laser
        cubeobj = np.load(file_cube_white)
        cubehyper = cubeobj['arr_0']
        
        threshold = threshold_ *t_i*1e-3/(np.shape(cubehyper)[0]*np.shape(cubehyper)[1]/16**2)  # absolute threshold 
        
        
        greyscale_img = np.sum(cubehyper, axis=2)
        
        mask = cv.threshold(greyscale_img, threshold, 1, cv.THRESH_BINARY) # thresholding function

        np.save(subpath +  type_reco + '_mask.npy', mask[1])





