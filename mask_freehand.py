# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 09:58:56 2025

@author: chiliaeva

Freehand mask selecting program 
when thresholding does not work
"""


import os
import numpy as np
import cv2 

from spas.metadata import read_metadata

# Use the "laser" hypercube
# Sum the wavelengths from 600 to 650 nm


#%%

# Select the wavelength interval in which
wvlgth_start = 600
wvlgth_stop = 650


# type_reco = 'had_reco'

file_cube_laser = 'C:/d/P64/obj_biopsy-1-lateral-portion_source_Laser_405nm_1.2W_A_0.14_f80mm-P2_Walsh_im_64x64_ti_15ms_zoom_x1/obj_biopsy-1-lateral-portion_source_Laser_405nm_1.2W_A_0.14_f80mm-P2_Walsh_im_64x64_ti_15ms_zoom_x1_nn_reco.npz'
subpath = 'C:/d/P63/obj_biopsy-9-intern-limit_source_white_LED_f80mm-P1_Walsh_im_64x64_ti_15ms_zoom_x1/'


file_metadata = 'C:/d/P63/obj_biopsy-4-contrast_source_Laser_405nm_1.2W_A_0.14+white_LED_f80mm-P2_Walsh_im_64x64_ti_20ms_zoom_x1/obj_biopsy-4-contrast_source_Laser_405nm_1.2W_A_0.14+white_LED_f80mm-P2_Walsh_im_64x64_ti_20ms_zoom_x1_metadata.json'

# type_reco = file_cube_laser[-12:-4]
type_reco = 'nn_reco'


metadata, acquisition_params, spectrometer_params, dmd_params = read_metadata(file_metadata)
wavelengths = acquisition_params.wavelengths

# Read hypercube laser
cubeobj = np.load(file_cube_laser)
cubehyper = cubeobj['arr_0']


index_start = np.digitize(wvlgth_start, wavelengths, right=True)
index_stop = np.digitize(wvlgth_stop, wavelengths, right=True)


image = np.sum(cubehyper[:,:,index_start:index_stop], axis=2)



#%% CV2 functions
# Global variables

drawing = False  # True if the mouse is pressed
points = []  # Stores the points of the ROI

def draw_freehand(event, x, y, flags, param):
    global drawing, points

    if event == cv2.EVENT_LBUTTONDOWN:  # Start drawing
        drawing = True
        points = [(x, y)]  # Initialize with first point

    elif event == cv2.EVENT_MOUSEMOVE and drawing:  # Draw continuously
        points.append((x, y))
        cv2.line(image, points[-2], points[-1], (0, 255, 0), 2)  # Draw line # color in BGR format
        cv2.imshow("Image", image)

    elif event == cv2.EVENT_LBUTTONUP:  # Finish drawing
        drawing = False
        points.append(points[0])  # Close the loop
        cv2.line(image, points[-2], points[-1], (0, 255, 0), 2)
        cv2.imshow("Image", image)



def extract_mask(image, points):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Create a mask
    pts = np.array(points, np.int32)  # Convert points to numpy array
    cv2.fillPoly(mask, [pts], 255)  # Fill the polygon area
    return mask



    
def extract_roi(image, points):
    mask = extract_mask(image, points)
    roi = cv2.bitwise_and(image, image, mask=mask)  # Apply mask
    return roi



#%%
cv2.imshow("Image", image) # cv2.imshow(window name, image)
cv2.setMouseCallback("Image", draw_freehand)

cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()

# Extract and display ROI
if len(points) > 2:
    mask = extract_mask(image, points)
    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
#%%

# Save mask 
if os.path.isfile(subpath + '/' + type_reco + '_mask.npy') == False :
    np.save(subpath + '/' +  type_reco + '_mask.npy', mask)
    print("mask saved")
else : 
    print("mask already exists")
    



























