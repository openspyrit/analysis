# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 11:32:59 2025

@author: chiliaeva

params_tabs[x,y,Nb]
Nb = 0 : coef 620 (a1)
Nb = 1 : coef 634 (a2)
Nb = 2 : coef lipo (a3)
Nb = 3 : shift 620
Nb = 4 : shift 634 
Nb = 5 : lambda_c
Nb = 6 : sigma 
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import matplotlib.colors as colors

from matplotlib.colors import TwoSlopeNorm



#%%

mksize = 4

savefig_maps = False
show_spectrum_pos = False

savefig_ratio = False
savenpy_ratio = True

# savefig_spectrum = False

type_reco = 'nn_reco'


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



num_patient = 'P60_'
num_biopsy = 'B4'



# Position of the spectrum to plot :
x = 11
y = 21
if type_reco == 'nn_reco' :
    x = 4*x
    y = 4*y



file_params = root_saveresults + num_patient +  type_reco + '/' + num_biopsy + '_' + type_reco + '_fit_params.npy'
params_tab = np.load(file_params)




min_ppix = np.amin([np.nanmin(params_tab[:,:,0]), np.nanmin(params_tab[:,:,1])]) # minimum for Protoporphyrin IX colormap
max_ppix = np.amax([np.nanmax(params_tab[:,:,0]), np.nanmax(params_tab[:,:,1])])



if type_reco == 'nn_reco':
    params_tab = cv.flip(params_tab, 0)
   
    

#%%
    
def func_plot_map_nb(params_tab, nb, show_spectrum_pos, x, y, c_lim):
    # plot abundance map number "nb"
    plt.figure("map nb" + str(nb))
    plt.clf()
    plt.title("map nb" + str(nb))
    plt.imshow(params_tab[:,:,nb])
    if c_lim == True :
        plt.clim(min_ppix, max_ppix)
    plt.colorbar()
    plt.grid()
    if show_spectrum_pos == True :
        plt.plot(y,x, "or", markersize = mksize)



#%%

c_lim = True  # colorbar between min(620, 634) and max(620, 634)
nb = 1 # number of the map 

func_plot_map_nb(params_tab, nb, show_spectrum_pos, 21, 20, c_lim)




#%% Save abundance map 

plt.gcf()
plt.savefig(root_savefig + num_patient + num_biopsy + '_' + type_reco + '_map_nb' + str(nb) + '.png', bbox_inches='tight')



#%%
'''
# Plot and save all the abundance maps 
for i in range(np.shape(params_tab)[2]) :
    plt.figure("map nb" + str(i))
    plt.clf()
    plt.title("map nb" + str(i))
    plt.imshow(params_tab[:,:,i])
    plt.clim(min_ppix, max_ppix)
    plt.colorbar()
    if show_spectrum_pos == True :
        plt.plot(y,x, "or", markersize = mksize)
    if savefig_maps == True :
        plt.savefig(root_savefig + num_patient + num_biopsy + '_' + type_reco + '_map_nb' + str(i) + '.png', bbox_inches='tight')
        
'''        
        
        

#%% Save maps for the entire 'fitresults' folder

root_maps = root_savefig + 'maps/'
if os.path.exists(root_maps) == False :
    os.mkdir(root_maps)


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
                subpath = os.path.join(path, file)
                coef_P620 = np.load(subpath)[:,:,0]
                coef_P634 = np.load(subpath)[:,:,1]
                coef_lipo = np.load(subpath)[:,:,2]
                shift620 = np.load(subpath)[:,:,3]
                shift634 = np.load(subpath)[:,:,4]
                lambd_lipo = np.load(subpath)[:,:,5]
                sigma_lipo = np.load(subpath)[:,:,6]
                
                min_ppix = np.amin([np.nanmin(coef_P620), np.nanmin(coef_P634)]) # minimum for Protoporphyrin IX colormap
                max_ppix = np.amax([np.nanmax(coef_P620), np.nanmax(coef_P634)])
        
                plt.figure()
                plt.imshow(coef_P620)
                plt.grid()
                plt.clim(min_ppix, max_ppix)
                plt.colorbar()
                plt.savefig(root_maps + num_patient + num_biops + '_' + type_reco + '_coef_P620_map.png', bbox_inches='tight')
                plt.close()
        
        
                plt.figure()
                plt.imshow(coef_P634)
                plt.grid()
                plt.clim(min_ppix, max_ppix)
                plt.colorbar()
                plt.savefig(root_maps + num_patient + num_biops + '_' + type_reco + '_coef_P634_map.png', bbox_inches='tight')
                plt.close()
    
    
                plt.figure()
                plt.imshow(coef_lipo)
                plt.grid()
                plt.colorbar()
                plt.savefig(root_maps + num_patient + num_biops + '_' + type_reco + '_coef_lipo_map.png', bbox_inches='tight')
                plt.close()
    
    
                plt.figure()
                plt.imshow(shift620)
                plt.grid()
                plt.colorbar()
                plt.savefig(root_maps + num_patient + num_biops + '_' + type_reco + '_shift620_map.png', bbox_inches='tight')
                plt.close()
    
                plt.figure()
                plt.imshow(shift634)
                plt.grid()
                plt.colorbar()
                plt.savefig(root_maps + num_patient + num_biops + '_' + type_reco + '_shift634_map.png', bbox_inches='tight')
                plt.close()
    
        
                plt.figure()
                plt.imshow(lambd_lipo)
                plt.grid()
                plt.colorbar()
                plt.savefig(root_maps + num_patient + num_biops + '_' + type_reco + '_lambd_lipo_map.png', bbox_inches='tight')
                plt.close()
    
    
                plt.figure()
                plt.imshow(sigma_lipo)
                plt.grid()
                plt.colorbar()
                plt.savefig(root_maps + num_patient + num_biops + '_' + type_reco + '_sigma_lipo_map.png', bbox_inches='tight')
                plt.close()



                


#%% Log transform the data, then use TwoSlopeNorm

vmax = 0.5
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
cmap = plt.get_cmap('viridis')

# lim = 1e-4 # values below this limit take this value


#########################################################################################

root_ratios = root_savefig + 'maps/ratios_log_scale_vmax=' + str(vmax) + '_green_background/'
if os.path.exists(root_ratios) == False :
    os.mkdir(root_ratios)


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
                
                
                

































