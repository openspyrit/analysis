# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 14:46:32 2025

@author: chiliaeva

This program loads :
- the lists of biopsies, 
- the ratios computed with hadamard reconstruction,
- the ratios computed with NN reconstruction
from the .xls file
"""


import os
import numpy as np
import pandas as pd



#%% Import CSV

root = 'C:/csv_files/'

file_ratios_had = root + 'ratios_had_header.csv'
file_ratios_nn = root + 'ratios_nn_header.csv'
file_list = root + 'list_biopsies_header.csv'


#%%

df_ratios_had = pd.read_csv(file_ratios_had)
df_ratios_nn = pd.read_csv(file_ratios_nn)
df_list_biopsies = pd.read_csv(file_list)


#%%

ratios_had = df_ratios_had['RatioHAD'].tolist()
ratios_nn = df_ratios_nn['RatioNN'].tolist()
list_biopsies = df_list_biopsies['RefBiopsy'].tolist()


#%%

save_fig = True
root_savefig = 'C:/ratio_results/'

import matplotlib as mpl 
import matplotlib.pyplot as plt


fontsize = 16
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['axes.labelsize'] = fontsize

# mpl.rcParams['figure.figsize'] = 12, 6  # figure size in inches

mpl.rcParams['figure.figsize'] = 20, 10  # figure size in inches



fig, ax = plt.subplots(1, 1)
ax.tick_params(axis='x', rotation=55, labelsize=10)
ax.plot(list_biopsies, ratios_had, 'bx', markersize=10, label='had reco') # plot mean ratios for each biopsy (hadamard reconstruction)
ax.plot(list_biopsies, ratios_nn, 'rx', markersize=10, label='nn reco') # plot mean ratios for each biopsy (NN reconstruction)
ax.vlines(list_biopsies, ratios_had, ratios_nn, linewidth=2, color='black')
plt.legend()
plt.grid()
if save_fig : 
    plt.savefig(root_savefig + 'mean_ratio_nn_had_compare.png', bbox_inches = 'tight')
















