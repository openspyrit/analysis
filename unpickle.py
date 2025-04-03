# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 15:12:58 2025

@author: chiliaeva
"""

import pickle


bounds_file = 'C:/fitresults_250313_full-spectra_/bounds.pickle'

with open(bounds_file, 'rb') as pickle_file:
    bounds = pickle.load(pickle_file)
    
    
    
    