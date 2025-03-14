# Analysis of the database

This module is intented for the interpretation of acquisition data from the pilot-warehouse medical database.

This module analyses the hyperspectral cubes obtained with reconstruction algorithms.

In this particular case, the spectra analysed are Protoporphyrin IX (PpIX) fluorescence spectra. The fluorescence of PpIX has two main states : one centered around 620 nm, the other around 634 nm. 
The program fits the spectra to reference spectra to determine which of the states is predominant.



## The Database
The database includes data acquired at the hospital from ex-vivo samples for 11 patients. For each patient, several biopsies were taken. Several measurements were performed for each biopsy using the single-pixel setup_v1.3.1.
The samples were illuminated fisrt with a laser (385nm or 405 nm), then with a white LED lamp. A backgroung measurement was also taken without any illumination. 
The raw spectra are stored in the files ending with "_spectraldata.npz"

The hyperspectral cubes were reconstructed using two methods : the Hadamard reconstruction, stored in the files : had_reco.npz 
and with neural network methods : nn_reco.npz.


## Prerequisites
Insall spyrit and spas on your computer following the procedure : LINK




## Contents 
The analysis module consists of 3 programs : 

* preprocessing.py 
This program has two main functions : 
    - apply a thresholding mask to the greyscale image to separate the sample from the background
    - preprocessing of the spectra : remove the operating room background and perform a smoothing of the spectrum
The spectra and the masks are saved to be used in the main program.

* main_fit.py


