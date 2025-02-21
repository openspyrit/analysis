# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 11:43:12 2025

@author: varvara

"""

import os


#%% List of subdirs in a dir, containing the string "had_reco" :


def find_subdirs_with_string(root_dir, search_string):
    subdirectories = []
    
    for dirpath, dirnames, _ in os.walk(root_dir):
       for dirname in dirnames:
           if search_string in dirname:
               subdirectories.append(os.path.join(dirpath, dirname))
   
    return subdirectories
    
    
# Example usage
root_directory = '/path/to/your/directory'  # Replace with your root directory path
search_string = 'had_reco'
result = find_subdirs_with_string(root_directory, search_string)

# Print the result
for subdir in result:
    print(subdir)    
    
    


def find_npy_files(subdirectories, search_string):
    ''' Arguments : 
    subdirectories : list of strings, full paths
    search_string : string to search in file names
    '''
    
    npy_files = []
    
    for subdir in subdirectories : 
        
        for filename in os.listdir(subdir):
            
            if filename.endswith('.npy') and search_string in filename :
                
                npy_files.append(os.path.join(subdir, filename))
                
    return npy_files
                
                
              
          
def get_next_n_chars(file_name, char, n):
    # Find the index of the given character in the file name
    char_index = file_name.find(char)
    
    # If the character is not found, return an empty string
    if char_index == -1:
        return ''
    
    # Return the next N characters after the given character
    return file_name[char_index : char_index + 1 + n]











    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    