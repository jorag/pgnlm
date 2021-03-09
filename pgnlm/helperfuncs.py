#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jorag
"""

import numpy as np


def length(x):
    """Returns length of input.
    
    Mimics MATLAB's length function.
    """
    if isinstance(x, (int, float, complex, np.int64)):
        return 1
    elif isinstance(x, np.ndarray):
        return max(x.shape)
    try:
        return len(x)
    except TypeError as e:
        print('In length, add handling for type ' + str(type(x)))
        raise e
        
   
def rand_coord(x_range, y_range, n_coords, unique_only=True):
    """Create a list of random coordinates from specified x and y range.
    
    Can be used for sampling random pixels from images.
    By default the draw is without replacement, but that can be changed by 
    setting unique_only = False
    """
    # Create inital list of coordinates
    x = np.random.randint(x_range[0], high=x_range[1], size=n_coords)
    y = np.random.randint(y_range[0], high=y_range[1], size=n_coords)
    
    # Initialize output
    coord_list = []
    if unique_only:
        # Combine and check
        for i_coord in range(0, length(x)):
            coord_candidate = (x[i_coord], y[i_coord])
            # Regenerate in case coordinate has been generated before
            while coord_candidate in coord_list:
                coord_candidate=(np.random.randint(x_range[0], high=x_range[1]),  
                                 np.random.randint(y_range[0], high=y_range[1]))
            # Add unique coordinate to list
            coord_list.append(coord_candidate)
    else:
        # Combine coordinates
        for i_coord in range(0, length(x)):
            coord_list.append((x[i_coord], y[i_coord]))
    
    return coord_list


def norm01(input_array, norm_type='global', min_cap=None, max_cap=None, min_cap_value=np.NaN, max_cap_value=np.NaN):
    """Normalise data.
    
    Parameters:
    norm_type:
        'none' - return input array 
        'channel' - min and max of each channel is 0 and 1
        'global' - min of array is 0, max is 1
    min_cap: Truncate values below this value to min_cap_value before normalising
    max_cap: Truncate values above this value to max_cap_value before normalising
    """
    
    # Ensure that original input is not modified
    output_array = np.array(input_array, copy=True)
    
    # Replace values outside envolope/cap with NaNs (or specifie value)
    if min_cap is not None:
        output_array[output_array   < min_cap] = min_cap_value
                   
    if max_cap is not None:
        output_array[output_array  > max_cap] = max_cap_value
    
    # Normalise data for selected normalisation option
    if norm_type.lower() in ['global', 'all', 'set']:
        # Normalise to 0-1 (globally)
        output_array = input_array - np.nanmin(input_array)
        output_array = output_array/np.nanmax(output_array)
    elif norm_type.lower() in ['band', 'channel']:
        # Normalise to 0-1 for each channel (assumed to be last index)
        # Get shape of input
        input_shape = input_array.shape
        output_array = np.zeros(input_shape)
        # Normalise each channel
        for i_channel in range(0, input_shape[2]):
            output_array[:,:,i_channel] = input_array[:,:,i_channel] - np.nanmin(input_array[:,:,i_channel])
            output_array[:,:,i_channel] = output_array[:,:,i_channel]/np.nanmax(output_array[:,:,i_channel])
        
    return output_array


def dB(x, ref=1, input_type='power'):
    """Return result, x, in decibels (dB) relative to reference, ref."""    
    if input_type.lower() in ['power', 'pwr']:
        a = 10
    elif input_type.lower() in ['amplitude', 'amp']:
        a = 20
    return a*(np.log10(x) - np.log10(ref))


def iq2complex(x, reciprocity=False):
    """Merge I and Q bands to complex valued array.
    
    Create an array with complex values from separate, real-vauled Inphase and 
    Quadrature components.
    """
    shape_in = x.shape
    # Number of bands determines from of expression
    if reciprocity and shape_in[2] == 8:
        # Initialise output
        array_out = np.zeros((shape_in[0], shape_in[1], 3), dtype=complex)
        # Input is real arrays: i_HH, q_HH, i_HV, q_HV, i_VH, q_VH, i_VV, q_VV
        array_out[:,:,0] = x[:,:,0] + 1j * x[:,:,1] # HH
        array_out[:,:,1] = (x[:,:,2] + 1j*x[:,:,3] + x[:,:,4] + 1j*x[:,:,5])/2 # HV (=VH)
        array_out[:,:,2] = x[:,:,6] + 1j * x[:,:,7] # VV
    elif not reciprocity and shape_in[2] == 8:
        # Initialise output
        array_out = np.zeros((shape_in[0], shape_in[1], 4), dtype=complex)
        # Input is real arrays: i_HH, q_HH, i_HV, q_HV, i_VH, q_VH, i_VV, q_VV
        array_out[:,:,0] = x[:,:,0] + 1j * x[:,:,1] # HH
        array_out[:,:,1] = x[:,:,2] + 1j * x[:,:,3] # HV
        array_out[:,:,2] = x[:,:,4] + 1j * x[:,:,5] # VH
        array_out[:,:,3] = x[:,:,6] + 1j * x[:,:,7] # VV
    elif shape_in[2] == 6:
        # Initialise output
        array_out = np.zeros((shape_in[0], shape_in[1], 3), dtype=complex)
        # Input is real arrays: i_HH, q_HH, i_HV, q_HV, i_VV, q_VV (reciprocity assumed) 
        array_out[:,:,0] = x[:,:,0] + 1j * x[:,:,1] # HH
        array_out[:,:,1] = x[:,:,2] + 1j * x[:,:,3] # HV (=VH)
        array_out[:,:,2] = x[:,:,4] + 1j * x[:,:,5] # VH
               
    return array_out 


def complex2real(c):
    """Divide complex array elements into Re and Im.

    Return real valued array.
    """
    shape_in = c.shape
    # Number of bands determines form of expression
    array_out = np.zeros((shape_in[0], shape_in[1], 2*shape_in[2]), dtype=np.float64)

    # Get real values of bands
    for i_element in range(shape_in[2]):
        array_out[:,:,2*i_element] = np.real(c[:,:,i_element])
        array_out[:,:,2*i_element+1] = np.imag(c[:,:,i_element])
               
    return array_out
