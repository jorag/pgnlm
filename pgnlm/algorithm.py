#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jorag
"""

import numpy as np
from tqdm import tqdm
from helperfuncs import length, rand_coord
import matplotlib.pyplot as plt


def run_pgnlm(polsar, optical, params):
    """Pad guide and PolSAR image with correct size and run PGNLM."""
    # Read parameters from object
    n_patches = params['n_patches']
    n_big = params['n_big'] 
    n_small = params['n_small'] 
    
    # Total number of patches (pixels) in search window
    n_patch_tot = (2*n_big+1)**2
    # Number of patches to use
    if n_patches > n_patch_tot:
        raise ValueError('Error, too many patches, '+ str(n_patches)+', specified! Max is '+str(n_patches_total))
    elif params['debug']:
        print('Patches in search window: ', n_patches) 
    
    # Precompute some constants
    n_pix_w = (2 * n_small + 1) ** 2 # Number of pixels in "small" window
    Nw = n_pix_w

    # Get shape of input
    input_shape = polsar.shape
    n_rows, n_cols = polsar.shape[0], polsar.shape[1]
    if length(input_shape) == 2:
        n_channels = 1
        polsar = np.expand_dims(polsar, axis=2)
    elif length(input_shape) == 3:
        n_channels = input_shape[2]

    # Create Tuple of the coordinate difference for the big patch 
    D = range(-n_big, n_big + 1)
    big_diff = [(r, c) for r in D for c in D] 

    # Precompute coordinate difference for the small patch
    small_rows, small_cols = np.indices((2 * n_small + 1, 2 * n_small + 1)) - n_small
    
    # Create padding tuple ((row left, row right), (col left, ...))
    padding = int(n_big + n_small) # also denotes extent for initial dissim building
    pad_tuple = ( (padding,padding), (padding,padding), (0,0) )
    # Pad image
    sar_pad = np.pad(polsar, pad_tuple, mode='wrap')
    # Pad guide
    opt_pad = np.pad(optical, pad_tuple, mode='wrap')
    
    
    # Find PolSAR threshold and guide T_opt 
    print('Computing thresholds...')
    sar_thresh, optical_thresh = _calcthrehs_patch(sar_pad, opt_pad, 
                big_diff, small_rows, small_cols, Nw, n_patch_tot, params)
    params['polsar_thresh'] = sar_thresh
    params['optical_thresh'] = optical_thresh
    
    # Initialise output image
    # The number of channels = the square of the number of scattering vector elements 
    c_mat = np.zeros((n_rows, n_cols, n_channels**2), dtype=polsar.dtype)

    # Initialise 2D array with weights
    weights_im = np.zeros((n_rows, n_cols))

    # Single or multicore processing
    n_cpu_avail = 1
    if params['cpu_cores'] > 3:
        print('Attempting to utilize 4 CPU cores...')
        
        import multiprocessing as mp
        n_cpu_avail = mp.cpu_count()
        print("Number of processors available: ", n_cpu_avail)
    
    if params['cpu_cores'] > 3 and n_cpu_avail > 3 :  
        print('Using 4 CPU cores.\nEstimating covariance matrices...') 
        # Init multiprocessing.Pool()
        pool = mp.Pool(min(4, mp.cpu_count()))
        
        # Find "stich" index, where images are merged together for the final output
        s_row = int(np.floor(n_rows/2)) # Row index to split
        s_col = int(np.floor(n_cols/2)) # Column index to split
        
        # Call multiple processes (limited by number of CPU cores) for each subpart
        # Extract extra pixels to correspond to padding
        res0 = pool.apply_async(
             _pgnlm, args=(sar_pad[0:s_row+padding*2, 0:s_col+padding*2, :], 
             opt_pad[0:s_row+padding*2, 0:s_col+padding*2, :], big_diff, 
             small_rows, small_cols, Nw, n_patch_tot, params))
        
        res1 = pool.apply_async(
             _pgnlm, args=(sar_pad[s_row:, 0:s_col+padding*2, :], 
             opt_pad[s_row:, 0:s_col+padding*2, :], big_diff, 
             small_rows, small_cols, Nw, n_patch_tot, params))
        
        res2 = pool.apply_async(
             _pgnlm, args=(sar_pad[0:s_row+padding*2, s_col:, :], 
             opt_pad[0:s_row+padding*2, s_col:, :], big_diff, 
             small_rows, small_cols, Nw, n_patch_tot, params))
        
        res3 = pool.apply_async(
             _pgnlm, args=(sar_pad[s_row:, s_col:, :], 
             opt_pad[s_row:, s_col:, :], big_diff, 
             small_rows, small_cols, Nw, n_patch_tot, params))
        
        # Close Pool and let all the processes complete    
        pool.close()
        pool.join()  # postpones the execution of next line of code until all processes in the queue are done.
        
        c_mat[0:s_row, 0:s_col, :], weights_im[0:s_row, 0:s_col] = res0.get(timeout=10)[0], res0.get(timeout=10)[1]
        c_mat[s_row:, 0:s_col, :], weights_im[s_row:, 0:s_col] = res1.get(timeout=10)[0], res1.get(timeout=10)[1]
        c_mat[0:s_row, s_col:, :], weights_im[0:s_row, s_col:] = res2.get(timeout=10)[0], res2.get(timeout=10)[1]
        c_mat[s_row:, s_col:, :], weights_im[s_row:, s_col:] = res3.get(timeout=10)[0], res3.get(timeout=10)[1]
                
        return c_mat, weights_im

    else:
        # Single CPU processing
        print('Using 1 CPU core.\nEstimating covariance matrices...')
        c_mat, weights_im = _pgnlm(
            sar_pad, opt_pad, big_diff, small_rows, small_cols, Nw, n_patch_tot, params)
        return c_mat, weights_im


def _d_polsar(centre_patch, cand_patch, Nw):
    """Calculate PolSAR dissimilarity""" 
    diff = cand_patch - centre_patch
    # Use Einstein notation, numerator stored temp in d 
    d = np.einsum('ijk,ijk->ij', diff.conj(), diff)
    # Find denominator
    denom = np.einsum('ijk,ijk->ij', centre_patch.conj(), centre_patch)
    denom += np.einsum('ijk,ijk->ij', cand_patch.conj(), cand_patch)
    d /= (0.5*denom)
    # Apply constants and ensure weight is real     
    d = np.sum(d.real)/Nw     
    # Return dissimilarity
    return d

def _d_optical(centre_patch, cand_patch, Nw):  
    """Calculate optical dissimilarity"""
    d = np.sqrt((cand_patch - centre_patch) ** 2) 
    # Sum and apply constants 
    d = np.sum(d)/Nw        
    # Return dissimilarity
    return d


def _pgnlm(polsar, optical, big_diff, small_rows, small_cols, Nw, n_patch_tot, params):
    """Run PGNLM to estimate covariance matrices."""
    
    # Read parameters from object
    n_patches = params['n_patches']
    n_big = params['n_big']
    n_small = params['n_small'] # Also index of the centre pixel
    gamma = params['gamma']
    lamb = params['lamb']
    polsar_thresh = params['polsar_thresh']
    optical_thresh = params['optical_thresh']
    # Set dissimilarity functions
    d_polsar = _d_polsar
    d_optical = _d_optical

    # Get shape of input, padding done in calling function
    input_shape = polsar.shape
    n_rows, n_cols = polsar.shape[0], polsar.shape[1]
    if length(input_shape) == 2:
        n_channels = 1
        polsar = np.expand_dims(polsar, axis=2)
    elif length(input_shape) == 3:
        n_channels = input_shape[2]

    # Initialise array of patches
    patch_candidates = np.zeros((2*n_small+1, 2*n_small + 1, n_channels, n_patch_tot), dtype=polsar.dtype) 
    # Initialise array of "normalised" dissimilarities
    d_pol_norm = np.zeros(n_patch_tot)
    d_opt_norm = np.zeros(n_patch_tot)

    # Create padding tuple ((row left, row right), (col left, ...))
    # Actual padding done in calling function
    padding = n_big + n_small # also denotes extent for initial dissim building
    
    # Initialise output image, n channels = square of the n of scattering vector elements 
    c_mat = np.zeros((n_rows, n_cols, n_channels**2), dtype=polsar.dtype)

    # Initialise 2D array with weights
    weights_im = np.zeros((n_rows, n_cols))
    
    # Main loop
    for r in tqdm(range(padding, n_rows-padding)): # loop over rows
        for c in range(padding, n_cols-padding): # loop over cols
            # Extract (small) patch centred on pixel (x_r, x_c)
            centre_patch_sar = polsar[small_rows + r, small_cols + c]
            centre_patch_opt = optical[small_rows + r, small_cols + c]

            for i_weights, d in enumerate(big_diff):
                # Get current candidate patch in PolSAR and optical guide
                cand_patch_sar = polsar[small_rows + r + d[0], small_cols + c + d[1]]
                cand_patch_opt = optical[small_rows + r + d[0], small_cols + c + d[1]]
                # Store current patch for filtering
                patch_candidates[:,:,:,i_weights] = cand_patch_sar
                # Get distance and weights for PolSAR image and opt guide
                d_pol = d_polsar(centre_patch_sar, cand_patch_sar, Nw)
                d_opt = d_optical(centre_patch_opt, cand_patch_opt, Nw)
                # Add to output arrays
                d_pol_norm[i_weights] = d_pol / polsar_thresh
                d_opt_norm[i_weights] = d_opt / optical_thresh
      
            # Use only patches where image dissimilarity is lower than threshold
            # When divided by threshold, check if less than 1
            valid_predictors = np.where(d_pol_norm < 1.0)[0]
            n_valid = length(valid_predictors)

            # Sort dissimilarities sums, from smallest to largest
            if n_valid > n_patches:
                # Restrict to N smallest guide dissimilarities
                sorted_valid_predict = d_opt_norm[valid_predictors].argsort()[:n_patches]
                sorted_valid_predict = valid_predictors[sorted_valid_predict]
            else:
                sorted_valid_predict = valid_predictors
            
            # Use sorted weights to index tuple of indices
            for i_sorted in sorted_valid_predict:
                # Multiply with conjugate transpose
                temp_s1 = patch_candidates[n_small,n_small,:,i_sorted]
                temp_s1 = temp_s1[:, np.newaxis] # Make row vector          
                # Find weights 
                w_use = np.exp(-lamb*(gamma*d_pol_norm[i_sorted] + (1-gamma)*d_opt_norm[i_sorted])) 
                # Filter
                filtered = w_use * np.matmul(temp_s1 , np.conj(temp_s1).T)
                # Add to output
                c_mat[r, c] += filtered.flatten()
                weights_im[r, c] += w_use


    return c_mat[padding:n_rows-padding, padding:n_cols-padding], weights_im[padding:n_rows-padding, padding:n_cols-padding]


def _calcthrehs_patch(polsar, optical, big_diff, small_rows, small_cols, Nw, n_patch_tot, params):
    """Calculate thresholds for SAR and optical guide based on percentiles."""
    
    # Read parameters from object
    n_big = params['n_big']
    n_small = params['n_small'] # Also index of the centre pixel
    # Set dissimilarity functions
    d_polsar = _d_polsar
    d_optical = _d_optical

    # Get shape of input
    input_shape = polsar.shape
    n_rows, n_cols = polsar.shape[0], polsar.shape[1]
    if length(input_shape) == 2:
        n_channels = 1
        polsar = np.expand_dims(polsar, axis=2)
    elif length(input_shape) == 3:
        n_channels = input_shape[2]

    # Initialise array of patches
    patch_candidates = np.zeros((2*n_small+1, 2*n_small + 1, n_channels, n_patch_tot), dtype=polsar.dtype) 
    # Initialise list of weight sums
    sar_dissim_list = []
    opt_dissim_list = []

    # Create padding tuple ((row left, row right), (col left, ...))
    padding = n_big + n_small # also denotes extent for initial dissim building
    
    range_use = min(n_rows-padding, n_cols-padding)
    
    # Create list of coordinates to calculate dissimilarities for
    if params['thresh_est'] in ['diag', 'diagonal']:
        coord_list = []
        for r in range(padding, range_use):
            coord_list.append((r,r))
    elif params['thresh_est'] in ['rand_diag', 'rdl']:
        # Create as many randomly drawn (without replacement) coordinates as diag
        coord_list = rand_coord([padding,n_rows-padding], [padding,n_cols-padding], 
                                range_use-padding, unique_only = True)
        
    # Go through list of coordinates
    for r, c in coord_list: 
        # Extract (small) patch centred on pixel (x_r, x_c)
        centre_patch_sar = polsar[small_rows + r, small_cols + c]
        centre_patch_opt = optical[small_rows + r, small_cols + c]

        for i_weights, d in enumerate(big_diff):
            # Get current patch in PolSAR and optical guide
            cand_patch_sar = polsar[small_rows + r + d[0], small_cols + c + d[1]]
            cand_patch_opt = optical[small_rows + r + d[0], small_cols + c + d[1]]
            # Calculate dissimilarities
            sar_dissim_list.append(d_polsar(centre_patch_sar, cand_patch_sar, Nw))
            opt_dissim_list.append(d_optical(centre_patch_opt, cand_patch_opt, Nw))

    # Convert to numpy arrays
    sar_dissim_list = np.array(sar_dissim_list)
    opt_dissim_list = np.array(opt_dissim_list)

    # Ensure there are no NaNs
    sar_dissim_list = sar_dissim_list[np.isfinite(sar_dissim_list)]
    opt_dissim_list = opt_dissim_list[np.isfinite(opt_dissim_list)]

    thresh_sar = np.percentile(sar_dissim_list, params['sar_percentile'])
    thresh_opt = np.percentile(opt_dissim_list, params['opt_percentile'])
    
    if params['debug']:
        # Plot histograms of dissimilarities
        print('thresh PolSAR = '+str(thresh_sar))
        plt.figure()
        plt.hist(sar_dissim_list[np.isfinite(sar_dissim_list)], 100, facecolor='g', alpha=0.75)
        plt.ylabel('Counts')
        plt.xlabel('Dissimilarity')
        plt.title('thresh PolSAR = '+str(thresh_sar))
        
        print('thresh OPT = '+str(thresh_opt))
        plt.figure()
        plt.hist(opt_dissim_list, 100, facecolor='r', alpha=0.75)
        plt.ylabel('Counts')
        plt.xlabel('Dissimilarity')
        plt.title('thresh OPT = '+str(thresh_opt))
    
    return thresh_sar, thresh_opt     
