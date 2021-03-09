#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jorag
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
#import gdal
from osgeo import gdal
import parameters
import datasets
from algorithm import run_pgnlm
from helperfuncs import length, norm01, dB, complex2real


#%% Path to directories
parent_dir = os.path.realpath('..') 
output_dir = os.path.join(parent_dir, 'output')
# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Which output to produce
save_geotiff = True
save_as_complex = False
plot_result = True

#%% Load SAT data
dataset_key = 'test'
polsar, optical, geotransform = datasets.fetch_dataset(dataset_key)

#%% Set parameters
params = parameters.get_params(debug=False)

#%% Time the processing
if params['debug']:
    print('Process start: '+time.strftime("%Y%m%d-%H%M"))
time_start = time.time()

#%% Run PGNLM
covmat_flat, weight_im = run_pgnlm(polsar, optical, params=params)

#%% Calculate final covariance matrices(divide by weight image)
if length(covmat_flat.shape) > length(weight_im.shape):
    covmat_flat = covmat_flat/weight_im[:,:,np.newaxis]
else:
    covmat_flat = covmat_flat/weight_im

# Reshape to 3x3 covariance matrices
c_size = covmat_flat.shape 
covmat_full = np.reshape(covmat_flat, (c_size[0],c_size[1],3,3))

# Time 
if params['debug']:
    print('Process end: '+time.strftime("%Y%m%d-%H%M"))
# Elapsed time
time_stop = time.time()
time_elapsed = time_stop - time_start
print('Time total = '+str(time_elapsed)+' s = '+str(time_elapsed/60)+' min')        


#%% Save flattened result as GeoTIFF
if save_geotiff:
    # Get SAR features
    if save_as_complex:
        covmat_im = covmat_full
    else:
        covmat_im = complex2real(covmat_flat)

    # Save parameters in .xls file and return the name for the GeoTIFF output
    tiff_fullfile = os.path.join(output_dir, dataset_key+'_covmat_out.tif')
    
    # Get size of filtered and guide data
    f1_size = covmat_im.shape
    n_optical_bands = optical.shape[2]
    
    # Create the n_bands_out-band raster file
    n_bands_out = f1_size[2] + n_optical_bands + 2
    if save_as_complex:
        dst_ds = gdal.GetDriverByName('GTiff').Create(tiff_fullfile, f1_size[1], f1_size[0], n_bands_out, gdal.GDT_CFloat64)
    else:
        dst_ds = gdal.GetDriverByName('GTiff').Create(tiff_fullfile, f1_size[1], f1_size[0], n_bands_out, gdal.GDT_Float64) 
        
    # Set geotransform, use pixel spacing etc. from original
    dst_ds.SetGeoTransform(geotransform)    
    
    # Establish encoding
    try:
        srs = gdal.SpatialReference()            
    except AttributeError:
        from osgeo import osr
        srs = osr.SpatialReference() 
    
    # WGS84 lat/long
    srs.ImportFromEPSG(4326) 
    # Export coords to file
    dst_ds.SetProjection(srs.ExportToWkt())
    # Write bands
    for i_band in range(f1_size[2]):
        # write filtered SAR band to the raster
        dst_ds.GetRasterBand(i_band+1).WriteArray(covmat_im[:,:,i_band])   
    for i_band in range(n_optical_bands):
        # write optical band to the raster
        dst_ds.GetRasterBand(i_band+f1_size[2]+1).WriteArray(optical[:,:,i_band])   

    # Write to disk
    dst_ds.FlushCache()
    dst_ds = None
    
#%% Plot result
if plot_result:        
    # Min dB value for contrast stretching
    dB_min_pgnlm = 0.001 # 0.001
    dB_min = dB_min_pgnlm
    # RGB bands for optical guide
    rgb_bands = [0,1,2] 
    
    # Plot original intensity data
    org_show = norm01(dB(np.abs(polsar)+dB_min), norm_type='channel')
    plt.figure()
    plt.imshow(org_show)
    plt.title('SAR intensity bands')

    # Plot estimated C diagonal (intensity bands)
    img_use = np.abs(covmat_flat[:,:, [0,4,8]])
    img_use = dB(img_use+dB_min)
    # Re-Normailze image so that values are between 0 to 1
    img_use = norm01(img_use, norm_type='channel')
    plt.figure()
    plt.imshow(img_use)
    plt.title(r'$C_{11}$, $C_{22}$, $C_{33}$')

    # Plot optical guide
    opt_im_show = norm01(optical[:,:, rgb_bands], norm_type='channel')
    plt.figure()
    plt.imshow(opt_im_show)
    plt.title('Guide RGB')
    
    # Plot sum of weights
    plt.figure()
    plt.imshow(weight_im)
    plt.colorbar()
    plt.title('Weight sum')