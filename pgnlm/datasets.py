#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jorag
"""

import numpy as np
import os 
from osgeo import gdal
from helperfuncs import iq2complex, norm01


def _test_dataset(load_options):
    """ Load and crop the simulated test dataset.

        Input:
            load_options - not used
        Output:
            polsar - simulated PolSAR image with complex target vectors, 
                    complex numpy tensor with shape (H, W, 3)
            optical - coregistered optical Sentinel-2 image, 
                    numpy tensor with shape (x, y, 4)
            geotransform - geotransform read from input is GeoTIFF,
                           tuple=(lon_min, xres, rot1, lat_min, rot2, yres)
    """
    
     # Path to directories and input file
    parent_dir = os.path.realpath('..') 
    data_dir = os.path.join(parent_dir, 'data')
    sat_file = os.path.join(data_dir, 'simulated_polsar.tif')

    # Data is a single GeoTIFF file, specify bands for the optical guide
    optical_bands = [6,7,8,9,10,11]

    # Data is a single GeoTIFF file, specify bands for guide SAR I and Q bands
    polsar_bands = [0, 1, 2, 3, 4, 5]

    # Load data 
    dataset = gdal.Open(sat_file)
    # Read all bands 
    raster_data = dataset.ReadAsArray()
    # Get geotransform
    lon_min, xres, rot1, lat_min, rot2, yres = dataset.GetGeoTransform()

    # Get geo bands to update geotransform
    lat = raster_data[12, :,:]
    lon = raster_data[13, :,:]

    # Indices for cropping data
    x_min = 50
    x_max = 250
    y_min = 200
    y_max = 400
    
    # Read input (SAR) and guide (Optical), lat and lon 
    polsar = raster_data[polsar_bands, x_min:x_max, y_min:y_max]
    optical = raster_data[optical_bands, x_min:x_max, y_min:y_max]
    lat = lat[x_min:x_max, y_min:y_max]
    lon = lon[x_min:x_max, y_min:y_max]

    # Change order to H, W, channels
    polsar = np.transpose(polsar, (1, 2, 0))
    optical = np.transpose(optical, (1, 2, 0))

    # Normalise guide so that values are between 0 to 1
    optical = norm01(optical, norm_type='global')

    # Convert input to complex data type
    polsar = iq2complex(polsar, reciprocity=True)

    # Calculate min and max of lat/lon
    xmin, ymin, xmax, ymax = np.min(lon), np.min(lat), np.max(lon), np.max(lat)
    # Set geotransform, use pixel spacing from original
    geotransform = (xmin, xres, rot1, ymax, rot2, yres)

    return polsar, optical, geotransform


def _full_test_dataset(load_options):
    """ Load the simulated test dataset.

        Input:
            load_options - not used
        Output:
            polsar - simulated PolSAR image with complex target vectors, 
                    complex numpy tensor with shape (H, W, 3)
            optical - coregistered optical Sentinel-2 image, 
                    numpy tensor with shape (x, y, 4)
            geotransform - geotransform read from input is GeoTIFF,
                           tuple=(lon_min, xres, rot1, lat_min, rot2, yres)
    """
    
    # Path to directories and input file
    parent_dir = os.path.realpath('..') 
    data_dir = os.path.join(parent_dir, 'data')
    sat_file = os.path.join(data_dir, 'simulated_polsar.tif')

    # Data is a single GeoTIFF file, specify bands for guide
    optical_bands = [6,7,8,9,10,11]

    # Data is a single GeoTIFF file, specify bands for SAR I and Q bands
    polsar_bands = [0, 1, 2, 3, 4, 5]

    # Load data 
    dataset = gdal.Open(sat_file)
    # Read all bands 
    raster_data = dataset.ReadAsArray()
    # Get geotransform
    geotransform = dataset.GetGeoTransform()

    # Get bands by indice, do not crop
    polsar = raster_data[polsar_bands, :,:]
    optical = raster_data[optical_bands, :,:]

    # Change order to H, W, channels
    polsar = np.transpose(polsar, (1, 2, 0))
    optical = np.transpose(optical, (1, 2, 0))

    # Normalise guide so that values are between 0 to 1
    optical = norm01(optical, norm_type='global')

    # Convert input to complex data type
    polsar = iq2complex(polsar, reciprocity=True)

    return polsar, optical, geotransform


# Dict linking each specific dataset with a load function
datasets = {
    'test': _test_dataset,
    'full_test': _full_test_dataset,
}


def fetch_dataset(name, load_options=None):
    """ Fetch the specified dataset.

        Input:
            name - dataset name, should be in datasets dict
            load_options - passed directly to the specific load methods,
                           can be none if not needed
        Output:
            polsar - PolSAR image with complex target vectors, 
                    complex numpy tensor with shape (H, W, n_chans),
                    n_chans = 3 (S_hh, S_hv/vh, S_vv)
            optical - coregistered optical image, 
                    numpy tensor with shape (x, y, bands)
            geotransform - geotransform for writing GeoTIFF output, 
                           can be read and reused if input is GeoTIFF,
                           tuple=(lon_min, xres, rot1, lat_min, rot2, yres)
    """
    polsar, optical, geotransform = datasets[name](load_options)

    return polsar, optical, geotransform

