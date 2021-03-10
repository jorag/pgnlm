# Polarimetric Guided Nonlocal Means (PGNLM)
PGNLM estimates PolSAR covariance matrices from single-look complex (SLC) using nonlocal means with a coregistered optical guide image.

Simulated PolSAR data with coregistered optical guide for testing the algorithm is provided in the data directory.

To run PGNLM on your own dataset, create a function for loading the coregistered PolSAR and optical data in pgnlm/datasets.py. See the functions for loading the test dataset for an example. The output should be a complex numpy array of size (H, W, 3) with the PolSAR data (S_hh, S_hv/vh, S_vv) and the coregistered optical guide image with as a numpy array with size (H, W, n_bands), where n_bands is the number of spectral channels in the optical image. Additionaly the geotransform must be provided if the output should be written as a GeoTIFF file, if not, geotransform can be set to a dummy value.
Next, give your dataset a key in the datasets dictionary and update the datasets dictionary with the loading function you created. Then the algorithm can be run on the dataset of your choice by passing the key when fetching the dataset in pgnlm/main.py.

## Requirements
Works with Python 3.4 or later. 
If using Conda, installing the following packages are sufficient:
- numpy 
- matplotlib (for plotting result)
- tqmd (for progress bar)
- gdal (for reading and writing GeoTIFF, required for example data)

Optional (but recomended for speeding up the proecssing by running on 4 CPU cores):
- multiprocessing
