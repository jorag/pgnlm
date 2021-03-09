# pgnlm
Estimate PolSAR covariance matrices from single-look complex (SLC) using nonlocal means with a coregistered optical guide image.

Simulated PolSAR data with coregistered optical guide for testing the algorithm is provided in the data directory.

Works with Python 3.4 or later. 
If using Conda, installing the following packages are sufficient:
- numpy 
- matplotlib (for plotting result)
- tqmd (for progress bar)
- gdal (for reading and writing GeoTIFF, required for example data)

Optional (but recomended for speeding up the proecssing by running on 4 CPU cores):
- multiprocessing
