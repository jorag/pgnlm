# pgnlm
Estimate PolSAR covariance matrices using nonlocal means with optical guide image.

Simulated PolSAR data for testing the algorithm is provided in the data directory.

Works with Python 3.4 or later. 
If using Conda, the following packages are required:
- numpy
- gdal
- matplotlib
- tqmd

Optional (but recomended for speeding up the proecssing by running on 4 CPU cores):
- multiprocessing
