#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jorag
"""

def get_params(debug=False):
    params = {
        # Extra printing and plotting
        'debug': debug,
        # Radius of search window
        'n_big': 19,
        # Radius of patches
        'n_small': 2,
        # Max number of predictors
        'n_patches': int(64),
        # Lambda for calculating weights
        'lamb': 2.0,
        # Balancing d_pol vs. d_opt i weight
        'gamma': 0.85,
        # Percentile for discarding predcitors
        'sar_percentile': 50,
        # Percentile for "normalising" d_opt
        'opt_percentile': 50,
        # Method for threshold estimation 
        'thresh_est': 'diag', # 'rand_diag'
        # Number of CPU cores to use
        'cpu_cores': 4, # 1
    }

    return params