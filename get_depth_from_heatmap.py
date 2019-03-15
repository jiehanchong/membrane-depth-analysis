#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 15:09:19 2019

@author: umjcho
"""

import numpy as np

hm = np.loadtxt("combined_lower_heatmap.dat")

hm_1d = np.reshape(hm, -1)
hm_no_nan = hm_1d[~np.isnan(hm_1d)]

centile90 = np.percentile(hm_no_nan, 90)

above90th = hm_no_nan[hm_no_nan>centile90]

depth_of_heatmap_membrane = np.mean(above90th) - np.min(hm_no_nan)
print(depth_of_heatmap_membrane)