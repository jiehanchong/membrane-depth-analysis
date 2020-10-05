#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:08:54 2019

@author: umjcho

Imports a depth map file dat file produced by depth_script.py, set in [file] in the parameters section of script
Plots an SVG, normalising the colour scale to start from 0, and end at [max], which is set in the parameters section of script
"""

#Parameters
max = 6.5
file = "combined_upper_heatmap.dat"
#####

import numpy as np
import matplotlib.pyplot as plt

hm = np.loadtxt(file)

hm2 = hm - np.nanmin(hm)

plt.imshow(hm2, vmax = max)
plt.colorbar()
plt.savefig("combined_upper_heatmap_rescaled.svg")
plt.close()
