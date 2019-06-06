#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:08:54 2019

@author: umjcho
"""

import numpy as np
import matplotlib.pyplot as plt

hm = np.loadtxt("combined_upper_heatmap.dat")

#Invert z-axis of heatmap
hm2 = (hm - np.nanmax(hm)) * -1

#Invert y-axis of heatmap
hm3 = np.flipud(hm2)

plt.imshow(hm3)
plt.colorbar()
plt.savefig("combined_upper_heatmap_zero_z_origin_z-inverted_y-inverted.svg")
plt.close()
