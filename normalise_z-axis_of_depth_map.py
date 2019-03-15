#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:08:54 2019

@author: umjcho
"""

import numpy as np
import matplotlib.pyplot as plt

hm = np.loadtxt("combined_upper_heatmap.dat")

hm2 = hm - np.nanmin(hm)

plt.imshow(hm2)
plt.colorbar()
plt.savefig("combined_upper_heatmap_10-16.svg")
plt.close()