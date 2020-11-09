#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:08:54 2019

@author: umjcho
"""
######### Parameters ##############
file = "combined_upper_heatmap.dat"
maxdepth = 6.5 # If set to 0, matplotlib will set colour scale to the full range of normalised depth values
##### Parameters end ##############

import numpy as np
import matplotlib
matplotlib.use('agg') #for use without X-window server, e.g. over ssh
import matplotlib.pyplot as plt

hm = np.loadtxt(file)

hm2 = hm - np.nanmin(hm)

if maxdepth = 0:
	plt.imshow(hm2)
	plt.colorbar()
	plt.savefig("combined_upper_heatmap_zero_z_origin.svg")
	plt.close()

else:
	plt.imshow(hm2, vmax=maxdepth)
	plt.colorbar()
	plt.savefig(file[:-4] + "_zero_z_origin.svg")
	plt.close()
