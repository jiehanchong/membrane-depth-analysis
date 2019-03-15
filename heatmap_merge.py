#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 17:05:01 2019

@author: umjcho

Script for combining depth.dat and heatmap.dat files produced by depth_script.py
Usage:
python3 heatmap_merge.py folder1 folder2 folder3 ...

v4 changes:
    Output raw heatmap data to allow subsequent transformation
    
v3 changes:
    Compatible with depth_script_v4.py
"""

import numpy as np
import sys
import matplotlib.pyplot as plt

folder_list = sys.argv[1:]

def join_heatmaps(file_name, list_of_folders_to_search):#Import and combine heatmaps matching the name given, in list of folders provided
    maplist = [None] * len(list_of_folders_to_search)

    for n in range(len(list_of_folders_to_search)):
        folder = list_of_folders_to_search[n - 1]
        if folder.endswith('/'):
            heatmap_file = file_name
        else:
            heatmap_file = '/' + file_name
        maplist[n] = np.loadtxt((folder + heatmap_file), delimiter=' ')

    non_zero_cells = np.zeros(np.shape(maplist[0]))    #array keeping track of number of simulations contributing to each cell

    for m in maplist:
        non_zero_cells = non_zero_cells + (m != 0)

    summed_heatmap = sum(maplist)
    normalised_heatmap = summed_heatmap / non_zero_cells #divide summed heights of each cell by number of contributing sims

    return normalised_heatmap

combined_upper_heatmap = join_heatmaps('upper_heatmap.dat', folder_list)
combined_lower_heatmap = join_heatmaps('lower_heatmap.dat', folder_list)

plot = plt.imshow(combined_upper_heatmap)
plt.colorbar(plot)
plt.savefig("combined_upper_heatmap.svg")
plt.savefig("combined_upper_heatmap.pdf")
plt.close()

plot = plt.imshow(combined_lower_heatmap)
plt.colorbar(plot)
plt.savefig("combined_lower_heatmap.svg")
plt.savefig("combined_lower_heatmap.pdf")
plt.close()

plot = plt.contourf(combined_upper_heatmap)
plt.colorbar(plot)
plt.savefig("combined_upper_contour.svg")
plt.savefig("combined_upper_contour.pdf")
plt.close()

plot = plt.contourf(combined_lower_heatmap)
plt.colorbar(plot)
plt.savefig("combined_lower_contour.svg")
plt.savefig("combined_lower_contour.pdf")
plt.close()

np.savetxt("combined_upper_heatmap.dat", combined_upper_heatmap)
np.savetxt("combined_lower_heatmap.dat", combined_lower_heatmap)

#Import and combine depths
for n in range(len(folder_list)):
    folder = folder_list[n - 1]
    if folder.endswith('/'):
        depths_file = 'depths.dat'
    else:
        depths_file = '/depths.dat'

    if n == 0:
        depthlist = np.loadtxt((folder + depths_file), delimiter=' ')
    else:
        depthlist = np.vstack((depthlist, np.loadtxt((folder + depths_file), delimiter=' ')))

average_upper_leaflet_depth = np.mean(depthlist[:, 1])
average_lower_leaflet_depth = np.mean(depthlist[:, 2])

f = open('average_depths.txt', 'w+')
f.write("Depth of upper leaflet dome over " + str(len(folder_list)) + " simulations = " + str(average_upper_leaflet_depth) + " nm\n" +
        "Depth of lower leaflet dome over " + str(len(folder_list)) + " simulations = " + str(average_lower_leaflet_depth) + " nm")
f.close()
