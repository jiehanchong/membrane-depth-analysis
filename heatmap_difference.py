"""
This script accepts a series of heatmaps of identical dimensions, performs pairwise comparison in the order provided, and outputs the absolute value of the difference between each pair


Created 29/01/2019 by Jiehan Chong
"""

import numpy as np
import sys
import matplotlib.pyplot as plt

heatmap_names = sys.argv[1:]
heatmap_list = []
difference_list = []

n = 0
for name in heatmap_names:
    heatmap_list = heatmap_list + [np.loadtxt(name)]
    n = n + 1

counter = 0
for heatmap in heatmap_list[1:]:
    difference_map = abs(heatmap - heatmap_list[counter])
    difference_list = difference_list + [difference_map]
    counter = counter + 1
    
start = 1
end = 2
for map in difference_list:
    
    
    np.savetxt("difference_map " + str(start) + "-" + str(end) + ".dat", map)
    plot = plt.imshow(map, cmap="gray_r", vmin=0, vmax=0.8)
    plt.colorbar(plot)
    plt.savefig("difference_map_" + str(start) + "-" + str(end) + ".svg")
    plt.savefig("difference_map_" + str(start) + "-" + str(end) + ".pdf")
    plt.close()
    
    start = start + 1
    end = end + 1
