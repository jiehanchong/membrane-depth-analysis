membrane-depth-analysis
A repository of tools for calculating the depth of a membrane, and manipulating the outputs

depth_script.py
  From a trajectory in the form of separate .gro files, generates a map of upper and lower leaflet depth. The depth is calculated from the PO4 CG beads in the simulation. Presence of PO4 beads not part of the bilayer may lead to incorrect results
    upper_heatmap.dat, lower_heatmap.dat - files containing a matrix of the depths at each point for the upper and lower leaflets respectively
    upper_heatmap.pdf, upper_heatmap.svg, lower_heatmap.pdf, lower_heatmap.svg - depth maps for the upper and lower leaflets
    upper_contour.pdf, upper_contour.svg, lower_contour.pdf, lower_contour.svg - contour maps for the upper and lower leaflets
    depths.txt - a text file stating the average depth of upper and lower leaflets
    depths.dat - a text file containing timestamp in first column, upper leaflet depth in second column, and lower leaflet depth in third column
    upper_leaflet_depth_over_time.svg, lower_leaflet_depth_over_time.pdf - the data from depths.dat plotted on a graph
    atoms_not_assigned.dat - any PO4 beads which did not get assigned to either leaflet, suggesting they are spatially removed from the membrane
    switched.dat - Any PO4 beads that switch leaflets during the simulation trajectory. Columns are  0-resid     1-x     2-y     3-z     4-lipid_type     5-timestamp     6-switched_to

heatmap_difference.py
  Takes depth maps provided as command line arguments, and returns new subtraction depth maps showing the pairwise difference between consecutive maps
  
heatmap_merge.py
  Takes a series of folders as command line arguments, looks for heatmp.dat and depths.dat files in those folders, and returns maps of the average of the files found
  
invert_and_normalise_z-axis_of_depth_map.py, invert_y_and_z_axis_and_normalise_z-axis_of_depth_map.py
  Looks through the working directory for combined_upper_heatmap.dat, and returns a new depth map with the axes treated as described in the script title
