#!/usr/bin/python3

"""
Created on Fri Dec 21 09:45:33 2018

@author: umjcho

This script analyses the depth of membrane deformation in a trajectory, which has been formatted as a series of separate .gro files
It should be run in a folder that contains these .gro files, and no other .gro files

Usage:
python3 depth_script.py

v4.6 changes:
	1.5 sized heatmap as that is all we need
	Number of bins reduced accordingly

v4.5 changes:
    Double sized heatmap, to capture the edges of the rotating membrane
    Suppress scientific notation when printing arrays to file
    Provides an escape mechanism for if there is no timestamp, allowing a heatmap to be drawn on a non-trajectory .gro file
    v4.4 introduced a bug where the output heatmap files for the upper and lower leaflet are identical. This is now fixed

v4.4 changes:
	For leftover atoms, uses only disc neighbours method (neighbours with distance < 4nm from missing atom, and with z-coordinates within a 1nm range of the missing atom) to assign to a leaflet. This is to avoid assigning PO4 atoms that have skipped the z-axis PBC boundary, as this would massively skew the leaflet depth in that region of the heatmap.
    Output list of atoms not assigned that was added in v4.3 now works
    Adds identification of lipids that switch leaflets

v4.3 retained changes:
    Writes atoms that are not assigned to a leaflet to a file for easier analysis

v4.2 retained changes:
    If atoms are missed by both branching directions, assign to leaflet of nearest neighbour which is already assigned a leaflet
    Adds a warning if more than 25% of atoms are missed despite branching in both directions

v4.1 retained changes:
    Changed criteria for detecting branching bleed, fixing bug where bleed detection would fail in cases of subtotal bleed
    Changed criteria for detecting branching failure, as original criteria only detected total branching failure, and ignored premature termination
    Added detection of branch failure for the check run

v4 retained changes:
    Leaflets are separated using a branching algorithm instead of separating by height
    If branching bleeds across leaflets, algorithm repeated with lower branching cutoff
    All analyses are performed on each leaflet separately

v2 retained changes:
    Calculates depth correctly regardless of whether dome points upward or downward
    Orders frames by timestamp instead of modification time, as simply copying files would break modification time
    If run on files without PO4 atoms, exits with an informative message instead of crashing
"""

import numpy as np
import sys, os
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

#Generate list of all the .gro files in the directory
gro_list = np.array([f for f in os.listdir('.') if f.endswith('.gro')])

bins = 75 #number of bins in heatmap

depths = np.zeros(3) #initialise table of depths

warnings = []
old_leaflet1 = np.array([])
old_leaflet2 = np.array([])
switched_to_leaflet1 = np.empty((0, 6))
switched_to_leaflet2 = np.empty((0, 6))

def initialize_heatmap():
    heatmap = np.zeros((bins, bins)) #initialise heatmap
    heatmap[:] = np.nan #initialise heatmap
    hm_residues_per_bin = np.zeros((bins, bins)) #initialise count of residues per bin for heatmap averaging
    return heatmap, hm_residues_per_bin #package heatmap and residues data into a tuple for feeding into function

upper_heatmap_data = initialize_heatmap()
lower_heatmap_data = initialize_heatmap()

frame_n = 1 #initialise frame counter

def separate_leaflets(membrane, timestamp=0):
    #input is array with column0=residue, column1=x, column2=y, column3=z
    #output is a tuple of (upper_leaflet, lower_leaflet)
    cutoff = 2 #nm
    warning_list = [] #initialise list of warnings for printing at the end

    def branching_function(membrane, start_from, cutoff): #starts at a lipid, extends a network of PO4 within cutoff of each other, and returns the network as one leaflet, assigning the remaining PO4 to the other leaflet
        coordinates = membrane[:, [1, 2, 3]]
        difference = coordinates - coordinates[start_from]
        distance = (difference[:, 0]**2 + difference[:, 1]**2 + difference[:, 2]**2)**0.5 
        in_leaflet1 = (distance < cutoff) & ~ (distance == 0) #exclude the starting atom so it can go to the head of the table later
        in_leaflet2 = distance > cutoff
        leaflet1 = np.vstack([membrane[start_from], membrane[in_leaflet1]]) #adds back the starting atom excluded previously
        leaflet2 = membrane[in_leaflet2]

        line_to_check = 1
        while line_to_check < len(leaflet1):

            reference = leaflet1[line_to_check][[1, 2, 3]]
            leaflet2_coordinates = leaflet2[:, [1, 2, 3]]

            difference = leaflet2_coordinates - reference
            distance = (difference[:, 0]**2 + difference[:, 1]**2 + difference[:, 2]**2)**0.5
            in_leaflet1 = distance < cutoff

            leaflet1 = np.vstack([leaflet1, leaflet2[in_leaflet1]])
            leaflet2 = leaflet2[np.invert(in_leaflet1)]

            line_to_check += 1

        return leaflet1, leaflet2

    first_run = branching_function(membrane, 0, cutoff)

    while len(first_run[1]) < (len(first_run[0]) * 0.5): #if branching has bled across bilayers, reduce cutoff and retry until bleeding stops
        cutoff = cutoff * 0.9
        first_run = branching_function(membrane, 0, cutoff)

    if len(first_run[0]) < (len(first_run[1]) * 0.5): #if branching fails, branch from elsewhere
        branch_start = 1
        while len(first_run[0]) < (len(first_run[1]) * 0.5):
            first_run = branching_function(membrane, branch_start, cutoff)
            branch_start += 1

    leaflet2_start = first_run[1][0] #run in opposite direction to ensure symmetry
    check_start = np.where((membrane == leaflet2_start).all(axis=1))
    check_run = branching_function(membrane, check_start, cutoff)

    if len(check_run[0]) < (len(check_run[1]) * 0.5): #if branching fails, branch from elsewhere
        check_index = 1
        while len(check_run[0]) < (len(check_run[1]) * 0.5):
            leaflet2_start = first_run[1][check_index]
            check_start = np.where((membrane == leaflet2_start).all(axis=1))
            check_run = branching_function(membrane, check_start, cutoff)
            check_index += 1

    leaflet1 = first_run[0]
    leaflet2 = check_run[0]

    if len(np.setdiff1d(leaflet1[:, 0], leaflet2[:, 0])) != len(leaflet1) or len(np.setdiff1d(leaflet2[:, 0], leaflet1[:, 0])) != len(leaflet2):
        print('FATAL ERROR: Despite bleed detection, leaflet1 and leaflet2 contain the same atoms at time ' + str(timestamp) + ' ps')
        sys.exit()

    branched_atoms = np.append(leaflet1[:, 0], leaflet2[:, 0])

    if len(np.setdiff1d(membrane[:, 0], branched_atoms)) != 0: #if atoms are missed by combination of forwards and backwards runs

        missed_atoms = np.setdiff1d(membrane[:, 0], branched_atoms)
        if len(missed_atoms) > (len(branched_atoms) * 0.25):
            print('WARNING: More than 25% of atoms missed during branching')

        def get_neighbours(search_centre, leaflet, search_radius=4): #gets number of neighbours in the leaflet provided.
            adjusted_x = leaflet[:, 1] - search_centre[0] #convert coordinate origin to location of missing residue
            adjusted_y = leaflet[:, 2] - search_centre[1]
            adjusted_z = leaflet[:, 3] - search_centre[2]

            dist = (adjusted_x**2 + adjusted_y**2 + adjusted_z**2)**0.5 # distance of PO4 from center missing residue

            search_sphere_contains = leaflet[dist < search_radius, :] #atoms in the search sphere
            search_disc_contains = search_sphere_contains[abs(search_sphere_contains[:, 3] - search_centre[2]) < 1, :]

            number_of_atoms_in_area = len(search_disc_contains)

            return number_of_atoms_in_area, dist


        for resid in missed_atoms:

            residue_is_on_row = membrane[:, 0] == resid
            search_centre = membrane[residue_is_on_row, [1,2,3]]

            leaflet1_get_neighbours = get_neighbours(search_centre, leaflet1)
            leaflet2_get_neighbours = get_neighbours(search_centre, leaflet2)

            leaflet1_distance = np.min(leaflet1_get_neighbours[1])
            leaflet2_distance = np.min(leaflet2_get_neighbours[1])

            leaflet1_neighbours = leaflet1_get_neighbours[0]
            leaflet2_neighbours = leaflet2_get_neighbours[0]

            if leaflet1_neighbours > leaflet2_neighbours:
                leaflet1 = np.vstack([leaflet1, membrane[residue_is_on_row, :]])
            elif leaflet2_neighbours > leaflet1_neighbours:
                leaflet2 = np.vstack([leaflet2, membrane[residue_is_on_row, :]])
            else:
                warning_details = [resid, timestamp]
                warning_list = warning_list + [warning_details]
                print('WARNING: Unable to assign residue ' + str(resid) + ' at time ' + str(timestamp))
                print('Distance from leaflet1 was ' + str(leaflet1_distance) + ' nm')
                print('Distance from leaflet2 was ' + str(leaflet2_distance) + ' nm')
                print('There were ' + str(leaflet1_neighbours) + ' neighbours in leaflet1')
                print('There were ' + str(leaflet2_neighbours) + ' neighbours in leaflet2')
                print('')

    return (leaflet1,  leaflet2, warning_list)



def get_depth(leaflet): #identifies the top and bottom 10% of residues by height, and returns the difference of their means
        top_cutoff = np.percentile(leaflet[:, 3], 90)

        top_ten_percent = leaflet[:, 3][leaflet[:, 3] > top_cutoff]

        average_top = np.mean(top_ten_percent)
        bottom = np.min(leaflet[:, 3])

        depth = average_top - bottom
        return depth

def switchers(source, destination, source_old, timestamp = 0):
    left_source = np.setdiff1d(source_old[:, 0], source[:, 0]) #check what lipids left source
    switched_to_destination = destination[np.isin(destination[:, 0], left_source)] #Entered destination having left source
    time_column = np.full((len(switched_to_destination), 1), timestamp) #column of time to add to switchlist
    switched_to_destination_with_time = np.hstack([switched_to_destination, time_column]) #add time column
    return switched_to_destination_with_time

lipid_dict = {
        "POPC ": 1,
        "POPE ": 2,
        "POPS ": 3,
        "POP2 ": 4,
        "DPSM ": 5,
        "CHOL ": 6
        }

for path in gro_list:

    #First we parse the resid and coordinates from .gro file
    frame = open(path)
    data = frame.read().splitlines() #read the frame to a list, one line per list
    numlen = 0

    coords = np.zeros((0,5))
    firstrow = data[0] # get timestamp
    time_starts_at = str.find(firstrow, ' t=') + 3 #find the position of the string 't=', which precedes the timestamp

    if time_starts_at > 10:     #this will be true if there is a timestamp
        timestamp = float(firstrow[time_starts_at:])
    else:   #if there is no timestamp, then set timestamp to 0
        timestamp = 0

    for row in data: #convert resID and coordinates into a numpy array

        if row[10:15] == '  PO4': #only import the PO4 coordinates

            if numlen == 0: #determine the length of coordinate string. Only do this the first time.
                first_dec = row.find('.')
                second_dec = row.find('.', first_dec +1 )
                numlen = second_dec - first_dec

            res = row[0:5]
            x = row[20: 20 + numlen]
            y = row[20 + numlen: 20 + numlen*2]
            z = row[20 + numlen*2: 20 + numlen*3]

            lipid_type = row[5:10]
            lipid_id = lipid_dict.get(lipid_type)
            row_coords = np.array([res, x , y, z, lipid_id], dtype='float')

            coords = np.vstack([coords, row_coords])

        else:
            continue

    frame.close()

    if len(coords) == 0: #if no coordinates have been imported (which only happens if there is no PO4), move to next file
        continue

    separated_leaflets = separate_leaflets(coords, timestamp) #separate leaflets

    warnings = warnings + separated_leaflets[2]

    leaflet1 = separated_leaflets[0]
    leaflet2 = separated_leaflets[1]
    leaflet1_mean_height = np.mean(leaflet1[:, 3])
    leaflet2_mean_height = np.mean(leaflet2[:, 3])
    if leaflet1_mean_height >  leaflet2_mean_height: #Assign leaflets to upper or lower
        upper_leaflet = leaflet1
        lower_leaflet = leaflet2
    elif leaflet2_mean_height > leaflet1_mean_height:
        upper_leaflet = leaflet2
        lower_leaflet = leaflet1
    elif leaflet1_mean_height == leaflet2_mean_height:
        print('Error: Leaflet heights are the same')
        sys.exit()

    if len(old_leaflet1) != 0 and len(old_leaflet2) != 0: #if this isn't the first run
        switched_to_leaflet1_this_frame = switchers(leaflet2, leaflet1, old_leaflet2, timestamp = timestamp)
        if len(switched_to_leaflet1_this_frame) > 0:
            switched_to_leaflet1 = np.vstack([switched_to_leaflet1, switched_to_leaflet1_this_frame])

        switched_to_leaflet2_this_frame = switchers(leaflet1, leaflet2, old_leaflet1, timestamp = timestamp)
        if len(switched_to_leaflet2_this_frame) > 0:
            switched_to_leaflet2 = np.vstack([switched_to_leaflet2, switched_to_leaflet2_this_frame])

    old_leaflet1 = leaflet1
    old_leaflet2 = leaflet2


    #Now obtain the depths
    upper_depth = get_depth(upper_leaflet)
    lower_depth = get_depth(lower_leaflet)

    depths = np.vstack((depths, [timestamp, upper_depth, lower_depth]))


    #Generate heatmap of z-coordinates over time
    lastline_split = np.array(data[-1].split(' ')) #extract box size from last line
    lastline_numbers = lastline_split[lastline_split != '']
    range_x = float(lastline_numbers[0])
    range_y = float(lastline_numbers[1])

    def make_heatmap(coords, heatmap_data, bins, range_x, range_y): #adds to a pre-initialised heatmap
        heatmap = heatmap_data[0]
        hm_residues_per_bin = heatmap_data[1]
        x_binwidth = range_x * 1.5 / bins
        y_binwidth = range_y * 1.5 / bins
        x_start = -0.25 * range_x #heatmap origin, which is outside the actual simulation in order to capture all information
        y_start = -0.25 * range_y

        for ybin in range(1, bins+1):                                               #in each row
            ybin_start = coords[:, 2] > (y_start + y_binwidth * (ybin - 1))   #boolean for y > start of bin
            ybin_end = coords[:, 2] <= (y_start + y_binwidth * ybin)              #boolean for y < end of bin
            ybin_selector = ybin_start * ybin_end
            this_ybin = coords[ybin_selector, :]                        #select residues in this ybin

            for xbin in range(1, bins+1):                                           #go through the bins
                xbin_start = this_ybin[:, 1] > (x_start + x_binwidth * (xbin - 1))
                xbin_end = this_ybin[:, 1] <= (x_start + x_binwidth * xbin)
                xbin_selector = xbin_start * xbin_end
                thisbin = this_ybin[xbin_selector, :]

                if np.isnan(heatmap[xbin - 1, ybin -1]): #if the heatmap bin is currently empty
                    total_z_this_bin = sum(thisbin[:, 3])
                    hm_residues_per_bin[xbin - 1, ybin -1] = len(thisbin[:, 3])
                else: #if the heatmap bin already contains data
                    total_z_this_bin = (heatmap[xbin - 1, ybin -1] * hm_residues_per_bin[xbin - 1, ybin -1]) + sum(thisbin[:, 3]) #generate a new total from the previous contents and the current frame
                    hm_residues_per_bin[xbin - 1, ybin -1] = hm_residues_per_bin[xbin - 1, ybin -1] + len(thisbin[:, 3]) #increase the residue count accordingly

                if len(thisbin[:, 3]) > 0: #if there was anything in this bin in this frame
                    heatmap[xbin - 1, ybin -1] = total_z_this_bin / hm_residues_per_bin[xbin - 1, ybin -1] #update the average value in the heatmap

        return (heatmap, hm_residues_per_bin)

    upper_heatmap_data = make_heatmap(upper_leaflet, upper_heatmap_data, bins, range_x, range_y)
    lower_heatmap_data = make_heatmap(lower_leaflet, lower_heatmap_data, bins, range_x, range_y)

    frame_n = frame_n + 1

if len(coords) == 0: #if having looped through all files, there are still no PO4 atoms, exit program
    print('No .gro files in this folder contain PO4 atoms')
    sys.exit()

depths = np.delete(depths, 0, axis=0) #remove initialisation column
depths = depths[depths[:, 0].argsort()] #sort depths by timestamp

#Export depths to file
np.savetxt("depths.dat", depths)

#Write mean depths to file
f = open('depths.txt', 'w+')
f.write("Depth of upper leaflet dome = " + str(np.mean(depths[:, 1])) + " nm\n" +
        "Depth of lower leaflet dome = " + str(np.mean(depths[:, 2])) + " nm")
f.close()

#Output graph of change in depth over course of simulation
u_DoT = plt.plot(depths[:, 0], depths[:, 1], linewidth=0.5)
plt.savefig("upper_leaflet_depth_over_time.svg")
plt.close()

l_DoT = plt.plot(depths[:, 0], depths[:, 2], linewidth=0.5)
plt.savefig("lower_leaflet_depth_over_time.svg")
plt.close()

#Output heatmap data
np.savetxt("upper_heatmap.dat", upper_heatmap_data[0])
np.savetxt("rescounts_upper_heatmap.dat", upper_heatmap_data[1])

np.savetxt("lower_heatmap.dat", lower_heatmap_data[0])
np.savetxt("rescounts_lower_heatmap.dat", lower_heatmap_data[1])

hmplot = plt.imshow(upper_heatmap_data[0])
plt.colorbar(hmplot)
plt.savefig("upper_heatmap.pdf")
plt.savefig("upper_heatmap.svg")
plt.close()

hmplot = plt.imshow(lower_heatmap_data[0])
plt.colorbar(hmplot)
plt.savefig("lower_heatmap.pdf")
plt.savefig("lower_heatmap.svg")
plt.close()

ctplot = plt.contourf(upper_heatmap_data[0])
plt.colorbar(ctplot)
plt.savefig("upper_contour.pdf")
plt.savefig("upper_contour.svg")
plt.close()

ctplot = plt.contourf(lower_heatmap_data[0])
plt.colorbar(ctplot)
plt.savefig("lower_contour.pdf")
plt.savefig("lower_contour.svg")
plt.close()

if len(warnings) > 0:
    warnings = np.array(warnings)
    np.savetxt("atoms_not_assigned.dat", warnings)

switched_to_leaflet1 = np.hstack([switched_to_leaflet1, np.full((len(switched_to_leaflet1), 1), 1)])
switched_to_leaflet2 = np.hstack([switched_to_leaflet2, np.full((len(switched_to_leaflet2), 1), 2)])
lipids_that_switched_leaflets = np.vstack([switched_to_leaflet1, switched_to_leaflet2])
np.savetxt("switched.dat", lipids_that_switched_leaflets) #columns are     0-resid     1-x     2-y     3-z     4-lipid_type     5-timestamp     6-switched_to
