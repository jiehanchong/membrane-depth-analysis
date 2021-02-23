'''
Author  : Jiehan Chong
e-Mail  : chongjiehan@gmail.com

This is a sped-up version of standard depth_script.
Algorithm is sped up by removing the leaflet separation step from the standard script.
Adds multiprocessing.
The code layout is also reorganised for simplicity.
'''

import numpy as np
import sys, glob, argparse
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)  # Suppress graphical output to allow running
                                    # on systems without graphcal interface, such
                                    # as over SSH

#### Create parser ####
p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='''
Calculate depth of lipid bilayer in a trajectory.
Trajectory should be in GRO format, one file per frame.
Script should be run in a folder where the only GRO files are part of the trajectory.

Outputs:
depths.dat            - csv with timestep (ps) in first column, depth (nm) in second column
average_depth.txt     - Text file stating the average depth over trajectory duration
mid-bilayer_depth.svg - plot of the data in depths.dat
heatmap.dat           - data of heatmap of average depths, in csv format
rescounts_heatmap.dat - number of residues in each bin of heatmap.dat
heatmap.pdf           - plot of data in heatmap.dat
heatmap.svg           - plot of data in heatmap.dat
contour.pdf           - contour map of data in heatmap.dat
contour.svg           - contour map of data in heatmap.dat

''')

p.add_argument('--bins',
               dest='bins',
               help='''Number of bins in X and Y axis for output depth map. 
                       It is assumed that the X and Y dimensions of the 
                       simulation box are the same. Default is 75.''',
               default=75,
               type=int)
p.add_argument('--mp',
               dest='mp',
               action='store_true',
               help='Turn on multiprocessing.')
p.add_argument('--time',
               action='store_true',
               help='Time the script')

args = p.parse_args()

if args.time == True:
    import datetime
    start_time = datetime.datetime.now()

def initialize_heatmap(bins):
    heatmap = np.empty((bins, bins))               # initialise heatmap
    heatmap[:] = np.nan                 
    hm_residues_per_bin = np.zeros((bins, bins))    # initialise count of residues per bin for heatmap averaging
    return heatmap, hm_residues_per_bin

def extract_data(gro_file):
    with open(gro_file) as frame:
        data = frame.read().splitlines() #read the frame to a list, one line per list
        numlen = 0

        # get timestamp
        firstrow = np.array(data[0].split(' '))         # Split first row by whitespace
        firstrow = firstrow[firstrow != '']             # Remove empty values created by long stretches of whitespace
        time_index = np.where(firstrow == 't=')[0]      # find the position of the string 't=',
        if len(time_index != 0):                        # If there is a timestamp in gro file
            timestamp = firstrow[time_index + 1].item() # save value to timestamp variable
            timestamp = int(float(timestamp))
        else:                                           # if there is no timestamp, 
            timestamp = 0                               # then set timestamp to 0

        coords = np.zeros((0,4))
        for row in data:                                # convert resID and coordinates into a numpy array
            if row[10:15] == '  PO4':                   # only import the PO4 coordinates
                if numlen == 0:                         # determine the length of coordinate string. Only do this the first time.
                    first_dec = row.find('.')
                    second_dec = row.find('.', first_dec +1 )
                    numlen = second_dec - first_dec

                res = row[0:5]
                x = row[20: 20 + numlen]
                y = row[20 + numlen: 20 + numlen*2]
                z = row[20 + numlen*2: 20 + numlen*3]

                row_coords = np.array([res, x , y, z], dtype='float')

                coords = np.vstack([coords, row_coords])

            else:
                continue

        # if having looped through all files, there are still no PO4 atoms, exit program
        if len(coords) == 0: 
            print('First frame analysis complete. No PO4 atoms present.')
            print('Exiting program.')
            sys.exit()
        
        # extract box size from last line
        lastline_split = np.array(data[-1].split(' '))
        lastline_numbers = lastline_split[lastline_split != '']
        range_x = float(lastline_numbers[0])
        range_y = float(lastline_numbers[1])

    return coords, timestamp, range_x, range_y

def get_depth(leaflet): 
    # identifies the top and bottom 10% of residues by height,
    # and returns the difference of their means
        top_cutoff = np.percentile(leaflet[:, 3], 90)

        top_ten_percent = leaflet[:, 3][leaflet[:, 3] > top_cutoff]

        average_top = np.mean(top_ten_percent)
        bottom = np.min(leaflet[:, 3])

        depth = average_top - bottom
        return depth

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

    return heatmap, hm_residues_per_bin

def make_heatmap_single_frame(coords, range_x, range_y, bins=args.bins):
    heatmap, hm_residues_per_bin = initialize_heatmap(bins)
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

            heatmap[xbin -1, ybin -1] = sum(thisbin[:, 3])
            hm_residues_per_bin[xbin - 1, ybin -1] = len(thisbin[:, 3])

    return heatmap, hm_residues_per_bin

def init_counter(x):
    # Store counter for later use
    global counter
    counter = x

def mp_wrapper(file):
    # Extract data
    coords, timestamp, range_x, range_y = extract_data(file)

    # Calculate depth
    this_frame_depth = get_depth(coords)

    # Make an array of time and depth
    this_frame_array = np.array([timestamp, this_frame_depth])

    # Produce depth map
    heatmap, residues_per_bin = make_heatmap_single_frame(coords, range_x, range_y)

    with counter.get_lock():
        counter.value += 1

    print(f'Processed frame {counter.value} with timestamp {timestamp} ps')

    return this_frame_array, heatmap, residues_per_bin

#### #### #### ####    
#### ALGORITHM ####
#### #### #### ####
if __name__ == '__main__':
    gro_list = glob.glob('*.gro')
    number_of_frames = len(gro_list)

    ## Analysis if single process ##
    if args.mp == False:
        print('Initialising')

        depth_array = np.zeros((0, 2))  # Initialise array of depth data
                                        # index 0 = timestamp (ps)
                                        # index 1 = depth (nm)

        hm_data = initialize_heatmap(bins=args.bins)

        print('Analysing gro files sequentially')

        for file in gro_list:
            # Extract data
            coords, timestamp, range_x, range_y = extract_data(file)

            # Calculate depth
            this_frame_depth = get_depth(coords)

            # Make an array of time and depth
            this_frame_array = np.array([timestamp, this_frame_depth])

            # Add this array to the existing array
            depth_array = np.vstack((depth_array, this_frame_array))

            # Produce depth map
            hm_data = make_heatmap(coords, hm_data, args.bins, range_x, range_y)

        heatmap = hm_data[0]
        residues_per_bin = hm_data[1]

    ## Analysis if multiprocess ##
    if args.mp == True:
        import multiprocessing, ctypes  # Import here to avoid import if not using    
        print('Analysing gro files in parallel')

        # Set up process counter
        counter = multiprocessing.Value(ctypes.c_int, 0)

        p_count = multiprocessing.cpu_count()

        with multiprocessing.Pool(processes=p_count, initializer=init_counter, initargs=(counter,)) as p:
            results = p.map(mp_wrapper, gro_list)

        print('Combining results from parallel processes')

        depth_array = np.vstack([item[0] for item in results])

        hm_total = sum([item[1] for item in results])

        residues_per_bin = sum([item[2] for item in results])

        np.seterr(divide='ignore', invalid='ignore')
        heatmap = hm_total / residues_per_bin
        np.seterr(divide='warn', invalid='warn')

    depth_array = depth_array[depth_array[:, 0].argsort()] # sort depths by timestamp

    ## Outputs ##
    print('Writing outputs')

    np.savetxt("depths.dat", depth_array)    # Export depths to file

    # Write mean depth to file
    with open('average_depth.txt', 'w+') as f:
        f.write(f"Depth of dome by mid-bilayer analysis = {str(np.mean(depth_array[:, 1]))} nm")

    # Output graph of change in depth over course of simulation
    depth_over_time = plt.plot(depth_array[:, 0], depth_array[:, 1], linewidth=0.5)
    plt.savefig("mid-bilayer_depth_over_time.svg")
    plt.close()

    # Output heatmap data
    np.savetxt("heatmap.dat", heatmap)
    np.savetxt("rescounts_heatmap.dat", residues_per_bin)

    # Plot heatmap data
    hmplot = plt.imshow(heatmap)
    plt.colorbar(hmplot)
    plt.savefig("heatmap.pdf")
    plt.savefig("heatmap.svg")
    plt.close()

    ctplot = plt.contourf(heatmap)
    plt.colorbar(ctplot)
    plt.savefig("contour.pdf")
    plt.savefig("contour.svg")
    plt.close()

    if args.time == True:
        run_time = datetime.datetime.now() - start_time
        with open('runtime.txt', 'w+') as f:
            f.write(str(run_time))