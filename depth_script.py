import glob, multiprocessing, datetime, ctypes, functools, argparse, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


##  ##  ##  ##  ##
##   FUNCTIONS  ##
##  ##  ##  ##  ##


def get_timestamp(frame):
    with open(frame, 'r') as f:
        first_line = f.readline()
    pars = np.array(first_line.split(' '))
    pars = pars[pars != '']
    t_index = np.where(pars == 't=')[0]      
    if len(t_index != 0):
        timestamp = pars[t_index + 1].item()
        timestamp = float(timestamp)
    else:                                           
        timestamp = 0
    return timestamp


def get_coord_string_length(file):
    with open(file, 'r') as f:
        line_found = False
        while line_found == False:
            line = f.readline()
            resid = line[0:5]
            resname = line[5:10]
            atomname = line[10:15]
            atomid = line[15:20]
            try:
                int(resid)
                int(atomid)
            except:
                continue
            try:
                int(resname)
                int(atomname)
            except:
                decimal_index1 = line.find('.')
                decimal_index2 = line.find('.', decimal_index1 + 1)
                coord_length = decimal_index2 - decimal_index1
                return coord_length


def get_coords(file, coord_length):
    arr = np.genfromtxt(file, delimiter=[5, 5, 5, 5, coord_length, coord_length, coord_length], dtype='S')
    
    po4_arr = arr[arr[:, 2] == b'  PO4']

    resid_coords_arr = po4_arr[:, [0,4,5,6]].astype('f')

    return resid_coords_arr


def get_depth(leaflet): 
    # Get average z of top 10%
    top_cutoff = np.percentile(leaflet[:, 3], 90)
    top_ten_percent = leaflet[:, 3][leaflet[:, 3] > top_cutoff]
    average_top = np.mean(top_ten_percent)

    # Get z of lowest as this is fixed relative to protein, to which trajectory is fitted
    bottom = np.min(leaflet[:, 3])

    # Calculate depth
    depth = average_top - bottom

    return depth


def bin_cutoffs(coords, axis, bins):
    axis_index = {'x': 1,
                    'y': 2}

    i = axis_index[axis]

    ax_max = coords[:, i].max()
    range_min = ax_max * -0.25
    range_max = ax_max * 1.25
    range_step = (range_max - range_min) / (bins + 1)

    return np.arange(range_min, range_max, range_step)


def binner(arr, bin_cutoffs, axis):
    axis_index = {'x': 1,
                  'y': 2}

    i = axis_index[axis]

    bin_starts = bin_cutoffs[:-1]
    bin_ends = bin_cutoffs[1:]

    split = [arr[(arr[:, i] >= bin_start) 
             & (arr[:, i] < bin_end)] 
             for bin_start, bin_end in zip(bin_starts, bin_ends)]

    return split

def separate_leaflets(coords, timestamp):
    X = coords[:, [1,2,3]]
    db = DBSCAN(eps=2, min_samples=6).fit(X)
    cluster1_len = len(db.labels_[db.labels_ == 0])
    cluster2_len = len(db.labels_[db.labels_ == 1])
    cluster_diff = abs(cluster1_len - cluster2_len)
    biggest_cluster_len = max(cluster1_len, cluster2_len)
    main_clusters_size_unmatched = cluster_diff > (biggest_cluster_len * 0.05)
    less_than_2_main_clusters = len(set(db.labels_)) < 3
    n = 7
    while less_than_2_main_clusters or main_clusters_size_unmatched:
        db = DBSCAN(eps=2, min_samples=n).fit(X)
        cluster1_len = len(db.labels_[db.labels_ == 0])
        cluster2_len = len(db.labels_[db.labels_ == 1])
        cluster_diff = abs(cluster1_len - cluster2_len)
        biggest_cluster_len = max(cluster1_len, cluster2_len)
        main_clusters_size_unmatched = cluster_diff > (biggest_cluster_len * 0.05)
        less_than_2_main_clusters = len(set(db.labels_)) < 3
        n += 1
    leaflet1 = coords[db.labels_ == 0]
    leaflet2 = coords[db.labels_ == 1]
    outliers = coords[(db.labels_ != 0) & (db.labels_ != 1)]

    to_leaflet1 = []
    to_leaflet2 = []
    unassigned = []
    for item in outliers:
        diff1 = leaflet1[:, [1,2,3]] - item[1:]
        dist1 = (diff1[:,0]**2 + diff1[:,1]**2 + diff1[:,2]**2)**0.5
        diff2 = leaflet2[:, [1,2,3]] - item[1:]
        dist2 = (diff2[:,0]**2 + diff2[:,1]**2 + diff2[:,2]**2)**0.5
        for n in range(4,7):
            neighbours1 = sum(dist1 <= n)
            neighbours2 = sum(dist2 <= n)
            if neighbours1 != neighbours2:
                break
        if neighbours1 > neighbours2:
            to_leaflet1.append(item)
        elif neighbours2 > neighbours1:
            to_leaflet2.append(item)
        elif neighbours1 == neighbours2:
            unassigned.append(item)
    if to_leaflet1:
        leaflet1 = np.vstack((leaflet1, to_leaflet1))
    if to_leaflet2:
        leaflet2 = np.vstack((leaflet2, to_leaflet2))
    if unassigned:
        unassigned_array = np.empty((len(unassigned), len(unassigned[0]) +1))
        unassigned_array[:, -1] = timestamp
        unassigned_array[:, :-1] = np.vstack(unassigned)
        unassigned = unassigned_array
        for item in unassigned:
            print('Unassigned residue')
            print(f'resid: {int(item[0])}')
            print(f'time : {int(timestamp)} ps')
            print('This will be logged in \"unassigned_lipids.dat\"')
            print()
    else:
        unassigned = np.array(unassigned)

    return leaflet1, leaflet2, unassigned


def make_depthmap(coords, x_bin_cutoffs, y_bin_cutoffs, bins):
    x_binned = binner(coords, x_bin_cutoffs, 'x')
    xy_binned = [binner(j, y_bin_cutoffs, 'y') for j in x_binned]

    depthsum = np.empty((bins, bins))
    rescount = np.empty((bins, bins))

    for x in range(0,bins):
        for y in range(0, bins):
            depthsum[x, y] = np.sum(xy_binned[x][y][:, 3])
            rescount[x, y] = len(xy_binned[x][y])

    return depthsum, rescount


def main_loop(file, bins):
    coord_length = get_coord_string_length(file)
    coords = get_coords(file, coord_length)

    timestamp = get_timestamp(file)

    leaflet1, leaflet2, unassigned = separate_leaflets(coords, timestamp)

    function_output = [timestamp]

    for leaflet in (leaflet1, leaflet2):
        depth = get_depth(leaflet)

        x_cutoffs = bin_cutoffs(leaflet, 'x', bins)
        y_cutoffs = bin_cutoffs(leaflet, 'y', bins)
        depthsum, rescount = make_depthmap(leaflet, x_cutoffs, y_cutoffs, bins)

        function_output.append((depth, depthsum, rescount))

    function_output.append(unassigned)

    return function_output # list of  index 0 = timestamp, 
                           #                1 = (depth, depthsum, rescount),
                           #                2 = (depth, depthsum, rescount)
                           #                3 = unassigned_items


def chunks(list, chunksize):
    for i in range(0, len(list), chunksize):
        yield list[i: i+chunksize]


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    np.set_printoptions(suppress=True)
    plt.switch_backend('agg')

    ## ARGUMENT PARSER ##
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, 
        description='''
            Calculate depth of lipid bilayer in a trajectory.
            Trajectory should be in GRO format, one file per frame.
            Script should be run in a folder where the only GRO files are part of the trajectory.

            Outputs:
            depths.dat                  - timestep (ps) in first column, depth (nm) in second column
            average_depth.txt           - Text file stating the average depth over trajectory duration
            depth_over_time.svg         - plot of the data in depths.dat
            depthmap.dat                - data of depthmap of average depths
            depthmap_bin_rescounts.dat  - number of residues in each bin of depthmap.dat
            depthmap.pdf                - plot of data in depthmap.dat
            depthmap.svg                - plot of data in depthmap.dat
            contour.pdf                 - contour map of data in depthmap.dat
            contour.svg                 - contour map of data in depthmap.dat
            unassigned_lipids.dat       - columns - 0: resid, 1: x, 2: y, 3: z, 4: time when not assigned
            running_time.txt            - performance metrics''')
    p.add_argument('--bins',
                '-b',
                help='''Number of bins in X and Y axis for output depth map. 
                        It is assumed that the X and Y dimensions of the 
                        simulation box are the same. Default is 75.''',
                default=75,
                type=int)
    p.add_argument('--chunksize',
                '-c',
                help='''Number of frames processed before outputs are
                        consolidated to avoid overloading system. Can be reduced 
                        for very large systems. Default is 100.''',
                type=int,
                default=100)
    p.add_argument('--nomp',
                dest='nomp',
                action='store_true',
                help='Turn off multiprocessing.')
    args = p.parse_args()


    ##  ##  ##  ##  ##
    ##  ALGORITHM   ##
    ##  ##  ##  ##  ##

    gro_list = glob.glob('*gro')

    counter = 0
    n_frames = len(gro_list)
    time_list = []
    depth_list1 = []
    depthsum1 = np.zeros((args.bins, args.bins))
    rescount1 = np.zeros((args.bins, args.bins))
    depth_list2 = []
    depthsum2 = np.zeros((args.bins, args.bins))
    rescount2 = np.zeros((args.bins, args.bins))
    unassigned = []

    if not args.nomp:
        # Fix start_time and n_frames in main loop as they are constant
        pain_loop = functools.partial(main_loop, bins=args.bins)

        # Split gro_list into chunks for analysis
        chunk_generator = chunks(gro_list, args.chunksize)
        for chunk in chunk_generator:
            with multiprocessing.Pool() as p:
                results = p.map(pain_loop, chunk)

            time_list += [item[0] for item in results]
            depth_list1 += [item[1][0] for item in results]
            depthsum1 += sum(item[1][1] for item in results)
            rescount1 += sum(item[1][2] for item in results)
            depth_list2 += [item[2][0] for item in results]
            depthsum2 += sum(item[2][1] for item in results)
            rescount2 += sum(item[2][2] for item in results)
            unassigned += [item[3] for item in results if item[3].any()]

            counter += 1
            if len(chunk) == args.chunksize:
                frames_analysed = counter * args.chunksize
                percent_progress = frames_analysed / n_frames * 100
                print(f'{round(percent_progress, 2)}% complete')
                elapsed_time = datetime.datetime.now() - start_time
                print(f'{frames_analysed}/{n_frames} frames analysed in {elapsed_time}')
                print(f'Average {elapsed_time/frames_analysed} per frame')
                remaining_time = elapsed_time / frames_analysed * (n_frames - frames_analysed)
                print(f'Estimated time remaining: {remaining_time}')
                print()
            else:
                print('100% complete')
                print(f'{n_frames}/{n_frames} frames analysed in {elapsed_time}')
                print(f'Average {elapsed_time/n_frames} per frame')

    if args.nomp:
        for file in gro_list:
            results = main_loop(file, args.bins)
            time_list.append(results[0])
            depth_list1.append(results[1][0])
            depthsum1 += results[1][1]
            rescount1 += results[1][2]
            depth_list2.append(results[2][0])
            depthsum2 += results[2][1]
            rescount2 += results[2][2]
            if results[3]:
                unassigned.append(results[3])

            counter += 1
            percent_progress = counter / n_frames * 100
            print(f'{round(percent_progress, 2)}% complete')
            elapsed_time = datetime.datetime.now() - start_time
            print(f'{counter}/{n_frames} analysed in {elapsed_time}')
            print(f'Average {counter/frames_analysed} per frame')
            remaining_time = elapsed_time / counter * (n_frames - counter)
            print(f'Estimated time remaining: {remaining_time}')
            print()

    np.seterr(divide='ignore', invalid='ignore')
    depthmap1 = depthsum1 / rescount1
    depthmap1 = depthmap1 - np.nanmin(depthmap1)
    depthmap2 = depthsum2 / rescount2
    depthmap2 = depthmap2 - np.nanmin(depthmap2)
    np.seterr(divide='warn', invalid='warn')

    ## Outputs ##

    # Write csv of depth over time
    time_depth_array = np.column_stack((time_list, depth_list1, depth_list2))
    np.savetxt('depths.dat', time_depth_array)

    # Output depthmap data
    np.savetxt("depthmap1.dat", depthmap1)
    np.savetxt("depthmap1_bin_rescounts.dat", rescount1)
    np.savetxt("depthmap2.dat", depthmap2)
    np.savetxt("depthmap2_bin_rescounts.dat", rescount2)

    # Output graph of change in depth over course of simulation
    plt.plot(time_list, depth_list1, linewidth=0.5)
    plt.plot(time_list, depth_list2, linewidth=0.5)
    plt.savefig("depth_over_time.svg")
    plt.close()

    # Plot depthmap data
    hmplot = plt.imshow(depthmap1)
    plt.colorbar(hmplot)
    plt.savefig("depthmap1.pdf")
    plt.savefig("depthmap1.svg")
    plt.close()

    ctplot = plt.contourf(depthmap1)
    plt.colorbar(ctplot)
    plt.savefig("contour1.pdf")
    plt.savefig("contour1.svg")
    plt.close()

    hmplot = plt.imshow(depthmap2)
    plt.colorbar(hmplot)
    plt.savefig("depthmap2.pdf")
    plt.savefig("depthmap2.svg")
    plt.close()

    ctplot = plt.contourf(depthmap2)
    plt.colorbar(ctplot)
    plt.savefig("contour2.pdf")
    plt.savefig("contour2.svg")
    plt.close()

    if unassigned:
        unassigned = np.vstack(unassigned)
        np.savetxt('unassigned_lipids.dat', unassigned)

    run_time = datetime.datetime.now() - start_time
    with open('running_time.txt', 'w+') as f:
        f.write(f'{n_frames} files processed over {run_time}\n')
        f.write(f'Average {run_time/n_frames} per frame')
        