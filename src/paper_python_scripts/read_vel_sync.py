# !/usr/bin/python
#
# Example code to read a velodyne_sync/[utime].bin file
# Plots the point cloud using matplotlib. Also converts
# to a CSV if desired.
#
# To call:
#
#   python read_vel_sync.py velodyne.bin [out.csv]
#

import sys
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def convert(x_s, y_s, z_s):

    scaling = 0.005 # 5 mm
    offset = -100.0

    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset

    return x, y, z

def main(args):

    if len(sys.argv) < 2:
        print 'Please specify velodyne file'
        return 1

    f_bin = open(sys.argv[1], "r")

    if len(sys.argv) >= 3:
        print 'Writing to ', sys.argv[2]
        f_csv = open(sys.argv[2], "w")
    else:
        f_csv = None

    hits = []

    while True:

        x_str = f_bin.read(2)
        if x_str == '': # eof
            break

        x = struct.unpack('<H', x_str)[0]
        y = struct.unpack('<H', f_bin.read(2))[0]
        z = struct.unpack('<H', f_bin.read(2))[0]
        i = struct.unpack('B', f_bin.read(1))[0]
        l = struct.unpack('B', f_bin.read(1))[0]

        x, y, z = convert(x, y, z)

        s = "%5.3f, %5.3f, %5.3f, %d, %d" % (x, y, z, i, l)

        if f_csv:
            f_csv.write('%s\n' % s)

        hits += [[x, y, z]]

    f_bin.close()

    if f_csv:
        f_csv.close()

    hits = np.asarray(hits)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(hits[:, 0], hits[:, 1], -hits[:, 2], c=-hits[:, 2], s=5, linewidths=0)
    plt.show()

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
