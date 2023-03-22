# !/usr/bin/python
#
# Example code to read and plot the odometry data.
#
# To call:
#
#   python read_odom.py odometry.csv
#

import sys
import matplotlib.pyplot as plt
import numpy as np

def main(args):

    if len(sys.argv) < 2:
        print 'Please specify odometry file'
        return 1

    odom = np.loadtxt(sys.argv[1], delimiter = ",")

    t = odom[:, 0]

    x = odom[:, 1]
    y = odom[:, 2]
    z = odom[:, 3]

    r = odom[:, 4]
    p = odom[:, 5]
    h = odom[:, 6]

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, 1, c=z, linewidth=0)
    plt.axis('equal')
    plt.title('Odometry position')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.plot(t, r, 'r')
    plt.plot(t, p, 'g')
    plt.plot(t, h, 'b')
    plt.legend(['Roll', 'Pitch', 'Heading'])
    plt.title('Odometry rph')
    plt.show()

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
