# !/usr/bin/python
#
# Example code to read and plot the microstrain data.
#
# To call:
#
#   python read_ms25.py ms25.csv
#

import sys
import matplotlib.pyplot as plt
import numpy as np

def main(args):

    if len(sys.argv) < 2:
        print 'Please specify microstrain file'
        return 1

    ms25 = np.loadtxt(sys.argv[1], delimiter = ",")

    t = ms25[:, 0]

    mag_x = ms25[:, 1]
    mag_y = ms25[:, 2]
    mag_z = ms25[:, 3]

    accel_x = ms25[:, 4]
    accel_y = ms25[:, 5]
    accel_z = ms25[:, 6]

    rot_r = ms25[:, 7]
    rot_p = ms25[:, 8]
    rot_h = ms25[:, 9]

    plt.figure()

    plt.subplot(1, 3, 1)
    plt.plot(t, mag_x, 'r')
    plt.plot(t, mag_y, 'g')
    plt.plot(t, mag_z, 'b')
    plt.legend(['X', 'Y', 'Z'])
    plt.title('Magnetic Field')
    plt.xlabel('utime (us)')
    plt.ylabel('Gauss')


    plt.subplot(1, 3, 2)
    plt.plot(t, accel_x, 'r')
    plt.plot(t, accel_y, 'g')
    plt.plot(t, accel_z, 'b')
    plt.legend(['X', 'Y', 'Z'])
    plt.title('Acceleration')
    plt.xlabel('utime (us)')
    plt.ylabel('m/s^2')

    plt.subplot(1, 3, 3)
    plt.plot(t, rot_r, 'r')
    plt.plot(t, rot_p, 'g')
    plt.plot(t, rot_h, 'b')
    plt.legend(['r', 'p', 'h'])
    plt.title('Angular Rotation Rate')
    plt.xlabel('utime (us)')
    plt.ylabel('rad/s')

    plt.show()

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
