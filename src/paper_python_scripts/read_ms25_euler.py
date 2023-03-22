# !/usr/bin/python
#
# Example code to read and plot the microstrain euler angles data.
#
# To call:
#
#   python read_ms25_euler.py ms25_euler.csv
#

import sys
import matplotlib.pyplot as plt
import numpy as np

def main(args):

    if len(sys.argv) < 2:
        print 'Please specify microstrain file'
        return 1

    euler = np.loadtxt(sys.argv[1], delimiter = ",")

    t = euler[:, 0]

    r = euler[:, 1]
    p = euler[:, 2]
    h = euler[:, 3]

    plt.figure()

    plt.plot(t, r, 'r')
    plt.plot(t, p, 'g')
    plt.plot(t, h, 'b')
    plt.legend(['r', 'p', 'h'])
    plt.title('Euler Angles')
    plt.xlabel('utime (us)')
    plt.ylabel('rad')

    plt.show()

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
