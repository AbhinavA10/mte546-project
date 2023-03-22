# !/usr/bin/python
#
# Example code to read and plot the gps data.
#
# To call:
#
#   python read_gps.py gps.csv
#

import sys
import matplotlib.pyplot as plt
import numpy as np

def main(args):

    if len(sys.argv) < 2:
        print 'Please specify gps file'
        return 1

    gps = np.loadtxt(sys.argv[1], delimiter = ",")

    num_sats = gps[:, 2]
    lat = gps[:, 3]
    lng = gps[:, 4]
    alt = gps[:, 5]

    lat0 = lat[0]
    lng0 = lng[0]

    dLat = lat - lat0
    dLng = lng - lng0

    r = 6400000 # approx. radius of earth (m)
    x = r * np.cos(lat0) * np.sin(dLng)
    y = r * np.sin(dLat)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, 1, c=alt, linewidth=0)
    plt.axis('equal')
    plt.title('By altitude')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.scatter(x, y, c=num_sats, linewidth=0)
    plt.axis('equal')
    plt.title('By number of satellites')
    plt.colorbar()

    plt.show()

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
