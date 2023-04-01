# !/usr/bin/python3
#
# Example code to read and plot the gps data.
#

import matplotlib.pyplot as plt
import numpy as np

USE_RTK = False # change as needed

# Accept a filepath to the CSV of interest and return Numpy array with data
def read_gps(dataset_date):
    filepath = f"dataset/{dataset_date}/gps.csv"
    if USE_RTK:
        filepath = f"dataset/{dataset_date}/gps_rtk.csv"
    
    gps = np.loadtxt(filepath, delimiter = ",")

    return gps

# Accepta a filepath to the CSV of interest and plot the FOG data
def plot_gps(filepath):
    gps = read_gps(filepath)

    # num_sats = gps[:, 2] # number of satetlite
    lat = gps[:, 3]
    lng = gps[:, 4]

    lat0 = lat[0]
    lng0 = lng[0]

    dLat = lat - lat0
    dLng = lng - lng0

    r = 6400000 # approx. radius of earth (m)
    x = r * np.cos(lat0) * np.sin(dLng)
    y = r * np.sin(dLat)

    plt.figure()
    plt.scatter(x, y, s=1, linewidth=0)
    plt.axis('equal')
    if USE_RTK:
        plt.title('GPS RTK Position')
    else:
        plt.title('Consumer GPS Position')

    plt.show()

if __name__ == "__main__":
    plot_gps("2013-04-05")