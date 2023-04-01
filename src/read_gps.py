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
    
    
    
    #TODO: make times relative
    # Calculate Hz of data
    t = gps[:,0]
    length = t[-1] - t[0]
    average_timestep = length/len(t)/1000000
    hz = 1/average_timestep
    print(f"Hz: {hz}") #  5.841021369762537
    
    #TODO: filter out rows where fix_mode<2
    fix_mode = gps[:,1] # >=2 means valid data
    
    return gps

# Accepta a filepath to the CSV of interest and plot the FOG data
def plot_gps(filepath):
    gps = read_gps(filepath)

    t = gps[:,0]
    lat = gps[:, 3]
    lng = gps[:, 4]

    # Convert GPS coordinates to linearized local frame
    lat0 = lat[0] # in radians
    lng0 = lng[0] # in radians

    dLat = lat - lat0
    dLng = lng - lng0

    r = 6400000 # approx. radius of earth (m)
    x = r * np.sin(dLat) # North
    y = r * np.cos(lat0) * np.sin(dLng) # East

    plt.figure()
    plt.scatter(y, x, s=1, linewidth=0)
    plt.axis('equal')
    if USE_RTK:
        plt.title('GPS RTK Position')
    else:
        plt.title('Consumer GPS Position')

    plt.show()
    #TODO: export to kml file

if __name__ == "__main__":
    plot_gps("2012-01-15")