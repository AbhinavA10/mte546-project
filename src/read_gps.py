# Example code to read and plot the gps data.

import matplotlib.pyplot as plt
import numpy as np
import utils

USE_RTK = False # change as needed

# Accept a filepath to the CSV of interest and return Numpy array with data
def read_gps(dataset_date):
    filepath = f"dataset/{dataset_date}/gps.csv"
    if USE_RTK:
        filepath = f"dataset/{dataset_date}/gps_rtk.csv"
    
    gps = np.loadtxt(filepath, delimiter = ",")
    
    if USE_RTK:
        utils.calculate_hz("GPS RTK", gps[:,0]) # 2.5 Hz
    else:
        utils.calculate_hz("GPS", gps[:,0]) # 5.85 Hz
    
    # Preprocessing
    #TODO: make timestamps relative
    #TODO: filter out rows where fix_mode<2
    fix_mode = gps[:,1] # >=2 means valid data
    #TODO: filter out position data outside of predefined range
    #TODO: perform conversion from lat/lon to local frame here, and make new minimal df to return
    
    return gps

# Accepta a filepath to the CSV of interest and plot the FOG data
def plot_gps(filepath):
    gps = read_gps(filepath)

    t = gps[:,0]
    lat = gps[:, 3]
    lng = gps[:, 4]

    x,y = utils.gps_to_local_coord(lat, lng) #North,East

    plt.figure()
    plt.scatter(y, x, s=1, linewidth=0) # plot flipped since North,East
    plt.axis('equal')
    if USE_RTK:
        plt.title('GPS RTK Position')
    else:
        plt.title('Consumer GPS Position')
    plt.show()    

if __name__ == "__main__":
    plot_gps("2013-04-05")