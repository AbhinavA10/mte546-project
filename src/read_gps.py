# Example code to read and plot the gps data.

import matplotlib.pyplot as plt
import numpy as np
import utils

USE_RTK = False # change as needed

# Accept a filepath to the CSV of interest and return Numpy array with data
def read_gps(dataset_date):
    """Read GPS Data
    Parameters: dataset date
    Returns: np.ndarray([timestamp, x, y])
    """
    filepath = f"dataset/{dataset_date}/gps.csv"
    if USE_RTK:
        filepath = f"dataset/{dataset_date}/gps_rtk.csv"
    
    gps = np.loadtxt(filepath, delimiter = ",")
    
    if USE_RTK:
        utils.calculate_hz("GPS RTK", gps[:,0]) # 2.5 Hz
    else:
        utils.calculate_hz("GPS", gps[:,0]) # 6 Hz
    
    # Preprocessing
    gps = np.delete(gps, np.where((gps[:,1] < 2 ))[0], axis=0) # filter out rows where fix_mode<2 (invalid data)
    # perform conversion from lat/lon to local frame
    t = gps[:,0]
    lat = gps[:, 3]
    lng = gps[:, 4]
    t = t-t[0] # make timestamps relative
    t = t/1000000
    x,y = utils.gps_to_local_coord(lat, lng) #North,East
    gps_data = np.array([])
    gps_data = np.vstack((t, x, y)).T
    # Filter data to campus map area
    gps_data = np.delete(gps_data, np.where((gps_data[:,1] < -450 ) | (gps_data[:,1] > 100 ))[0], axis=0) # x
    gps_data = np.delete(gps_data, np.where((gps_data[:,2] < -900 ) | (gps_data[:,2] > 50 ))[0], axis=0) # y

    # timestamp | x | y
    return gps_data

# Accepta a filepath to the CSV of interest and plot the GPS data
def plot_gps(filepath):
    gps = read_gps(filepath)
    t = gps[:,0]
    x = gps[:,1]
    y = gps[:,2]

    plt.figure()
    plt.scatter(y, x, s=1, linewidth=0) # plot flipped since North,East
    plt.xlabel('East [m]')
    plt.ylabel('North [m]')
    plt.axis('equal')
    if USE_RTK:
        plt.title('GPS RTK Position')
    else:
        plt.title('Consumer GPS Position')
    plt.show()    

if __name__ == "__main__":
    plot_gps("2013-04-05")