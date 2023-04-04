# Example code to read and plot the gps data.

import matplotlib.pyplot as plt
import numpy as np
import utils

# Accept a filepath to the CSV of interest and return Numpy array with data
def read_gps(dataset_date, use_rtk=False):
    """Read GPS Data
    Parameters: dataset date
    Returns: np.ndarray([timestamp, x, y])
    """
    filepath = f"dataset/{dataset_date}/gps.csv"
    if use_rtk:
        filepath = f"dataset/{dataset_date}/gps_rtk.csv"
    
    gps = np.loadtxt(filepath, delimiter = ",")
    # Preprocessing
    gps = np.delete(gps, np.where((gps[:,1] < 3 ))[0], axis=0) # filter out rows where fix_mode < 3 (invalid data)
    # perform conversion from lat/lon to local frame
    t = gps[:,0]
    lat = gps[:, 3]
    lng = gps[:, 4]
    # t = t-t[0] # make timestamps relative
    t = t/1000000
    if use_rtk:
        utils.calculate_hz("GPS RTK", t) # 2.5 Hz
    else:
        utils.calculate_hz("GPS", t) # 4 Hz
    
    x,y = utils.gps_to_local_coord(lat, lng) #North,East

    x = x + 76.50582406697139 # Offset to adjust with ground truth initial position
    y = y + 108.31373031919006 # Offset to adjust with ground truth initial position

    gps_data = np.array([])
    gps_data = np.vstack((t, x, y)).T

    # Filter data to campus map area
    gps_data = np.delete(gps_data, np.where((gps_data[:,1] < -350 ) | (gps_data[:,1] > 120 ))[0], axis=0) # x
    gps_data = np.delete(gps_data, np.where((gps_data[:,2] < -750 ) | (gps_data[:,2] > 150 ))[0], axis=0) # y
    
    # Manually filter poor GPS readings - tuned for 2013-04-05 data
    if dataset_date == "2013-04-05":
        gps_data = np.delete(gps_data, slice(2350,2650), axis=0)
        gps_data = np.delete(gps_data, slice(13800,14550), axis=0)

    # timestamp | x | y
    return gps_data

# Accepta a filepath to the CSV of interest and plot the GPS data
def plot_gps(filepath, use_rtk=False):
    gps = read_gps(filepath)
    # print(gps.shape)
    # gps = gps[:14550,:] # Use this for tuning bad gps to filter
    t = gps[:,0]
    x = gps[:,1]
    y = gps[:,2]

    plt.figure()
    # plt.scatter(y, x, s=1, linewidth=0) # plot flipped since North,East
    plt.scatter(y, x, c=t, s=5, linewidth=0) # Plot color over time
    plt.colorbar()
    plt.xlabel('East [m]')
    plt.ylabel('North [m]')
    plt.axis('equal')
    if use_rtk:
        plt.title('GPS RTK Position')
    else:
        plt.title('Consumer GPS Position')
    plt.show()    

if __name__ == "__main__":
    plot_gps("2013-04-05", use_rtk=False)