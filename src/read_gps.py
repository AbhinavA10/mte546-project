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

    ### GPS -> local Frame ###
    lat0 = lat[0] # in radians
    lng0 = lng[0] # in radians
    # Compute radii of Earth at origin of linearization
    re = 6378135 # Earth Equatorial Radius [m]
    rp = 6356750 # Earth Polar Radius [m]
    r_ns = pow(re*rp,2)/pow(pow(re*np.cos(lat0),2)+pow(rp*np.sin(lat0),2),3/2)
    r_ew = pow(re,2)/pow(pow(re*np.cos(lat0),2)+pow(rp*np.sin(lat0),2),1/2)
    # Convert GPS coordinates to linearized local frame
    x = np.sin(lat - lat0) * r_ns # North
    y = np.sin(lng - lng0) * r_ew * np.cos(lat0) # East

    plt.figure()
    plt.scatter(y, x, s=1, linewidth=0)
    plt.axis('equal')
    if USE_RTK:
        plt.title('GPS RTK Position')
    else:
        plt.title('Consumer GPS Position')

    plt.show()
    #TODO: export to kml file

    ### Local frame --> GPS
    la = np.arcsin(x/r_ns) + lat0
    lon = np.arcsin(y/(r_ew*np.cos(lat0))) + lng0
    la = np.rad2deg(la) # Latitude, in degrees
    lon = np.rad2deg(lon) # Longitude, in degrees


if __name__ == "__main__":
    plot_gps("2012-01-15")