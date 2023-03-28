# !/usr/bin/python3

# General imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Imports for reading ground truth
import sys
import scipy.interpolate

# Accept a filepath to the CSV of interest and return Numpy array with data
def read_FOG(filepath):
    data = pd.read_csv("dataset/2013-04-05_sen/kvh.csv", header=None)
    data = data - [data.iloc[0,0], 0]
    data[2] = data[1].rolling(1000).mean()
    return(data.to_numpy())

# Accept a filepath to the CSV of interest and plot the FOG data
def plot_FOG(filepath):
    array = read_FOG(filepath)

    plt.figure()
    plt.plot(array[:,0], array[:,1])
    plt.plot(array[:,0], array[:,2])
    
    plt.yticks(np.arange(0, 7, step=np.pi/8), fontsize=15)
    plt.xticks(fontsize=15)

    plt.legend(["Original Angle Data","Angle Data Moving Average"], fontsize=20)
    plt.title("FOG Angle versus Time", fontsize=30)
    plt.xlabel("Time [s]", fontsize=20)
    plt.ylabel("Angle [rads]", fontsize=20)

    plt.grid()
    plt.show()
    
# Accept a a filepath to the CSVs of interest (ground truth and covariance) and provide access to it
def read_groundtruth(gt_filepath, cov_filepath):

    gt  = np.loadtxt(gt_filepath, delimiter = ",")
    cov = np.loadtxt(cov_filepath, delimiter = ",")
    t_cov = cov[:, 0]

    # Note: Interpolation is not needed, this is done as a convience
    interp = scipy.interpolate.interp1d(gt[:, 0], gt[:, 1:], kind='nearest', axis=0, fill_value="extrapolate")
    pose_gt = interp(t_cov)

    # NED (North, East Down)
    x = pose_gt[:, 0]
    y = pose_gt[:, 1]
    z = pose_gt[:, 2]
    r = pose_gt[:, 3]
    p = pose_gt[:, 4]
    h = pose_gt[:, 5]

    return pose_gt

# Accept a filepath to the CSVs of interest (ground truth and covariance) and plot it
def plot_groundtruth(gt_filepath, cov_filepath):
    pose = read_groundtruth(gt_filepath, cov_filepath)

    x = pose[:, 0]
    y = pose[:, 1]
    z = pose[:, 2]
    r = pose[:, 3]
    p = pose[:, 4]
    h = pose[:, 5]

    plt.figure()
    plt.scatter(y, x, 1, c=-z, linewidth=0)
    plt.axis('equal')
    plt.title('Ground Truth Position of Nodes in SLAM Graph')
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.colorbar()
    plt.grid()

    plt.show()


# NEED FUNCTIONS TO READ IN REQUIRED DATA

# 1. GPS
# 2. GPS-RTK
# 3. WHEEL ODOMETRY
# 4. IMU


if __name__ == "__main__":
    plot_FOG("dataset/2013-04-5_sen")

    plot_groundtruth("dataset/groundtruth_2013-04-05.csv", "dataset/cov_2013-04-05.csv")