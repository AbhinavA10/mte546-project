# !/usr/bin/python3

# General imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

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

# accept a filepath to GPS csv of interest (either consumer or RTK) and provide access to it
def read_gps(filepath):
    gps = np.loadtxt(filepath, delimiter = ",")

    return gps
    
# Accepta a filepath to the CSV of interest and plot the GPS data (RTK or consumer)
# use_RTK is a bool that specifies whether you're reading consumer or RTK GPS file
def plot_gps(filepath, use_RTK):
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
    if use_RTK:
        plt.title('GPS RTK Position')
    else:
        plt.title('Consumer GPS Position')

    plt.show()   

# accept a file path to IMU csv of interest and provide access to it
def read_IMU(filepath):
    ms25 = np.loadtxt(filepath, delimiter = ",")
    
    return ms25

# plot desired IMU data
def plot_IMU(filepath):
    ms25 = read_IMU(filepath)
    # time in us 
    t = ms25[:, 0]

    # acceleration
    accel_x = ms25[:, 4] # m/s^2
    accel_y = ms25[:, 5] # m/s^2
    
    # angular rotation rate 
    rot_yaw = ms25[:, 9] # rate of change of yaw (about z) in rad/s
    
    plt.figure()
    # accel x
    plt.subplot(1, 3, 1)
    plt.plot(t, accel_x, 'b')
    plt.title('Acceleration x-dir')
    plt.xlabel('time (us)')
    plt.ylabel('m/s^2')    
    # accel y
    plt.subplot(1, 3, 2)
    plt.plot(t, accel_y, 'b')
    plt.title('Acceleration y-dir')
    plt.xlabel('time (us)')
    plt.ylabel('m/s^2')        
    # yaw
    plt.subplot(1, 3, 3)
    plt.title('Angular Rotation Rate Heading (z)')
    plt.plot(t, rot_yaw, 'b')
    plt.xlabel('time (us)')
    plt.ylabel('rad/s')
    plt.show()

# accept a file path to wheel velocity csv of interest and provide access to it
def read_wheel_vel(filepath):
    wheel_vel = np.loadtxt(filepath, delimiter = ",")
    
    return wheel_vel

# plot wheel velocities
def plot_wheel_vel(filepath):
    wheel_vel = read_wheel_vel(filepath)
    # time in us 
    t = wheel_vel[:, 0]

    # left and right wheel velocities in m/s
    left_wheel_vel = wheel_vel[:, 1] 
    right_wheel_vel = wheel_vel[:, 2] 
    
    plt.figure()
    # left wheel velocity
    plt.subplot(1, 2, 1)
    plt.plot(t, left_wheel_vel, 'b')
    plt.title('Left Wheel Velocity')
    plt.xlabel('time (us)')
    plt.ylabel('m/s')    
    # right wheel velocity
    plt.subplot(1, 2, 2)
    plt.plot(t, right_wheel_vel, 'b')
    plt.title('Right Wheel Velocity')
    plt.xlabel('time (us)')
    plt.ylabel('m/s')
    plt.show()        

# update measurement models
def calc_z_hat_wheel(vel_x, vel_y, yaw_dot, dt):
    v_c = math.sqrt(vel_x**2 + vel_y**2)
    v_right = v_c + (dt*yaw_dot)/2
    v_left = v_c - (dt*yaw_dot)/2
    
    z_wheel_vel = [v_left, v_right]
    return z_wheel_vel


if __name__ == "__main__":
    #plot_FOG("dataset/2013-04-5_sen")
    #plot_groundtruth("dataset/groundtruth_2013-04-05.csv", "dataset/cov_2013-04-05.csv")
    #plot_IMU('ms25.csv')
    #plot_wheel_vel('wheels.csv')