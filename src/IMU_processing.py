# Script to read and process Microstrain IMU data

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

file_name = 'ms25.csv'
#file_name_euler = '../Data/Sensor/2012-01-08/ms25_euler.csv'

def read_process_ms25(file_name):

    ms25 = np.loadtxt(file_name, delimiter = ",")
    # time in us 
    t = ms25[:, 0]
    # adjust to start at time 0 (for now)
    t = t-t[0]
    # convert from us to s
    t_sec = []
    for time in t:
        t_new = time*10**-6
        t_sec.append(t_new)
        
    # magnetic field (probs not needed)
    mag_x = ms25[:, 1]
    mag_y = ms25[:, 2]
    mag_z = ms25[:, 3]

    # acceleration
    accel_x_OG = ms25[:, 4]
    accel_y_OG = ms25[:, 5]
    accel_z_OG = ms25[:, 6]
    # apply rolling average to accelerations
    accel_x_df = pd.DataFrame(accel_x_OG)
    accel_x = accel_x_df.rolling(1000, min_periods=1).mean()
    accel_x = accel_x.to_numpy()
    accel_y_df = pd.DataFrame(accel_y_OG)
    accel_y = accel_y_df.rolling(1000, min_periods=1).mean()
    accel_y = accel_y.to_numpy()
    accel_z_df = pd.DataFrame(accel_z_OG)
    accel_z = accel_z_df.rolling(1000, min_periods=1).mean()
    accel_z = accel_z.to_numpy()
    
    # angular rotation rate 
    rot_r_OG = ms25[:, 7] # roll (about x)
    rot_p_OG = ms25[:, 8] # pitch (about y)
    rot_h_OG = ms25[:, 9] # heading (about z) -> this is the one we care about (I think)
    # apply rolling average to angular rotations
    rot_r_df = pd.DataFrame(rot_r_OG)
    rot_r = rot_r_df.rolling(1000, min_periods=1).mean()
    rot_r = rot_r.to_numpy()
    rot_p_df = pd.DataFrame(rot_p_OG)
    rot_p = rot_p_df.rolling(1000, min_periods=1).mean()
    rot_p = rot_p.to_numpy()
    rot_h_df = pd.DataFrame(rot_h_OG)
    rot_h = rot_h_df.rolling(1000, min_periods=1).mean()
    rot_h = rot_h.to_numpy()  

    # calculate velocity from accelerations (don't care about z-vel)
    vel_x = []
    vel_y = []
    prev_vel_x = 0; # assume initial velocity is 0
    prev_vel_y = 0; # assume initial velocity is 0
    prev_t = t_sec[0] - 0.02017 # sample rate is approx 49.6 Hz
    for i in range(0, len(t_sec)):
        dt = t_sec[i] - prev_t
        vel_x.append(accel_x[i]*dt + prev_vel_x)
        vel_y.append(accel_y[i]*dt  + prev_vel_y)
        prev_vel_x = vel_x[i]
        if i == 0:
            print(accel_x[i][0])
        prev_vel_y = vel_y[i]
        prev_t = t_sec[i]
        
    # calculate position from velocities
    pos_x = []
    pos_y = []
    prev_pos_x = 0; # assume initial position is at (0,0)
    prev_pos_y = 0; # assume initial velocity is at (0,0)
    prev_t = t_sec[0] - 0.02017 # sample rate is approx 49.6 Hz
    for i in range(0,len(t_sec)):
        dt = t_sec[i] - prev_t
        pos_x.append(vel_x[i]*dt + prev_pos_x)
        pos_y.append(vel_y[i]*dt  + prev_pos_y)
        prev_pos_x = pos_x[i]
        prev_pos_y = pos_y[i]
        prev_t = t_sec[i]
    
    estimate_IMU_noise(accel_y_OG, accel_y)
    
    # # # plot mag field 
    # # plt.figure()
    # # plt.plot(t, mag_x, 'r')
    # # plt.plot(t, mag_y, 'g')
    # # plt.plot(t, mag_z, 'b')
    # # plt.legend(['X', 'Y', 'Z'])
    # # plt.title('Magnetic Field')
    # # plt.xlabel('utime (us)')
    # # plt.ylabel('Gauss')

    # # plot actual accel vs smoothed accel
    plt.figure()
    # accel x
    plt.subplot(1, 2, 1)
    plt.plot(t_sec, accel_x_OG, 'b')
    plt.plot(t_sec, accel_x, 'r')
    plt.legend(['Original', 'Smooth'])
    plt.title('Acceleration x-dir')
    plt.xlabel('time (s)')
    plt.ylabel('m/s^2')    
    # accel y
    plt.subplot(1, 2, 2)
    plt.plot(t_sec, accel_y_OG, 'b')
    plt.plot(t_sec, accel_y, 'r')
    plt.legend(['Original', 'Smooth'])
    plt.title('Acceleration y-dir')
    plt.xlabel('time (s)')
    plt.ylabel('m/s^2')       
    # accel z
    # plt.subplot(1, 3, 3)
    # plt.plot(t_sec, accel_z_OG, 'b')
    # plt.plot(t_sec, accel_z, 'r')
    # plt.legend(['Original', 'Smooth'])
    # plt.title('Acceleration z-dir')
    # plt.xlabel('time (s)')
    # plt.ylabel('m/s^2')    
    # plt.show()
       
    # # plot accel, vel, pos
    # plt.figure()
    # # accel 
    # plt.subplot(1, 3, 1)
    # plt.plot(t_sec, accel_x, 'r')
    # plt.plot(t_sec, accel_y, 'g')
    # # plt.plot(t_sec, accel_z, 'b')
    # plt.legend(['X', 'Y'])
    # plt.title('Acceleration')
    # plt.xlabel('time (s)')
    # plt.ylabel('m/s^2')
    # # vel
    # plt.subplot(1, 3, 2)
    # plt.plot(t_sec, vel_x, 'r')
    # plt.plot(t_sec, vel_y, 'g')
    # # plt.plot(t, vel_z, 'b')'
    # plt.legend(['X', 'Y'])
    # plt.title('Velocity')
    # plt.xlabel('time (s)')
    # plt.ylabel('m/s')
    # # pos
    # plt.subplot(1, 3, 3)
    # plt.plot(t_sec, pos_x, 'r')
    # plt.plot(t_sec, pos_y, 'g')
    # #plt.plot(t_sec, pos_z, 'b')
    # plt.legend(['X', 'Y'])
    # plt.title('Position')
    # plt.xlabel('time (s)')
    # plt.ylabel('m')
    # plt.show()

    # plot angular rotation
    plt.figure()
    # plt.subplot(1, 3, 1)
    # plt.title('Angular Rotation Rate Roll (x)')
    # plt.plot(t, rot_r_OG, 'b')
    # plt.plot(t, rot_r, 'r')
    # plt.legend(['Original', 'Smooth'])
    # plt.subplot(1, 3, 2)
    # plt.title('Angular Rotation Rate Pitch (y)')
    # plt.plot(t, rot_p_OG, 'b')
    # plt.plot(t, rot_p, 'r')
    # plt.legend(['Original', 'Smooth'])
    # plt.subplot(1, 3, 3)
    plt.title('Angular Rotation Rate Yaw (z)')
    plt.plot(t_sec, rot_h_OG, 'b')
    plt.plot(t_sec, rot_h, 'r')
    plt.legend(['Original', 'Smooth'])
    plt.xlabel('time (s)')
    plt.ylabel('rad/s')
    plt.show()
    


def read_process_ms25_euler(file_name_euler):
    
    euler = np.loadtxt(file_name_euler, delimiter = ",")

    t = euler[:, 0]
    # adjust to start at time 0 (for now)
    t = t-t[0]
    # convert from us to s
    t_sec = []
    for time in t:
        t_new = time*10**-6
        t_sec.append(t_new)

    r_OG = euler[:, 1] # roll (x)
    p_OG = euler[:, 2] # pitch (y)
    h_OG = euler[:, 3] # heading (z)
    
    # apply rolling average to Euler angle measurements
    r_df = pd.DataFrame(r_OG)
    r = r_df.rolling(1000, min_periods=1).mean()
    r = r.to_numpy()
    p_df = pd.DataFrame(p_OG)
    p = p_df.rolling(1000, min_periods=1).mean()
    p = p.to_numpy()
    h_df = pd.DataFrame(h_OG)
    h = h_df.rolling(1000, min_periods=1).mean()
    h = h.to_numpy()

    # plot heading angle (smoothed)
    plt.figure()
    plt.plot(t_sec, h_OG, 'b')
    plt.plot(t_sec, h, 'r')
    plt.title('Euler Angle (Heading, z)')
    plt.legend(['Original', 'Smooth'])
    plt.xlabel('time (s)')
    plt.ylabel('rad')
    plt.show()

def estimate_IMU_noise(original_data, smooth_data):
    print("estimating noise")
    error = []
    print(len(original_data))
    for i in range(0, len(original_data)):
        error.append(original_data[i] - smooth_data[i])
        
    error = np.array(error)
    
    # norm fit of data
    (mu, sigma) = norm.fit(error)
    print(f'sigma = {sigma}')
    # plot
    plt.figure()
    n, bins, patches = plt.hist(error, bins=30, alpha=0.75, facecolor='blue')
    # x = norm.rvs(size=100000)
    # y = np.linspace(-4,4, 1000)
    # bin_width = (x.max() - x.min())/30
    # plt.plot(y, norm.pdf(y) * 100000 * 30)
    x = np.linspace(-3, 3, 100)
    y = norm.pdf(x, mu, sigma)
    plt.plot(x, y, 'r')
    plt.xlabel('IMU Accel Error (m/s^2)')
    plt.ylabel('Count')
    plt.title('IMU Error Histogram (y)')
    plt.grid(True)
    plt.show()
    
    
if __name__ == '__main__':
    read_process_ms25(file_name)
    read_process_ms25_euler(file_name_euler)