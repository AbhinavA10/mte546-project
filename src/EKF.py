# !/usr/bin/python3

# General imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from IPython.display import display

# Imports for sympy
import sympy as sp
import sys
import mpmath
sys.modules['sympy.mpmath'] = mpmath

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

# wrap yaw measurements to [-pi, pi]. Accepts an angle measurement in radians and returns an angle measurement in radians
def wraptopi(x):
    if x > np.pi:
        x = x - (np.floor(x / (2 * np.pi)) + 1) * 2 * np.pi
    elif x < -np.pi:
        x = x + (np.floor(x / (-2 * np.pi)) + 1) * 2 * np.pi
    return x

def motion_jacobian(ax, ay, psidot, delta_t):
    x_k, y_k, x_dot_k, y_dot_k, psi_k = sp.symbols('x_k, y_k, x_dot_k, y_dot_k, psi_k', real=True)

    f1 = x_k     + x_dot_k*delta_t + 0.5*ax*delta_t^2
    f2 = y_k     + y_dot_k*delta_t + 0.5*ay*delta_t^2
    f3 = x_dot_k + ax*delta_t
    f4 = y_dot_k + ay*delta_t
    f5 = psi_k   + psidot*delta_t

    f = sp.Matrix([f1, f2, f3, f4, f5]).jacobian([x_k, y_k, x_dot_k, y_dot_k, psi_k])

    return f

# PLEASE ENSURE THAT ALL STATES AND INPUTS TO THE SYSTEM ARE IN THE GLOBAL FRAME OF REFERENCE :-)
# PLEASE MAKE USE OF WRAPTOPI() WHEN SUPPLYING ROBOT HEADING OR ROBOT ANGULAR VELOCITY

# CURRENT STATE MODEL IS: X = [x, y, xdot, ydot, psi, psidot] 
# CURRENT INPUT MODEL IS: U = [ax, ay, yaw_dot]

if __name__ == "__main__":

    ######################### 0. INITIALIZE IMPORTANT VARIABLES #################################################################

    # IMPORTANT TO-DO: NEED TO INITIALIZE Q AND R COVARIANCE MATRICES

    Q = np.diag([]) # input noise covariance
    R = np.diag([])  # measurement noise covariance

    # TO-DO: NEED TO DEFINE GROUND TRUTH ARRAYS & ARRAY OF MEASUREMENTS
    x_true = ...
    y_true = ...
    measurements = ...

    # TO-DO: NEED TO DEFINE LENGTH OF STATES WE'LL BE ESTIMATING
    N = len(measurements)
    x_est = np.zeros([N, 6])  # estimated states, x, y, and theta
    P_est = np.zeros([N, 6, 6])  # state covariance matrices

    # TO-DO: NEED TO DEFINE INTIIAL STATE AND COVARIANCE MATRIX
    x_est[0] = np.array([]) # initial state
    P_est[0] = np.diag([]) # initial state covariance

    ################################ 1. MAIN FILTER LOOP ##########################################################################

    # TO-DO: NEED TO DEFINE ARRAY OF LINEARLY SPACED TIME INCREMENTS
    #        THIS WILL DEFINE DELTA T...
    
    t = []

    for k in range(1, len(t)):  # Start at 1 because we have initial prediction from ground truth.

        delta_t = t[k] - t[k - 1]  # time step (difference between timestamps)

        # TO-DO: I THINK WE HAVE COME TO THE CONSENSUS THAT WE'RE NOT USING ADDITIVE NOISE BUT IF WE ARE WE
        #        NEED TO DEFINE IT HERE WITH APPROPRIATE DIMENSIONING. MULTIVARIATE NORMAL WITH A SCALED
        #        IDENTITY COVARIANCE RETURNS AN Nx1 VECTOR OF 0-MEAN GAUSSIAN NOISE DEPENDING ON DIAG. OF R MATRIX
            
        w_v_k = np.random.multivariate_normal([0,0], R)

        # 1-1. INITIAL UPDATE OF THE ROBOT STATE USING MEASUREMENTS (IMU, ETC.) 
        x_check = x_est[k-1] + delta_t*...

        # 1-2 Linearize Motion Model
        # Compute the Jacobian of f w.r.t. the last state.

        # TO-DO: DETERMINE MOTION MODEL EQUATIONS
        # Sam's note: where to get ax, ay, and psidot? Does it even matter if it's removed by the Jacobian?
        x_k, y_k, x_dot_k, y_dot_k, psi_k = sp.symbols('x_k, y_k, x_dot_k, y_dot_k, psi_k', real=True)

        f1 = x_k     + x_dot_k*delta_t + 0.5*ax*delta_t^2
        f2 = y_k     + y_dot_k*delta_t + 0.5*ay*delta_t^2
        f3 = x_dot_k + ax*delta_t
        f4 = y_dot_k + ay*delta_t
        f5 = psi_k   + psidot*delta_t

        f = sp.Matrix([f1, f2, f3, f4, f5]).jacobian([x_k, y_k, x_dot_k, y_dot_k, psi_k])
        F = np.array(f.subs([(x_k,      x_est[k-1,0]),
                             (y_k,      x_est[k-1,1]),
                             (x_dot_k,  x_est[k-1,2]),
                             (y_dot_k,  x_est[k-1,3]),
                             (psi_k,    x_est[k-1,4]),
                            ])).astype(np.float64)
        

        # IGNORE FOR NOW.....
        # Compute the Jacobian w.r.t. the noise variables.
        # n_v, n_w = sp.symbols('n_v n_w', real=True)
        
        # l1 = x_k     + delta_t*(v[k]  + n_v)*sp.cos(theta_k)
        # l2 = y_k     + delta_t*(v[k]  + n_v)*sp.sin(theta_k)
        # l3 = theta_k + delta_t*(om[k] + n_w)

        # l_small = sp.Matrix([l1, l2, l3]).jacobian([n_v, n_w])

        # L = np.array(l_small.subs([(x_k,     x_est[k-1,0]),
        #                         (y_k,     x_est[k-1,1]),
        #                         (theta_k, x_est[k-1,2]),
        #                         (n_v,     w_v_k[0]),
        #                         (n_w,     w_v_k[1])])).astype(np.float64)
        
        # 2. Propagate uncertainty by updating the covariance
        P_check = np.matmul(np.matmul(F,P_est[k-1]),np.transpose(F)) + np.matmul(np.matmul(L,Q),np.transpose(L))

        # 3. Update state estimate using available landmark measurements r[k], b[k].
        # for i in range(len(r[k])):
        #     x_check, P_check = measurement_update(l[i], r[k,i], b[k,i], P_check, x_check)

        # Set final state predictions for this kth timestep.
        x_est[k] = x_check
        P_est[k] = P_check
