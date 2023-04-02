# !/usr/bin/python3

# General imports
import scipy.interpolate
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

# Accept a filepath to the CSV of interest and return Numpy array with data


def read_FOG(filepath):
    data = pd.read_csv("dataset/2013-04-05_sen/kvh.csv", header=None)
    data = data - [data.iloc[0, 0], 0]
    data[2] = data[1].rolling(1000).mean()
    return (data.to_numpy())

# Accept a filepath to the CSV of interest and plot the FOG data


def plot_FOG(filepath):
    array = read_FOG(filepath)

    plt.figure()
    plt.plot(array[:, 0], array[:, 1])
    plt.plot(array[:, 0], array[:, 2])

    plt.yticks(np.arange(0, 7, step=np.pi/8), fontsize=15)
    plt.xticks(fontsize=15)

    plt.legend(["Original Angle Data", "Angle Data Moving Average"], fontsize=20)
    plt.title("FOG Angle versus Time", fontsize=30)
    plt.xlabel("Time [s]", fontsize=20)
    plt.ylabel("Angle [rads]", fontsize=20)

    plt.grid()
    plt.show()

# Accept a a filepath to the CSVs of interest (ground truth and covariance) and provide access to it


def read_groundtruth(gt_filepath, cov_filepath):

    gt = np.loadtxt(gt_filepath, delimiter=",")
    cov = np.loadtxt(cov_filepath, delimiter=",")
    t_cov = cov[:, 0]

    # Note: Interpolation is not needed, this is done as a convience
    interp = scipy.interpolate.interp1d(
        gt[:, 0], gt[:, 1:], kind='nearest', axis=0, fill_value="extrapolate")
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
    gps = np.loadtxt(filepath, delimiter=",")

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

    r = 6400000  # approx. radius of earth (m)
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
    ms25 = np.loadtxt(filepath, delimiter=",")

    accel_x_OG = ms25[:, 4]
    accel_y_OG = ms25[:, 5]
    rot_h_OG = ms25[:, 9]

    accel_x_df = pd.DataFrame(accel_x_OG)
    accel_x = accel_x_df.rolling(1000, min_periods=1).mean()
    accel_x = accel_x.to_numpy()

    accel_y_df = pd.DataFrame(accel_y_OG)
    accel_y = accel_y_df.rolling(1000, min_periods=1).mean()
    accel_y = accel_y.to_numpy()

    rot_h_df = pd.DataFrame(rot_h_OG)
    rot_h = rot_h_df.rolling(1000, min_periods=1).mean()
    rot_h = rot_h.to_numpy()

    return ms25, accel_x, accel_y, rot_h

# plot desired IMU data


def plot_IMU(filepath):
    ms25, _, _, _ = read_IMU(filepath)
    # time in us
    t = ms25[:, 0]

    # acceleration
    accel_x = ms25[:, 4]  # m/s^2
    accel_y = ms25[:, 5]  # m/s^2

    # angular rotation rate
    rot_yaw = ms25[:, 9]  # rate of change of yaw (about z) in rad/s

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
    wheel_vel = np.loadtxt(filepath, delimiter=",")

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
    x_k, y_k, x_dot_k, y_dot_k, psi_k = sp.symbols(
        'x_k, y_k, x_dot_k, y_dot_k, psi_k', real=True)

    f1 = x_k + x_dot_k*delta_t + 0.5*ax*delta_t ^ 2
    f2 = y_k + y_dot_k*delta_t + 0.5*ay*delta_t ^ 2
    f3 = x_dot_k + ax*delta_t
    f4 = y_dot_k + ay*delta_t
    f5 = psi_k + psidot*delta_t

    f = sp.Matrix([f1, f2, f3, f4, f5]).jacobian(
        [x_k, y_k, x_dot_k, y_dot_k, psi_k])

    # TO DO sub in previous x_est...
    # Output of the function would be a NxN np.array of floating point numbers

    return f

# GPS goes here (x,y)
# wheel velocities goes here (xdot, ydot)
# heading??? might want to update it somehow?

<<<<<<< Updated upstream
def measurement_jacobian():
    return H

def measurement_update(measurements, P_check, x_check):
=======

def measurement_update(lk, rk, bk, P_check, x_check):
>>>>>>> Stashed changes
    # 3.1 Compute measurement Jacobian using the landmarks and the current estimated state.
    H = measurement_jacobian(lk[0], lk[1], x_check)

    # 3.2 Compute the Kalman gain.
    K = np.matmul(np.matmul(P_check, np.transpose(H)),
                  np.linalg.inv(np.matmul((np.matmul(H, P_check)), np.transpose(H)) + R))

    n_w_L = np.random.multivariate_normal([0, 0], Q)

    x_check[2] = wraptopi(x_check[2])

    # 3.3 Correct the predicted state.
    # NB : Make sure to use wraptopi() when computing the bearing estimate!
<<<<<<< Updated upstream
    h = np.array([])
=======
    h = np.array([((lk[0] - x_check[0] - d[0]*np.cos(x_check[2]))**2
                 + (lk[1] - x_check[1] - d[0]*np.sin(x_check[2]))**2)**0.5,
                  (np.arctan2(lk[1] - x_check[1] - d[0]*np.sin(x_check[2]),
                              lk[0] - x_check[0] - d[0]*np.cos(x_check[2])) - x_check[2])])
>>>>>>> Stashed changes

    y_check = h

    y_check[1] = wraptopi(y_check[1])

    y = np.array([rk, bk])

    x_check = x_check + np.matmul(K, (y - y_check))

    x_check[2] = wraptopi(x_check[2])

    # 3.4 Correct the covariance.
    P_check = np.matmul((np.identity(3) - np.matmul(K, H)), P_check)

    return x_check, P_check


# PLEASE ENSURE THAT ALL STATES AND INPUTS TO THE SYSTEM ARE IN THE GLOBAL FRAME OF REFERENCE :-)
# PLEASE MAKE USE OF WRAPTOPI() WHEN SUPPLYING ROBOT HEADING OR ROBOT ANGULAR VELOCITY

# CURRENT STATE MODEL IS: X = [x, y, xdot, ydot, psi, psidot]
# CURRENT INPUT MODEL IS: U = [ax, ay, yaw_dot]

if __name__ == "__main__":

    ######################### 0. INITIALIZE IMPORTANT VARIABLES #################################################################

    # TO-DO: NEED TO INITIALIZE Q AND R COVARIANCE MATRICES

    Q = np.diag([])  # input noise covariance
    R = np.diag([])  # measurement noise covariance

    # TO-DO: NEED TO DEFINE GROUND TRUTH ARRAYS & ARRAY OF MEASUREMENTS
    x_true = ...
    y_true = ...

    # MEASUREMENTS CAN BE HELD AS A???
    measurements = ...

    # TO-DO: NEED TO DEFINE LENGTH OF STATES WE'LL BE ESTIMATING
    N = len(measurements)
    x_est = np.zeros([N, 5])  # estimated states, x, y, and theta
    P_est = np.zeros([N, 5, 5])  # state covariance matrices

    # TO-DO: NEED TO DEFINE INTIIAL STATE AND COVARIANCE MATRIX
    x_est[0] = np.array([])  # initial state
    P_est[0] = np.diag([])  # initial state covariance

    ################################ 1. MAIN FILTER LOOP ##########################################################################

    # TO-DO: NEED TO DEFINE ARRAY OF LINEARLY SPACED TIME INCREMENTS
    #        THIS WILL DEFINE DELTA T...

    t = []

    # TO-DO: Specify file path (are we doing a specific file or everything at once?)
    _, accel_x_vec, accel_y_vec, yaw_vec = read_IMU(filepath)

    # Start at 1 because we have initial prediction from ground truth.
    for k in range(1, len(t)):

        delta_t = t[k] - t[k - 1]  # time step (difference between timestamps)

        # TO-DO: - READ IN IMU FILE
        #        - APPLY TRANSFORMATION/ROTATION MATRIX TO ACCELERATION
        ax_orig = accel_x_vec[k]
        ay_orig = accel_y_vec[k]
        yaw_rate_orig = yaw_vec[k]

        yaw = x_est[k-1][4]  # state estimation of psi

        rotation_matrix = np.array(
            [np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)])

        acc_vec_imu = np.array([ax_orig], [ay_orig])
        acc_tranfs = np.matmul(rotation_matrix, acc_vec_imu)  # 2x1 matrix

        ax = acc_vec_imu[0]
        ay = acc_vec_imu[1]
        psidot = wraptopi(yaw_rate_orig)

        # 1-1. INITIAL UPDATE OF THE ROBOT STATE USING MEASUREMENTS (IMU, ETC.)
        x_check = x_est[k-1] + delta_t*np.array([x_dot_k + 0.5*ax*delta_t,
                                                 y_dot_k + 0.5*ay*delta_t,
                                                 ax,
                                                 ay,
                                                 psidot])

        # 1-2 Linearize Motion Model
        # Compute the Jacobian of f w.r.t. the last state.
        # TO-DO: DETERMINE MOTION MODEL EQUATIONS
        x_k, y_k, x_dot_k, y_dot_k, psi_k = sp.symbols(
            'x_k, y_k, x_dot_k, y_dot_k, psi_k', real=True)

<<<<<<< Updated upstream
        f1 = x_k      +  x_dot_k*delta_t + 0.5*ax*delta_t^2
        f2 = y_k      +  y_dot_k*delta_t + 0.5*ay*delta_t^2
        f3 = x_dot_k  +  ax*delta_t # might want acceleration local in here idk
        f4 = y_dot_k  +  ay*delta_t # TO-DO: maybe change
        f5 = psi_k    +  psidot*delta_t
=======
        f1 = x_k + x_dot_k*delta_t + 0.5*ax*delta_t ^ 2
        f2 = y_k + y_dot_k*delta_t + 0.5*ay*delta_t ^ 2
        f3 = x_dot_k + ax*delta_t
        f4 = y_dot_k + ay*delta_t
        f5 = psi_k + psidot*delta_t

        f = sp.Matrix([f1, f2, f3, f4, f5]).jacobian(
            [x_k, y_k, x_dot_k, y_dot_k, psi_k])
        F = np.array(f.subs([(x_k,      x_est[k-1, 0]),
                             (y_k,      x_est[k-1, 1]),
                             (x_dot_k,  x_est[k-1, 2]),
                             (y_dot_k,  x_est[k-1, 3]),
                             (psi_k,    x_est[k-1, 4])
                             ])).astype(np.float64)
>>>>>>> Stashed changes

        # 2. Propagate uncertainty by updating the covariance
        P_check = np.matmul(np.matmul(F, P_est[k-1]), np.transpose(F)) + Q

        # TO-DO: - GRAB THE DATA BASED ON TIMESTAMPS AND SAMPLING RATES....
        #        - THEN, UPDATE MEASUREMENTS ACCORDING TO WHICH PIECE OF DATA YOU GET
        for i in range(len(r[k])):
            x_check, P_check = measurement_update(
                l[i], r[k, i], b[k, i], P_check, x_check)

        # Set final state predictions for this kth timestep.
        x_est[k] = x_check
        P_est[k] = P_check

# TO-DO: PLOT DELIVERABLES #########################################################################################

# 1. PLOT FUSED LOCATION DATA
# 2. PLOT MSE FROM GROUND TRUTH (EUCLIDEAN DISTANCE)
# 3. PLOT GROUND TRUTH FOR COMPARISON
