# !/usr/bin/python3

# General imports
import numpy as np
import matplotlib.pyplot as plt
import math

import sympy as sp
import sys
import utils

# Imports for reading ground truth
import read_imu
import read_wheels
import read_gps
import read_ground_truth 

R_WHEEL = np.diag([1, 1])  # measurement noise covariance, Guess
R_GPS = np.diag([10, 10])  # measurement noise covariance, Guess

# wrap theta measurements to [-pi, pi].
# Accepts an angle measurement in radians and returns an angle measurement in radians
def wraptopi(x):
    if x > np.pi:
        x = x - (np.floor(x / (2 * np.pi)) + 1) * 2 * np.pi
    elif x < -np.pi:
        x = x + (np.floor(x / (-2 * np.pi)) + 1) * 2 * np.pi
    return x

##### Symbolic Variables #####
x_k, y_k, x_dot_k, y_dot_k, theta_k, omega_k = sp.symbols(
    'x_k, y_k, x_dot_k, y_dot_k, theta_k, omega_k', real=True)
ax_imu, ay_imu, omega_imu, dt_sym = sp.symbols(
    'ax_imu, ay_imu, omega_imu, dt', real=True)

##### Symbolic Jacobian for Motion Model #####
ax_global = ax_imu*sp.cos(-theta_k) - ay_imu*sp.sin(-theta_k)
ay_global = ax_imu*sp.sin(-theta_k) - ay_imu*sp.cos(-theta_k)
f1 = x_k + x_dot_k*dt_sym + 0.5*ax_global*dt_sym**2
f2 = y_k + y_dot_k*dt_sym + 0.5*ay_global*dt_sym**2
f3 = x_dot_k + ax_global*dt_sym
f4 = y_dot_k + ay_global*dt_sym
f5 = theta_k + omega_imu*dt_sym
f6 = omega_imu
F_JACOB = sp.Matrix([f1, f2, f3, f4, f5, f6]).jacobian([x_k, y_k, x_dot_k, y_dot_k, theta_k, omega_k])

### Symbolic Jacobian for Wheel Measurement
ROBOT_WIDTH_WHEEL_BASE = 0.562356 # T [m], From SolidWorks Model
v_c     = sp.sqrt(x_dot_k**2 + y_dot_k**2)
v_left  = v_c - (ROBOT_WIDTH_WHEEL_BASE*omega_k)/2
v_right = v_c + (ROBOT_WIDTH_WHEEL_BASE*omega_k)/2
h1 = v_left
h2 = v_right

H_JACOB = sp.Matrix([h1, h2]).jacobian([x_k, y_k, x_dot_k, y_dot_k, theta_k, omega_k])

def motion_jacobian(_ax, _ay, _omega, _dt, state_vector): # IMU inputs in robot frame
    """Numerical Jacobian for Motion Model"""
    return np.array(F_JACOB.subs([(x_k,      state_vector[0]),
                         (y_k,      state_vector[1]),
                         (x_dot_k,  state_vector[2]),
                         (y_dot_k,  state_vector[3]),
                         (theta_k,    state_vector[4]),
                         (omega_k,    state_vector[5]),
                         (ax_imu,  _ax),
                         (ay_imu,    _ay),
                         (omega_imu,    _omega),
                         (dt_sym,    _dt)
                         ])).astype(np.float64)

# Measurement models
def predict_z_hat_wheel(state_vector):
    """Predict Z_hat for Wheel Velocity Measurements"""
    vel_x, vel_y, omega = state_vector[2], state_vector[3], state_vector[5]
    v_c     = math.sqrt(vel_x**2 + vel_y**2)
    v_left  = v_c - (ROBOT_WIDTH_WHEEL_BASE*omega)/2
    v_right = v_c + (ROBOT_WIDTH_WHEEL_BASE*omega)/2
    z_wheel_vel = np.array([v_left, v_right])
    return z_wheel_vel

def predict_z_hat_gps(state_vector):
    """Predict Z_hat for GPS measurements"""
    return np.array([state_vector[0], state_vector[1]])

def measurement_jacobian_wheel(state_vector):
    """Numerical Jacobian for Wheel Velocity Measurement Model"""
    return np.array(H_JACOB.subs([(x_k,      state_vector[0]),
                            (y_k,      state_vector[1]),
                            (x_dot_k,  state_vector[2]),
                            (y_dot_k,  state_vector[3]),
                            (theta_k,    state_vector[4]),
                            (omega_k,    state_vector[5])
                            ])).astype(np.float64)

def measurement_update_wheel(wheel_vel_left, wheel_vel_right, P_pred, x_pred):
    """Perform Correction using Wheel Velocity Measurement"""

    # Compute measurement Jacobian using current estimated state.
    H = measurement_jacobian_wheel(x_pred)

    # Correct the predicted state.
    z_hat = predict_z_hat_wheel(x_pred)
    z = np.array([wheel_vel_left, wheel_vel_right])
    # Compute the Kalman gain.
    K = np.matmul(np.matmul(P_pred, np.transpose(H)),
                  np.linalg.inv(np.matmul((np.matmul(H, P_pred)), np.transpose(H)) + R_WHEEL))
    x_corrected = x_pred + np.matmul(K, (z - z_hat))
    x_corrected[4] = wraptopi(x_corrected[4])

    # Correct the covariance.
    P_corrected = np.matmul((np.identity(6) - np.matmul(K, H)), P_pred)

    return x_corrected, P_corrected

def measurement_jacobian_gps(state_vector):
    """Compute Jacobian for GPS Measurement Model"""
    # x_k, y_k, x_dot_k, y_dot_k, theta_k, omega_k = sp.symbols(
    #     'x_k, y_k, x_dot_k, y_dot_k, theta_k, omega_k', real=True)

    # Note: GPS Measurement Model is linear -> Jacobian actually looks like a C Matrix
    # h1 = x_k
    # h2 = y_k

    # Results in Matrix shown below
    # H = sp.Matrix([h1, h2]).jacobian([x_k, y_k, x_dot_k, y_dot_k, theta_k, omega_k])
    # H = np.array(H.subs([(x_k,      state_vector[0]),
    #                      (y_k,      state_vector[1]),
    #                      (x_dot_k,  state_vector[2]),
    #                      (y_dot_k,  state_vector[3]),
    #                      (theta_k,    state_vector[4]),
    #                      (omega_k,    state_vector[5])
    #                      ])).astype(np.float64)
    return np.array([[1, 0, 0, 0, 0, 0,],
                     [0, 1, 0, 0, 0, 0,]])

def measurement_update_gps(x, y, P_pred, x_pred):
    """Perform Correction using GPS Measurement"""

    # Compute measurement Jacobian using current estimated state.
    H = measurement_jacobian_gps(x_pred)

    # Correct the predicted state.
    z_hat = predict_z_hat_wheel(x_pred)
    z = np.array([x, y])
    # Compute the Kalman gain.
    K = np.matmul(np.matmul(P_pred, np.transpose(H)),
                  np.linalg.inv(np.matmul((np.matmul(H, P_pred)), np.transpose(H)) + R_GPS))
    x_corrected = x_pred + np.matmul(K, (z - z_hat))
    x_corrected[4] = wraptopi(x_corrected[4])

    # Correct the covariance.
    P_corrected = np.matmul((np.identity(6) - np.matmul(K, H)), P_pred)

    return x_corrected, P_corrected

# Find nearest index to value in array
def find_nearest_index(array:np.ndarray, time): # array of timesteps, time to search for
    """Find closest time in array, that has already passed"""
    diff_arr = array - time
    idx = np.where(diff_arr <= 0, diff_arr, -np.inf).argmax()
    return idx

# [-0.02 +0.02 +2]
# [-0.02  -inf -inf]

if __name__ == "__main__":

    ######################### 0. INITIALIZE IMPORTANT VARIABLES #################################################################
    
    # CURRENT STATE MODEL IS: X = [x, y, x_dot, y_dot, theta, omega]
    # CURRENT INPUT MODEL IS: U = [ax, ay, omega]

    Q = np.diag([0.1, 0.1, 1, 1, 0.1, 1])  # input noise covariance, Guess

    file_date = ["2013-04-05", "", ""]

    ground_truth = read_ground_truth.read_ground_truth(file_date[0]) # 107 Hz
    gps_data     = read_gps.read_gps(file_date[0]) # 2.5 or 6 Hz
    imu_data     = read_imu.read_imu(file_date[0]) # 47 Hz
    wheel_data   = read_wheels.read_wheels(file_date[0]) # 37 Hz
    #Truncate data to first 20000 datapoints, for testing
    ground_truth = ground_truth[:20000,:]
    gps_data     = gps_data[:20000,:]
    imu_data     = imu_data[:20000,:]
    wheel_data   = wheel_data[:20000,:]
    # Using the original Unix Timestamp for timesyncing

    x_true   = ground_truth[:, 1] # North
    y_true   = ground_truth[:, 2] # East
    theta_true = ground_truth[:, 3] # Heading

    N     = len(x_true)
    x_est = np.zeros([N, 6]) 
    P_est = np.zeros([N, 6, 6])  # state covariance matrices

    # x_est = x | y | xdot | ydot | theta | omega
    x_est[0] = np.array([x_true[0], y_true[0], 0, 0, theta_true[0], 0])  # initial state
    P_est[0] = np.diag([1, 1, 1, 1, 1, 1])  # initial state covariance TO-DO: TUNE THIS TO TRAIN

    ################################ 1. MAIN FILTER LOOP ##########################################################################

    # Generate list of timesteps, from 0 to last timestep in ground_truth
    dt = 1/50
    t = np.arange(ground_truth[0,0], ground_truth[-1,0], dt)
    a_x           =   imu_data[:,1]
    a_y           =   imu_data[:,2]
    omega         =   imu_data[:,3]
    gps_x         =   gps_data[:,1]
    gps_y         =   gps_data[:,2]
    #v_robot       = wheel_data[:,1]
    v_left_wheel  = wheel_data[:,2]
    v_right_wheel = wheel_data[:,3]

    gps_times     =   gps_data[:,0]
    wheel_times   = wheel_data[:,0]
    imu_times     =   imu_data[:,0]
    gps_counter   = 0
    wheel_counter = 0
    imu_counter   = 0
    prev_gps_counter   = -1
    prev_wheel_counter = -1
    prev_imu_counter   = -1

    # Start at 1 because we have initial prediction from ground truth.
    for k in range(1, len(t)):
        print(k)

        # PREDICTION - UPDATE OF THE ROBOT STATE USING MOTION MODEL AND INPUTS (IMU)
        ax_global = a_x[imu_counter]*sp.cos(-x_est[k-1, 4]) - a_y[imu_counter]*sp.sin(-x_est[k-1, 4])
        ay_global = a_x[imu_counter]*sp.sin(-x_est[k-1, 4]) - a_y[imu_counter]*sp.cos(-x_est[k-1, 4])

        x_predicted = x_est[k-1] + dt*np.array([x_est[k-1,2] + 0.5*ax_global*dt,
                                                x_est[k-1,3] + 0.5*ay_global*dt,
                                                ax_global,
                                                ay_global,
                                                omega[imu_counter],
                                                0])
        
        x_predicted[4] = wraptopi(x_predicted[4])
        x_predicted[5] = omega[imu_counter]
        # Compute the Jacobian of f w.r.t. the last state.
        F = motion_jacobian(a_x[imu_counter], a_y[imu_counter], omega[imu_counter], dt, x_est[k-1])
        # Propagate uncertainty by updating the covariance
        P_predicted = np.matmul(np.matmul(F, P_est[k-1]), np.transpose(F)) + Q

        imu_counter = find_nearest_index(imu_times, t[k]) # Grab closest IMU data
        gps_counter = find_nearest_index(gps_times, t[k]) # Grab closest IMU data
        wheel_counter = find_nearest_index(wheel_times, t[k]) # Grab closest IMU data
        
        #print("Times", t[k], wheel_times[wheel_counter])

        # CORRECTION - Correct with measurement models if available
        if gps_counter != prev_gps_counter: # Use GPS data only if new
            # print("Used GPS Data: ", gps_counter, prev_gps_counter)
            x_predicted, P_predicted = measurement_update_gps(gps_x[gps_counter], gps_y[gps_counter], P_predicted, x_predicted)
            prev_gps_counter = gps_counter

        # CORRECTION - Correct with measurement models if available
        if wheel_counter != prev_wheel_counter:
            #print("Used Wheel data: ", wheel_counter, prev_wheel_counter)
            x_predicted, P_predicted = measurement_update_wheel(v_left_wheel[wheel_counter], v_right_wheel[wheel_counter], P_predicted, x_predicted)
            prev_wheel_counter = wheel_counter
        
        # Set final state predictions for this kth timestep.
        x_est[k] = x_predicted
        P_est[k] = P_predicted

    print('Done! Plotting now.')
    ###### PLOT DELIVERABLES #########################################################################################
    # 1. PLOT FUSED LOCATION DATA
    utils.export_to_kml(x_est[:,0], x_est[:,1], ground_truth[:,1], ground_truth[:,2])
    utils.plot_state_comparison(x_est[:,0], x_est[:,1], ground_truth[:,1], ground_truth[:,2])
    # TODO 2. PLOT MSE FROM GROUND TRUTH (EUCLIDEAN DISTANCE)
    # TODO 3. PLOT GROUND TRUTH FOR COMPARISON
