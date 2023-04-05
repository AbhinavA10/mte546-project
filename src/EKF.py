
import math
import numpy as np
import sympy as sp

import utils
import read_imu
import read_wheels
import read_gps
import read_ground_truth 
import read_FOG

USE_RTK = False # Whether or not to use GPS RTK
TRUNCATION_END = -1 # Ground Truth has 500000 data points, filter for testing. Set to -1 for all data
USE_WHEEL_AS_INPUT = False # False = Use IMU acceleration as Input for Motion Model. True = Use Wheel Velocity and IMU Theta as Input for Motion Model
USE_GPS_FOR_CORRECTION = True # Correct prediction with GPS measurements
USE_WHEEL_FOR_CORRECTION = True # Correct Prediction with Wheel velocity measurement
USE_GPS_AS_INPUT = False # True = override state prediction with GPS measurement, for GPS error calculation
KALMAN_FILTER_RATE = 1 # Keep at 1Hz

R_WHEEL = np.eye(2) * 0.00001 # measurement noise, Guess
R_GPS   = np.eye(2) * np.power(10, 2)  # measurement noise, 10m^2
if USE_RTK:
    R_GPS = np.eye(2) * 0.1  # measurement noise, covariance, Guess

# Q For IMU as Input to Motion Model
Q = np.diag([10,  # x
             10,  # y
             10000, # x_dot
             10000, # y_dot
             0.001,  # theta
             0.01])   # omega
if USE_WHEEL_AS_INPUT:
    # Q For Wheel Velocity as Input to Motion Model
    Q = np.diag([1,  # x
                 1,  # y
                 0.001, # x_dot
                 0.001, # y_dot
                 0.001,  # theta
                 0.01])   # omega

FILE_DATES = ["2012-01-08", "2012-01-15", "2012-01-22", "2012-02-02", "2012-02-04", "2012-02-05", "2012-02-12", 
              "2012-02-18", "2012-02-19", "2012-03-17", "2012-03-25", "2012-03-31", "2012-04-29", "2012-05-11", 
              "2012-05-26", "2012-06-15", "2012-08-04", "2012-08-20", "2012-09-28", "2012-10-28", "2012-11-04", 
              "2012-11-16", "2012-11-17", "2012-12-01", "2013-01-10", "2013-02-23", "2013-04-05"]
ROBOT_WIDTH_WHEEL_BASE = 0.562356 # T [m], From SolidWorks Model

def wraptopi(x):
    """
    Wrap theta measurements to [-pi, pi].
    Accepts an angle measurement in radians and returns an angle measurement in radians
    """
    if x > np.pi:
        x = x - (np.floor(x / (2 * np.pi)) + 1) * 2 * np.pi
    elif x < -np.pi:
        x = x + (np.floor(x / (-2 * np.pi)) + 1) * 2 * np.pi
    return x

##### Symbolic Variables #####
x_k, y_k, x_dot_k, y_dot_k, theta_k, omega_k, dt_sym = sp.symbols(
    'x_k, y_k, x_dot_k, y_dot_k, theta_k, omega_k, dt', real=True)

##### Symbolic Jacobian for Motion Model #####
# IMU Based
ax_imu, ay_imu, heading_imu, omega_imu = sp.symbols(
    'ax_imu, ay_imu, theta_imu, omega_imu', real=True)
ax_global = ax_imu*sp.cos(-theta_k) - ay_imu*sp.sin(-theta_k)
ay_global = ax_imu*sp.sin(-theta_k) + ay_imu*sp.cos(-theta_k)
f1 = x_k + x_dot_k*dt_sym + 0.5*ax_global*dt_sym*dt_sym
f2 = y_k + y_dot_k*dt_sym + 0.5*ay_global*dt_sym*dt_sym
f3 = x_dot_k + ax_global*dt_sym
f4 = y_dot_k + ay_global*dt_sym
f5 = heading_imu # Theta estimate from IMU
f6 = omega_imu
f_imu_input=sp.Matrix([f1, f2, f3, f4, f5, f6])
F_JACOB_IMU = f_imu_input.jacobian([x_k, y_k, x_dot_k, y_dot_k, theta_k, omega_k])

def motion_update_imu_input(_ax, _ay, _theta, _omega, _dt, prev_state): # IMU inputs in robot frame
    """IMU for Motion Model
    Returns updated state and Jacobian wrt previous state
    """
    # Use motion model to predict state
    f = np.array(f_imu_input.subs([(x_k,         prev_state[0]),
                                   (y_k,         prev_state[1]),
                                   (x_dot_k,     prev_state[2]),
                                   (y_dot_k,     prev_state[3]),
                                   (theta_k,     prev_state[4]),
                                   (omega_k,     prev_state[5]),
                                   (ax_imu,      _ax),
                                   (ay_imu,      _ay),
                                   (heading_imu, _theta),
                                   (omega_imu,   _omega),
                                   (dt_sym,      _dt)
                                   ])).astype(np.float64).flatten()
    f[4] = wraptopi(f[4])
    # Compute Jacobian wrt previous state
    F = np.array(F_JACOB_IMU.subs([(x_k,         prev_state[0]),
                                   (y_k,         prev_state[1]),
                                   (x_dot_k,     prev_state[2]),
                                   (y_dot_k,     prev_state[3]),
                                   (theta_k,     prev_state[4]),
                                   (omega_k,     prev_state[5]),
                                   (ax_imu,      _ax),
                                   (ay_imu,      _ay),
                                   (heading_imu, _theta),
                                   (omega_imu,   _omega),
                                   (dt_sym,      _dt)
                                    ])).astype(np.float64)
    return (f,F)

# Wheel Based
vl_wheel, vr_wheel = sp.symbols(
    'vl_wheel, vr_wheel', real=True)
v_c = 0.5*(vl_wheel+vr_wheel)
f1 = x_k + v_c*sp.cos(heading_imu)*dt_sym
f2 = y_k + v_c*sp.sin(heading_imu)*dt_sym
f3 = v_c*sp.cos(heading_imu) # Redundant State
f4 = v_c*sp.sin(heading_imu) # Redundant State
f5 = heading_imu # Theta estimate from IMU
f6 = omega_imu
f_wheel_input=sp.Matrix([f1, f2, f3, f4, f5, f6])
F_JACOB_WHEEL = f_wheel_input.jacobian([x_k, y_k, x_dot_k, y_dot_k, theta_k, omega_k])

def motion_update_wheel_input(_vel_left_wheel, _vel_right_wheel, _theta, _omega, _dt, prev_state): # IMU inputs in robot frame
    """Wheel-based Motion Model
    Returns updated state and Jacobian wrt previous state
    """
    # Use motion model to predict state
    f = np.array(f_wheel_input.subs([(x_k,         prev_state[0]),
                                     (y_k,         prev_state[1]),
                                     (x_dot_k,     prev_state[2]),
                                     (y_dot_k,     prev_state[3]),
                                     (theta_k,     prev_state[4]),
                                     (omega_k,     prev_state[5]),
                                     (vl_wheel,    _vel_left_wheel),
                                     (vr_wheel,    _vel_right_wheel),
                                     (heading_imu, _theta),
                                     (omega_imu,   _omega),
                                     (dt_sym,      _dt)
                                      ])).astype(np.float64).flatten()
    f[4] = wraptopi(f[4])
    # Compute Jacobian wrt previous state
    F = np.array(F_JACOB_WHEEL.subs([(x_k,         prev_state[0]),
                                     (y_k,         prev_state[1]),
                                     (x_dot_k,     prev_state[2]),
                                     (y_dot_k,     prev_state[3]),
                                     (theta_k,     prev_state[4]),
                                     (omega_k,     prev_state[5]),
                                     (vl_wheel,    _vel_left_wheel),
                                     (vr_wheel,    _vel_right_wheel),
                                     (heading_imu, _theta),
                                     (omega_imu,   _omega),
                                     (dt_sym,      _dt)
                                      ])).astype(np.float64)
    return (f,F)

# Measurement models
# Symbolic Jacobian for Wheel Measurement
v_c     = sp.sqrt(x_dot_k**2 + y_dot_k**2)
v_left  = v_c - (ROBOT_WIDTH_WHEEL_BASE*omega_k)/2
v_right = v_c + (ROBOT_WIDTH_WHEEL_BASE*omega_k)/2
h1 = v_left
h2 = v_right

H_JACOB = sp.Matrix([h1, h2]).jacobian([x_k, y_k, x_dot_k, y_dot_k, theta_k, omega_k])

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
    z_hat = predict_z_hat_gps(x_pred)
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

    gps_data     = read_gps.read_gps(FILE_DATES[-1], USE_RTK) # 2.5 or 6 Hz
    imu_data     = read_imu.read_imu(FILE_DATES[-1]) # 47 Hz
    euler_data     = read_imu.read_euler(FILE_DATES[-1]) # 47 Hz
    fog_data     = read_FOG.read_FOG(FILE_DATES[-1]) # 97 Hz
    wheel_data   = read_wheels.read_wheels(FILE_DATES[-1]) # 37 Hz
    ground_truth = read_ground_truth.read_ground_truth(FILE_DATES[-1], truncation=TRUNCATION_END) # 107 Hz
    #Truncate data to first few datapoints, for testing
    ground_truth = ground_truth[:TRUNCATION_END,:]
    gps_data     = gps_data[:TRUNCATION_END,:]
    imu_data     = imu_data[:TRUNCATION_END,:]
    fog_data     = fog_data[:TRUNCATION_END,:]
    euler_data   = euler_data[:TRUNCATION_END,:]
    wheel_data   = wheel_data[:TRUNCATION_END,:]
    # Using the original Unix Timestamp for timesyncing

    x_true     = ground_truth[:, 1] # North
    y_true     = ground_truth[:, 2] # East
    theta_true = ground_truth[:, 3] # Heading
    true_times = ground_truth[:, 0]

    # Generate list of timesteps, from 0 to last timestep in ground_truth
    dt = 1/KALMAN_FILTER_RATE # 1/Hz = seconds
    t = np.arange(ground_truth[0,0], ground_truth[-1,0], dt)
    N     = len(t)
    x_est = np.zeros([N, 6]) 
    P_est = np.zeros([N, 6, 6])  # state covariance matrices

    # x_est = x | y | xdot | ydot | theta | omega
    x_est[0] = np.array([x_true[0], y_true[0], 0, 0, theta_true[0], 0])  # initial state
    P_est[0] = np.diag([1, 1, 1, 1, 1, 1])  # initial state covariance TO-DO: TUNE THIS TO TRAIN
    
    x_true_arr        = np.zeros([N]) # Keep track of corresponding truths
    y_true_arr        = np.zeros([N])
    theta_true_arr    = np.zeros([N])
    x_true_arr[0]     = x_true[0]  # initial state
    y_true_arr[0]     = y_true[0]
    theta_true_arr[0] = theta_true[0]

    ################################ 1. MAIN FILTER LOOP ##########################################################################

    a_x           =   imu_data[:,1]
    a_y           =   imu_data[:,2]
    omega         =   imu_data[:,3]
    omega_fog     = fog_data[:,1]
    theta_imu    = euler_data[:,1]

    gps_x         =   gps_data[:,1]
    gps_y         =   gps_data[:,2]
    robot_vel     = wheel_data[:,1]
    v_left_wheel  = wheel_data[:,2]
    v_right_wheel = wheel_data[:,3]

    gps_times     =   gps_data[:,0]
    wheel_times   = wheel_data[:,0]
    imu_times     =   imu_data[:,0]
    fog_times     =   fog_data[:,0]
    euler_times   =   euler_data[:,0]
    gps_counter   = 0
    wheel_counter = 0
    imu_counter   = 0
    fog_counter   = 0
    euler_counter   = 0
    ground_truth_counter = 0

    prev_gps_counter   = -1
    prev_wheel_counter = -1
    prev_imu_counter   = -1
    prev_fog_counter   = -1
    prev_euler_counter   = -1

    # Plot IMU estimated Theta vs Ground Truth  
    # plt.figure()
    # plt.plot(euler_times,theta_imu, label="IMU")
    # plt.plot(true_times,theta_true, label="Ground Truth")
    # plt.legend()
    # plt.xlabel('Time [s]')
    # plt.ylabel('Theta [rad]')
    # plt.show()

    # Start at 1 because we have initial prediction from ground truth.
    for k in range(1, len(t)):
        print(k, "/", len(t))

        # PREDICTION - UPDATE OF THE ROBOT STATE USING MOTION MODEL AND INPUTS (IMU)
        if USE_GPS_AS_INPUT:
            # GPS-ONLY Mode - override Prediction
            x_predicted = x_est[k-1,:]
            x_predicted[0] = gps_x[gps_counter]
            x_predicted[1] = gps_y[gps_counter]
        elif USE_WHEEL_AS_INPUT:
            # WHEEL VELOCITY BASED MODEL
            x_predicted, F = motion_update_wheel_input(v_left_wheel[wheel_counter], v_right_wheel[wheel_counter], theta_imu[euler_counter], omega[imu_counter], dt, x_est[k-1])
            P_predicted = np.matmul(np.matmul(F, P_est[k-1]), np.transpose(F)) + Q
        
        else:
            # IMU BASED MODEL
            x_predicted, F = motion_update_imu_input(a_x[imu_counter], a_y[imu_counter], theta_imu[euler_counter], omega[imu_counter], dt, x_est[k-1])
            P_predicted = np.matmul(np.matmul(F, P_est[k-1]), np.transpose(F)) + Q

        imu_counter = find_nearest_index(imu_times, t[k]) # Grab closest IMU data
        fog_counter = find_nearest_index(fog_times, t[k]) # Grab closest FOG data
        euler_counter = find_nearest_index(euler_times, t[k]) # Grab closest Theta correction data
        gps_counter = find_nearest_index(gps_times, t[k]) # Grab closest GPS data
        wheel_counter = find_nearest_index(wheel_times, t[k]) # Grab closest Wheel Velocity data
        ground_truth_counter = find_nearest_index(true_times, t[k]) # Grab closest Ground Truth data

        # CORRECTION - Correct with measurement models if available
        if USE_GPS_FOR_CORRECTION and not USE_GPS_AS_INPUT:
            if gps_counter != prev_gps_counter: # Use GPS data only if new
                # print("Used GPS Data: ", gps_counter, prev_gps_counter)
                x_predicted, P_predicted = measurement_update_gps(gps_x[gps_counter], gps_y[gps_counter], P_predicted, x_predicted)
                prev_gps_counter = gps_counter

        # CORRECTION - Correct with measurement models if available
        if USE_WHEEL_FOR_CORRECTION and not USE_WHEEL_AS_INPUT and not USE_GPS_AS_INPUT:
            if wheel_counter != prev_wheel_counter:
                #print("Used Wheel data: ", wheel_counter, prev_wheel_counter)
                x_predicted, P_predicted = measurement_update_wheel(v_left_wheel[wheel_counter], v_right_wheel[wheel_counter], P_predicted, x_predicted)
                prev_wheel_counter = wheel_counter
        

        # Set final state predictions for this kth timestep.
        x_est[k] = x_predicted
        P_est[k] = P_predicted
        
        # Keep track of corresponding Ground Truths at the same timestep
        x_true_arr[k] = x_true[ground_truth_counter]
        y_true_arr[k] = y_true[ground_truth_counter]
        theta_true_arr[k] = theta_true[ground_truth_counter]

    print('Done! Plotting now.')
    ###### PLOT DELIVERABLES #########################################################################################
    utils.export_to_kml(x_est[:,0], x_est[:,1], x_true_arr, y_true_arr, "Estimated", "Ground Truth")
    utils.plot_position_comparison_2D(x_est[:,0], x_est[:,1], x_true_arr, y_true_arr, "Estimated", "Ground Truth")
    # utils.plot_states(x_est, P_est, x_true_arr, y_true_arr, theta_true_arr, t)
