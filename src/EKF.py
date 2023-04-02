# !/usr/bin/python3

# General imports
import numpy as np
import matplotlib.pyplot as plt
import math

# Imports for sympy
import sympy as sp
import sys
import mpmath
sys.modules['sympy.mpmath'] = mpmath

# Imports for reading ground truth
import read_imu
import read_wheels
import read_gps
import read_ground_truth 

# wrap yaw measurements to [-pi, pi].
# Accepts an angle measurement in radians and returns an angle measurement in radians
def wraptopi(x):
    if x > np.pi:
        x = x - (np.floor(x / (2 * np.pi)) + 1) * 2 * np.pi
    elif x < -np.pi:
        x = x + (np.floor(x / (-2 * np.pi)) + 1) * 2 * np.pi
    return x

def motion_jacobian(ax_imu, ay_imu, psidot, dt, state_vector): # IMU inputs in robot frame
    # state_vector = [x_k, y_k, x_dot_k, y_dot_k, psi_k]
    x_k, y_k, x_dot_k, y_dot_k, psi_k = sp.symbols(
        'x_k, y_k, x_dot_k, y_dot_k, psi_k', real=True)
    
    ax_global = ax_imu*sp.cos(-psi_k) - ay_imu*sp.sin(-psi_k)
    ay_global = ax_imu*sp.sin(-psi_k) - ay_imu*sp.cos(-psi_k)
    f1 = x_k + x_dot_k*dt + 0.5*ax_global*dt**2
    f2 = y_k + y_dot_k*dt + 0.5*ay_global*dt**2
    f3 = x_dot_k + ax_global*dt
    f4 = y_dot_k + ay_global*dt
    f5 = psi_k + psidot*dt

    F = sp.Matrix([f1, f2, f3, f4, f5]).jacobian([x_k, y_k, x_dot_k, y_dot_k, psi_k])

    F = np.array(F.subs([(x_k,      state_vector[0]),
                         (y_k,      state_vector[1]),
                         (x_dot_k,  state_vector[2]),
                         (y_dot_k,  state_vector[3]),
                         (psi_k,    state_vector[4])
                         ])).astype(np.float64)
    return F

# TODO
def measurement_jacobian():
    return 0

# update measurement models
def z_hat_wheel(state_vector, dt):
    vel_x, vel_y, yaw_dot = state_vector #TODO
    v_c = math.sqrt(vel_x**2 + vel_y**2)
    v_right = v_c + (dt*yaw_dot)/2
    v_left = v_c - (dt*yaw_dot)/2

    z_wheel_vel = [v_left, v_right]
    return z_wheel_vel

def z_hat_wheel(state_vector, dt):
    v_c = math.sqrt(vel_x**2 + vel_y**2)
    v_right = v_c + (dt*yaw_dot)/2
    v_left = v_c - (dt*yaw_dot)/2

    z_wheel_vel = [v_left, v_right]
    return z_wheel_vel


# TODO
def measurement_update(measurements, P_check, x_check):
    # 3.1 Compute measurement Jacobian using the landmarks and the current estimated state.
    H = measurement_jacobian(lk[0], lk[1], x_check)

    # 3.2 Compute the Kalman gain.
    K = np.matmul(np.matmul(P_check, np.transpose(H)),
                  np.linalg.inv(np.matmul((np.matmul(H, P_check)), np.transpose(H)) + R))

    n_w_L = np.random.multivariate_normal([0, 0], Q)

    x_check[2] = wraptopi(x_check[2])

    # 3.3 Correct the predicted state.
    # NB : Make sure to use wraptopi() when computing the bearing estimate!
    h = np.array([])

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

# CURRENT STATE MODEL IS: X = [x, y, xdot, ydot, psi]
# CURRENT INPUT MODEL IS: U = [ax, ay, yaw_dot]

if __name__ == "__main__":

    ######################### 0. INITIALIZE IMPORTANT VARIABLES #################################################################

    Q = np.diag([0.1, 0.1, 1, 1, 0.1])  # input noise covariance, Guess
    R_gps = np.diag([10, 10])  # measurement noise covariance, Guess
    R_wheels = np.diag([1, 1])  # measurement noise covariance, Guess

    file_date = ["", "", ""]

    ground_truth = read_ground_truth(file_date[0]) # 107 Hz
    gps_data     = read_gps(file_date[0]) # 2.5 or 6 Hz
    imu_data     = read_imu(file_date[0]) # 47 Hz
    wheel_data   = read_wheels(file_date[0]) # 37 Hz

    x_true   = ground_truth[:, 1] # North
    y_true   = ground_truth[:, 2] # East
    psi_true = ground_truth[:, 3] # Heading

    N     = len(x_true)
    x_est = np.zeros([N, 5]) 
    P_est = np.zeros([N, 5, 5])  # state covariance matrices

    # x_est = x | y | xdot | ydot | psi
    x_est[0] = np.array([x_true[0], y_true[0], 0, 0, psi_true[0]])  # initial state
    P_est[0] = np.diag([1, 1, 1, 1, 1])  # initial state covariance TO-DO: TUNE THIS TO TRAIN

    ################################ 1. MAIN FILTER LOOP ##########################################################################

    # Generate list of timesteps, from 0 to last timestep in ground_truth
    dt = 1/50
    t = np.linspace(0, ground_truth[-1,0], ground_truth[-1,0]/dt)

    a_x           =   imu_data[:,1]
    a_y           =   imu_data[:,2]
    psi_dot       =   imu_data[:,3]
    gps_x         =   gps_data[:,1]
    gps_y         =   gps_data[:,2]
    v_robot       = wheel_data[:,1]
    v_left_wheel  = wheel_data[:,2]
    v_right_wheel = wheel_data[:,3]

    # Start at 1 because we have initial prediction from ground truth.
    for k in range(1, len(t)):

        acc_vec_imu = np.array([a_x[k-1], a_y[k-1]])

        rotation_matrix = np.array([np.cos(x_est[k-1,4]), -np.sin(x_est[k-1,4])],
                                   [np.sin(x_est[k-1,4]),  np.cos(x_est[k-1,4])])

        a_global = np.matmul(rotation_matrix, acc_vec_imu)

        ax = a_global[0]
        ay = a_global[1]

        psidot = wraptopi(psi_dot[k])

        # 1-1. INITIAL UPDATE OF THE ROBOT STATE USING MEASUREMENTS (IMU, ETC.)
        vel_vec_wheels = np.array([v_robot[k-1], 0])
        
        vel_global = np.matmul(rotation_matrix, vel_vec_wheels)

        x_dot_k = vel_global[0]
        y_dot_k = vel_global[1]

        x_check = x_est[k-1] + dt*np.array([x_dot_k + 0.5*ax*dt,
                                            y_dot_k + 0.5*ay*dt,
                                            ax,
                                            ay,
                                            psidot])

        # 1-2 Linearize Motion Model
        # Compute the Jacobian of f w.r.t. the last state. TODO
        F = motion_jacobian()

        # 2. Propagate uncertainty by updating the covariance
        P_check = np.matmul(np.matmul(F, P_est[k-1]), np.transpose(F)) + Q

        # TODO: - GRAB THE DATA BASED ON TIMESTAMPS AND SAMPLING RATES....
        #        - THEN, UPDATE MEASUREMENTS ACCORDING TO WHICH PIECE OF DATA YOU GET
        for i in range(len(r[k])):
            x_check, P_check = measurement_update(
                l[i], r[k, i], b[k, i], P_check, x_check)

        # Set final state predictions for this kth timestep.
        x_est[k] = x_check
        P_est[k] = P_check

# TODO: PLOT DELIVERABLES #########################################################################################

# 1. PLOT FUSED LOCATION DATA
# 2. PLOT MSE FROM GROUND TRUTH (EUCLIDEAN DISTANCE)
# 3. PLOT GROUND TRUTH FOR COMPARISON
