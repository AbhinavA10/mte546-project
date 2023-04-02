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

# wrap theta measurements to [-pi, pi].
# Accepts an angle measurement in radians and returns an angle measurement in radians
def wraptopi(x):
    if x > np.pi:
        x = x - (np.floor(x / (2 * np.pi)) + 1) * 2 * np.pi
    elif x < -np.pi:
        x = x + (np.floor(x / (-2 * np.pi)) + 1) * 2 * np.pi
    return x

def motion_jacobian(ax_imu, ay_imu, omega_imu, dt, state_vector): # IMU inputs in robot frame
    """Compute Jacobian for Motion Model"""
    x_k, y_k, x_dot_k, y_dot_k, theta_k, omega_k = sp.symbols(
        'x_k, y_k, x_dot_k, y_dot_k, theta_k, omega_k', real=True)
    
    ax_global = ax_imu*sp.cos(-theta_k) - ay_imu*sp.sin(-theta_k)
    ay_global = ax_imu*sp.sin(-theta_k) - ay_imu*sp.cos(-theta_k)
    
    f1 = x_k + x_dot_k*dt + 0.5*ax_global*dt**2
    f2 = y_k + y_dot_k*dt + 0.5*ay_global*dt**2
    f3 = x_dot_k + ax_global*dt
    f4 = y_dot_k + ay_global*dt
    f5 = theta_k + omega_imu*dt
    f6 = omega_imu

    F = sp.Matrix([f1, f2, f3, f4, f5, f6]).jacobian([x_k, y_k, x_dot_k, y_dot_k, theta_k, omega_k])

    F = np.array(F.subs([(x_k,      state_vector[0]),
                         (y_k,      state_vector[1]),
                         (x_dot_k,  state_vector[2]),
                         (y_dot_k,  state_vector[3]),
                         (theta_k,    state_vector[4])
                         (omega_k,    state_vector[5])
                         ])).astype(np.float64)
    return F

# Measurement models
#TODO: take jacobians, since wheel velocity measurement is nonlinear
def z_hat_wheel(state_vector, dt):
    """Calculate Z_hat for Wheel Velocity Measurements"""
    ROBOT_WIDTH_WHEEL_BASE = 0.562356 # T [m], From SolidWorks Model
    vel_x, vel_y, omega = state_vector[2], state_vector[3], state_vector[5]
    v_c     = math.sqrt(vel_x**2 + vel_y**2)
    v_right = v_c + (ROBOT_WIDTH_WHEEL_BASE*omega)/2
    v_left  = v_c - (ROBOT_WIDTH_WHEEL_BASE*omega)/2
    z_wheel_vel = [v_left, v_right]
    return z_wheel_vel

def z_hat_gps(state_vector):
    """Calculate Z_hat for GPS measurements"""
    x, y = state_vector[0], state_vector[1]
    z_gps = [x, y]
    return z_gps

def measurement_jacobian_wheel(state_vector, dt):
    pass

def measurement_jacobian_gps(state_vector, dt):
    pass

# TODO
def measurement_update_wheel(measurements, P_check, x_check, R):
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

# CURRENT STATE MODEL IS: X = [x, y, x_dot, y_dot, theta, omega]
# CURRENT INPUT MODEL IS: U = [ax, ay, omega]

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
    theta_true = ground_truth[:, 3] # Heading

    N     = len(x_true)
    x_est = np.zeros([N, 5]) 
    P_est = np.zeros([N, 5, 5])  # state covariance matrices

    # x_est = x | y | xdot | ydot | theta | omega
    x_est[0] = np.array([x_true[0], y_true[0], 0, 0, theta_true[0]])  # initial state
    P_est[0] = np.diag([1, 1, 1, 1, 1])  # initial state covariance TO-DO: TUNE THIS TO TRAIN

    ################################ 1. MAIN FILTER LOOP ##########################################################################

    # Generate list of timesteps, from 0 to last timestep in ground_truth
    dt = 1/50
    t = np.linspace(0, ground_truth[-1,0], ground_truth[-1,0]/dt)

    a_x           =   imu_data[:,1]
    a_y           =   imu_data[:,2]
    omega         =   imu_data[:,3]
    gps_x         =   gps_data[:,1]
    gps_y         =   gps_data[:,2]
    v_robot       = wheel_data[:,1]
    v_left_wheel  = wheel_data[:,2]
    v_right_wheel = wheel_data[:,3]

    gps_times   = gps_data[:,0]
    wheel_times = wheel_data[:,0]
    imu_times   = imu_data[:,0]
    gps_counter   = 0
    wheel_counter = 0
    imu_counter   = 0

    # Start at 1 because we have initial prediction from ground truth.
    for k in range(1, len(t)):

        if t[k] >= imu_times[imu_counter]:
            acc_vec_imu = np.array([a_x[k-1], a_y[k-1]])

            rotation_matrix = np.array([np.cos(x_est[k-1,4]), -np.sin(x_est[k-1,4])],
                                    [np.sin(x_est[k-1,4]),  np.cos(x_est[k-1,4])])

            a_global = np.matmul(rotation_matrix, acc_vec_imu)

            ax = a_global[0]
            ay = a_global[1]

            omega = wraptopi(omega[k])

            # 1-1. INITIAL UPDATE OF THE ROBOT STATE USING MOTION MODEL AND INPUTS (IMU)
            ax_global = ax*sp.cos(-x_est[k-1, 5]) - ay*sp.sin(-x_est[k-1, 5])
            ay_global = ax*sp.sin(-x_est[k-1, 5]) - ay*sp.cos(-x_est[k-1, 5])

            x_check = x_est[k-1] + dt*np.array([x_est[k-1,2] + 0.5*ax*dt,
                                                x_est[k-1,3] + 0.5*ay*dt,
                                                ax,
                                                ay,
                                                omega,
                                                0])
            x_check[6] = omega

            # 1-2 Linearize Motion Model
            # Compute the Jacobian of f w.r.t. the last state.
            F = motion_jacobian(acc_vec_imu[0], acc_vec_imu[1], omega, dt, x_est[k-1])

            # 2. Propagate uncertainty by updating the covariance
            P_check = np.matmul(np.matmul(F, P_est[k-1]), np.transpose(F)) + Q

        if t[k] >= gps_times[gps_counter]:
            x_check, P_check = gps_measurement_update()
            gps_counter += 1

        if t[k] >= wheel_times[wheel_counter]:
            x_check, P_check = wheel_measurement_update()
            wheel_counter += 1

        # Set final state predictions for this kth timestep.
        x_est[k] = x_check
        P_est[k] = P_check

# TODO: PLOT DELIVERABLES #########################################################################################

# 1. PLOT FUSED LOCATION DATA
# 2. PLOT MSE FROM GROUND TRUTH (EUCLIDEAN DISTANCE)
# 3. PLOT GROUND TRUTH FOR COMPARISON



