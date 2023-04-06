import numpy as np
import utils
import pandas as pd

# Read IMU using dataset date and return NP array providing
def read_imu(dataset_date):
    filepath = f"dataset/{dataset_date}/ms25.csv"
    ms25 = np.loadtxt(filepath, delimiter = ",")
    accel_x_OG = ms25[:, 4]
    accel_y_OG = ms25[:, 5]
    rot_h_OG   = ms25[:, 9]
    t          = ms25[:, 0]

    # Attempt to Remove bias by estimating it using the first 2 seconds of stationary data
    # print("a_x Bias", np.average(accel_x_OG[:100]))
    # print("a_y Bias", np.average(accel_y_OG[:100]))
    # accel_x_OG -= np.average(accel_x_OG[:100])
    # accel_y_OG -= np.average(accel_y_OG[:100])
    
    # apply rolling average to accelerations, to smooth noise
    accel_x_df = pd.DataFrame(accel_x_OG)
    accel_x_rolling = accel_x_df.rolling(50, min_periods=1).mean()
    accel_x_rolling = accel_x_rolling.to_numpy().flatten()
    accel_y_df = pd.DataFrame(accel_y_OG)
    accel_y_rolling = accel_y_df.rolling(50, min_periods=1).mean()
    accel_y_rolling = accel_y_rolling.to_numpy().flatten()
    
    # Relative timestamps
    # t = t-t[0]
    t = t/1000000
    utils.calculate_hz("IMU Accel and Omega", t) # 47 Hz

    # have the following format:
    # timestamp | ax_robot | ay_robot | omega
    imu_data = np.array([])
    imu_data = np.vstack((t, accel_x_rolling, accel_y_rolling, rot_h_OG)).T
    return imu_data

def read_euler(dataset_date):
    filepath = f"dataset/{dataset_date}/ms25_euler.csv"
    euler = np.loadtxt(filepath, delimiter = ",")

    t          = euler[:, 0]
    h_OG       = euler[:, 3] # heading (z)
    
    # Relative timestamps
    # t = t-t[0]
    t = t/1000000
    utils.calculate_hz("IMU Euler", t) # 47 Hz

    # have the following format:
    # timestamp | ax_robot | ay_robot | omega
    euler_data = np.array([])
    euler_data = np.vstack((t, h_OG)).T
    return euler_data

# Accept a filepath to the CSV of interest and plot the IMU data
def plot_imu(filepath):
    imu_data = read_imu(filepath)
    pass

if __name__ == "__main__":
    plot_imu("2013-04-05")
