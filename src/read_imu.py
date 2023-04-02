import numpy as np
import os

# Read IMU using dataset date and return NP array providing
def read_imu(dataset_date):
    filepath = f"dataset/{dataset_date}/ms25.csv"
    ms25 = np.loadtxt(filepath, delimiter = ",")
    accel_x_OG = np.array(ms25[:, 4])
    accel_y_OG = np.array(ms25[:, 5])
    rot_h_OG   = np.array(ms25[:, 9])
    t          = np.array(ms25[:, 0])
    
    # Relative timestamps
    t = t-t[0]

    # have the following format:
    # timestamp | ax_robot | ay_robot | psi_dot
    imu_data = np.concatenate((t, accel_x_OG, accel_y_OG, rot_h_OG), axis=1)




