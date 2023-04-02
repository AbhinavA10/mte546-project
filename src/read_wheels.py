import numpy as np
import utils

# Read wheels velocity using dataset date
def read_wheels(dataset_date):
    filepath = f"dataset/{dataset_date}/wheels.csv"
    wheel_vel = np.loadtxt(filepath, delimiter = ",")

    t = wheel_vel[:, 0]

    left_wheel_vel  = wheel_vel[:, 1] 
    right_wheel_vel = wheel_vel[:, 2] 
    robot_vel = 0.5*(wheel_vel[:, 1] + wheel_vel[:, 2])
    
    # Relative timestamps
    t = t-t[0]
    t = t/1000000

    utils.calculate_hz("Wheel Odometry", t) # 5.85 Hz

    # have the following format:
    # timestamp | robot velocity | left wheel velocity | right wheel velocity
    wheel_data = np.array([])
    wheel_data = np.vstack((t, robot_vel, left_wheel_vel, right_wheel_vel)).T
    return wheel_data

