# Read and plot the ground truth data.
#
# Note: The ground truth data is provided at a high rate of about 100 Hz. To
# generate this high rate ground truth, a SLAM solution was used. Nodes in the
# SLAM graph were not added at 100 Hz, but rather about every 8 meters. In
# between the nodes in the SLAM graph, the odometry was used to interpolate and
# provide a high rate ground truth.

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import utils

# Accept a filepath to the CSV of interest and return Numpy array with data
def read_ground_truth(dataset_date):
    """Read Ground Truth Data
    Parameters: dataset date
    Returns: np.ndarray([timestamp, x, y, yaw])
    """
    filepath_cov = f"dataset/{dataset_date}/odometry_cov_100hz.csv"
    filepath_gt = f"dataset/{dataset_date}/groundtruth_{dataset_date}.csv"
    
    gt = np.loadtxt(filepath_gt, delimiter = ",")
    cov = np.loadtxt(filepath_cov, delimiter = ",")
    
    t = cov[:, 0]

    interp = scipy.interpolate.interp1d(gt[:, 0], gt[:, 1:], kind='nearest', axis=0, fill_value="extrapolate")
    pose_gt = interp(t)
    t = t-t[0] # Make timestamps relative
    t = t/1000000
    x = pose_gt[:, 0] # North
    y = pose_gt[:, 1] # East
    yaw = pose_gt[:, 5]
    utils.calculate_hz("Ground Truth", t) # 107 Hz
    
    ground_truth = np.array([])
    ground_truth = np.vstack((t, x, y, yaw)).T 

    # timestamp | x | y | yaw
    return ground_truth

# Accept a filepath to the CSV of interest and plot the Ground Truth data
def plot_ground_truth(filepath):
    ground_truth = read_ground_truth(filepath)
    t = ground_truth[:,0]
    x = ground_truth[:, 1] # North
    y = ground_truth[:, 2] # East
    yaw = ground_truth[:, 3]

    utils.export_to_kml(None,None,x,y)

    plt.figure()
    plt.scatter(y, x, s=1, linewidth=0)
    plt.axis('equal')
    plt.title('Ground Truth Position')
    plt.xlabel('East [m]')
    plt.ylabel('North [m]')
    
    plt.figure()
    plt.plot(t, yaw)
    plt.title('Ground Truth Heading')
    plt.xlabel('Time [s]')
    plt.ylabel('Angle [rad]')
    plt.show()

if __name__ == "__main__":
    plot_ground_truth("2013-04-05")