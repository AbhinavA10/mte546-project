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
    #t = cov[:,0]
    interp = scipy.interpolate.interp1d(gt[:, 0], gt[:, 1:], kind='nearest', axis=0, fill_value="extrapolate")
    pose_gt = interp(t)
    # t = t-t[0] # Make timestamps relative
    t = t/1000000
    x = pose_gt[:, 0] # North
    y = pose_gt[:, 1] # East

    print("Ground truth x0: ", x[0])
    print("Ground truth y0: ", y[0])

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

    # utils.export_to_kml(None,None,x,y)

    plt.figure(figsize=(15, 10), dpi=300)
    # plt.scatter(y, x, s=1, linewidth=0)
    plt.scatter(y, x, c=t, s=1, linewidth=0)
    plt.colorbar(label='Time')
    plt.axis('equal')
    plt.title('Ground Truth Position')
    plt.xlabel('East [m]')
    plt.ylabel('North [m]')
    # fig = plt.gcf()
    # fig.savefig(f"{filepath}_Ground_Truth_Over_Time.png", dpi=300)
    
    # plt.figure()
    # plt.plot(t, yaw)
    # plt.title('Ground Truth Heading')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Angle [rad]')
    plt.show()

if __name__ == "__main__":
    
    FILE_DATES = ["2012-01-08", "2012-01-15", "2012-01-22", "2012-02-02", "2012-02-04", "2012-02-05", "2012-02-12", 
              "2012-02-18", "2012-02-19", "2012-03-17", "2012-03-25", "2012-03-31", "2012-04-29", "2012-05-11", 
              "2012-05-26", "2012-06-15", "2012-08-04", "2012-08-20", "2012-09-28", "2012-10-28", "2012-11-04", 
              "2012-11-16", "2012-11-17", "2012-12-01", "2013-01-10", "2013-02-23", "2013-04-05"]
    for file in FILE_DATES:
        plot_ground_truth(file)