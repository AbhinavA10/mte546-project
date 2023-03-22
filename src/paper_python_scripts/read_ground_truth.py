# !/usr/bin/python
#
# Example code to read and plot the ground truth data.
#
# Note: The ground truth data is provided at a high rate of about 100 Hz. To
# generate this high rate ground truth, a SLAM solution was used. Nodes in the
# SLAM graph were not added at 100 Hz, but rather about every 8 meters. In
# between the nodes in the SLAM graph, the odometry was used to interpolate and
# provide a high rate ground truth. If precise pose is desired (e.g., for
# accumulating point clouds), then we recommend using only the ground truth
# poses that correspond to the nodes in the SLAM graph. This can be found by
# inspecting the timestamps in the covariance file.
#
# To call:
#
#   python read_ground_truth.py groundtruth.csv covariance.csv
#

import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

def main(args):

    if len(sys.argv) < 3:
        print 'Please specify ground truth and covariance files'
        return 1

    gt = np.loadtxt(sys.argv[1], delimiter = ",")
    cov = np.loadtxt(sys.argv[2], delimiter = ",")

    t_cov = cov[:, 0]

    # Note: Interpolation is not needed, this is done as a convience
    interp = scipy.interpolate.interp1d(gt[:, 0], gt[:, 1:], kind='nearest', axis=0)
    pose_gt = interp(t_cov)

    # NED (North, East Down)
    x = pose_gt[:, 0]
    y = pose_gt[:, 1]
    z = pose_gt[:, 2]

    r = pose_gt[:, 3]
    p = pose_gt[:, 4]
    h = pose_gt[:, 5]

    plt.figure()
    plt.scatter(y, x, 1, c=-z, linewidth=0)    # Note Z points down
    plt.axis('equal')
    plt.title('Ground Truth Position of Nodes in SLAM Graph')
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.colorbar()

    plt.show()

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
