# !/usr/bin/python
#
# Convert the ms25_euler csv files to a rosbag
#
# To call:
#
#   python ms25_euler_to_rosbag.py ms25_euler.csv ms25_euler.bag
#

import rosbag, rospy
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout

import sys
import numpy as np

def main(args):

    if len(sys.argv) < 2:
        print 'Please specify ms25_euler file'
        return 1

    if len(sys.argv) < 3:
        print 'Please specify output rosbag file'
        return 1

    ms25_euler = np.loadtxt(sys.argv[1], delimiter = ",")

    utimes= ms25_euler[:, 0]

    rs = ms25_euler[:, 1]
    ps = ms25_euler[:, 2]
    hs = ms25_euler[:, 3]

    bag = rosbag.Bag(sys.argv[2], 'w')

    try:

        for i, utime in enumerate(utimes):

            timestamp = rospy.Time.from_sec(utime/1e6)

            layout_rph = MultiArrayLayout()
            layout_rph.dim = [MultiArrayDimension()]
            layout_rph.dim[0].label = "rph"
            layout_rph.dim[0].size = 3
            layout_rph.dim[0].stride = 1

            euler = Float64MultiArray()
            euler.data = [rs[i], ps[i], hs[i]]
            euler.layout = layout_rph

            bag.write('ms25_euler', euler, t=timestamp)

    finally:
        bag.close()


    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
