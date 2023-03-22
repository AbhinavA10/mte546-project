# !/usr/bin/python
#
# Convert the ms25 csv files to a rosbag
#
# To call:
#
#   python ms25_to_rosbag.py ms25.csv ms25.bag
#

import rosbag, rospy
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout

import sys
import numpy as np

def main(args):

    if len(sys.argv) < 2:
        print 'Please specify ms25 file'
        return 1

    if len(sys.argv) < 3:
        print 'Please specify output rosbag file'
        return 1

    ms25 = np.loadtxt(sys.argv[1], delimiter = ",")

    utimes= ms25[:, 0]

    mag_xs = ms25[:, 1]
    mag_ys = ms25[:, 2]
    mag_zs = ms25[:, 3]

    accel_xs = ms25[:, 4]
    accel_ys = ms25[:, 5]
    accel_zs = ms25[:, 6]

    rot_rs = ms25[:, 7]
    rot_ps = ms25[:, 8]
    rot_hs = ms25[:, 9]

    bag = rosbag.Bag(sys.argv[2], 'w')

    try:

        for i, utime in enumerate(utimes):

            timestamp = rospy.Time.from_sec(utime/1e6)

            layout = MultiArrayLayout()
            layout.dim = [MultiArrayDimension()]
            layout.dim[0].label = "xyz"
            layout.dim[0].size = 3
            layout.dim[0].stride = 1

            mag = Float64MultiArray()
            mag.data = [mag_xs[i], mag_ys[i], mag_zs[i]]
            mag.layout = layout

            accel = Float64MultiArray()
            accel.data = [accel_xs[i], accel_ys[i], accel_zs[i]]
            accel.layout = layout

            layout_rph = MultiArrayLayout()
            layout_rph.dim = [MultiArrayDimension()]
            layout_rph.dim[0].label = "rph"
            layout_rph.dim[0].size = 3
            layout_rph.dim[0].stride = 1

            rot = Float64MultiArray()
            rot.data = [rot_rs[i], rot_ps[i], rot_hs[i]]
            rot.layout = layout_rph

            bag.write('mag', mag, t=timestamp)
            bag.write('accel', accel, t=timestamp)
            bag.write('rot', rot, t=timestamp)

    finally:
        bag.close()


    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
