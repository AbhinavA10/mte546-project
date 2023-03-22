# !/usr/bin/python
#
# Convert the hokuyo_30m binary files to a rosbag
#
# To call:
#
#   python hokuyo_30m_to_rosbag.py hokuyo_30m.bin hokuyo_30m.bag
#

import rosbag, rospy
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout

import sys
import numpy as np
import struct

def convert(x_s):

    scaling = 0.005 # 5 mm
    offset = -100.0

    x = x_s * scaling + offset

    return x

def main(args):

    if len(sys.argv) < 2:
        print 'Please specify hokuyo_30m file'
        return 1

    if len(sys.argv) < 3:
        print 'Please specify output rosbag file'
        return 1

    # hokuyo_30m always has 1081 hits
    num_hits = 1081

    f_bin = open(sys.argv[1], "r")

    bag = rosbag.Bag(sys.argv[2], 'w')

    try:

        while True:

            # Read timestamp
            utime = struct.unpack('<Q', f_bin.read(8))[0]

            r = np.zeros(num_hits)

            for i in range(num_hits):
                s = struct.unpack('<H', f_bin.read(2))[0]
                r[i] = convert(s)

            # Now write out to rosbag
            timestamp = rospy.Time.from_sec(utime/1e6)

            layout = MultiArrayLayout()
            layout.dim = [MultiArrayDimension()]
            layout.dim[0].label = "r"
            layout.dim[0].size = num_hits
            layout.dim[0].stride = 1

            hits = Float64MultiArray()
            hits.data = r
            hits.layout = layout

            bag.write('hokuyo_30m_packet', hits, t=timestamp)

    except Exception:
        print 'End of File'
    finally:
        f_bin.close()
        bag.close()

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
