# !/usr/bin/python
#
# Convert the velodyne_hits binary files to a rosbag
#
# To call:
#
#   python vel_to_rosbag.py velodyne_hits.bin vel.bag
#

import rosbag, rospy
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout

import sys
import numpy as np
import struct

def convert(x_s, y_s, z_s):

    scaling = 0.005 # 5 mm
    offset = -100.0

    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset

    return x, y, z

def verify_magic(s):

    magic = 44444

    m = struct.unpack('<HHHH', s)

    return len(m)>=3 and m[0] == magic and m[1] == magic and m[2] == magic and m[3] == magic

def main(args):

    if len(sys.argv) < 2:
        print 'Please specify velodyne hits file'
        return 1

    if len(sys.argv) < 3:
        print 'Please specify output rosbag file'
        return 1

    f_bin = open(sys.argv[1], "r")

    bag = rosbag.Bag(sys.argv[2], 'w')

    try:
        while True:

            magic = f_bin.read(8)
            if magic == '': # eof
                break

            if not verify_magic(magic):
                print "Could not verify magic"

            num_hits = struct.unpack('<I', f_bin.read(4))[0]
            utime = struct.unpack('<Q', f_bin.read(8))[0]

            f_bin.read(4) # padding

            # Read all hits
            data = []
            for i in range(num_hits):

                x = struct.unpack('<H', f_bin.read(2))[0]
                y = struct.unpack('<H', f_bin.read(2))[0]
                z = struct.unpack('<H', f_bin.read(2))[0]
                i = struct.unpack('B', f_bin.read(1))[0]
                l = struct.unpack('B', f_bin.read(1))[0]

                x, y, z = convert(x, y, z)

                data += [x, y, z, float(i), float(l)]

            # Now write out to rosbag
            timestamp = rospy.Time.from_sec(utime/1e6)

            layout = MultiArrayLayout()
            layout.dim = [MultiArrayDimension(), MultiArrayDimension()]
            layout.dim[0].label = "hits"
            layout.dim[0].size = num_hits
            layout.dim[0].stride = 5
            layout.dim[1].label = "xyzil"
            layout.dim[1].size = 5
            layout.dim[1].stride = 1

            vel = Float64MultiArray()
            vel.data = data
            vel.layout = layout

            bag.write('velodyne_packet', vel, t=timestamp)
    except Exception:
        print 'End of File'
    finally:
        f_bin.close()
        bag.close()

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
