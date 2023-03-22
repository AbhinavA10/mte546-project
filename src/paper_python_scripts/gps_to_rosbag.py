# !/usr/bin/python
#
# Convert the gps csv files to a rosbag
#
# To call:
#
#   python gps_to_rosbag.py gps.csv gps.bag
#

import rosbag, rospy
from std_msgs.msg import Float64, UInt16
from sensor_msgs.msg import NavSatStatus, NavSatFix

import sys
import numpy as np

def main(args):

    if len(sys.argv) < 2:
        print 'Please specify gps file'
        return 1

    if len(sys.argv) < 3:
        print 'Please specify output rosbag file'
        return 1

    gps = np.loadtxt(sys.argv[1], delimiter = ",")

    utimes = gps[:, 0]
    modes = gps[:, 1]
    num_satss = gps[:, 2]
    lats = gps[:, 3]
    lngs = gps[:, 4]
    alts = gps[:, 5]
    tracks = gps[:, 6]
    speeds = gps[:, 7]

    bag = rosbag.Bag(sys.argv[2], 'w')

    try:

        for i, utime in enumerate(utimes):

            timestamp = rospy.Time.from_sec(utime/1e6)

            status = NavSatStatus()

            if modes[i]==0 or modes[i]==1:
                status.status = NavSatStatus.STATUS_NO_FIX
            else:
                status.status = NavSatStatus.STATUS_FIX

            status.service = NavSatStatus.SERVICE_GPS

            num_sats = UInt16()
            num_sats.data = num_satss[i]

            fix = NavSatFix()
            fix.status = status

            fix.latitude = np.rad2deg(lats[i])
            fix.longitude = np.rad2deg(lngs[i])
            fix.altitude = alts[i]

            track = Float64()
            track.data = tracks[i]

            speed = Float64()
            speed.data = speeds[i]

            bag.write('fix', fix, t=timestamp)
            bag.write('track', track, t=timestamp)
            bag.write('speed', speed, t=timestamp)

    finally:
        bag.close()


    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
