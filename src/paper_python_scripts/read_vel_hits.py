# !/usr/bin/python
#
# Example code to go through the velodyne_hits.bin
# file and read timestamps, number of hits, and the
# hits in each packet.
#
#
# To call:
#
#   python read_vel_hits.py velodyne.bin
#

import sys
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

    return len(m)>=4 and m[0] == magic and m[1] == magic and m[2] == magic and m[3] == magic

def main(args):

    if len(sys.argv) < 2:
        print "Please specifiy input bin file"
        return 1

    f_bin = open(sys.argv[1], "r")

    total_hits = 0
    first_utime = -1
    last_utime = -1

    while True:

        magic = f_bin.read(8)
        if magic == '': # eof
            break

        if not verify_magic(magic):
            print "Could not verify magic"

        num_hits = struct.unpack('<I', f_bin.read(4))[0]
        utime = struct.unpack('<Q', f_bin.read(8))[0]

        padding = f_bin.read(4) # padding

        print "Have %d hits for utime %ld" % (num_hits, utime)

        total_hits += num_hits
        if first_utime == -1:
            first_utime = utime
        last_utime = utime

        for i in range(num_hits):

            x = struct.unpack('<H', f_bin.read(2))[0]
            y = struct.unpack('<H', f_bin.read(2))[0]
            z = struct.unpack('<H', f_bin.read(2))[0]
            i = struct.unpack('B', f_bin.read(1))[0]
            l = struct.unpack('B', f_bin.read(1))[0]

            x, y, z = convert(x, y, z)
            s = "%5.3f, %5.3f, %5.3f, %d, %d" % (x, y, z, i, l)

            print s

        raw_input("Press enter to continue...")

    f_bin.close()

    print "Read %d total hits from %ld to %ld" % (total_hits, first_utime, last_utime)

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
