import rosbag
import sys

channels = ['fix_mode', 'num_sats', 'latitude', 'longitude', 'altitude', 'track', 'speed', 'mag', 'accel', 'rot']
channels += ['euler']
channels += ['velodyne_packet']
channels += ['hokuyo_30m_packet']
channels += ['hokuyo_4m_packet']

print channels
print sys.argv[1]

bag = rosbag.Bag(sys.argv[1])
for topic, msg, t in bag.read_messages(topics=channels):
    print topic, msg, t
    print type(t)
    raw_input()
bag.close()
