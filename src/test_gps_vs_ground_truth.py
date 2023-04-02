# General imports
import utils

# Imports for reading ground truth
import read_gps
import read_ground_truth 

file_date = "2013-04-05"

ground_truth = read_ground_truth.read_ground_truth(file_date) # 107 Hz
gps_data     = read_gps.read_gps(file_date, use_rtk=False) # 2.5 or 6 Hz
x_true     = ground_truth[:, 1] # North
y_true     = ground_truth[:, 2] # East
gps_x         =   gps_data[:,1]
gps_y         =   gps_data[:,2]

utils.plot_position_comparison_2D_scatter(gps_x, gps_y, x_true, y_true, "Consumer GPS", "Ground Truth")
#utils.export_to_kml(gps_x, gps_y, x_true, y_true, "Consumer GPS", "Ground Truth", subsample=True)

ground_truth = read_ground_truth.read_ground_truth(file_date) # 107 Hz
gps_data     = read_gps.read_gps(file_date, use_rtk=True) # 2.5 or 6 Hz
x_true     = ground_truth[:, 1] # North
y_true     = ground_truth[:, 2] # East
gps_x         =   gps_data[:,1]
gps_y         =   gps_data[:,2]

utils.plot_position_comparison_2D_scatter(gps_x, gps_y, x_true, y_true, "GPS RTK", "Ground Truth")
utils.export_to_kml(gps_x, gps_y, x_true, y_true, "GPS RTK", "Ground Truth", subsample=True)
