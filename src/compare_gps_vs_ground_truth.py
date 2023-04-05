# General imports
import utils

# Imports for reading ground truth
import read_gps
import read_ground_truth 
import numpy as np
import pandas as pd 

file_date = "2013-04-05"

ground_truth = read_ground_truth.read_ground_truth(file_date) # 107 Hz
gps_data     = read_gps.read_gps(file_date, use_rtk=False) # 2.5 or 6 Hz
x_true     = ground_truth[:, 1] # North
y_true     = ground_truth[:, 2] # East
t = ground_truth[:,0]
gt_data = np.array([])
gt_data = np.vstack((t, x_true, y_true)).T
pd.DataFrame(gt_data).to_csv("gt_data.csv", header=None, index=None)

t = gps_data[:,0]
t = t-t[0]
gps_x         =   gps_data[:,1]
gps_y         =   gps_data[:,2]
gps_data = np.array([])
gps_data = np.vstack((t, gps_x, gps_y)).T
pd.DataFrame(gps_data).to_csv("gps_data.csv", header=None, index=None)

#utils.plot_position_comparison_2D_scatter(gps_x, gps_y, x_true, y_true, "Consumer GPS", "Ground Truth")
#utils.export_to_kml(gps_x, gps_y, x_true, y_true, "Consumer GPS", "Ground Truth", subsample=True)

gps_data     = read_gps.read_gps(file_date, use_rtk=True) # 2.5 or 6 Hz
gps_x         =   gps_data[:,1]
gps_y         =   gps_data[:,2]

#utils.plot_position_comparison_2D_scatter(gps_x, gps_y, x_true, y_true, "GPS RTK", "Ground Truth")
# utils.export_to_kml(gps_x, gps_y, x_true, y_true, "GPS RTK", "Ground Truth", subsample=True)
