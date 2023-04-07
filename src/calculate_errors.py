# Read Error CSV and analyze Error metrics
import pandas as pd

df = pd.read_csv("output/error_results.csv", header=None, names=["date", "type", "mean_error", "std_dev_error"])
ekf_modes = df["type"].unique() # list of EKF modes in 2nd column of csv

print(df.head())

#TODO: Cycle through each configs, and calculate mean of error metrics, across all paths/dates
#TODO: Draw bar chart for a single date or few dates, showing trend of error across EKF modes