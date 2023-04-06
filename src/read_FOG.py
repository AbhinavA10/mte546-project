# !/usr/bin/python3

# Need to add requirements:

import numpy as np
import utils
import matplotlib.pyplot as plt

# Accept a filepath to the CSV of interest and return Numpy array with data
# def read_FOG(filepath):
#     data = pd.read_csv("dataset/2013-04-05/kvh.csv", header=None)
#     data = data - [data.iloc[0,0], 0]
#     data[2] = data[1].rolling(1000).mean()
#     return(data.to_numpy())

def read_FOG(dataset_date):
    filepath = f"dataset/{dataset_date}/kvh.csv"
    fog = np.loadtxt(filepath, delimiter = ",")
    rot_h_OG   = fog[:, 1]
    t          = fog[:, 0]
    
    # Relative timestamps
    # t = t-t[0]
    t = t/1000000
    utils.calculate_hz("FOG", t) # 47 Hz

    # have the following format:
    # timestamp | ax_robot | ay_robot | omega
    fog_data = np.array([])
    fog_data = np.vstack((t, rot_h_OG)).T
    return fog_data

# Accepta a filepath to the CSV of interest and plot the FOG data
def plot_FOG(filepath):
    array = read_FOG(filepath)

    plt.plot(array[:,0], array[:,1])
    plt.plot(array[:,0], array[:,2])
    
    plt.yticks(np.arange(0, 7, step=np.pi/8), fontsize=15)
    plt.xticks(fontsize=15)

    plt.legend(["Original Angle Data","Angle Data Moving Average"], fontsize=20)
    plt.title("FOG Angle versus Time", fontsize=30)
    plt.xlabel("Time [s]", fontsize=20)
    plt.ylabel("Angle [rads]", fontsize=20)

    plt.grid()
    plt.show()
    

if __name__ == "__main__":
    plot_FOG("2013-04-05")
