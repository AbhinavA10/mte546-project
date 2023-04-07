import numpy as np

import utils

def load_results(filename):
    """Load EKF results from exported numpy file, for plotting"""
    
    with open(f"./output/{filename}", 'rb') as f:
        x_est           = np.load(f)
        P_est           = np.load(f)
        x_true_arr      = np.load(f)
        y_true_arr      = np.load(f)
        theta_true_arr  = np.load(f)
        t               = np.load(f)
            
    utils.plot_position_comparison_2D(x_est[:,0], x_est[:,1], x_true_arr, y_true_arr, filename, "Ground Truth")
    utils.plot_states(x_est, P_est, x_true_arr, y_true_arr, theta_true_arr, t)    

if __name__=="__main__":
    load_results("2012-08-04_Estimated - Wheels with GPS 1Hz.npy")
