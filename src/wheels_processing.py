
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

file_name = 'wheels.csv'

def read_process_wheels(file_name):
    
    wheels = np.loadtxt(file_name, delimiter = ",")
    # time in us 
    t = wheels[:, 0]
    # adjust to start at time 0 (for now)
    t = t-t[0]
    # convert from us to s
    t_sec = []
    for time in t:
        t_new = time*10**-6
        t_sec.append(t_new)
        
    t = wheels[:, 0]

    leftv_OG  = wheels[:, 1] 
    rightv_OG = wheels[:, 2] 

    # apply rolling average
    leftv_df = pd.DataFrame(leftv_OG)
    leftv = leftv_df.rolling(1000, min_periods=1).mean()
    leftv = leftv.to_numpy()
    rightv_df = pd.DataFrame(rightv_OG)
    rightv = rightv_df.rolling(1000, min_periods=1).mean()
    rightv = rightv.to_numpy()
    
    # # plot actual accel vs smoothed accel
    plt.figure()
    # left wheel
    plt.subplot(1, 2, 1)
    plt.plot(t_sec, leftv_OG, 'b')
    plt.plot(t_sec, leftv, 'r')
    plt.legend(['Original', 'Smooth'])
    plt.title('Left Wheel Velocity')
    plt.xlabel('time (s)')
    plt.ylabel('m/s')    
    # right wheel
    plt.subplot(1, 2, 2)
    plt.plot(t_sec, rightv_OG, 'b')
    plt.plot(t_sec, rightv, 'r')
    plt.legend(['Original', 'Smooth'])
    plt.title('Right Wheel Velocity')
    plt.xlabel('time (s)')
    plt.ylabel('m/s')   
    plt.show()
    
    estimate_wheel_noise(rightv_OG, rightv)    
       
def estimate_wheel_noise(original_data, smooth_data):
    print("estimating noise")
    error = []
    print(len(original_data))
    for i in range(0, len(original_data)):
        error.append(original_data[i] - smooth_data[i])
        
    error = np.array(error)
    
    # norm fit of data
    (mu, sigma) = norm.fit(error)
    print(f'sigma = {sigma}')
    # plot
    plt.figure()
    n, bins, patches = plt.hist(error, bins=30, alpha=0.75, facecolor='blue')
    # x = norm.rvs(size=100000)
    # y = np.linspace(-4,4, 1000)
    # bin_width = (x.max() - x.min())/30
    # plt.plot(y, norm.pdf(y) * 100000 * 30)
    x = np.linspace(-3, 3, 100)
    y = norm.pdf(x, mu, sigma)
    plt.plot(x, y, 'r')
    plt.xlabel('Velocity Error (m/s)')
    plt.ylabel('Count')
    plt.title('Encoder Error Histogram (Right Wheel)')
    plt.grid(True)
    plt.show()
    
if __name__ == '__main__':
    read_process_wheels(file_name)