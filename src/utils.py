# Helper functions

import numpy as np
from lxml import etree
import matplotlib.pyplot as plt

def _compute_gps_conversion_params():
    # Compute radii of Earth at origin of linearization
    LAT_0 = 0.738167915410646 # in radians, lat[0]
    LON_0 = -1.46098650670922 # in radians, lon[0]
    re = 6378135 # Earth Equatorial Radius [m]
    rp = 6356750 # Earth Polar Radius [m]
    r_ns = pow(re*rp,2)/pow(pow(re*np.cos(LAT_0),2)+pow(rp*np.sin(LAT_0),2),3/2)
    r_ew = pow(re,2)/pow(pow(re*np.cos(LAT_0),2)+pow(rp*np.sin(LAT_0),2),1/2)
    return (r_ns, r_ew, LAT_0, LON_0)

def calculate_hz(sensor_name:str, timestamps: list):
    """Calculate Hz of Sensor Data"""
    length = timestamps[-1] - timestamps[0]
    average_timestep = length/len(timestamps)
    hz = 1/average_timestep
    print(f"{sensor_name} data, Hz: {hz}")

def gps_to_local_coord(lat: list, lon:list):
    """Convert list of latitude/longitude to local frame
    Parameters: latitude, longitudes in radians
    Returns: local frame coords (x,y) = (North, East) [meters]
    """
    r_ns, r_ew, LAT_0, LON_0 = _compute_gps_conversion_params()
    # Convert GPS coordinates to linearized local frame
    x = np.sin(lat - LAT_0) * r_ns # North
    y = np.sin(lon - LON_0) * r_ew * np.cos(LAT_0) # East
    return (x,y)

def local_to_gps_coord(x: list, y:list):
    """Convert list of local frame coords to GPS latitude/longitude
    Parameters: local frame coords (x,y) = (North, East) [meters]
    Returns: GPS lat/lon in degrees
    """
    r_ns, r_ew, LAT_0, LON_0 = _compute_gps_conversion_params()
    # Convert linearized local frame to GPS
    x = x - 76.50582406697139 # Tuned offset to adjust with Google earth
    y = y - 108.31373031919006 # Tuned offset to adjust with Google earth
    lat = np.arcsin(x/r_ns) + LAT_0
    lon = np.arcsin(y/(r_ew*np.cos(LAT_0))) + LON_0
    lat = np.rad2deg(lat) # Latitude, in degrees
    lon = np.rad2deg(lon) # Longitude, in degrees
    return (lat,lon)

def _format_lat_lon(lat: list, lon:list):
    """Format coords for KML file"""
    l = ["            {},{},1".format(lo, la) for la, lo in zip(lat, lon)]
    return "\n".join(l)

def export_to_kml(x1: list, y1:list, x2: list, y2:list, label1:str, label2:str, subsample=False):
    """Export list of local frame ground truth and estimated coords to KML file
    Parameters: 
    - local frame estimated coords (x,y) = (North, East) [meters]
    - local frame ground truth coords (x_gt,y_gt) = (North, East) [meters]
    Returns: KML file export
    """
    root = etree.parse('template.kml').getroot()
    tags = root.findall('.//name', {None : 'http://www.opengis.net/kml/2.2'}) # recurisvely find all coordinate tags in namespace
    name_tag_1 = tags[1] # 2nd name tag in kml
    name_tag_2 = tags[2] # 3rd name tag in kml
    name_tag_1.text = label1
    name_tag_2.text = label2
    tags = root.findall('.//coordinates', {None : 'http://www.opengis.net/kml/2.2'}) # recurisvely find all coordinate tags in namespace
    coord_tag_1 = tags[0]
    coord_tag_2 = tags[1]
    if x1 is not None:
        # if subsample:
            # x1 = x1[1::200] # sample every 200th point
            # y1 = y1[1::200] # sample every 200th point
        lat1,lon1 = local_to_gps_coord(x1,y1)
        formatted_coords1 = _format_lat_lon(lat1, lon1)
        coord_tag_1.text = formatted_coords1
    if x2 is not None:
        if subsample:
            # Ground truth has ~500,000 points
            x2 = x2[1::200] # sample every 200th point
            y2 = y2[1::200] # sample every 200th point
        lat2,lon2 = local_to_gps_coord(x2,y2)
        formatted_coords2 = _format_lat_lon(lat2, lon2)
        coord_tag_2.text = formatted_coords2 
    with open('output.kml', 'wb') as f:
        f.write(etree.tostring(root, xml_declaration=True, encoding='UTF-8', pretty_print=True))

def plot_position_comparison_2D(x1: list, y1:list, x2: list, y2:list, label1:str, label2:str):
    """Plot local frame ground truth and estimated coords on x-y line plot
    Parameters: 
    - local frame estimated coords (x,y) = (North, East) [meters]
    - local frame ground truth coords (x_gt,y_gt) = (North, East) [meters]
    """
    plt.figure()
    plt.plot(y1, x1, c='b', linewidth=1, label=label1) # plot flipped since North,East
    plt.plot(y2, x2, c='r', linewidth=1, label=label2)
    plt.axis('equal')
    plt.legend()
    plt.title('Comparison')
    plt.xlabel('East [m]')
    plt.ylabel('North [m]')
    plt.show()


def plot_position_comparison_2D_scatter(x1: list, y1:list, x2: list, y2:list, label1:str, label2:str):
    """Plot local frame ground truth and estimated coords on x-y sacatter plot
    Parameters: 
    - local frame estimated coords (x,y) = (North, East) [meters]
    - local frame ground truth coords (x_gt,y_gt) = (North, East) [meters]
    """
    plt.figure()
    plt.scatter(y1, x1, s=1, c='b', linewidth=0, label=label1) # plot flipped since North,East
    plt.scatter(y2, x2, c='r', s=1, linewidth=0, label=label2)
    plt.axis('equal')
    plt.legend()
    plt.title('Comparison')
    plt.xlabel('East [m]')
    plt.ylabel('North [m]')
    plt.show(),

def plot_states(x_est:np.ndarray, P_est:np.ndarray, x_true_arr:np.ndarray, y_true_arr:np.ndarray, theta_true_arr:np.ndarray, t:np.ndarray):

    # x_est = x | y | xdot | ydot | theta | omega
    x       = x_est[:,0]
    y       = x_est[:,1]
    xdot    = x_est[:,2]
    ydot    = x_est[:,3]
    theta   = x_est[:,4]
    omega   = x_est[:,5]
    
    P = P_est.diagonal(0,1,2)  # diagonals represent uncertainities
    P_x     = P[:,0]
    P_y     = P[:,1]
    P_xdot  = P[:,2]
    P_ydot  = P[:,3]
    P_theta = P[:,4]
    P_omega = P[:,5]
    
    # x,y,theta over time vs Ground Truth, with uncertainties
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t,x, label="x")
    plt.plot(t,x_true_arr, label="x_true")
    plt.plot(t,x+np.sqrt(P_x),'r--', label="_Uncertainity Bound") # _ hides from legend
    plt.plot(t,x-np.sqrt(P_x),'r--', label="_Uncertainity Bound")
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('X Position [m]')
    plt.subplot(3, 1, 2)
    plt.plot(t,y, label="y")
    plt.plot(t,y_true_arr, label="y_true")
    plt.plot(t,y+np.sqrt(P_y),'r--', label="_Uncertainity Bound")
    plt.plot(t,y-np.sqrt(P_y),'r--', label="_Uncertainity Bound")
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Y Position [m]')
    plt.subplot(3, 1, 3)
    plt.plot(t,theta, label="theta")
    plt.plot(t,theta_true_arr, label="theta_true")
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Theta [rad]')

    # Plot other states over time, with uncertainites
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t,xdot, label="x_dot")
    plt.plot(t,xdot+np.sqrt(P_xdot),'r--', label="_Uncertainity Bound")
    plt.plot(t,xdot-np.sqrt(P_xdot),'r--', label="_Uncertainity Bound")
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('X Velocity [m/s]')
    plt.subplot(3, 1, 2)
    plt.plot(t,ydot, label="y_dot")
    plt.plot(t,ydot+np.sqrt(P_ydot),'r--', label="_Uncertainity Bound")
    plt.plot(t,ydot-np.sqrt(P_ydot),'r--', label="_Uncertainity Bound")
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Y Velocity [m/s]')
    plt.subplot(3, 1, 3)
    plt.plot(t,omega, label="omega")
    plt.plot(t,omega+np.sqrt(P_omega),'r--', label="_Uncertainity Bound")
    plt.plot(t,omega-np.sqrt(P_omega),'r--', label="_Uncertainity Bound")
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Omega [rad/s]')

    # plot errors
    euclidean_error = np.sqrt(np.power(x_est[:,0] - x_true_arr, 2) + np.power(x_est[:,1] - y_true_arr,2))
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t,euclidean_error)
    plt.title('Euclidean Distance Error from Ground Truth')
    plt.xlabel('Time [s]')
    plt.ylabel('Euclidean Dist. From Ground Truth [m]')
    heading_error = x_est[:,4] - theta_true_arr
    plt.subplot(2, 1, 2)
    plt.plot(t, heading_error)
    plt.title('Error in Theta from Ground Truth')
    plt.xlabel('Time [s]')
    plt.ylabel('Error in Theta [rad]')
    
    plt.show()