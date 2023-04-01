# Helper functions

import numpy as np
from lxml import etree

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
    average_timestep = length/len(timestamps)/1000000
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
    x = x-72.5 # Tuned offset to adjust with Google Earth
    y = y-107 # Tuned offset to adjust with Google Earth
    lat = np.arcsin(x/r_ns) + LAT_0
    lon = np.arcsin(y/(r_ew*np.cos(LAT_0))) + LON_0
    lat = np.rad2deg(lat) # Latitude, in degrees
    lon = np.rad2deg(lon) # Longitude, in degrees
    return (lat,lon)

def _format_lat_lon(lat: list, lon:list):
    """Format coords for KML file"""
    l = ["            {},{},1".format(lo, la) for la, lo in zip(lat, lon)]
    return "\n".join(l)

def export_to_kml(x: list, y:list, x_gt: list, y_gt:list):
    """Export list of local frame ground truth and estimated coords to KML file
    Parameters: 
    - local frame estimated coords (x,y) = (North, East) [meters]
    - local frame ground truth coords (x_gt,y_gt) = (North, East) [meters]
    Returns: KML file export
    """
    root = etree.parse('template.kml').getroot()
    tags = root.findall('.//coordinates', {None : 'http://www.opengis.net/kml/2.2'}) # recurisvely find all coordinate tags in namespace
    ground_truth_tag = tags[0]
    estimation_tag = tags[1]
    if x_gt is not None:
        #TODO: skip datapoints in list to reduce file size. Ground truth has ~500,000 points
        x_gt = x_gt[1::200] # sample every 200th point
        y_gt = y_gt[1::200] # sample every 200th point
        print(len(x_gt))
        lat_gt,lon_gt = local_to_gps_coord(x_gt,y_gt)
        formatted_coords_gt = _format_lat_lon(lat_gt, lon_gt)
        ground_truth_tag.text = formatted_coords_gt 
    if x is not None:
        lat_est,lon_est = local_to_gps_coord(x,y)
        formatted_coords_est = _format_lat_lon(lat_est, lon_est)
        estimation_tag.text = formatted_coords_est
    with open('output.kml', 'wb') as f:
        f.write(etree.tostring(root, xml_declaration=True, encoding='UTF-8', pretty_print=True))

def plot_state_comparison():
    # Plot EKF states over time vs. Ground Truth
    #TODO
    pass
