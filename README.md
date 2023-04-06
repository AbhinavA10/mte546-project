# MTE 546 Final Project
- Localizing Robot on U of M campus

## Setup
- Unzip `dataset.zip` into `./src/dataset`
- `pip install matplotlib numpy pandas sympy scipy lxml`

## Running:

From `src` folder, 
- `python read_ground_truth.py`
- `python read_gps.py`
- `python read_wheels.py`
- `python read_imu.py`
- `python IMU_processing.py`
- `python EKF.py 2013-04-05`
- `python run_all.py`

 Other Examples:
- `python .\examples\read_gps.py .\dataset\2012-01-08\gps.csv`
- `python .\examples\read_gps.py .\dataset\2012-01-08\gps_rtk.csv`
- `python .\examples\read_ground_truth.py .\dataset\2012-01-08\groundtruth_2012-01-08.csv .\dataset\2012-01-08\odometry_cov_100hz.csv`
- `python .\examples\export_kml.py` Sample script for exporting data to kml


## EKF Configuration

| `USE_WHEEL_AS_INPUT` | `USE_GPS_FOR_CORRECTION` | `USE_WHEEL_FOR_CORRECTION` | `USE_GPS_AS_INPUT` | Configuration Meaning |
|---|---|---|---|---|
| x | x | x | 1 | Use only GPS to estimate state |
| 0 | 0 | 0 | 0 | Use IMU as input, no corrections |
| 0 | 0 | 1 | 0 | Use IMU as input, correct with Wheels |
| 0 | 1 | 1 | 0 | Use IMU as input, correct with GPS and Wheels |
| 1 | 0 | x | 0 | Use Wheel as input, no corrections. Implicitly uses IMU's theta |
| 1 | 1 | x | 0 | Use Wheel as input, correct with GPS |
