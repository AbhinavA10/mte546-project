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
- `python EKF.py 2013-04-05`: Run EKF with config given in `EKF.py` for the given path
- `python run_all.py`: Run EKF with config given in `EKF.py` for all paths in the dataset

## EKF Configuration

| `USE_WHEEL_AS_INPUT` | `USE_GPS_FOR_CORRECTION` | `USE_WHEEL_FOR_CORRECTION` | `USE_GPS_AS_INPUT` | Configuration Meaning |
|---|---|---|---|---|
| x | x | x | 1 | Use only GPS to estimate state |
| 0 | 0 | 0 | 0 | Use IMU as input, no corrections |
| 0 | 0 | 1 | 0 | Use IMU as input, correct with Wheels |
| 0 | 1 | 1 | 0 | Use IMU as input, correct with GPS and Wheels |
| 1 | 0 | x | 0 | Use Wheel as input, no corrections. Implicitly uses IMU's theta |
| 1 | 1 | x | 0 | Use Wheel as input, correct with GPS |


## Paths
The following paths do not have readable wheel velocities:
- `2012-01-08`
- `2012-01-22`
- `2012-02-12`
- `2012-03-17`
- `2012-05-26`
- `2012-06-15`