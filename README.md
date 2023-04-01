# MTE 546 Final Project
- Localizing Robot on U of M campus

## Setup
- Unzip `dataset.zip` into `./src/dataset`
- `pip install matplotlib numpy pandas sympi scipy`

## Running:
From `src` folder, 
- `python .\examples\read_gps.py .\dataset\2012-01-08\gps.csv`
- `python .\examples\read_gps.py .\dataset\2012-01-08\gps_rtk.csv`
- `python .\examples\read_ground_truth.py .\dataset\2012-01-08\groundtruth_2012-01-08.csv .\dataset\2012-01-08\odometry_cov_100hz.csv`