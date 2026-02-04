# ECE 276A Project 1: Orientation Tracking

## Overview

This project presents a method for estimating the 3D orientation trajectory of a rigid body using datasets from an Inertial Measurement Unit (IMU). While gyroscopes provide angular velocity, the collected data tends to drift over time. To address this, the Projected Gradient Descent (PGD) algorithm is employed to minimize a cost function under the unit quaternion transformation. Furthermore, the optimized orientation estimates are applied to build panorama images from the associated camera data.

## Requirements

- **Python**: 3.8 or higher
- **Dependencies**: See `requirements.txt`

### Installation

```bash
pip install -r requirements.txt
```

**Required Libraries:**

- NumPy
- Pandas
- Matplotlib
- SciPy
- OpenCV
- PyTorch
- transforms3d

## Project Structure

```
ECE276A_PR1/
├── code/
│   ├── PR1.py                    # Main processing pipeline
│   ├── quaternion.py             # Quaternion operations (NumPy)
│   ├── quaternion_torch.py       # Quaternion operations (PyTorch)
│   ├── plot.py                   # Visualization functions
│   └── panorama.py               # Panorama stitching
├── data/
│   ├── trainset/                 # Training datasets (1-9)
│   │   ├── cam/                  # Camera images
│   │   ├── imu/                  # IMU raw data
│   │   └── vicon/                # Vicon ground truth
│   ├── testset/                  # Test datasets (10-11)
│   │   ├── cam/                  # Camera images
│   │   └── imu/                  # IMU raw data
│   ├── 1/                        # Outputs for dataset 1
│   ├── 2/                        # Outputs for dataset 2
│   ├── ...
│   ├── 9/                        # Outputs for dataset 9
│   ├── 10/                       # Outputs for dataset 10
│   └── 11/                       # Outputs for dataset 11
└── README.md
```

## Usage

### 1. Process All Datasets

```bash
cd code
python PR1.py
```

This will:

- Load IMU and Vicon data
- Save denoised raw data to csv files for each dataset
- Perform gradient descent optimization (800 iterations) (with `quaternion_torch.py`)
- Generate orientation plots and cost curves (with `plot.py`)
- Save optimized orientations to CSV

### 2. Generate Panoramas

```bash
python panorama.py
```

Creates panoramic images using optimized orientations.

1. Load camera images
2. Align camera timestamps with IMU timestamps
3. Project pixels onto unit sphere using FOV (60° horizontal, 45° vertical)
4. Rotate to world frame using optimized orientation
5. Map to rectangular projection

## Quaternion Operation

`quaternion_torch.py`

Contains PyTorch-based quaternion operations and a cost function for projected gradient descent.

**Functions:**

- `quaternion_multiplication(q, p)`: Quaternion multiplication
- `quaternion_exponential(v)`: Quaternion exponential
- `quaternion_inverse(q)`: Quaternion inverse
- `quaternion_log(q)`: Quaternion logarithm
- `cost_function(qt, omega, at, tau)`: Cost function for orientation optimization
  - Motion cost + observation cost

**Hyperparameters:**

- Iterations: 800
- Learning rate: 0.001

## Output Files

For each dataset `d`:

```
data/{d}/
├── orientation_data.csv              # Estimated orientation
├── orientation_data_optimized.csv    # Optimized orientation
├── RPY_groundtruth_estimated.png     # Estimated vs Ground Truth (trainset)
├── RPY_groundtruth_optimized.png     # Optimized vs Ground Truth (trainset)
├── RPY_optimized.png                 # Optimized only (testset)
├── cost_curve.png                    # Optimization cost over iterations
└── panorama_optimized.png            # Panorama image
```

## Visualization

Saved Figures in each result folder:

- `RPY_groundtruth_estimated.png`: Estimated orientation vs Vicon ground truth (for trainsets)
- `RPY_groundtruth_optimized.png`: Optimized orientation vs Vicon ground truth (for trainsets)
- `RPY_optimized.png`: Optimized orientation only (for testsets)
- `cost_curve.png`: Optimization cost over iterations
- `panorama_optimized.png`: Panorama image

### Orientation Plots

3 plots from top to bottom: **Roll**, **Pitch**, **Yaw** in each figure:

- **Red**: Estimated/Optimized orientation
- **Blue**: Ground Truth (Vicon, trainset only)
- **X-axis**: Time (seconds)
- **Y-axis**: [-180 (degree), 180 (degree)]

### Cost Curve

- Shows optimization cost over iterations

### Panorama

- Shows projection of estimated sensor trajectory with stitched RGB images

## Author

Yih-Cherng Lin
Department of Electrical and Computer Engineering
University of California, San Diego
<yil388@ucsd.edu>

ECE 276A - Sensing & Estimation in Robotics  
University of California, San Diego
