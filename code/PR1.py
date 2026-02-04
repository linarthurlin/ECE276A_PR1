import pickle
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from transforms3d.euler import quat2euler, mat2euler
from quaternion import update_quaternion
from quaternion_torch import cost_function
from plot import plot_ground_truth_estimated_RPY, plot_ground_truth_optimized_RPY, plot_optimized_RPY

# from rotplot import rotplot

def tic():
    return time.time()

def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

def read_data(fname):
    d = []
    with open(fname, 'rb') as f:
        if sys.version_info[0] < 3:
            d = pickle.load(f)
        else:
            d = pickle.load(f, encoding='latin1')
    return d

def save_to_csv(data_dict, filename):
    import os
    df = pd.DataFrame(data_dict)

    global dataset
    output_dir = f'../data/{dataset}/'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f'{output_dir}{filename}.csv'
    df.to_csv(output_file, index=False)
    return output_file

def plot_cost_curve(cost_history, dataset):
    output_dir = f'../data/{dataset}/'
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, 'b-', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cost', fontsize=12)
    plt.title(f'Cost Function Convergence - Dataset {dataset}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = f'{output_dir}cost_curve.png'
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Cost curve saved to {output_file}")

# Camera data
def cam_processing(camd):
    cam = camd['cam']  # Camera images/data
    cam_ts = camd['ts']  # Camera timestamps
    
    cam_data = {
        'timestamp': cam_ts.flatten()
    }
    
    save_to_csv(cam_data, 'cam_data')
    
    return cam, cam_ts

# IMU data
def imu_processing(imud):
    imu_ts = imud[0, :]
    imu_Ax = imud[1, :]
    imu_Ay = imud[2, :]
    imu_Az = imud[3, :]
    imu_Wx = imud[4, :]
    imu_Wy = imud[5, :]
    imu_Wz = imud[6, :]

    acc_sensitivity = 330  # (mV/g)
    gyro_sensitivity = 3.33 * 180 / np.pi  # (mV/rad/sec)
    acc_scale_factor = 3300 / 1023 / acc_sensitivity
    gyro_scale_factor = 3300 / 1023 / gyro_sensitivity

    # Compute Biases
    bias_frames = 500
    acc_Ax_bias = np.mean(imu_Ax[:bias_frames])
    acc_Ay_bias = np.mean(imu_Ay[:bias_frames])
    acc_Az_bias = np.mean(imu_Az[:bias_frames])
    gyro_Wx_bias = np.mean(imu_Wx[:bias_frames])
    gyro_Wy_bias = np.mean(imu_Wy[:bias_frames])
    gyro_Wz_bias = np.mean(imu_Wz[:bias_frames])
    
    # Compute Value
    imu_Ax = (imu_Ax - acc_Ax_bias) * acc_scale_factor
    imu_Ay = (imu_Ay - acc_Ay_bias) * acc_scale_factor
    imu_Az = (imu_Az - acc_Az_bias) * acc_scale_factor + 1
    imu_Wx = (imu_Wx - gyro_Wx_bias) * gyro_scale_factor
    imu_Wy = (imu_Wy - gyro_Wy_bias) * gyro_scale_factor
    imu_Wz = (imu_Wz - gyro_Wz_bias) * gyro_scale_factor
    
    # Save IMU data to CSV
    imu_data = {
        'timestamp': imu_ts,
        'accel_x': imu_Ax,
        'accel_y': imu_Ay,
        'accel_z': imu_Az,
        'gyro_x': imu_Wx,
        'gyro_y': imu_Wy,
        'gyro_z': imu_Wz
    }

    save_to_csv(imu_data, 'imu_data')

    return imu_ts, imu_Ax, imu_Ay, imu_Az, imu_Wx, imu_Wy, imu_Wz

# Vicon data
def vicon_processing(vicd):
    vic_rots = vicd['rots']  # Rotation matrices (3, 3, N)
    vic_ts = vicd['ts']  # Vicon timestamps
    
    n_samples = vic_rots.shape[2]
    vicon_data = {
        'timestamp': vic_ts.flatten(),
        'r00': vic_rots[0, 0, :],
        'r01': vic_rots[0, 1, :],
        'r02': vic_rots[0, 2, :],
        'r10': vic_rots[1, 0, :],
        'r11': vic_rots[1, 1, :],
        'r12': vic_rots[1, 2, :],
        'r20': vic_rots[2, 0, :],
        'r21': vic_rots[2, 1, :],
        'r22': vic_rots[2, 2, :]
    }
    
    save_to_csv(vicon_data, 'vicon_data')
    
    roll = np.zeros(n_samples)
    pitch = np.zeros(n_samples)
    yaw = np.zeros(n_samples)
    
    for i in range(n_samples):
        R = vic_rots[:, :, i]
        roll[i], pitch[i], yaw[i] = mat2euler(R, 'sxyz')
    
    roll = np.rad2deg(roll)
    pitch = np.rad2deg(pitch)
    yaw = np.rad2deg(yaw)
    
    vicon_euler_data = {
        'timestamp': vic_ts.flatten(),
        'roll': roll,
        'pitch': pitch,
        'yaw': yaw
    }
    
    save_to_csv(vicon_euler_data, 'vicon_euler')

    return vic_rots, vic_ts

def imu_quaternion(ts, Wx, Wy, Wz, save=True):
    n_samples = len(ts)
    quaternions = np.zeros((n_samples, 4))
    quaternions[0] = np.array([1.0, 0.0, 0.0, 0.0])
    
    for i in range(1, n_samples):
        tau = ts[i] - ts[i-1]
        quaternions[i] = update_quaternion(quaternions[i-1], tau, Wx[i], Wy[i], Wz[i])
    
    n_samples = quaternions.shape[0]
    roll = np.zeros(n_samples)
    pitch = np.zeros(n_samples)
    yaw = np.zeros(n_samples)

    for i in range(n_samples):
        roll[i], pitch[i], yaw[i] = quat2euler(np.asarray(quaternions[i]), 'sxyz')
    
    roll = np.rad2deg(roll)
    pitch = np.rad2deg(pitch)
    yaw = np.rad2deg(yaw)

    orientation_data = {
        'timestamp': ts,
        'roll': roll,
        'pitch': pitch,
        'yaw': yaw
    }

    if save == True:
        save_to_csv(orientation_data, 'orientation_data')

    return quaternions

def process_trainset(d):
    global dataset
    dataset = str(d)
    cfile = "../data/trainset/cam/cam" + dataset + ".p"
    ifile = "../data/trainset/imu/imuRaw" + dataset + ".p"
    vfile = "../data/trainset/vicon/viconRot" + dataset + ".p"
    # camd = read_data(cfile)
    imud = read_data(ifile)
    vicd = read_data(vfile)
    
    # cam, cam_ts = cam_processing(camd)
    imu_ts, imu_Ax, imu_Ay, imu_Az, imu_Wx, imu_Wy, imu_Wz = imu_processing(imud)
    vic_rots, vic_ts = vicon_processing(vicd)
    qt = imu_quaternion(imu_ts, imu_Wx, imu_Wy, imu_Wz, save=False)

    acceleration_data = np.stack([imu_Ax, imu_Ay, imu_Az], axis=1)
    rotation_data = np.stack([imu_Wx, imu_Wy, imu_Wz], axis=1)

    tau = np.diff(imu_ts)
    q_optimized, cost_history = projected_gradient_descent(
        qt=qt,
        at= acceleration_data,
        omega= rotation_data,
        tau=tau,
    )
    
    plot_cost_curve(cost_history, dataset)
    
    n_samples = q_optimized.shape[0]
    roll_opt = np.zeros(n_samples)
    pitch_opt = np.zeros(n_samples)
    yaw_opt = np.zeros(n_samples)
    
    for i in range(n_samples):
        q_copy = np.asarray(q_optimized[i], dtype=np.float64)
        roll_opt[i], pitch_opt[i], yaw_opt[i] = quat2euler(q_copy, 'sxyz')
    
    roll_opt = np.rad2deg(roll_opt)
    pitch_opt = np.rad2deg(pitch_opt)
    yaw_opt = np.rad2deg(yaw_opt)
    
    orientation_data_optimized = {
        'timestamp': imu_ts,
        'roll': roll_opt,
        'pitch': pitch_opt,
        'yaw': yaw_opt
    }
    
    save_to_csv(orientation_data_optimized, 'orientation_data_optimized')
    print("Optimized orientation data saved.")

    plot_ground_truth_estimated_RPY(dataset)
    plot_ground_truth_optimized_RPY(dataset)

def process_testset(d):
    global dataset
    dataset = str(d)
    cfile = "../data/testset/cam/cam" + dataset + ".p"
    ifile = "../data/testset/imu/imuRaw" + dataset + ".p"
    camd = read_data(cfile)
    imud = read_data(ifile)
    
    cam, cam_ts = cam_processing(camd)
    imu_ts, imu_Ax, imu_Ay, imu_Az, imu_Wx, imu_Wy, imu_Wz = imu_processing(imud)
    qt = imu_quaternion(imu_ts, imu_Wx, imu_Wy, imu_Wz, save=False)

    acceleration_data = np.stack([imu_Ax, imu_Ay, imu_Az], axis=1)
    rotation_data = np.stack([imu_Wx, imu_Wy, imu_Wz], axis=1)

    tau = np.diff(imu_ts)
    q_optimized, cost_history = projected_gradient_descent(
        qt=qt,
        at= acceleration_data,
        omega= rotation_data,
        tau=tau,
    )
    
    plot_cost_curve(cost_history, dataset)
    
    n_samples = q_optimized.shape[0]
    roll_opt = np.zeros(n_samples)
    pitch_opt = np.zeros(n_samples)
    yaw_opt = np.zeros(n_samples)
    
    for i in range(n_samples):
        q_copy = np.asarray(q_optimized[i], dtype=np.float64)
        roll_opt[i], pitch_opt[i], yaw_opt[i] = quat2euler(q_copy, 'sxyz')
    
    roll_opt = np.rad2deg(roll_opt)
    pitch_opt = np.rad2deg(pitch_opt)
    yaw_opt = np.rad2deg(yaw_opt)
    
    orientation_data_optimized = {
        'timestamp': imu_ts,
        'roll': roll_opt,
        'pitch': pitch_opt,
        'yaw': yaw_opt
    }
    
    save_to_csv(orientation_data_optimized, 'orientation_data_optimized')
    print("Optimized orientation data saved.")

    plot_optimized_RPY(dataset)

def projected_gradient_descent(qt, at, omega, tau):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    qt = torch.tensor(qt, dtype=torch.float32, device=device)
    at = torch.tensor(at, dtype=torch.float32, device=device)
    omega = torch.tensor(omega, dtype=torch.float32, device=device)
    tau = torch.tensor(tau, dtype=torch.float32, device=device)

    q_estimate = qt.clone().detach().requires_grad_(True)
    
    cost_history = []

    for i in range(ITERATION):
        cost_func = lambda q: cost_function(q, omega, at, tau)
        
        grad = torch.autograd.functional.jacobian(cost_func, q_estimate)

        with torch.no_grad():
            q_estimate = q_estimate - LEARNING_RATE * grad.squeeze(0)
            
            q_estimate = q_estimate / (torch.norm(q_estimate, dim=-1, keepdim=True) + 1e-8)
            
            q_estimate = q_estimate.detach().requires_grad_(True)
            
            current_cost = cost_func(q_estimate)
            cost_history.append(current_cost.item())
            
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}: Cost {current_cost.item():.6f}")

    return q_estimate.detach().cpu().numpy(), cost_history
              

if __name__ == "__main__":
    ITERATION = 800
    LEARNING_RATE = 0.001
    # Read Trainset
    for d in range(1, 10):
        process_trainset(d)

    # Read Testset
    for d in range(10, 12):
        process_testset(d)