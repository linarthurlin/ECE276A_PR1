import numpy as np
import pandas as pd
import pickle
import sys
import os
import cv2
from transforms3d.quaternions import quat2mat
from transforms3d.euler import euler2quat

def read_data(fname):
    d = []
    with open(fname, 'rb') as f:
        if sys.version_info[0] < 3:
            d = pickle.load(f)
        else:
            d = pickle.load(f, encoding='latin1')
    return d

def aligned_orientation(cam_ts, imu_ts, q_estimate):
    idx = np.searchsorted(imu_ts, cam_ts, side='right') - 1
    idx = max(0, min(idx, len(q_estimate) - 1))
    return q_estimate[idx]


def build_panorama(images, cam_timestamps, imu, imu_timestamps):
    h_canvas, w_canvas = 480, 960
    pano = np.zeros((h_canvas, w_canvas, 3), dtype=np.uint8)
    
    HFOV = np.deg2rad(60)
    VFOV = np.deg2rad(45)
    
    h, w = images[0].shape[:2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Spherical to Cartesian
    lam = (u / w - 0.5) * HFOV
    phi = -(v / h - 0.5) * VFOV

    x = np.cos(phi) * np.cos(lam)
    y = np.cos(phi) * np.sin(lam)
    z = np.sin(phi)
    cam_points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    for i, img in enumerate(images):
        q = aligned_orientation(cam_timestamps[i], imu_timestamps, imu)
        
        R = quat2mat(q)
        
        world_points = cam_points @ R.T
        
        # Cartesian to Cylindrical
        longitude = np.arctan2(world_points[:, 1], world_points[:, 0])
        latitude = np.arcsin(world_points[:, 2] / np.linalg.norm(world_points, axis=1))

        v = ((longitude + np.pi) / (2 * np.pi) * (w_canvas - 1)).astype(int)
        u = ((-latitude + np.pi/2) / np.pi * (h_canvas - 1)).astype(int)
        
        px = np.clip(v, 0, w_canvas - 1)
        py = np.clip(u, 0, h_canvas - 1)

        pano[py, px] = img.reshape(-1, 3)
        
    return pano
    
def process_dataset(datasets=[1, 2, 8, 9], mode='trainset'):
    base_path = "../data"

    for d in datasets:
        d_str = str(d)
        print(f"\n=== Processing {mode} Dataset {d_str} ===")
        
        orientation_file = f'../data/{d_str}/orientation_data_optimized.csv'
        orientation_data = pd.read_csv(orientation_file)
        
        cam_file = f"{base_path}/{mode}/cam/cam{d_str}.p"
        camd = read_data(cam_file)
        
        cam_images = camd['cam']
        cam_ts = camd['ts'].flatten()
        
        print(f"Camera images shape: {cam_images.shape}")
        
        imu_ts = orientation_data['timestamp'].values
        roll = np.deg2rad(orientation_data['roll'].values)
        pitch = np.deg2rad(orientation_data['pitch'].values)
        yaw = np.deg2rad(orientation_data['yaw'].values)
        
        n_samples = len(imu_ts)
        q_traj = np.zeros((n_samples, 4))

        for i in range(n_samples):
            q_traj[i] = euler2quat(roll[i], pitch[i], yaw[i])
        
        n_images = cam_images.shape[-1]
        images = [cam_images[:, :, :, i] for i in range(n_images)]
        
        print(f"Building panorama with {n_images} images...")
        panorama = build_panorama(images, cam_ts, q_traj, imu_ts)

        output_dir = f'{base_path}/{d_str}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_file = f'{output_dir}/panorama_optimized.png'
        cv2.imwrite(output_file, cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR))
        print(f"Panorama saved to {output_file}")

if __name__ == "__main__":
    # Trainset
    process_dataset(datasets=[1, 2, 8, 9], mode='trainset')

    # Testset
    process_dataset(datasets=[10, 11], mode='testset')