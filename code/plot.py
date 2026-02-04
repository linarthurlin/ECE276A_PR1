import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_ground_truth_estimated_RPY(dataset):
    orientation_data = pd.read_csv(f'../data/{dataset}/orientation_data.csv')
    vicon_data = pd.read_csv(f'../data/{dataset}/vicon_euler.csv')
    
    t_orient = (orientation_data.iloc[:, 0] - orientation_data.iloc[0, 0])
    t_vicon = (vicon_data.iloc[:, 0] - vicon_data.iloc[0, 0])

    fig, axes = plt.subplots(3, 1, figsize=(16, 8))

    # Plot Roll
    axes[0].plot(t_orient, orientation_data.iloc[:, 1], 'r-', label='Estimated Roll')
    axes[0].plot(t_vicon, vicon_data.iloc[:, 1], 'b-', label='Ground Truth Roll')
    axes[0].set_ylabel('Roll (degree)')
    axes[0].set_title('Roll')
    axes[0].set_ylim([-180, 180])
    axes[0].legend()
    axes[0].grid(True)

    # Plot Pitch
    axes[1].plot(t_orient, orientation_data.iloc[:, 2], 'r-', label='Estimated Pitch')
    axes[1].plot(t_vicon, vicon_data.iloc[:, 2], 'b-', label='Ground Truth Pitch')
    axes[1].set_ylabel('Pitch (degree)')
    axes[1].set_title('Pitch')
    axes[1].set_ylim([-180, 180])
    axes[1].legend()
    axes[1].grid(True)

    # Plot Yaw
    axes[2].plot(t_orient, orientation_data.iloc[:, 3], 'r-', label='Estimated Yaw')
    axes[2].plot(t_vicon, vicon_data.iloc[:, 3], 'b-', label='Ground Truth Yaw')
    axes[2].set_xlabel('Time (s))')
    axes[2].set_ylabel('Yaw (degree)')
    axes[2].set_title('Yaw')
    axes[2].set_ylim([-180, 180])
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(f'../data/{dataset}/RPY_groundtruth_estimated.png')
    plt.close()

def plot_ground_truth_optimized_RPY(dataset):
    orientation_data = pd.read_csv(f'../data/{dataset}/orientation_data_optimized.csv')
    vicon_data = pd.read_csv(f'../data/{dataset}/vicon_euler.csv')
    
    t_orient = (orientation_data.iloc[:, 0] - orientation_data.iloc[0, 0])
    t_vicon = (vicon_data.iloc[:, 0] - vicon_data.iloc[0, 0])

    fig, axes = plt.subplots(3, 1, figsize=(16, 8))

    # Plot Roll
    axes[0].plot(t_orient, orientation_data.iloc[:, 1], 'r-', label='Estimated Roll')
    axes[0].plot(t_vicon, vicon_data.iloc[:, 1], 'b-', label='Ground Truth Roll')
    axes[0].set_ylabel('Roll (degree)')
    axes[0].set_title('Roll')
    axes[0].set_ylim([-180, 180])
    axes[0].legend()
    axes[0].grid(True)

    # Plot Pitch
    axes[1].plot(t_orient, orientation_data.iloc[:, 2], 'r-', label='Estimated Pitch')
    axes[1].plot(t_vicon, vicon_data.iloc[:, 2], 'b-', label='Ground Truth Pitch')
    axes[1].set_ylabel('Pitch (degree)')
    axes[1].set_title('Pitch')
    axes[1].set_ylim([-180, 180])
    axes[1].legend()
    axes[1].grid(True)

    # Plot Yaw
    axes[2].plot(t_orient, orientation_data.iloc[:, 3], 'r-', label='Estimated Yaw')
    axes[2].plot(t_vicon, vicon_data.iloc[:, 3], 'b-', label='Ground Truth Yaw')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Yaw (degree)')
    axes[2].set_title('Yaw')
    axes[2].set_ylim([-180, 180])
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(f'../data/{dataset}/RPY_groundtruth_optimized.png')
    plt.close()

def plot_optimized_RPY(dataset):
    orientation_data = pd.read_csv(f'../data/{dataset}/orientation_data_optimized.csv')

    t_orient = (orientation_data.iloc[:, 0] - orientation_data.iloc[0, 0])

    fig, axes = plt.subplots(3, 1, figsize=(16, 8))

    # Plot Roll
    axes[0].plot(t_orient, orientation_data.iloc[:, 1], 'r-', label='Estimated Roll')
    axes[0].set_ylabel('Roll (degree)')
    axes[0].set_title('Roll')
    axes[0].set_ylim([-180, 180])
    axes[0].legend()
    axes[0].grid(True)

    # Plot Pitch
    axes[1].plot(t_orient, orientation_data.iloc[:, 2], 'r-', label='Estimated Pitch')
    axes[1].set_ylabel('Pitch (degree)')
    axes[1].set_title('Pitch')
    axes[1].set_ylim([-180, 180])
    axes[1].legend()
    axes[1].grid(True)

    # Plot Yaw
    axes[2].plot(t_orient, orientation_data.iloc[:, 3], 'r-', label='Estimated Yaw')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Yaw (degree)')
    axes[2].set_title('Yaw')
    axes[2].set_ylim([-180, 180])
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(f'../data/{dataset}/RPY_optimized.png')
    plt.close()