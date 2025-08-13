# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
NTU RGB+D Dataset Preprocessing Module

This module processes the NTU RGB+D skeletal action recognition dataset by:
- Removing NaN frames from skeleton sequences
- Performing sequence and frame-level translations/normalizations
- Aligning sequences to uniform frame lengths
- Splitting data into train/test sets for cross-subject (CS) and cross-view (CV) evaluations
"""

import os
import os.path as osp
import numpy as np
import pickle
import logging
import h5py
from typing import List, Tuple, Union, Optional
from sklearn.model_selection import train_test_split

# File paths configuration
root_path: str = './'
stat_path: str = osp.join(root_path, 'statistics')
setup_file: str = osp.join(stat_path, 'setup.txt')
camera_file: str = osp.join(stat_path, 'camera.txt')
performer_file: str = osp.join(stat_path, 'performer.txt')
replication_file: str = osp.join(stat_path, 'replication.txt')
label_file: str = osp.join(stat_path, 'label.txt')
skes_name_file: str = osp.join(stat_path, 'skes_available_name.txt')

denoised_path: str = osp.join(root_path, 'denoised_data')
raw_skes_joints_pkl: str = osp.join(denoised_path, 'raw_denoised_joints.pkl')
frames_file: str = osp.join(denoised_path, 'frames_cnt.txt')

save_path: str = './'

if not osp.exists(save_path):
    os.mkdir(save_path)


def remove_nan_frames(ske_name: str, ske_joints: np.ndarray, nan_logger: logging.Logger) -> np.ndarray:
    """
    Remove frames containing NaN values from skeleton joint data.
    
    Args:
        ske_name: Name identifier of the skeleton sequence
        ske_joints: Skeleton joint data array of shape (num_frames, num_joints * 3)
                   where each joint has (x, y, z) coordinates
        nan_logger: Logger instance for recording NaN frame information
        
    Returns:
        np.ndarray: Cleaned skeleton joint data with NaN frames removed
        
    Note:
        Logs information about removed frames including skeleton name, frame number,
        and indices of joints containing NaN values.
    """
    num_frames: int = ske_joints.shape[0]
    valid_frames: List[int] = []

    for f in range(num_frames):
        if not np.any(np.isnan(ske_joints[f])):
            valid_frames.append(f)
        else:
            nan_indices: np.ndarray = np.where(np.isnan(ske_joints[f]))[0]
            nan_logger.info('{}\t{:^5}\t{}'.format(ske_name, f + 1, nan_indices))

    return ske_joints[valid_frames]


def seq_translation(skes_joints: List[np.ndarray]) -> List[np.ndarray]:
    """
    Perform sequence-level translation by setting spine base (joint-2) as origin.
    
    This function normalizes all skeleton sequences by translating them so that
    the spine base joint of the first valid frame becomes the origin (0,0,0).
    
    Args:
        skes_joints: List of skeleton joint arrays, each of shape (num_frames, 75 or 150)
                    75 for single person, 150 for two persons (75 joints * 2)
                    
    Returns:
        List[np.ndarray]: Translated skeleton sequences with spine base as origin
        
    Note:
        - For single person: 75 values (25 joints × 3 coordinates)
        - For two persons: 150 values (50 joints × 3 coordinates)
        - Missing body data is handled by preserving zero values
    """
    for idx, ske_joints in enumerate(skes_joints):
        num_frames: int = ske_joints.shape[0]
        num_bodies: int = 1 if ske_joints.shape[1] == 75 else 2
        
        if num_bodies == 2:
            # Find frames where each body is missing (all zeros)
            missing_frames_1: np.ndarray = np.where(ske_joints[:, :75].sum(axis=1) == 0)[0]
            missing_frames_2: np.ndarray = np.where(ske_joints[:, 75:].sum(axis=1) == 0)[0]
            cnt1: int = len(missing_frames_1)
            cnt2: int = len(missing_frames_2)

        # Find the first frame with valid actor1 data
        i: int = 0
        while i < num_frames:
            if np.any(ske_joints[i, :75] != 0):
                break
            i += 1

        # Set origin to spine base joint (joint-2: indices 3:6)
        origin: np.ndarray = np.copy(ske_joints[i, 3:6])

        # Translate all frames to new origin
        for f in range(num_frames):
            if num_bodies == 1:
                ske_joints[f] -= np.tile(origin, 25)  # 25 joints × 3 coords
            else:  # for 2 actors
                ske_joints[f] -= np.tile(origin, 50)  # 50 joints × 3 coords

        # Restore zero values for missing bodies
        if (num_bodies == 2) and (cnt1 > 0):
            ske_joints[missing_frames_1, :75] = np.zeros((cnt1, 75), dtype=np.float32)

        if (num_bodies == 2) and (cnt2 > 0):
            ske_joints[missing_frames_2, 75:] = np.zeros((cnt2, 75), dtype=np.float32)

        skes_joints[idx] = ske_joints

    return skes_joints


def frame_translation(skes_joints: List[np.ndarray], skes_name: np.ndarray, 
                     frames_cnt: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Perform frame-level translation and normalization based on spine length.
    
    This function normalizes each frame by:
    1. Setting spine middle (joint-2) as origin
    2. Scaling by the distance between spine base (joint-1) and spine (joint-21)
    3. Removing frames containing NaN values
    
    Args:
        skes_joints: List of skeleton joint arrays
        skes_name: Array of skeleton sequence names for logging
        frames_cnt: Array tracking frame counts for each sequence
        
    Returns:
        Tuple containing:
        - List[np.ndarray]: Normalized skeleton sequences
        - np.ndarray: Updated frame counts after NaN removal
        
    Note:
        Creates a log file 'nan_frames.log' recording removed NaN frames.
        The normalization uses spine length as a scale factor for pose normalization.
    """
    # Setup logging for NaN frames
    nan_logger: logging.Logger = logging.getLogger('nan_skes')
    nan_logger.setLevel(logging.INFO)
    nan_logger.addHandler(logging.FileHandler("./nan_frames.log"))
    nan_logger.info('{}\t{}\t{}'.format('Skeleton', 'Frame', 'Joints'))

    for idx, ske_joints in enumerate(skes_joints):
        num_frames: int = ske_joints.shape[0]
        
        # Calculate spine length (distance between spine base and spine top)
        j1: np.ndarray = ske_joints[:, 0:3]    # spine base (joint-1)
        j21: np.ndarray = ske_joints[:, 60:63]  # spine (joint-21)
        dist: np.ndarray = np.sqrt(((j1 - j21) ** 2).sum(axis=1))

        # Normalize each frame
        for f in range(num_frames):
            origin: np.ndarray = ske_joints[f, 3:6]  # spine middle (joint-2)
            
            if (ske_joints[f, 75:] == 0).all():  # single person
                ske_joints[f, :75] = (ske_joints[f, :75] - np.tile(origin, 25)) / \
                                    dist[f] + np.tile(origin, 25)
            else:  # two persons
                ske_joints[f] = (ske_joints[f] - np.tile(origin, 50)) / \
                               dist[f] + np.tile(origin, 50)

        # Remove NaN frames and update frame count
        ske_name: str = skes_name[idx].decode('utf-8') if isinstance(skes_name[idx], bytes) else skes_name[idx]
        ske_joints = remove_nan_frames(ske_name, ske_joints, nan_logger)
        frames_cnt[idx] = ske_joints.shape[0]  # update with valid frame count
        skes_joints[idx] = ske_joints

    return skes_joints, frames_cnt


def align_frames(skes_joints: List[np.ndarray], frames_cnt: np.ndarray) -> np.ndarray:
    """
    Align all sequences to the same frame length by zero-padding.
    
    Args:
        skes_joints: List of skeleton joint arrays with varying frame lengths
        frames_cnt: Array of frame counts for each sequence
        
    Returns:
        np.ndarray: Aligned skeleton data of shape (num_sequences, max_frames, 150)
                   where 150 = 2 bodies × 25 joints × 3 coordinates
                   
    Note:
        - Single-person sequences are padded with zeros for the second person
        - All sequences are zero-padded to match the maximum frame length
        - Output always has 150 features (space for 2 people)
    """
    num_skes: int = len(skes_joints)
    max_num_frames: int = frames_cnt.max()  # typically 300 for NTU dataset
    aligned_skes_joints: np.ndarray = np.zeros((num_skes, max_num_frames, 150), dtype=np.float32)

    for idx, ske_joints in enumerate(skes_joints):
        num_frames: int = ske_joints.shape[0]
        num_bodies: int = 1 if ske_joints.shape[1] == 75 else 2
        
        if num_bodies == 1:
            # Pad single person data with zeros for second person
            aligned_skes_joints[idx, :num_frames] = np.hstack((ske_joints,
                                                               np.zeros_like(ske_joints)))
        else:
            # Two person data fits directly
            aligned_skes_joints[idx, :num_frames] = ske_joints

    return aligned_skes_joints


def one_hot_vector(labels: np.ndarray) -> np.ndarray:
    """
    Convert integer labels to one-hot encoded vectors.
    
    Args:
        labels: Array of integer labels in range [0, 59] for NTU-60 dataset
        
    Returns:
        np.ndarray: One-hot encoded labels of shape (num_samples, 60)
                   where each row has exactly one 1.0 and rest are 0.0
                   
    Example:
        labels = [0, 5, 59] -> [[1,0,0,...,0], [0,0,0,0,0,1,0,...,0], [0,0,...,0,1]]
    """
    num_skes: int = len(labels)
    labels_vector: np.ndarray = np.zeros((num_skes, 60))
    for idx, l in enumerate(labels):
        labels_vector[idx, l] = 1

    return labels_vector


def split_train_val(train_indices: np.ndarray, method: str = 'sklearn', 
                   ratio: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split training data into train and validation sets.
    
    Args:
        train_indices: Array of training sample indices
        method: Split method - 'sklearn' (uses train_test_split) or 'numpy' (manual split)
        ratio: Fraction of training data to use for validation (default: 0.05 = 5%)
        
    Returns:
        Tuple containing:
        - np.ndarray: Updated training indices after removing validation samples
        - np.ndarray: Validation indices
        
    Note:
        Both methods use random_state/seed=10000 for reproducibility.
        The sklearn method is recommended and set as default.
    """
    if method == 'sklearn':
        return train_test_split(train_indices, test_size=ratio, random_state=10000)
    else:
        np.random.seed(10000)
        np.random.shuffle(train_indices)
        val_num_skes: int = int(np.ceil(ratio * len(train_indices)))
        val_indices: np.ndarray = train_indices[:val_num_skes]
        train_indices = train_indices[val_num_skes:]
        return train_indices, val_indices


def split_dataset(skes_joints: np.ndarray, label: np.ndarray, performer: np.ndarray, 
                 camera: np.ndarray, evaluation: str, save_path: str) -> None:
    """
    Split dataset into train/test sets and save in NPZ format.
    
    Args:
        skes_joints: Aligned skeleton joint data of shape (num_samples, max_frames, 150)
        label: Action labels array of shape (num_samples,)
        performer: Subject/performer IDs for each sample
        camera: Camera IDs for each sample  
        evaluation: Evaluation protocol - 'CS' (Cross-Subject) or 'CV' (Cross-View)
        save_path: Directory path where to save the output files
        
    Saves:
        NPZ file named 'NTU60_{evaluation}.npz' containing:
        - x_train: Training skeleton data
        - y_train: Training labels (one-hot encoded)
        - x_test: Test skeleton data  
        - y_test: Test labels (one-hot encoded)
        
    Note:
        - Cross-Subject (CS): Splits based on performer IDs
        - Cross-View (CV): Splits based on camera IDs
        - Labels are converted to one-hot encoding with 60 classes
    """
    train_indices, test_indices = get_indices(performer, camera, evaluation)
    
    # Optional: Select validation set from training set
    # m = 'sklearn'  # 'sklearn' or 'numpy'
    # train_indices, val_indices = split_train_val(train_indices, m)

    # Extract labels for each split
    train_labels: np.ndarray = label[train_indices]
    test_labels: np.ndarray = label[test_indices]

    # Extract and encode data
    train_x: np.ndarray = skes_joints[train_indices]
    train_y: np.ndarray = one_hot_vector(train_labels)
    test_x: np.ndarray = skes_joints[test_indices]
    test_y: np.ndarray = one_hot_vector(test_labels)

    # Save as NPZ file
    save_name: str = 'NTU60_%s.npz' % evaluation
    np.savez(save_name, x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y)

    # Alternative: Save as HDF5 file (commented out)
    # h5file = h5py.File(osp.join(save_path, 'NTU_%s.h5' % (evaluation)), 'w')
    # h5file.create_dataset('x', data=skes_joints[train_indices])
    # h5file.create_dataset('y', data=one_hot_vector(train_labels))
    # h5file.create_dataset('test_x', data=skes_joints[test_indices])
    # h5file.create_dataset('test_y', data=one_hot_vector(test_labels))
    # h5file.close()


def get_indices(performer: np.ndarray, camera: np.ndarray, 
               evaluation: str = 'CS') -> Tuple[np.ndarray, np.ndarray]:
    """
    Get training and testing indices based on evaluation protocol.
    
    Args:
        performer: Array of subject/performer IDs (1-40 for NTU dataset)
        camera: Array of camera IDs (1-3 for NTU dataset)  
        evaluation: Evaluation protocol:
                   - 'CS': Cross-Subject evaluation
                   - 'CV': Cross-View evaluation
                   
    Returns:
        Tuple containing:
        - np.ndarray: Training sample indices
        - np.ndarray: Test sample indices
        
    Cross-Subject Protocol:
        - Training: Subjects [1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38]
        - Testing: Subjects [3,6,7,10,11,12,20,21,22,23,24,26,29,30,32,33,36,37,39,40]
        
    Cross-View Protocol:
        - Training: Camera views [2, 3]
        - Testing: Camera view [1]
        
    Note:
        These splits follow the standard NTU RGB+D evaluation protocols
        for fair comparison with other methods.
    """
    test_indices: np.ndarray = np.empty(0, dtype=int)
    train_indices: np.ndarray = np.empty(0, dtype=int)

    if evaluation == 'CS':  # Cross Subject (Subject IDs)
        train_ids: List[int] = [1,  2,  4,  5,  8,  9,  13, 14, 15, 16,
                               17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
        test_ids: List[int] = [3,  6,  7,  10, 11, 12, 20, 21, 22, 23,
                              24, 26, 29, 30, 32, 33, 36, 37, 39, 40]

        # Get indices of test data
        for idx in test_ids:
            temp: np.ndarray = np.where(performer == idx)[0]  # 0-based index
            test_indices = np.hstack((test_indices, temp)).astype(int)

        # Get indices of training data
        for train_id in train_ids:
            temp = np.where(performer == train_id)[0]  # 0-based index
            train_indices = np.hstack((train_indices, temp)).astype(int)
            
    else:  # Cross View (Camera IDs)
        train_ids: List[int] = [2, 3]
        test_ids: int = 1
        
        # Get indices of test data
        temp = np.where(camera == test_ids)[0]  # 0-based index
        test_indices = np.hstack((test_indices, temp)).astype(int)

        # Get indices of training data
        for train_id in train_ids:
            temp = np.where(camera == train_id)[0]
            train_indices = np.hstack((train_indices, temp)).astype(int)

    return train_indices, test_indices


if __name__ == '__main__':
    """
    Main execution pipeline for NTU RGB+D dataset preprocessing.
    
    Pipeline:
    1. Load metadata (camera IDs, performer IDs, action labels, frame counts)
    2. Load raw skeleton joint data
    3. Perform sequence-level translation (set spine base as origin)
    4. Align all sequences to same frame length with zero-padding
    5. Split dataset using both Cross-Subject and Cross-View protocols
    6. Save processed data as NPZ files
    """
    # Load metadata
    camera: np.ndarray = np.loadtxt(camera_file, dtype=int)      # camera id: 1, 2, 3
    performer: np.ndarray = np.loadtxt(performer_file, dtype=int) # subject id: 1~40
    label: np.ndarray = np.loadtxt(label_file, dtype=int) - 1     # action label: 0~59 (originally 1~60)

    frames_cnt: np.ndarray = np.loadtxt(frames_file, dtype=int)   # frame counts per sequence
    skes_name: np.ndarray = np.loadtxt(skes_name_file, dtype='S') # sequence names

    # Load raw skeleton joint data
    with open(raw_skes_joints_pkl, 'rb') as fr:
        skes_joints: List[np.ndarray] = pickle.load(fr)

    # Preprocessing pipeline
    skes_joints = seq_translation(skes_joints)
    skes_joints = align_frames(skes_joints, frames_cnt)

    # Generate both evaluation splits
    evaluations: List[str] = ['CS', 'CV']
    for evaluation in evaluations:
        split_dataset(skes_joints, label, performer, camera, evaluation, save_path)