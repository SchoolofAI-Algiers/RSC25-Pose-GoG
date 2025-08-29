# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Skeleton Data Reader Module

This module provides functionality to read and parse raw skeleton data files from 
human action recognition datasets (particularly NTU RGB+D dataset). It processes 
.skeleton files to extract joint positions, color coordinates, and temporal information 
for multiple actors across video sequences.

The module handles:
- Reading .skeleton files with multiple bodies per frame
- Extracting 3D joint positions (25 joints per body)
- Extracting 2D color coordinates for joint visualization
- Tracking temporal intervals for each body ID
- Computing motion statistics for multi-body sequences
- Handling missing frames and data validation

Author: Microsoft Corporation
License: MIT
"""

import os.path as osp
import os
import numpy as np
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from project_setup import DATA_STATS , DATA_RAW , RAW_SKELS

# Type definitions for better code readability
BodyData = Dict[str, Union[np.ndarray, List[int], float]]
BodiesDataDict = Dict[str, BodyData]
SkeletonSequence = Dict[str, Union[str, BodiesDataDict, int]]
FramesDropDict = Dict[str, np.ndarray]


def get_raw_bodies_data(
    skes_path: str, 
    ske_name: str, 
    frames_drop_skes: FramesDropDict, 
    frames_drop_logger: logging.Logger
) -> SkeletonSequence:
    """
    Extract raw skeleton data from a single .skeleton file.
    
    Parses a .skeleton file to extract joint positions, color coordinates, and temporal
    information for all bodies present in the sequence. Handles missing frames and
    computes motion statistics for multi-body sequences.
    
    Args:
        skes_path (str): Directory path containing skeleton files
        ske_name (str): Name of the skeleton file (without .skeleton extension)
        frames_drop_skes (FramesDropDict): Dictionary to store information about dropped frames
        frames_drop_logger (logging.Logger): Logger for recording dropped frame information
    
    Returns:
        SkeletonSequence: Dictionary containing:
            - 'name' (str): Skeleton filename
            - 'data' (BodiesDataDict): Dictionary mapping body IDs to their data
            - 'num_frames' (int): Number of valid frames after dropping empty frames
    
    Each body's data dictionary contains:
        - 'joints' (np.ndarray): 3D joint positions, shape (num_frames x 25, 3)
        - 'colors' (np.ndarray): 2D color coordinates, shape (num_frames, 25, 2)
        - 'interval' (List[int]): List of frame indices where this body appears
        - 'motion' (float): Motion amount (only for sequences with 2+ bodies)
    
    Raises:
        AssertionError: If skeleton file doesn't exist or all frames are missing
    
    Side Effects:
        - Logs dropped frame information to frames_drop_logger
        - Updates frames_drop_skes dictionary with dropped frame indices
    
    File Format:
        .skeleton files contain:
        - Line 1: Total number of frames
        - For each frame:
            - Number of bodies in this frame
            - For each body:
                - Body ID and metadata
                - Number of joints (always 25)
                - For each joint: X Y Z confidence colorX colorY
    
    Note:
        - Frames with 0 bodies are automatically dropped
        - Joint coordinates are in 3D world space
        - Color coordinates are 2D pixel locations for visualization
        - Motion is computed as sum of variance across all joint coordinates
    """
    ske_file = osp.join(skes_path, ske_name + '.skeleton')
    assert osp.exists(ske_file), 'Error: Skeleton file %s not found' % ske_file
    
    # Read all data from .skeleton file into a list (in string format)
    print('Reading data from %s' % ske_file[-29:])
    with open(ske_file, 'r') as fr:
        str_data = fr.readlines()

    num_frames = int(str_data[0].strip('\r\n'))
    frames_drop: List[int] = []
    bodies_data: BodiesDataDict = dict()
    valid_frames = -1  # 0-based index for valid frames
    current_line = 1

    # Process each frame in the sequence
    for f in range(num_frames):
        num_bodies = int(str_data[current_line].strip('\r\n'))
        current_line += 1

        if num_bodies == 0:  # No body data in this frame, drop it
            frames_drop.append(f)  # 0-based index
            continue

        valid_frames += 1
        joints = np.zeros((num_bodies, 25, 3), dtype=np.float32)
        colors = np.zeros((num_bodies, 25, 2), dtype=np.float32)

        # Process each body in the current frame
        for b in range(num_bodies):
            # Extract body ID from the data line
            bodyID = str_data[current_line].strip('\r\n').split()[0]
            current_line += 1
            num_joints = int(str_data[current_line].strip('\r\n'))  # Should be 25
            current_line += 1

            # Extract joint data for this body
            for j in range(num_joints):
                temp_str = str_data[current_line].strip('\r\n').split()
                # Extract 3D joint position (X, Y, Z)
                joints[b, j, :] = np.array(temp_str[:3], dtype=np.float32)
                # Extract 2D color coordinates (colorX, colorY) - skip confidence at index 3,4
                colors[b, j, :] = np.array(temp_str[5:7], dtype=np.float32)
                current_line += 1

            # Update or create body data
            if bodyID not in bodies_data:  # Add new body's data
                body_data: BodyData = dict()
                body_data['joints'] = joints[b]  # ndarray: (25, 3)
                body_data['colors'] = colors[b, np.newaxis]  # ndarray: (1, 25, 2)
                body_data['interval'] = [valid_frames]  # Index of the first frame
            else:  # Update existing body's data
                body_data = bodies_data[bodyID]
                # Stack each body's data along the frame dimension
                body_data['joints'] = np.vstack((body_data['joints'], joints[b]))
                body_data['colors'] = np.vstack((body_data['colors'], colors[b, np.newaxis]))
                pre_frame_idx = body_data['interval'][-1]
                body_data['interval'].append(pre_frame_idx + 1)  # Add consecutive frame index

            bodies_data[bodyID] = body_data  # Update bodies_data dictionary

    # Validate that we have some valid frames
    num_frames_drop = len(frames_drop)
    assert num_frames_drop < num_frames, \
        'Error: All frames data (%d) of %s is missing or lost' % (num_frames, ske_name)
    
    # Log information about dropped frames
    if num_frames_drop > 0:
        frames_drop_skes[ske_name] = np.array(frames_drop, dtype=np.int32)
        frames_drop_logger.info('{}: {} frames missed: {}\n'.format(ske_name, num_frames_drop,
                                                                    frames_drop))

    # Calculate motion statistics for multi-body sequences
    if len(bodies_data) > 1:
        for body_data in bodies_data.values():
            # Motion computed as sum of variance across all joint coordinates
            body_data['motion'] = np.sum(np.var(body_data['joints'], axis=0))

    return {'name': ske_name, 'data': bodies_data, 'num_frames': num_frames - num_frames_drop}


def get_raw_skes_data() -> None:
    """
    Process all skeleton files and save raw skeleton data.
    
    Main processing function that:
    1. Reads list of available skeleton files
    2. Processes each skeleton file to extract raw body data
    3. Saves processed data to pickle files
    4. Generates frame count statistics
    5. Saves information about dropped frames
    
    This function relies on global variables set in the main execution block:
    - skes_path: Path to directory containing .skeleton files
    - skes_name_file: Text file listing available skeleton file names
    - save_data_pkl: Output path for processed skeleton data
    - frames_drop_pkl: Output path for dropped frames information
    - frames_drop_logger: Logger for recording dropped frame events
    - frames_drop_skes: Dictionary to collect dropped frame information
    
    Output Files:
        - raw_skes_data.pkl: List of SkeletonSequence dictionaries
        - frames_cnt.txt: Text file with frame counts for each sequence
        - frames_drop_skes.pkl: Dictionary mapping sequence names to dropped frame indices
        - frames_drop.log: Log file with detailed dropped frame information
    
    Side Effects:
        - Prints progress information every 1000 processed files
        - Creates output files in specified directories
        - Logs dropped frame statistics
    
    Data Structure:
        The output raw_skes_data.pkl contains a list where each element is a dictionary:
        {
            'name': 'sequence_name',
            'data': {
                'body_id_1': {
                    'joints': np.ndarray,    # (num_frames, 25, 3)
                    'colors': np.ndarray,    # (num_frames, 25, 2)
                    'interval': List[int],   # Frame indices
                    'motion': float          # Motion amount (multi-body only)
                },
                'body_id_2': { ... },
                ...
            },
            'num_frames': int  # Total valid frames
        }
    
    Note:
        - Processes files in the order listed in skes_name_file
        - Progress is reported every 1000 files for large datasets
        - Frame counts exclude dropped frames with no body data
    """
    # Load list of available skeleton file names
    skes_name = np.loadtxt(skes_name_file, dtype=str)

    num_files = skes_name.size
    print('Found %d available skeleton files.' % num_files)

    # Initialize data containers
    raw_skes_data: List[SkeletonSequence] = []
    frames_cnt = np.zeros(num_files, dtype=np.int32)

    # Process each skeleton file
    for (idx, ske_name) in enumerate(skes_name):
        bodies_data = get_raw_bodies_data(skes_path, ske_name, frames_drop_skes, frames_drop_logger)
        raw_skes_data.append(bodies_data)
        frames_cnt[idx] = bodies_data['num_frames']
        
        # Report progress every 1000 files
        if (idx + 1) % 1000 == 0:
            print('Processed: %.2f%% (%d / %d)' % 
                  (100.0 * (idx + 1) / num_files, idx + 1, num_files))

    # Save processed skeleton data
    with open(save_data_pkl, 'wb') as fw:
        pickle.dump(raw_skes_data, fw, pickle.HIGHEST_PROTOCOL)
    
    # Save frame count statistics
    np.savetxt(osp.join(save_path, 'raw_data', 'frames_cnt.txt'), frames_cnt, fmt='%d')

    # Print summary statistics
    print('Saved raw bodies data into %s' % save_data_pkl)
    print('Total frames: %d' % np.sum(frames_cnt))

    # Save dropped frames information
    with open(frames_drop_pkl, 'wb') as fw:
        pickle.dump(frames_drop_skes, fw, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    """
    Main execution block for skeleton data reading and processing.
    
    Sets up all necessary paths and configurations, then processes skeleton files
    to extract raw body data. This script is specifically designed for processing
    NTU RGB+D dataset skeleton files.
    
    Configuration:
        - save_path: Base directory for output files
        - skes_path: Directory containing input .skeleton files  
        - stat_path: Directory containing statistics and file lists
        - skes_name_file: Text file listing available skeleton files
        
    Output Structure:
        ./raw_data/
        ├── raw_skes_data.pkl      # Main processed skeleton data
        ├── frames_cnt.txt         # Frame counts for each sequence
        ├── frames_drop_skes.pkl   # Information about dropped frames
        └── frames_drop.log        # Detailed log of dropped frames
    
    Expected Input Files:
        - statistics/skes_available_name.txt: List of skeleton file names
        - ../nturgbd_raw/nturgb+d_skeletons/*.skeleton: Raw skeleton files
    
    Usage:
        python skeleton_reader.py
        
    Note:
        - Creates raw_data directory if it doesn't exist
        - Assumes skeleton files follow NTU RGB+D naming convention
        - Processes all files listed in skes_available_name.txt
        - Logs detailed information about any issues encountered
    """
    # Configuration paths

    save_path = RAW_SKELS
    skes_path = DATA_RAW
    stat_path = DATA_STATS
    

    # Define input and output file paths
    skes_name_file: str = osp.join(stat_path, 'skes_available_name.txt')
    save_data_pkl: str = osp.join(save_path, 'raw_skes_data.pkl')
    frames_drop_pkl: str = osp.join(save_path, 'frames_drop_skes.pkl')

    # Setup logging for dropped frames
    frames_drop_logger = logging.getLogger('frames_drop')
    frames_drop_logger.setLevel(logging.INFO)
    frames_drop_logger.addHandler(
        logging.FileHandler(osp.join(save_path, 'frames_drop.log'))
    )
    frames_drop_skes: FramesDropDict = dict()

    # Execute main processing pipeline
    get_raw_skes_data()

    # Final save of dropped frames information
    with open(frames_drop_pkl, 'wb') as fw:
        pickle.dump(frames_drop_skes, fw, pickle.HIGHEST_PROTOCOL)