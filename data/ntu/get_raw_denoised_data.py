# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Skeleton Data Denoising and Processing Module

This module provides functionality to denoise and process raw skeleton data from human action
recognition datasets. It filters out noisy skeleton sequences based on frame length, spatial
spread, and motion characteristics to produce clean skeleton joint positions and color data.

The denoising process includes:
1. Length-based filtering: Remove sequences with insufficient frames
2. Spread-based filtering: Remove sequences with unrealistic joint spreads
3. Motion-based filtering: Remove sequences with motion outside acceptable ranges

Author: Microsoft Corporation
License: MIT
"""

import os
import os.path as osp
import numpy as np
import pickle
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from project_setup import RAW_SKELS , DENOISED_SKELS

# Type definitions for better code readability
BodyData = Dict[str, Union[np.ndarray, List[int], float]]
BodiesData = Dict[str, BodyData]
SkeletonData = Dict[str, Union[str, BodiesData, int]]
DenoiseResult = Tuple[Union[Dict[str, BodyData], List[Tuple[str, BodyData]]], str]
JointsColorsResult = Tuple[np.ndarray, np.ndarray]

# Configuration constants
root_path: str = './'
raw_data_file: str = osp.join(RAW_SKELS, 'raw_skes_data.pkl')
save_path = DENOISED_SKELS

if not osp.exists(save_path):
    os.mkdir(save_path)

rgb_ske_path: str = osp.join(save_path, 'rgb+ske')
if not osp.exists(rgb_ske_path):
    os.mkdir(rgb_ske_path)

actors_info_dir: str = osp.join(save_path, 'actors_info')
if not osp.exists(actors_info_dir):
    os.mkdir(actors_info_dir)

# Global counters and thresholds
missing_count: int = 0
noise_len_thres: int = 11  # Minimum frame length threshold
noise_spr_thres1: float = 0.8  # X-Y spread ratio threshold
noise_spr_thres2: float = 0.69754  # Noise frame ratio threshold
noise_mot_thres_lo: float = 0.089925  # Minimum motion threshold
noise_mot_thres_hi: float = 2  # Maximum motion threshold

# Logger setup for different types of noise detection
noise_len_logger = logging.getLogger('noise_length')
noise_len_logger.setLevel(logging.INFO)
noise_len_logger.addHandler(logging.FileHandler(osp.join(save_path, 'noise_length.log')))
noise_len_logger.info('{:^20}\t{:^17}\t{:^8}\t{}'.format('Skeleton', 'bodyID', 'Motion', 'Length'))

noise_spr_logger = logging.getLogger('noise_spread')
noise_spr_logger.setLevel(logging.INFO)
noise_spr_logger.addHandler(logging.FileHandler(osp.join(save_path, 'noise_spread.log')))
noise_spr_logger.info('{:^20}\t{:^17}\t{:^8}\t{:^8}'.format('Skeleton', 'bodyID', 'Motion', 'Rate'))

noise_mot_logger = logging.getLogger('noise_motion')
noise_mot_logger.setLevel(logging.INFO)
noise_mot_logger.addHandler(logging.FileHandler(osp.join(save_path, 'noise_motion.log')))
noise_mot_logger.info('{:^20}\t{:^17}\t{:^8}'.format('Skeleton', 'bodyID', 'Motion'))

fail_logger_1 = logging.getLogger('noise_outliers_1')
fail_logger_1.setLevel(logging.INFO)
fail_logger_1.addHandler(logging.FileHandler(osp.join(save_path, 'denoised_failed_1.log')))

fail_logger_2 = logging.getLogger('noise_outliers_2')
fail_logger_2.setLevel(logging.INFO)
fail_logger_2.addHandler(logging.FileHandler(osp.join(save_path, 'denoised_failed_2.log')))

missing_skes_logger = logging.getLogger('missing_frames')
missing_skes_logger.setLevel(logging.INFO)
missing_skes_logger.addHandler(logging.FileHandler(osp.join(save_path, 'missing_skes.log')))
missing_skes_logger.info('{:^20}\t{}\t{}'.format('Skeleton', 'num_frames', 'num_missing'))

missing_skes_logger1 = logging.getLogger('missing_frames_1')
missing_skes_logger1.setLevel(logging.INFO)
missing_skes_logger1.addHandler(logging.FileHandler(osp.join(save_path, 'missing_skes_1.log')))
missing_skes_logger1.info('{:^20}\t{}\t{}\t{}\t{}\t{}'.format('Skeleton', 'num_frames', 'Actor1',
                                                              'Actor2', 'Start', 'End'))

missing_skes_logger2 = logging.getLogger('missing_frames_2')
missing_skes_logger2.setLevel(logging.INFO)
missing_skes_logger2.addHandler(logging.FileHandler(osp.join(save_path, 'missing_skes_2.log')))
missing_skes_logger2.info('{:^20}\t{}\t{}\t{}'.format('Skeleton', 'num_frames', 'Actor1', 'Actor2'))


def denoising_by_length(ske_name: str, bodies_data: BodiesData) -> DenoiseResult:
    """
    Remove skeleton sequences based on frame length filtering.
    
    Filters out body IDs that have frame sequences shorter than or equal to the
    predefined threshold (noise_len_thres).
    
    Args:
        ske_name (str): Name of the skeleton sequence file
        bodies_data (BodiesData): Dictionary mapping body IDs to their data
            Each body_data contains:
            - 'interval': List of frame indices
            - 'motion': Float representing motion amount
            - 'joints': Array of joint positions
            - 'colors': Array of color locations
    
    Returns:
        DenoiseResult: Tuple containing:
            - Filtered bodies_data dictionary
            - String with noise information for logging
    
    Side Effects:
        - Logs filtered sequences to noise_len_logger
        - Modifies input bodies_data dictionary in-place
    """
    noise_info = str()
    new_bodies_data = bodies_data.copy()
    
    for (bodyID, body_data) in new_bodies_data.items():
        length = len(body_data['interval'])
        if length <= noise_len_thres:
            noise_info += 'Filter out: %s, %d (length).\n' % (bodyID, length)
            noise_len_logger.info('{}\t{}\t{:.6f}\t{:^6d}'.format(ske_name, bodyID,
                                                                  body_data['motion'], length))
            del bodies_data[bodyID]
    
    if noise_info != '':
        noise_info += '\n'

    return bodies_data, noise_info


def get_valid_frames_by_spread(points: np.ndarray) -> List[int]:
    """
    Identify valid frames based on the spatial spread of joint coordinates.
    
    A frame is considered valid if the ratio of X-coordinate spread to Y-coordinate
    spread is within acceptable bounds (X_spread <= noise_spr_thres1 * Y_spread).
    
    Args:
        points (np.ndarray): Joint coordinates with shape (num_frames, num_joints, 3)
            where the last dimension contains (x, y, z) coordinates
    
    Returns:
        List[int]: List of frame indices that are considered valid
    
    Note:
        Uses noise_spr_thres1 (0.8) as the threshold for X/Y spread ratio
    """
    num_frames = points.shape[0]
    valid_frames = []
    
    for i in range(num_frames):
        x = points[i, :, 0]  # X coordinates for all joints in frame i
        y = points[i, :, 1]  # Y coordinates for all joints in frame i
        
        if (x.max() - x.min()) <= noise_spr_thres1 * (y.max() - y.min()):  # 0.8
            valid_frames.append(i)
    
    return valid_frames


def denoising_by_spread(ske_name: str, bodies_data: BodiesData) -> Tuple[BodiesData, str, bool]:
    """
    Remove skeleton sequences based on spatial spread analysis.
    
    Filters out body IDs where the ratio of noisy frames (determined by joint spread)
    exceeds the predefined threshold. Also updates motion values for remaining sequences.
    
    Args:
        ske_name (str): Name of the skeleton sequence file
        bodies_data (BodiesData): Dictionary with at least 2 body IDs and their data
    
    Returns:
        Tuple[BodiesData, str, bool]: Tuple containing:
            - Filtered bodies_data dictionary
            - String with noise information for logging
            - Boolean indicating if any sequence was denoised by spread
    
    Side Effects:
        - Logs filtered sequences to noise_spr_logger
        - Updates motion values for valid sequences
        - Modifies input bodies_data dictionary in-place
    
    Note:
        Requires at least 2 body IDs in input. Will preserve at least one body ID.
    """
    noise_info = str()
    denoised_by_spr = False  # mark if this sequence has been processed by spread.

    new_bodies_data = bodies_data.copy()
    
    for (bodyID, body_data) in new_bodies_data.items():
        if len(bodies_data) == 1:
            break
            
        valid_frames = get_valid_frames_by_spread(body_data['joints'].reshape(-1, 25, 3))
        num_frames = len(body_data['interval'])
        num_noise = num_frames - len(valid_frames)
        
        if num_noise == 0:
            continue

        ratio = num_noise / float(num_frames)
        motion = body_data['motion']
        
        if ratio >= noise_spr_thres2:  # 0.69754
            del bodies_data[bodyID]
            denoised_by_spr = True
            noise_info += 'Filter out: %s (spread rate >= %.2f).\n' % (bodyID, noise_spr_thres2)
            noise_spr_logger.info('%s\t%s\t%.6f\t%.6f' % (ske_name, bodyID, motion, ratio))
        else:  # Update motion
            joints = body_data['joints'].reshape(-1, 25, 3)[valid_frames]
            body_data['motion'] = min(motion, np.sum(np.var(joints.reshape(-1, 3), axis=0)))
            noise_info += '%s: motion %.6f -> %.6f\n' % (bodyID, motion, body_data['motion'])

    if noise_info != '':
        noise_info += '\n'

    return bodies_data, noise_info, denoised_by_spr


def denoising_by_motion(ske_name: str, bodies_data: BodiesData, 
                       bodies_motion: Dict[str, float]) -> Tuple[List[Tuple[str, BodyData]], str]:
    """
    Filter skeleton sequences based on motion amount thresholds.
    
    Sorts bodies by motion amount and filters out those with motion outside
    the acceptable range [noise_mot_thres_lo, noise_mot_thres_hi]. Always
    preserves the body with the largest motion.
    
    Args:
        ske_name (str): Name of the skeleton sequence file
        bodies_data (BodiesData): Dictionary mapping body IDs to their data
        bodies_motion (Dict[str, float]): Dictionary mapping body IDs to motion amounts
    
    Returns:
        Tuple[List[Tuple[str, BodyData]], str]: Tuple containing:
            - List of tuples (bodyID, body_data) for valid sequences
            - String with noise information for logging
    
    Side Effects:
        - Logs filtered sequences to noise_mot_logger
    
    Note:
        The body with the largest motion is always preserved regardless of thresholds
    """
    # Sort bodies based on motion in descending order
    bodies_motion_sorted = sorted(bodies_motion.items(), key=lambda x: x[1], reverse=True)

    # Reserve the body data with the largest motion
    denoised_bodies_data = [(bodies_motion_sorted[0][0], bodies_data[bodies_motion_sorted[0][0]])]
    noise_info = str()

    for (bodyID, motion) in bodies_motion_sorted[1:]:
        if (motion < noise_mot_thres_lo) or (motion > noise_mot_thres_hi):
            noise_info += 'Filter out: %s, %.6f (motion).\n' % (bodyID, motion)
            noise_mot_logger.info('{}\t{}\t{:.6f}'.format(ske_name, bodyID, motion))
        else:
            denoised_bodies_data.append((bodyID, bodies_data[bodyID]))
    
    if noise_info != '':
        noise_info += '\n'

    return denoised_bodies_data, noise_info


def denoising_bodies_data(bodies_data: SkeletonData) -> Tuple[List[Tuple[str, BodyData]], str]:
    """
    Apply comprehensive denoising pipeline to skeleton data.
    
    Performs multi-stage denoising using heuristic methods:
    1. Length-based filtering (removes short sequences)
    2. Spread-based filtering (removes sequences with poor joint spread)
    3. Motion-based sorting (sorts remaining sequences by motion amount)
    
    Args:
        bodies_data (SkeletonData): Dictionary containing:
            - 'name': Skeleton sequence name
            - 'data': Dictionary of body IDs and their corresponding data
            - 'num_frames': Total number of frames
    
    Returns:
        Tuple[List[Tuple[str, BodyData]], str]: Tuple containing:
            - List of tuples (bodyID, body_data) for denoised sequences
            - Combined noise information string from all denoising steps
    
    Note:
        This function implements a conservative approach that may not be
        correct for all samples, as noted in the original code comments.
    """
    ske_name = bodies_data['name']
    bodies_dict = bodies_data['data']

    # Step 1: Denoising based on frame length
    bodies_dict, noise_info_len = denoising_by_length(ske_name, bodies_dict)

    if len(bodies_dict) == 1:  # only has one bodyID left after step 1
        return list(bodies_dict.items()), noise_info_len

    # Step 2: Denoising based on spread
    bodies_dict, noise_info_spr, denoised_by_spr = denoising_by_spread(ske_name, bodies_dict)

    if len(bodies_dict) == 1:
        return list(bodies_dict.items()), noise_info_len + noise_info_spr

    # Sort bodies based on motion for final output
    bodies_motion = {bodyID: body_data['motion'] for bodyID, body_data in bodies_dict.items()}
    bodies_motion_sorted = sorted(bodies_motion.items(), key=lambda x: x[1], reverse=True)
    
    denoised_bodies_data = [(bodyID, bodies_dict[bodyID]) for bodyID, _ in bodies_motion_sorted]

    return denoised_bodies_data, noise_info_len + noise_info_spr


def get_one_actor_points(body_data: BodyData, num_frames: int) -> JointsColorsResult:
    """
    Extract joint positions and color data for a single actor.
    
    Converts skeleton data for one actor into standardized numpy arrays.
    Missing frames are filled with zeros for joints and NaN for colors.
    
    Args:
        body_data (BodyData): Dictionary containing:
            - 'interval': List of valid frame indices
            - 'joints': Array of joint positions (frames x 25 joints x 3)
            - 'colors': Array of color coordinates (frames x 25 x 2)
        num_frames (int): Total number of frames in the sequence
    
    Returns:
        JointsColorsResult: Tuple containing:
            - joints (np.ndarray): Shape (num_frames, 75) - flattened joint positions
            - colors (np.ndarray): Shape (num_frames, 1, 25, 2) - color coordinates
    
    Note:
        Each frame contains 75 values (25 joints × 3 coordinates) for joints.
        Color data contains 25 × 2 (X, Y) coordinates per frame.
    """
    joints = np.zeros((num_frames, 75), dtype=np.float32)
    colors = np.ones((num_frames, 1, 25, 2), dtype=np.float32) * np.nan
    
    start, end = body_data['interval'][0], body_data['interval'][-1]
    joints[start:end + 1] = body_data['joints'].reshape(-1, 75)
    colors[start:end + 1, 0] = body_data['colors']

    return joints, colors


def remove_missing_frames(ske_name: str, joints: np.ndarray, colors: np.ndarray) -> JointsColorsResult:
    """
    Remove frames where all joint positions are zeros (missing data).
    
    Identifies and removes frames with missing skeleton data. For sequences with
    two actors, also logs statistics about missing frames for each actor separately.
    
    Args:
        ske_name (str): Name of the skeleton sequence file
        joints (np.ndarray): Joint positions array, shape depends on number of actors:
            - Single actor: (num_frames, 75)
            - Two actors: (num_frames, 150)
        colors (np.ndarray): Color coordinates array with shape (num_frames, num_actors, 25, 2)
    
    Returns:
        JointsColorsResult: Tuple containing:
            - joints (np.ndarray): Filtered joint positions with missing frames removed
            - colors (np.ndarray): Color array with missing frames marked as NaN
    
    Side Effects:
        - Updates global missing_count
        - Logs missing frame statistics to various loggers
        - For two-actor sequences, logs detailed missing frame analysis
    
    Note:
        For two-actor data, joints[:, :75] represents actor 1, joints[:, 75:] represents actor 2
    """
    num_frames = joints.shape[0]
    num_bodies = colors.shape[1]  # 1 or 2

    # Debug logging for two-actor sequences
    if num_bodies == 2:
        missing_indices_1 = np.where(joints[:, :75].sum(axis=1) == 0)[0]
        missing_indices_2 = np.where(joints[:, 75:].sum(axis=1) == 0)[0]
        cnt1 = len(missing_indices_1)
        cnt2 = len(missing_indices_2)

        start = 1 if 0 in missing_indices_1 else 0
        end = 1 if num_frames - 1 in missing_indices_1 else 0
        
        if max(cnt1, cnt2) > 0:
            if cnt1 > cnt2:
                info = '{}\t{:^10d}\t{:^6d}\t{:^6d}\t{:^5d}\t{:^3d}'.format(
                    ske_name, num_frames, cnt1, cnt2, start, end)
                missing_skes_logger1.info(info)
            else:
                info = '{}\t{:^10d}\t{:^6d}\t{:^6d}'.format(ske_name, num_frames, cnt1, cnt2)
                missing_skes_logger2.info(info)

    # Find valid frame indices where at least one actor has data
    valid_indices = np.where(joints.sum(axis=1) != 0)[0]  # 0-based index
    missing_indices = np.where(joints.sum(axis=1) == 0)[0]
    num_missing = len(missing_indices)

    if num_missing > 0:  # Update joints and colors
        joints = joints[valid_indices]
        colors[missing_indices] = np.nan
        
        global missing_count
        missing_count += 1
        missing_skes_logger.info('{}\t{:^10d}\t{:^11d}'.format(ske_name, num_frames, num_missing))

    return joints, colors


def get_bodies_info(bodies_data: BodiesData) -> str:
    """
    Generate formatted information string about skeleton bodies.
    
    Creates a summary table showing body IDs, their frame intervals, and motion amounts.
    
    Args:
        bodies_data (BodiesData): Dictionary mapping body IDs to their data
            Each body_data contains 'interval' and 'motion' keys
    
    Returns:
        str: Formatted string containing tabulated body information with headers
    
    Example Output:
        ```
            bodyID          Interval     Motion  
        body_001        [10, 120]    1.234567
        body_002        [15, 110]    0.987654
        ```
    """
    bodies_info = '{:^17}\t{}\t{:^8}\n'.format('bodyID', 'Interval', 'Motion')
    
    for (bodyID, body_data) in bodies_data.items():
        start, end = body_data['interval'][0], body_data['interval'][-1]
        bodies_info += '{}\t{:^8}\t{:f}\n'.format(bodyID, str([start, end]), body_data['motion'])

    return bodies_info + '\n'


def get_two_actors_points(bodies_data: SkeletonData) -> JointsColorsResult:
    """
    Extract and process joint positions and colors for two-actor sequences.
    
    Handles complex multi-actor scenarios by:
    1. Denoising the input data
    2. Selecting primary and secondary actors based on motion
    3. Handling temporal overlaps between actors
    4. Generating detailed logging information
    
    Args:
        bodies_data (SkeletonData): Dictionary containing:
            - 'name': Skeleton sequence name
            - 'data': Dictionary of body IDs and their data
            - 'num_frames': Total number of frames
    
    Returns:
        JointsColorsResult: Tuple containing:
            - joints (np.ndarray): Shape (num_frames, 150) for two actors or (num_frames, 75) for one
            - colors (np.ndarray): Shape (num_frames, 2, 25, 2) or (num_frames, 1, 25, 2)
    
    Side Effects:
        - Saves detailed actor information to text files
        - Logs denoising failures for single vs. two-subject actions
        - Applies denoising pipeline which may modify input data
    
    Note:
        For two-actor output:
        - joints[:, :75] contains actor 1 data
        - joints[:, 75:] contains actor 2 data
        - Actors are selected based on motion amount and temporal non-overlap
    """
    ske_name = bodies_data['name']
    label = int(ske_name[-2:])  # Extract action label from filename
    num_frames = bodies_data['num_frames']
    bodies_info = get_bodies_info(bodies_data['data'])

    # Apply comprehensive denoising pipeline
    denoised_data, noise_info = denoising_bodies_data(bodies_data)
    bodies_info += noise_info

    denoised_data = list(denoised_data)
    
    if len(denoised_data) == 1:  # Only one actor remains after denoising
        if label >= 50:  # Two-subject actions typically have labels >= 50
            fail_logger_2.info(ske_name)  # Log denoising failure for two-subject action

        bodyID, body_data = denoised_data[0]
        joints, colors = get_one_actor_points(body_data, num_frames)
        bodies_info += 'Main actor: %s' % bodyID
        
    else:  # Multiple actors - process as two-actor sequence
        if label < 50:  # Single-subject actions typically have labels < 50
            fail_logger_1.info(ske_name)  # Log denoising failure for single-subject action

        joints = np.zeros((num_frames, 150), dtype=np.float32)
        colors = np.ones((num_frames, 2, 25, 2), dtype=np.float32) * np.nan

        # Process primary actor (highest motion)
        bodyID, actor1 = denoised_data[0]
        start1, end1 = actor1['interval'][0], actor1['interval'][-1]
        joints[start1:end1 + 1, :75] = actor1['joints'].reshape(-1, 75)
        colors[start1:end1 + 1, 0] = actor1['colors']
        
        actor1_info = '{:^17}\t{}\t{:^8}\n'.format('Actor1', 'Interval', 'Motion') + \
                      '{}\t{:^8}\t{:f}\n'.format(bodyID, str([start1, end1]), actor1['motion'])
        del denoised_data[0]

        # Initialize secondary actor tracking
        actor2_info = '{:^17}\t{}\t{:^8}\n'.format('Actor2', 'Interval', 'Motion')
        start2, end2 = [0, 0]  # Initial interval for actor2

        # Process remaining actors, assigning to actor1 or actor2 slots based on temporal overlap
        while len(denoised_data) > 0:
            bodyID, actor = denoised_data[0]
            start, end = actor['interval'][0], actor['interval'][-1]
            
            # Check temporal overlap with existing actors
            if min(end1, end) - max(start1, start) <= 0:  # No overlap with actor1
                joints[start:end + 1, :75] = actor['joints'].reshape(-1, 75)
                colors[start:end + 1, 0] = actor['colors']
                actor1_info += '{}\t{:^8}\t{:f}\n'.format(bodyID, str([start, end]), actor['motion'])
                # Update actor1 interval
                start1 = min(start, start1)
                end1 = max(end, end1)
            elif min(end2, end) - max(start2, start) <= 0:  # No overlap with actor2
                joints[start:end + 1, 75:] = actor['joints'].reshape(-1, 75)
                colors[start:end + 1, 1] = actor['colors']
                actor2_info += '{}\t{:^8}\t{:f}\n'.format(bodyID, str([start, end]), actor['motion'])
                # Update actor2 interval
                start2 = min(start, start2)
                end2 = max(end, end2)
            
            del denoised_data[0]

        bodies_info += ('\n' + actor1_info + '\n' + actor2_info)

    # Save detailed actor information to file
    with open(osp.join(actors_info_dir, ske_name + '.txt'), 'w') as fw:
        fw.write(bodies_info + '\n')

    return joints, colors


def get_raw_denoised_data() -> None:
    """
    Main processing function to denoise and extract skeleton data from raw sequences.
    
    Processes all skeleton sequences in the raw data file through the complete pipeline:
    1. Loads raw skeleton data from pickle file
    2. Applies appropriate processing based on number of actors
    3. Removes missing frames
    4. Saves processed data to pickle files
    5. Generates comprehensive logging and statistics
    
    The function handles both single-actor and multi-actor scenarios:
    - Single actor: Direct processing with get_one_actor_points
    - Multiple actors: Complex processing with get_two_actors_points
    
    Output Files:
        - raw_denoised_joints.pkl: Joint position data for all sequences
        - raw_denoised_colors.pkl: Color coordinate data for all sequences  
        - frames_cnt.txt: Frame count for each sequence
        - Various log files for debugging and analysis
    
    Side Effects:
        - Creates multiple output files in save_path directory
        - Updates global missing_count
        - Generates extensive logging information
        - Prints progress information to console
    
    Data Format:
        Each skeleton sequence frame contains:
        - Single actor: 75-dim vector (25 joints × 3 coordinates)
        - Two actors: 150-dim vector (2 actors × 75 dimensions)
        - Missing actor data is zero-padded
    
    Note:
        Processes sequences in batches of 1000 for progress reporting.
        The function assumes raw_data_file contains a list of SkeletonData dictionaries.
    """
    # Load raw skeleton data
    with open(raw_data_file, 'rb') as fr:
        raw_skes_data: List[SkeletonData] = pickle.load(fr)

    num_skes = len(raw_skes_data)
    print('Found %d available skeleton sequences.' % num_skes)

    # Initialize output containers
    raw_denoised_joints: List[np.ndarray] = []
    raw_denoised_colors: List[np.ndarray] = []
    frames_cnt: List[int] = []

    # Process each skeleton sequence
    for (idx, bodies_data) in enumerate(raw_skes_data):
        ske_name = bodies_data['name']
        print('Processing %s' % ske_name)
        num_bodies = len(bodies_data['data'])

        if num_bodies == 1:  # Single actor sequence
            num_frames = bodies_data['num_frames']
            body_data = list(bodies_data['data'].values())[0]
            joints, colors = get_one_actor_points(body_data, num_frames)
        else:  # Multi-actor sequence - select two main actors
            joints, colors = get_two_actors_points(bodies_data)
            # Remove frames with missing data
            joints, colors = remove_missing_frames(ske_name, joints, colors)
            num_frames = joints.shape[0]  # Update frame count after removal

        raw_denoised_joints.append(joints)
        raw_denoised_colors.append(colors)
        frames_cnt.append(num_frames)

        # Progress reporting
        if (idx + 1) % 1000 == 0:
            print('Processed: %.2f%% (%d / %d), Missing count: %d' % 
                  (100.0 * (idx + 1) / num_skes, idx + 1, num_skes, missing_count))

    # Save processed joint positions
    raw_skes_joints_pkl = osp.join(save_path, 'raw_denoised_joints.pkl')
    with open(raw_skes_joints_pkl, 'wb') as f:
        pickle.dump(raw_denoised_joints, f, pickle.HIGHEST_PROTOCOL)

    # Save processed color coordinates
    raw_skes_colors_pkl = osp.join(save_path, 'raw_denoised_colors.pkl')
    with open(raw_skes_colors_pkl, 'wb') as f:
        pickle.dump(raw_denoised_colors, f, pickle.HIGHEST_PROTOCOL)

    # Save frame count statistics
    frames_cnt_array = np.array(frames_cnt, dtype=np.int32)
    np.savetxt(osp.join(save_path, 'frames_cnt.txt'), frames_cnt_array, fmt='%d')

    # Final summary
    total_frames = np.sum(frames_cnt_array)
    print('Saved raw denoised positions of {} frames into {}'.format(total_frames, raw_skes_joints_pkl))
    print('Found %d files that have missing data' % missing_count)


if __name__ == '__main__':
    """
    Main execution block for skeleton data denoising.
    
    When run as a script, this will:
    1. Process all skeleton sequences in the raw data file
    2. Apply comprehensive denoising pipeline
    3. Save processed data and generate detailed logs
    4. Print summary statistics
    
    Expected Input:
        - raw_skes_data.pkl: Pickle file containing list of skeleton sequences
        - Each sequence should be a dictionary with 'name', 'data', and 'num_frames'
    
    Generated Output:
        - raw_denoised_joints.pkl: Processed joint position data
        - raw_denoised_colors.pkl: Processed color coordinate data
        - frames_cnt.txt: Frame counts for each sequence
        - Multiple log files for analysis and debugging
        - Individual actor info files in actors_info/ directory
    """
    get_raw_denoised_data()