import random
from typing import List, Tuple, Union, Optional
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F

'''
This module provides various utility functions for processing skeleton data , icnluding:
- Temporal processing: cropping, resizing, downsampling, slicing, random choosing, shifting
- Spatial augmentation: random rotation, random movement
- Data normalization: mean subtraction, auto padding
- Multi-person handling: OpenPose matching
'''

# Temporal Processing


def valid_crop_resize(
    data_numpy: np.ndarray,
    valid_frame_num: int,
    p_interval: Union[List[float], Tuple[float, ...]],
    window: int
) -> np.ndarray:
    """
    Crop a portion of the sequence and resize it to a fixed window size.
    
    This function performs temporal cropping followed by resizing using bilinear interpolation.
    It can handle both fixed cropping (center crop) and random cropping with variable ratios.
    
    Args:
        data_numpy: Input skeleton data with shape (C, T, V, M) where:
                   C = channels (typically 3 for x,y,confidence or x,y,z coordinates)
                   T = temporal frames
                   V = number of joints/vertices
                   M = number of persons
        valid_frame_num: Number of valid frames in the sequence
        p_interval: Cropping ratio interval. If single value, performs center crop.
                   If two values [min, max], performs random crop with ratio in this range.
        window: Target temporal window size after resizing
        
    Returns:
        Processed data with shape (C, window, V, M)
        
    Example:
        >>> data = np.random.rand(3, 100, 25, 2)  # 3D skeleton data
        >>> result = valid_crop_resize(data, 100, [0.8, 1.0], 64)
        >>> print(result.shape)  # (3, 64, 25, 2)
    """
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin

    #crop
    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1-p) * valid_size/2)
        data = data_numpy[:, begin+bias:end-bias, :, :]# center_crop
        cropped_length = data.shape[1]
    else:
        p = np.random.rand(1)*(p_interval[1]-p_interval[0])+p_interval[0]
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size*p)),64), valid_size)# constraint cropped_length lower bound as 64
        bias = np.random.randint(0,valid_size-cropped_length+1)
        data = data_numpy[:, begin+bias:begin+bias+cropped_length, :, :]
        if data.shape[1] == 0:
            print(cropped_length, bias, valid_size)

    # resize
    data = torch.tensor(data,dtype=torch.float)
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
    data = data[None, None, :, :]
    data = F.interpolate(data, size=(C * V * M, window), mode='bilinear',align_corners=False).squeeze() # could perform both up sample and down sample
    data = data.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous().numpy()

    return data


def downsample(
    data_numpy: np.ndarray, 
    step: int, 
    random_sample: bool = True
) -> np.ndarray:
    """
    Downsample the temporal dimension by taking every step-th frame.
    
    Args:
        data_numpy: Input skeleton data with shape (C, T, V, M)
        step: Downsampling step size (e.g., step=2 takes every 2nd frame)
        random_sample: If True, randomly choose starting frame within step range.
                      If False, always start from frame 0.
                      
    Returns:
        Downsampled data with shape (C, T//step, V, M)
        
    Example:
        >>> data = np.random.rand(3, 100, 25, 2)
        >>> downsampled = downsample(data, step=2)
        >>> print(downsampled.shape)  # (3, 50, 25, 2)
    """
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy: np.ndarray, step: int) -> np.ndarray:
    """
    Slice the temporal dimension into chunks and reshape for multi-stream processing.
    
    This function groups consecutive frames into chunks of size 'step' and treats
    each chunk as additional persons in the M dimension.
    
    Args:
        data_numpy: Input skeleton data with shape (C, T, V, M)
        step: Size of temporal chunks
        
    Returns:
        Reshaped data with shape (C, T//step, V, step*M)
        
    Note:
        T must be divisible by step for this function to work correctly.
        
    Example:
        >>> data = np.random.rand(3, 100, 25, 2)
        >>> sliced = temporal_slice(data, step=4)
        >>> print(sliced.shape)  # (3, 25, 25, 8)
    """
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T // step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T // step, V, step * M)

def random_choose(
    data_numpy: np.ndarray, 
    size: int, 
    auto_pad: bool = True
) -> np.ndarray:
    """
    Randomly select a temporal segment of specified size from the sequence.
    
    Args:
        data_numpy: Input skeleton data with shape (C, T, V, M)
        size: Target temporal size
        auto_pad: If True, pad sequences shorter than size with random positioning.
                 If False, return original data when T < size.
                 
    Returns:
        Processed data with shape (C, size, V, M) or original shape if shorter and auto_pad=False
        
    Note:
        This function may not handle zero-padded frames optimally as mentioned in comments.
        
    Example:
        >>> data = np.random.rand(3, 200, 25, 2)
        >>> segment = random_choose(data, size=64)
        >>> print(segment.shape)  # (3, 64, 25, 2)
    """
    # input: C,T,V,M 
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]


def random_shift(data_numpy: np.ndarray) -> np.ndarray:
    """
    Randomly shift the temporal position of valid frames within the sequence.
    
    This function identifies the valid frame range (non-zero frames) and randomly
    repositions this segment within the full temporal dimension, padding with zeros.
    
    Args:
        data_numpy: Input skeleton data with shape (C, T, V, M)
        
    Returns:
        Temporally shifted data with same shape (C, T, V, M)
        
    Example:
        >>> data = np.zeros((3, 100, 25, 2))
        >>> data[:, 20:60, :, :] = np.random.rand(3, 40, 25, 2)  # Valid frames 20-60
        >>> shifted = random_shift(data)
        >>> # Valid frames now at random position within 0-100
    """
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift

# Data Normalization

def mean_subtractor(
    data_numpy: np.ndarray, 
    mean: Union[float, np.ndarray]
) -> Optional[np.ndarray]:
    """
    Subtract mean values from the skeleton data for normalization.
    
    Args:
        data_numpy: Input skeleton data with shape (C, T, V, M)
        mean: Mean value(s) to subtract. Can be scalar or array matching data shape.
              If 0, no operation is performed.
              
    Returns:
        Mean-subtracted data or None if mean=0
        
    Note:
        Only processes frames within the valid range (non-zero frames).
        
    Example:
        >>> data = np.random.rand(3, 100, 25, 2) + 10
        >>> normalized = mean_subtractor(data, mean=10.0)
    """
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_pading(
    data_numpy: np.ndarray, 
    size: int, 
    random_pad: bool = False
) -> np.ndarray:
    """
    Pad the temporal dimension with zeros if sequence is shorter than target size.
    
    Args:
        data_numpy: Input skeleton data with shape (C, T, V, M)
        size: Target temporal size
        random_pad: If True, randomly position the sequence within padded array.
                   If False, place sequence at the beginning.
                   
    Returns:
        Padded data with shape (C, size, V, M) if T < size, otherwise original data
        
    Example:
        >>> data = np.random.rand(3, 50, 25, 2)
        >>> padded = auto_pading(data, size=100, random_pad=True)
        >>> print(padded.shape)  # (3, 100, 25, 2)
    """
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy




# Spatial augmentation
def _rot(rot: torch.Tensor) -> torch.Tensor:
    """
    Create 3D rotation matrices from Euler angles.
    
    Constructs rotation matrices for each frame using XYZ Euler angle convention.
    The rotation is applied as R = Rz * Ry * Rx.
    
    Args:
        rot: Rotation angles with shape (T, 3) where T is number of frames
             and the 3 dimensions are rotations around X, Y, Z axes respectively
             
    Returns:
        Rotation matrices with shape (T, 3, 3)
        
    Example:
        >>> angles = torch.randn(100, 3) * 0.1  # Small random rotations
        >>> rotation_matrices = _rot(angles)
        >>> print(rotation_matrices.shape)  # torch.Size([100, 3, 3])
    """
    cos_r, sin_r = rot.cos(), rot.sin()  # T,3
    zeros = torch.zeros(rot.shape[0], 1)  # T,1
    ones = torch.ones(rot.shape[0], 1)  # T,1

    r1 = torch.stack((ones, zeros, zeros),dim=-1)  # T,1,3
    rx2 = torch.stack((zeros, cos_r[:,0:1], sin_r[:,0:1]), dim = -1)  # T,1,3
    rx3 = torch.stack((zeros, -sin_r[:,0:1], cos_r[:,0:1]), dim = -1)  # T,1,3
    rx = torch.cat((r1, rx2, rx3), dim = 1)  # T,3,3

    ry1 = torch.stack((cos_r[:,1:2], zeros, -sin_r[:,1:2]), dim =-1)
    r2 = torch.stack((zeros, ones, zeros),dim=-1)
    ry3 = torch.stack((sin_r[:,1:2], zeros, cos_r[:,1:2]), dim =-1)
    ry = torch.cat((ry1, r2, ry3), dim = 1)

    rz1 = torch.stack((cos_r[:,2:3], sin_r[:,2:3], zeros), dim =-1)
    r3 = torch.stack((zeros, zeros, ones),dim=-1)
    rz2 = torch.stack((-sin_r[:,2:3], cos_r[:,2:3],zeros), dim =-1)
    rz = torch.cat((rz1, rz2, r3), dim = 1)

    rot = rz.matmul(ry).matmul(rx)
    return rot


def random_rot(data_numpy: np.ndarray, theta: float = 0.3) -> torch.Tensor:
    """
    Apply random 3D rotations to skeleton data for augmentation.
    
    Generates random rotation angles within [-theta, theta] range and applies
    the same rotation to all frames. Useful for 3D skeleton data augmentation.
    
    Args:
        data_numpy: Input skeleton data with shape (C, T, V, M) where C >= 3
                   First 3 channels should be x, y, z coordinates
        theta: Maximum rotation angle in radians for each axis
        
    Returns:
        Rotated data as torch.Tensor with same shape (C, T, V, M)
        
    Note:
        The function assumes the first 3 channels are spatial coordinates (x, y, z).
        All frames receive the same random rotation.
        
    Example:
        >>> data = np.random.rand(3, 100, 25, 2)
        >>> rotated = random_rot(data, theta=0.2)
        >>> print(rotated.shape)  # torch.Size([3, 100, 25, 2])
    """
    data_torch = torch.from_numpy(data_numpy)
    C, T, V, M = data_torch.shape
    data_torch = data_torch.permute(1, 0, 2, 3).contiguous().view(T, C, V*M)  # T,3,V*M
    rot = torch.zeros(3).uniform_(-theta, theta)
    rot = torch.stack([rot, ] * T, dim=0)
    rot = _rot(rot)  # T,3,3
    data_torch = torch.matmul(rot, data_torch)
    data_torch = data_torch.view(T, C, V, M).permute(1, 0, 2, 3).contiguous()

    return data_torch

def random_move(
    data_numpy: np.ndarray,
    angle_candidate: List[float] = [-10., -5., 0., 5., 10.],
    scale_candidate: List[float] = [0.9, 1.0, 1.1],
    transform_candidate: List[float] = [-0.2, -0.1, 0.0, 0.1, 0.2],
    move_time_candidate: List[int] = [1]
) -> np.ndarray:
    """
    Apply random spatial transformations (rotation, scaling, translation) to skeleton data.
    
    This function applies smooth temporal variations of 2D transformations including:
    - Rotation around the origin
    - Uniform scaling 
    - Translation in x and y directions
    
    Args:
        data_numpy: Input skeleton data with shape (C, T, V, M).
                   First 2 channels should be x,y coordinates.
        angle_candidate: List of rotation angles in degrees to choose from
        scale_candidate: List of scale factors to choose from  
        transform_candidate: List of translation offsets to choose from
        move_time_candidate: List of number of temporal control points for smooth interpolation
        
    Returns:
        Transformed data with same shape (C, T, V, M)
        
    Note:
        Only the first 2 channels (x, y coordinates) are transformed.
        Other channels (e.g., confidence, z-coordinate) remain unchanged.
        
    Example:
        >>> data = np.random.rand(3, 100, 25, 2)
        >>> augmented = random_move(data, angle_candidate=[-15, 0, 15])
        >>> print(augmented.shape)  # (3, 100, 25, 2)
    """
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy

# Multi-person handling

def openpose_match(data_numpy: np.ndarray) -> np.ndarray:
    """
    Match and track multiple persons across frames using OpenPose confidence scores.
    
    This function solves the person tracking problem by:
    1. Ranking persons by confidence scores in each frame
    2. Finding optimal person assignments between consecutive frames using distance
    3. Maintaining consistent person IDs throughout the sequence
    4. Sorting final output by overall confidence scores
    
    Args:
        data_numpy: Input skeleton data with shape (C, T, V, M) where:
                   C must be 3 with channels [x, y, confidence]
                   T = temporal frames
                   V = number of joints
                   M = maximum number of persons
                   
    Returns:
        Person-matched data with same shape (C, T, V, M) where person IDs 
        are consistent across frames and sorted by confidence
        
    Note:
        Requires confidence scores in the 3rd channel (index 2).
        Uses Hungarian-like algorithm for optimal person matching.
        
    Example:
        >>> # Data with x, y coordinates and confidence scores
        >>> data = np.random.rand(3, 100, 25, 4)  # 4 persons max
        >>> data[2, :, :, :] = np.random.rand(100, 25, 4)  # Confidence scores
        >>> matched = openpose_match(data)
        >>> print(matched.shape)  # (3, 100, 25, 4)
    """
    C, T, V, M = data_numpy.shape
    assert (C == 3)
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert (np.all(forward_map >= 0))

    # string data
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[
                                                             t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy