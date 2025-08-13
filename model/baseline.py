# baseline.py

"""
Spatial-Temporal Graph Convolutional Network (ST-GCN) implementation.
This module implements ST-GCN for skeleton-based action recognition, combining
graph convolutional networks (GCN) for spatial modeling and temporal convolutional
networks (TCN) for temporal modeling.
"""

import math
from typing import Callable, Optional, Dict, Any, Union

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def import_class(name: str) -> type:
    """
    Dynamically import a class from a module path string.
    
    Args:
        name: Dot-separated module path (e.g., 'graph.ntu_rgb_d.Graph')
        
    Returns:
        The imported class
        
    Example:
        >>> Graph = import_class('graph.ntu_rgb_d.Graph')
    """
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv: nn.Conv2d, branches: int) -> None:
    """
    Initialize convolutional layer weights for branched architecture.
    
    Uses normal initialization with variance scaled by the number of branches
    to maintain proper signal propagation in multi-branch networks.
    
    Args:
        conv: Convolutional layer to initialize
        branches: Number of branches in the architecture
    """
    weight = conv.weight
    n = weight.size(0)  # Output channels
    k1 = weight.size(1)  # Input channels
    k2 = weight.size(2)  # Kernel height
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def conv_init(conv: nn.Conv2d) -> None:
    """
    Initialize convolutional layer with Kaiming normal initialization.
    
    Args:
        conv: Convolutional layer to initialize
    """
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn: nn.BatchNorm2d, scale: float) -> None:
    """
    Initialize batch normalization layer.
    
    Args:
        bn: Batch normalization layer to initialize
        scale: Scale factor for the weight initialization
    """
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    """
    Temporal Convolutional Network unit.
    
    Applies 1D convolution along the temporal dimension with batch normalization.
    This unit captures temporal dependencies in the skeleton sequence.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the temporal convolution kernel (default: 5)
        stride: Stride of the convolution (default: 1)
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        
        # Temporal convolution: (kernel_size, 1) means only temporal dimension
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=(kernel_size, 1), 
            padding=(pad, 0),
            stride=(stride, 1)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Initialize layers
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of TCN unit.
        
        Args:
            x: Input tensor of shape (N, C, T, V) where:
               N = batch size, C = channels, T = time steps, V = vertices
               
        Returns:
            Output tensor of shape (N, out_channels, T', V) where T' depends on stride
        """
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    """
    Graph Convolutional Network unit.
    
    Applies graph convolution using adjacency matrices to model spatial relationships
    between skeleton joints. Supports adaptive adjacency matrices that are learned
    during training.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels  
        A: Adjacency matrix of shape (num_subsets, V, V) where V is number of vertices
        adaptive: Whether to use learnable adjacency matrices (default: True)
    """
    
    def __init__(self, in_channels: int, out_channels: int, A: np.ndarray, adaptive: bool = True):
        super(unit_gcn, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = A.shape[0]  # Number of adjacency matrix subsets
        self.adaptive = adaptive
        
        if adaptive:
            # Learnable adjacency matrices
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)
        else:
            # Fixed adjacency matrices
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)

        # Separate convolution for each adjacency matrix subset
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        # Residual connection with dimension matching
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Initialize all layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        
        # Initialize branch convolutions
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def L2_norm(self, A: torch.Tensor) -> torch.Tensor:
        """
        Apply L2 normalization to adjacency matrices.
        
        Args:
            A: Adjacency matrices of shape (N, V, V)
            
        Returns:
            L2-normalized adjacency matrices
        """
        A_norm = torch.norm(A, 2, dim=1, keepdim=True) + 1e-4  # N, 1, V
        A = A / A_norm
        return A

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GCN unit.
        
        Args:
            x: Input tensor of shape (N, C, T, V)
            
        Returns:
            Output tensor of shape (N, out_channels, T, V)
        """
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            A = self.PA
            A = self.L2_norm(A)
        else:
            A = self.A.cuda(x.get_device())
            
        # Apply graph convolution for each adjacency matrix subset
        for i in range(self.num_subset):
            A1 = A[i]  # Shape: (V, V)
            A2 = x.view(N, C * T, V)  # Reshape for matrix multiplication
            # Graph convolution: A2 @ A1, then apply 1x1 conv
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)  # Residual connection
        y = self.relu(y)

        return y


class TCN_GCN_unit(nn.Module):
    """
    Combined Temporal-Spatial Graph Convolutional unit.
    
    This is the core building block of ST-GCN, combining spatial modeling (GCN)
    followed by temporal modeling (TCN) with residual connections.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        A: Adjacency matrix for graph convolution
        stride: Stride for temporal convolution (default: 1)
        residual: Whether to use residual connections (default: True)
        adaptive: Whether to use adaptive adjacency matrices (default: True)
    """
    
    def __init__(self, in_channels: int, out_channels: int, A: np.ndarray, 
                 stride: int = 1, residual: bool = True, adaptive: bool = True):
        super(TCN_GCN_unit, self).__init__()
        
        # Spatial modeling
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        # Temporal modeling  
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        
        # Configure residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            # Use 1x1 temporal conv to match dimensions
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining spatial and temporal modeling.
        
        Args:
            x: Input tensor of shape (N, C, T, V)
            
        Returns:
            Output tensor with spatial-temporal features extracted
        """
        # Spatial → Temporal → Residual connection → Activation
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


class Model(nn.Module):
    """
    ST-GCN model for skeleton-based action recognition.
    
    This model uses a series of ST-GCN blocks (TCN_GCN_unit) to extract
    spatial-temporal features from skeleton sequences, followed by global
    average pooling and a linear classifier.
    
    Args:
        num_class: Number of action classes to predict (default: 60)
        num_point: Number of skeleton joints/vertices (default: 25)
        num_person: Maximum number of people in the scene (default: 2)
        graph: Graph class path string (e.g., 'graph.ntu_rgb_d.Graph')
        graph_args: Arguments for graph construction (default: empty dict)
        in_channels: Number of input channels (default: 3 for x,y,z coordinates)
        drop_out: Dropout probability (default: 0)
        adaptive: Whether to use adaptive adjacency matrices (default: True)
        num_set: Number of adjacency matrix subsets (default: 3)
        
    Raises:
        ValueError: If graph is None
    """
    
    def __init__(self, num_class: int = 60, num_point: int = 25, num_person: int = 2, 
                 graph: Optional[str] = None, graph_args: Dict[str, Any] = dict(), 
                 in_channels: int = 3, drop_out: float = 0, adaptive: bool = True, 
                 num_set: int = 3):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError("Graph class must be specified")
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        # Initialize adjacency matrices as identity matrices
        A = np.stack([np.eye(num_point)] * num_set, axis=0)
        
        self.num_class = num_class
        self.num_point = num_point
        
        # Input batch normalization
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # ST-GCN layers with increasing channel dimensions
        self.l1 = TCN_GCN_unit(3, 64, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, adaptive=adaptive)  # Temporal downsampling
        self.l6 = TCN_GCN_unit(128, 128, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(128, 128, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2, adaptive=adaptive)  # Temporal downsampling
        self.l9 = TCN_GCN_unit(256, 256, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(256, 256, A, adaptive=adaptive)
        
        # Final classification layer
        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        
        # Dropout for regularization
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ST-GCN model.
        
        Args:
            x: Input tensor of shape (N, C, T, V, M) where:
               N = batch size
               C = input channels (typically 3 for x,y,z coordinates)
               T = time steps/frames
               V = number of vertices/joints
               M = number of people
               
        Returns:
            Tensor of shape (N, num_class) containing class logits
        """
        N, C, T, V, M = x.size()
        
        # Reshape for batch normalization: (N, M*V*C, T)
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        
        # Reshape to (N*M, C, T, V) for processing each person separately
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        
        # Pass through ST-GCN layers
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # Global average pooling and classification
        # x shape: (N*M, C, T, V)
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)  # (N, M, C, T*V)
        x = x.mean(3).mean(1)        # Global average pooling: (N, C)
        x = self.drop_out(x)

        return self.fc(x)