"""
Graph in Graph (GiG) ST-GCN implementation for skeleton-based action recognition.
This module implements an advanced ST-GCN architecture with Graph in Graph mechanism,
featuring:
- SGU (Spatial Graph Unit): Standard spatial-temporal graph convolution
- GVU (Graph Vertex Update): Vertex update mechanism with representative vertices
- GGU (Graph Global Update): Global graph update for enhanced feature learning
The architecture uses multi-scale temporal convolutions, channel-wise topology
refinement graph convolution (CTRGC), and hierarchical graph processing.
"""


import math
import pdb
from typing import List, Optional, Dict, Any, Tuple, Union, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchsummary import summary

# Global constants
num_frames: int = 64
num_vertices: int = 25
g_num_vertices: int = num_vertices + 1  # Including representative vertex
adj_dim: int = 16


def import_class(name: str) -> type:
    """
    Dynamically import a class from a module path string.
    
    Args:
        name: Dot-separated module path (e.g., 'graph.ntu_rgb_d.Graph')
        
    Returns:
        The imported class
    """
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv: nn.Conv2d, branches: int) -> None:
    """
    Initialize convolutional layer weights for branched architecture.
    
    Args:
        conv: Convolutional layer to initialize
        branches: Number of branches in the architecture
    """
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
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


def weights_init(m: nn.Module) -> None:
    """
    General weight initialization function for different layer types.
    
    Args:
        m: PyTorch module to initialize
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


def act(x: torch.Tensor) -> torch.Tensor:
    """
    Mish activation function: x * tanh(softplus(x)).
    
    Args:
        x: Input tensor
        
    Returns:
        Activated tensor
    """
    return x * (torch.tanh(F.softplus(x)))


class Mish(nn.Module):
    """
    Mish activation function module.
    
    Mish is a smooth, non-monotonic activation function that tends to produce
    better results than ReLU in many deep learning tasks.
    """
    
    def __init__(self):
        super(Mish, self).__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Mish activation: x * tanh(softplus(x)).
        
        Args:
            x: Input tensor
            
        Returns:
            Activated tensor
        """
        return x * (torch.tanh(F.softplus(x)))


class TemporalConv(nn.Module):
    """
    Temporal convolution module with dilation support.
    
    Performs 1D convolution along the temporal dimension with configurable
    dilation for multi-scale temporal modeling.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolution kernel
        stride: Stride of the convolution (default: 1)
        dilation: Dilation factor for dilated convolution (default: 1)
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, dilation: int = 1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of temporal convolution.
        
        Args:
            x: Input tensor of shape (N, C, T, V)
            
        Returns:
            Output tensor with temporal features
        """
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    """
    Multi-scale temporal convolution with multiple dilated branches.
    
    This module applies temporal convolutions with different dilation rates
    to capture multi-scale temporal patterns, similar to Inception modules.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size(s) for temporal convolution (default: 3)
        stride: Stride for temporal convolution (default: 1)
        dilations: List of dilation rates for different branches (default: [1,2,3,4])
        residual: Whether to use residual connections (default: True)
        residual_kernel_size: Kernel size for residual connection (default: 1)
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, List[int]] = 3,
                 stride: int = 1, dilations: List[int] = [1,2,3,4], residual: bool = True,
                 residual_kernel_size: int = 1):
        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)
            
        # Temporal convolution branches with different dilations
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(branch_channels),
                Mish(),
                TemporalConv(branch_channels, branch_channels, kernel_size=ks, 
                           stride=stride, dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional max pooling branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            Mish(),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
        ))

        # Additional 1x1 convolution branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, 
                                       kernel_size=residual_kernel_size, stride=stride)

        # Initialize weights
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of multi-scale temporal convolution.
        
        Args:
            x: Input tensor of shape (N, C, T, V)
            
        Returns:
            Multi-scale temporal features concatenated along channel dimension
        """
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class Bi_Inter(nn.Module):
    def __init__(self,in_channels):
        super(Bi_Inter, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = in_channels // 2
        self.conv_c = nn.Sequential(
            nn.Conv2d(self.in_channels,self.inter_channels ,kernel_size=1),
            nn.BatchNorm2d(self.inter_channels),
            nn.GELU(),
            nn.Conv2d(self.inter_channels,self.in_channels,kernel_size=1)
        )
        self.conv_s = nn.Conv2d(self.in_channels, 1, kernel_size=1)

        self.conv_t = nn.Conv2d(self.in_channels, 1, kernel_size=(9,1), padding=(4,0))

        self.sigmoid = nn.Sigmoid()

    def forward(self,x,mode):
        if mode == 'channel':
            x_res = x.mean(-1, keepdim=True).mean(-2, keepdim=True) # N,C,1,1
            x_res = self.sigmoid(self.conv_c(x_res))
        elif mode == 'spatial':
            x_res = x.mean(2,keepdim=True) # N,C,1,V
            x_res = self.sigmoid(self.conv_s(x_res))
        else:
            x_res = x.mean(-1,keepdim=True) # N,C,T,1
            x_res = self.sigmoid(self.conv_t(x_res))
        return x_res



class SelfGCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(SelfGCN_Block, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv1_1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2_2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels,self.out_channels,kernel_size=1)
        self.att_t = nn.Conv2d(self.rel_channels,self.out_channels,kernel_size=1)
        self.Bi_inter = Bi_Inter(self.out_channels)

        # self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.conv5 = nn.Conv2d(2 * self.rel_channels, self.rel_channels, kernel_size=1, groups=self.rel_channels)
        self.beita = nn.Parameter(torch.zeros(1))
        self.tanh = nn.Tanh()
        self.bn = nn.BatchNorm2d(self.out_channels)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        N,C,T,V = x.size()
        x1, x2 = self.conv1(x), self.conv2(x)
        x1_mean, x2_mean = x1.mean(-2), x2.mean(-2)
        x1_max, _ = x1.max(-2)
        x2_max, _ = x2.max(-2)
        x_max_res = x1_max.unsqueeze(-1) - x2_max.unsqueeze(-2)
        x1_mean_res = x1_mean.unsqueeze(-1) - x2_mean.unsqueeze(-2)
        N1, C1, V1, V2 = x1_mean_res.shape
        x_result = torch.cat((x1_mean_res, x_max_res), 2)  # (N,rel_channels,2V,V)
        x_result = x_result.reshape(N1, 2 * C1, V1, V2)
        x_1 = self.tanh(self.conv5(x_result)) * alpha # (N,rel_channels,V,V)
        x_1 = self.conv4(x_1) # out_channels

        x3 = self.conv3(x) # out_channels
        # A.unsqueeze(0).unsqueeze(0).shape : (1,1,25,25)

        x1_res = x_1 + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,out_channels,V,V

        q , k = x1, x2 # rel_channels
        att = self.tanh(torch.einsum('nctu,nctv->ncuv',[q,k])/T) # rel_channels
        att = self.att_t(att) # out_channels
        global_res = torch.einsum('nctu,ncuv->nctv',x3,att)
        c_att = self.Bi_inter(global_res,'channel')
        x1_res = c_att * x1_res

        x1_res = torch.einsum('ncuv,nctv->nctu', x1_res, x3) # out_channels
        s_att = self.Bi_inter(x1_res,'spatial')
        global_res = global_res * s_att
        x1_res = x1_res + global_res



        # N, C, T, V = x1_1.size()
        x1_1, x2_2 = self.conv1_1(x), self.conv2_2(x)
        x1_trans = x1_1.permute(0, 2, 3, 1).contiguous()  # (N,T,V,C)
        x2_trans = x2_2.permute(0, 2, 1, 3).contiguous()  # (N,T,C,V)
        xt = self.tanh(torch.einsum('ntvc,ntcu->ntvu',x1_trans,x2_trans)/self.rel_channels) #(N,T,V,V)
        t_res = torch.einsum('nctu,ntuv->nctv',x3,xt) # 特定于时间的结果(N,C,T,V)
        res = t_res.permute(0, 2, 3, 1).contiguous() * self.beita + x1_res.permute(0, 2, 3, 1).contiguous()  # (N,T,V,C)
        res = res.permute(0, 3, 1, 2).contiguous()

        return res



class CTRGC(nn.Module):
    """
    Channel-wise Topology Refinement Graph Convolution.
    
    This module refines the graph topology in a channel-wise manner,
    learning adaptive adjacency matrices for better spatial modeling.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stage: Current stage number for different processing (default: 1)
        rel_reduction: Reduction factor for relation channels (default: 8)
        mid_reduction: Reduction factor for middle channels (default: 1)
    """
    
    def __init__(self, in_channels: int, out_channels: int, stage: int = 1, 
                 rel_reduction: int = 8, mid_reduction: int = 1):
        super(CTRGC, self).__init__()
        self.stage = stage
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Channel reduction for efficiency
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction

        # Convolution layers for topology refinement
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)

        self.tanh = nn.Tanh()
        
        # Initialize layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x: torch.Tensor, A: Optional[torch.Tensor] = None, 
                alpha: float = 1) -> torch.Tensor:
        """
        Forward pass of CTRGC.
        
        Args:
            x: Input tensor of shape (N, C, T, V)
            A: Optional adjacency matrix
            alpha: Scaling factor for learned adjacency
            
        Returns:
            Graph convolution output with refined topology
        """
        x1 = self.conv1(x).mean(-2)  # Global average pooling over time
        x2 = self.conv2(x).mean(-2)  # Global average pooling over time
        x3 = self.conv3(x)
        
        # Compute pairwise relationships
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)
        
        # Apply graph convolution using Einstein summation
        x1 = torch.einsum('ncvu,nctv->nctu', x1, x3)
        return x1


class unit_tcn(nn.Module):
    """
    Basic temporal convolution unit.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the temporal kernel (default: 9)
        stride: Stride of the convolution (default: 1)
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9, stride: int = 1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), 
                             padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of temporal convolution unit.
        
        Args:
            x: Input tensor of shape (N, C, T, V)
            
        Returns:
            Temporal convolution output
        """
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    """
    Advanced graph convolution unit with CTRGC and adaptive adjacency.
    
    This unit combines multiple CTRGC modules to process different subsets
    of the adjacency matrix, enabling more sophisticated spatial modeling.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        A: Adjacency matrix of shape (num_subsets, V, V)
        stage: Processing stage number (default: 1)
        rel_reduction: Channel reduction factor for CTRGC (default: 8)
        coff_embedding: Coefficient for channel embedding (default: 4)
        adaptive: Whether to use learnable adjacency matrices (default: True)
        residual: Whether to use residual connections (default: True)
    """
    
    def __init__(self, in_channels: int, out_channels: int, A: np.ndarray, stage: int = 1,
                 rel_reduction: int = 8, coff_embedding: int = 4, adaptive: bool = True, 
                 residual: bool = True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        
        # CTRGC modules for each adjacency subset
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(SelfGCN_Block(in_channels, out_channels))

        # Residual connection
        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
            
        # Adjacency matrix parameters
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
            
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)

        # Initialize layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of advanced GCN unit.
        
        Args:
            x: Input tensor of shape (N, C, T, V)
            
        Returns:
            Graph convolution output with enhanced spatial modeling
        """
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
            
        # Apply CTRGC for each adjacency subset
        for i in range(self.num_subset):
            # self.convs[0](x,A[0],self.alpha)   A[0] 是 I
            # self.convs[1](x,A[1],self.alpha)   A[1] 是inward
            # self.convs[2](x,A[2],self.alpha)   A[2] 是outward
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
            
        y = self.bn(y)
        y += self.down(x)  # Residual connection
        y = act(y)  # Mish activation

        return y


class TCN_GCN_unit(nn.Module):
    """
    Combined temporal-spatial graph convolution unit for GiG architecture.
    
    This unit combines advanced GCN with multi-scale temporal convolution,
    supporting different stages and configurations for hierarchical processing.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        A: Adjacency matrix
        stage: Processing stage number (default: 1)
        rel_reduction: Channel reduction factor (default: 8)
        stride: Temporal stride (default: 1)
        residual: Whether to use residual connections (default: True)
        adaptive: Whether to use adaptive adjacency (default: True)
        kernel_size: Temporal kernel size (default: 5)
        dilations: Dilation rates for multi-scale temporal conv (default: [1,2])
    """
    
    def __init__(self, in_channels: int, out_channels: int, A: np.ndarray, stage: int = 1,
                 rel_reduction: int = 8, stride: int = 1, residual: bool = True, 
                 adaptive: bool = True, kernel_size: int = 5, dilations: List[int] = [1,2]):
        super(TCN_GCN_unit, self).__init__()
        
        # Spatial modeling with advanced GCN
        self.gcn1 = unit_gcn(in_channels, out_channels, A, stage=stage, 
                           rel_reduction=rel_reduction, adaptive=adaptive)
        
        # Multi-scale temporal modeling
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, 
                                          kernel_size=kernel_size, stride=stride, 
                                          dilations=dilations, residual=False)

        # Residual connection configuration
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining spatial and temporal modeling.
        
        Args:
            x: Input tensor of shape (N, C, T, V)
            
        Returns:
            Enhanced spatial-temporal features
        """
        y = act(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


class Model(nn.Module):
    """
    Graph in Graph (GiG) ST-GCN model for skeleton-based action recognition.
    
    This model implements a hierarchical architecture with three main components:
    1. SGU (Spatial Graph Units): Standard spatial-temporal processing
    2. GVU (Graph Vertex Update): Introduces representative vertices  
    3. GGU (Graph Global Update): Global graph-level feature refinement
    
    The architecture uses different adjacency matrices for different stages,
    enabling progressive feature abstraction and global context modeling.
    
    Args:
        num_class: Number of action classes (default: 60)
        num_point: Number of skeleton joints (default: 25)
        num_person: Maximum number of people (default: 2)
        graph: Graph class path string
        graph_args: Arguments for graph construction (default: empty dict)
        in_channels: Number of input channels (default: 3)
        drop_out: Dropout probability (default: 0)
        adaptive: Whether to use adaptive adjacency matrices (default: True)
    """
    
    def __init__(self, num_class: int = 60, num_point: int = 25, num_person: int = 2,
                 graph: Optional[str] = None, graph_args: Dict[str, Any] = dict(),
                 in_channels: int = 3, drop_out: float = 0, adaptive: bool = True):
        super(Model, self).__init__()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if graph is None:
            raise ValueError("Graph class must be specified")
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        # Different adjacency matrices for different stages
        A1 = self.graph.A  # Shape: (3, 25, 25) - Standard skeleton adjacency
        A2 = self.get_adjacency(2)  # Representative vertex adjacency
        A3 = self.get_adjacency(3)  # Hierarchical adjacency for global modeling
        A4 = self.get_adjacency(4)  # Final global adjacency

        self.num_class = num_class
        self.num_point = num_point
        
        # Input normalization with representative vertex
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * (num_point + 1))

        base_channel = 64
        
        # SGU (Spatial Graph Units) - Standard spatial-temporal processing
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A1, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A1, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A1, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A1, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A1, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A1, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A1, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A1, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A1, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A1, adaptive=adaptive)
        
        # GVU (Graph Vertex Update) - Representative vertex processing
        self.l11 = TCN_GCN_unit(base_channel*4, base_channel*4, A=A2, stage=2, 
                               kernel_size=7, rel_reduction=32, adaptive=adaptive)
        
        # GGU Part 1 - Global graph modeling
        self.l12 = TCN_GCN_unit(base_channel*4, base_channel*4, A=A3, stage=3,
                               kernel_size=7, rel_reduction=32, adaptive=adaptive)
        
        # GGU Part 2 - Final global update
        self.l13 = TCN_GCN_unit(base_channel*4, base_channel*4, A=A4, stage=4,
                               kernel_size=7, rel_reduction=32, adaptive=adaptive)

        # Final classification layer
        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        
        # Dropout for regularization
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def normalize_digraph(self, A: torch.Tensor) -> torch.Tensor:
        """
        Normalize directed graph adjacency matrix.
        
        Args:
            A: Adjacency matrix tensor
            
        Returns:
            Normalized adjacency matrix
        """
        A = A.clone().detach().cpu().numpy()
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return torch.from_numpy(AD).float()

    @staticmethod
    def k_adjacency(A: np.ndarray, k: int, with_self: bool = False, 
                   self_factor: float = 1) -> np.ndarray:
        """
        Compute k-hop adjacency matrix.
        
        Args:
            A: Base adjacency matrix
            k: Number of hops
            with_self: Whether to include self-connections
            self_factor: Self-connection weight factor
            
        Returns:
            k-hop adjacency matrix
        """
        assert isinstance(A, np.ndarray)
        I = np.eye(len(A), dtype=A.dtype)
        if k == 0:
            return I
        Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
           - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
        if with_self:
            Ak += (self_factor * I)
        return Ak

    def get_adjacency(self, stage: int) -> np.ndarray:
        """
        Generate different adjacency matrices for different processing stages.
        
        Args:
            stage: Processing stage number (2, 3, or 4)
            
        Returns:
            Adjacency matrix for the specified stage
        """
        if stage == 2:
            # Stage 2: Representative vertex adjacency
            # Creates connections from all vertices to a representative vertex
            diag = torch.ones([g_num_vertices])
            adj = torch.diag_embed(diag)
            last_row = torch.ones([g_num_vertices])
            adj[25, :] = last_row  # Representative vertex connects to all
            adj = self.normalize_digraph(adj).unsqueeze(0)
            
        elif stage == 3:
            # Stage 3: Hierarchical adjacency for global modeling
            # Upper triangular, lower triangular, and diagonal matrices
            adj1 = self.normalize_digraph(torch.triu(torch.ones((adj_dim, adj_dim)), diagonal=1))
            adj2 = torch.tril(torch.ones((adj_dim, adj_dim)), diagonal=0)
            diag = torch.diag(adj2)
            adj2 = adj2 - diag
            adj2 = self.normalize_digraph(adj2)
            adj3 = torch.diag_embed(diag)
            adj = torch.stack((adj3, adj1, adj2))
            
        elif stage == 4:
            # Stage 4: Global adjacency with representative vertex
            # All vertices connect to the representative vertex
            diag = torch.ones([g_num_vertices])
            adj = torch.diag_embed(diag)
            src = torch.arange(num_vertices)
            tgt = torch.full([num_vertices,], num_vertices)
            ind = (src, tgt)
            value = torch.ones([src.shape[0]]) 
            adj = self.normalize_digraph(adj.index_put_(ind, value)).unsqueeze(0)
            
        return adj.numpy()

    def representative_vertex_generate(self, h: torch.Tensor) -> torch.Tensor:
        """
        Generate representative vertices for Graph Vertex Update (GVU).
        
        This method adds a representative vertex to each instance that
        summarizes the global information of the skeleton.
        
        Args:
            h: Input tensor of shape (N, C, T, V, M)    
        Returns:
            Tensor with representative vertices added, shape (N, C, T, V+1, M)
        """
        h_view = h.permute(0, 4, 1, 3, 2).contiguous()  # (N, M, C, V, T)
        h_view_shape = h_view.shape
        h_view = h_view.tolist()
        
        # Process each batch, instance, and channel
        for num_batch in range(h_view_shape[0]):
            for num_instance in range(h_view_shape[1]):
                for axis in range(h_view_shape[2]):
                    working_h = torch.tensor(h_view[num_batch][num_instance][axis])
                    # Create representative vertex as mean of all vertices
                    representative_vertex = working_h.mean(dim=0).reshape(1, num_frames)
                    representative_vertex = nn.init.kaiming_uniform_(representative_vertex)
                    list_h = working_h.tolist()
                    list_h.extend(representative_vertex.tolist())
                    h_view[num_batch][num_instance][axis] = list_h
                    
        h_view = torch.tensor(h_view).permute(0, 2, 4, 3, 1).contiguous()
        return h_view.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Graph in Graph ST-GCN model.
        
        The forward pass consists of three main stages:
        1. SGU (Spatial Graph Units): Standard spatial-temporal feature extraction
        2. GVU (Graph Vertex Update): Representative vertex generation and processing
        3. GGU (Graph Global Update): Global graph-level feature refinement
        
        Args:
            x: Input tensor. Can be either:
               - Shape (N, T, V*C): Flattened skeleton sequence
               - Shape (N, C, T, V, M): Standard skeleton tensor format
               where N=batch, C=channels, T=time, V=vertices, M=persons
               
        Returns:
            Class prediction logits of shape (N, num_class)
        """
        # Handle input format conversion if needed
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        
        # Generate representative vertices for GVU
        x = self.representative_vertex_generate(x)
        N, C, T, V, M = x.size()

        # Input normalization
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        
        # SGU (Spatial Graph Units) - Standard spatial-temporal processing
        # Progressive feature extraction with increasing channel dimensions
        h1 = self.l1(x)      # Initial feature extraction
        h1 = self.l2(h1)     # Feature refinement
        h1 = self.l3(h1)     # Feature refinement  
        h1 = self.l4(h1)     # Feature refinement
        h1 = self.l5(h1)     # Temporal downsampling + channel expansion
        h1 = self.l6(h1)     # Feature refinement
        h1 = self.l7(h1)     # Feature refinement
        h1 = self.l8(h1)     # Temporal downsampling + channel expansion
        h1 = self.l9(h1)     # Feature refinement
        h1 = self.l10(h1)    # Feature refinement
        
        # GVU (Graph Vertex Update) - Representative vertex processing
        # Process features with representative vertex adjacency
        h2 = self.l11(h1)
        
        # GGU Part 1 - Global graph modeling
        # Extract and process representative vertex features separately
        N2, C2, T2, _ = h2.size()
        h3 = h2[:, :, :, 25].clone().view(N2, C2, 1, T2)  # Extract representative vertex
        h3 = self.l12(h3)  # Process global features
        
        # GGU Part 2 - Final global update
        # Integrate processed global features back into the full representation
        h4 = h2.clone()
        h4[:, :, :, 25] = h3.reshape(N2, C2, T2)  # Update representative vertex
        h4 = self.l13(h4)  # Final global processing

        # Global average pooling and classification
        c_new = h4.size(1)
        h4 = h4.view(N, M, c_new, -1)  # Reshape to separate persons
        h4 = h4.mean(3).mean(1)        # Global average pooling over space and persons
        h4 = self.drop_out(h4)         # Apply dropout

        return self.fc(h4)
    

if __name__ == '__main__':
    model = Model(num_class=60, num_point=25, num_person=2,
                  graph='graph.ntu_rgb_d.Graph',
                  in_channels=3, drop_out=0, adaptive=True).to('cuda:0')

    # summary of torch model
    summary(model,input_size=(3, 64, 25, 2))