import numpy as np
from typing import List, Tuple

'''
This module provides utility functions for graph operations :
- Building adjacency matrices from edge lists
- Normalizing adjacency matrices for directed and undirected graphs
- Creating spatial graphs with self-loops, inward, and outward connections
- Generating k-hop adjacency matrices
- Subgraph projection matrices
- Multi-scale spatial graphs
- Uniform graph adjacency matrices
- k-adjacency matrices
- Uniform graph adjacency matrices
'''


def get_subgraph_projection_mat(num_in: int, num_out: int, link: List[Tuple[int, int]]) -> np.ndarray:
    """
    Build and column-normalize a subgraph projection matrix.

    Args:
        num_in (int): Number of input nodes (rows).
        num_out (int): Number of output nodes (columns).
        link (List[Tuple[int, int]]): List of (input_node, output_node) edges.

    Returns:
        np.ndarray: Shape (num_in, num_out), normalized so each column sums to 1.

    Purpose:
        Used to map from one set of nodes to another in graph-based models,
        ensuring equal weight distribution to each output node's incoming edges.


    Example:
    Suppose we have:
        - num_in = 25 joints of a human skeleton (input nodes),
        - num_out = 5 body parts like head, torso, arms, legs, etc. (output nodes),
        - A link list that maps which joints belong to which parts.

        We can use this function to project features from joints to body parts 
        before doing high-level reasoning. The result is a body-part-level representation.
    """
    A = np.zeros((num_in, num_out))
    for i, j in link:
        A[i, j] = 1
    A_norm = A / np.sum(A, axis=0, keepdims=True)
    return A_norm


def edges_to_adj_mat(link: List[Tuple[int, int]], num_node: int) -> np.ndarray:
    """
    Convert an edge list into an adjacency matrix.

    Args:
        link (List[Tuple[int, int]]): List of edges (source_node, target_node).
        num_node (int): Total number of nodes.

    Returns:
        np.ndarray: Adjacency matrix of shape (num_node, num_node).
    """
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


# for directed graphs, use this normalization
def normalize_digraph(A: np.ndarray) -> np.ndarray:
    """
    Normalize a directed graph adjacency matrix column-wise.

    Args:
        A (np.ndarray): Adjacency matrix.

    Returns:
        np.ndarray: Column-normalized adjacency matrix.
    """
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


# for undirected graphs, use this normalization
def normalize_adjacency_matrix(A: np.ndarray) -> np.ndarray:
    """
    Symmetrically normalize an adjacency matrix for GCN.

    Args:
        A (np.ndarray): Adjacency matrix.

    Returns:
        np.ndarray: Normalized adjacency matrix (float32).
    """
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)


def get_k_scale_graph(scale: int, A: np.ndarray) -> np.ndarray:
    """
    Get the k-hop reachability adjacency matrix.

    Args:
        scale (int): Maximum hop distance to consider.
        A (np.ndarray): Base adjacency matrix.

    Returns:
        np.ndarray: Binary adjacency showing reachability within 'scale' hops.
    """
    if scale == 1:
        return A
    An = np.zeros_like(A)
    A_power = np.eye(A.shape[0])
    for _ in range(scale):
        A_power = A_power @ A
        An += A_power
    An[An > 0] = 1
    return An

def k_adjacency(A: np.ndarray, k: int, with_self: bool = False, self_factor: float = 1) -> np.ndarray:
    """
    Get the adjacency matrix for exactly k-hop neighbors.

    Args:
        A (np.ndarray): Adjacency matrix.
        k (int): Hop distance.
        with_self (bool): Whether to include self-loops.
        self_factor (float): Weight of self-loops if included.

    Returns:
        np.ndarray: k-hop adjacency matrix.
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



def get_uniform_graph(num_node: int,
                      self_link: List[Tuple[int, int]],
                      neighbor: List[Tuple[int, int]]) -> np.ndarray:
    """
    Build a single-channel uniform graph adjacency matrix.

    Args:
        num_node (int): Number of nodes.
        self_link (List[Tuple[int, int]]): Self-loop edges.
        neighbor (List[Tuple[int, int]]): Neighbor edges.

    Returns:
        np.ndarray: Normalized adjacency matrix of shape (num_node, num_node).
    """
    A = normalize_digraph(edges_to_adj_mat(neighbor + self_link, num_node))
    return A


def get_spatial_graph(num_node: int,
                      self_link: List[Tuple[int, int]],
                      inward: List[Tuple[int, int]],
                      outward: List[Tuple[int, int]]) -> np.ndarray:
    """
    Build a 3-channel spatial graph adjacency tensor.

    Args:
        num_node (int): Number of nodes.
        self_link (List[Tuple[int, int]]): Self-loop edges.
        inward (List[Tuple[int, int]]): Inward edges.
        outward (List[Tuple[int, int]]): Outward edges.

    Returns:
        np.ndarray: Shape (3, num_node, num_node).
            Channel 0: Self-loops
            Channel 1: Inward normalized adjacency
            Channel 2: Outward normalized adjacency
    """
    I = edges_to_adj_mat(self_link, num_node)
    In = normalize_digraph(edges_to_adj_mat(inward, num_node))
    Out = normalize_digraph(edges_to_adj_mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A

def get_multiscale_spatial_graph(num_node: int,
                                 self_link: List[Tuple[int, int]],
                                 inward: List[Tuple[int, int]],
                                 outward: List[Tuple[int, int]]
                                 ) -> np.ndarray:
    """
    Build a 5-channel multi-scale spatial adjacency tensor.

    Args:
        num_node (int): Number of nodes.
        self_link (List[Tuple[int, int]]): Self-loop edges.
        inward (List[Tuple[int, int]]): Inward edges.
        outward (List[Tuple[int, int]]): Outward edges.

    Returns:
        np.ndarray: Shape (5, num_node, num_node).
            Channels:
            0 - Self-loops
            1 - Inward 1-hop normalized adjacency
            2 - Outward 1-hop normalized adjacency
            3 - Inward 2-hop normalized adjacency
            4 - Outward 2-hop normalized adjacency
    """

    # self-link
    I = edges_to_adj_mat(self_link, num_node)
    # inward and outward edges
    A1 = edges_to_adj_mat(inward, num_node)
    A2 = edges_to_adj_mat(outward, num_node)
    # 2-hop adjacency for both inward and outward 
    A3 = k_adjacency(A1, 2)
    A4 = k_adjacency(A2, 2)
    # normalization
    A1 = normalize_digraph(A1)
    A2 = normalize_digraph(A2)
    A3 = normalize_digraph(A3)
    A4 = normalize_digraph(A4)
    A = np.stack((I, A1, A2, A3, A4))
    return A
