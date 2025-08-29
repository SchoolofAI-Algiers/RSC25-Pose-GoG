# feeder_ntu.py

from typing import List, Tuple, Union, Optional, Any, Iterator
import numpy as np
from torch.utils.data import Dataset
from data_feeders import data_utils


class DatasetFeeder(Dataset):
    """
    PyTorch Dataset class for loading and preprocessing NTU RGB+D skeleton data.
    
    This class handles loading skeleton sequences from NPZ files and applies various
    data augmentation techniques including temporal sampling, spatial transformations,
    bone vector computation, and velocity estimation.
    
    Attributes:
        debug (bool): If True, only use first 100 samples for debugging
        data_path (str): Path to the NPZ data file
        label_path (Optional[str]): Path to label file (currently unused)
        split (str): Dataset split ('train' or 'test')
        random_choose (bool): Enable random temporal cropping
        random_shift (bool): Enable random temporal shifting
        random_move (bool): Enable random spatial transformations
        random_rot (bool): Enable random 3D rotations
        window_size (int): Target sequence length (-1 for original length)
        normalization (bool): Enable data normalization
        use_mmap (bool): Use memory mapping for data loading
        p_interval (Union[float, List[float]]): Temporal cropping ratio(s)
        bone (bool): Compute bone vectors instead of joint positions
        vel (bool): Compute velocity vectors (temporal differences)
        data (np.ndarray): Loaded skeleton data with shape (N, C, T, V, M)
        label (np.ndarray): Action labels
        sample_name (List[str]): Sample identifiers
        mean_map (Optional[np.ndarray]): Mean values for normalization
        std_map (Optional[np.ndarray]): Standard deviation values for normalization
    """
    
    def __init__(
        self, 
        data_path: str, 
        label_path: Optional[str] = None, 
        p_interval: Union[float, List[float]] = 1, 
        split: str = 'train', 
        random_choose: bool = False, 
        random_shift: bool = False,
        random_move: bool = False, 
        random_rot: bool = False, 
        window_size: int = -1, 
        normalization: bool = False, 
        debug: bool = False, 
        use_mmap: bool = False,
        bone: bool = False, 
        vel: bool = False
    ) -> None:
        """
        Initialize the NTU RGB+D dataset feeder.
        
        Args:
            data_path: Path to the NPZ file containing skeleton data
            label_path: Path to label file (optional, currently not used)
            p_interval: Temporal cropping ratio. Single value for center crop,
                       list of two values [min, max] for random crop ratio
            split: Dataset split, either 'train' or 'test'
            random_choose: If True, randomly choose a portion of input sequence
            random_shift: If True, randomly pad zeros at beginning or end of sequence
            random_move: If True, apply random 2D spatial transformations
            random_rot: If True, rotate skeleton around xyz axes
            window_size: Target sequence length. -1 keeps original length
            normalization: If True, normalize input sequences using dataset statistics
            debug: If True, only use first 100 samples for debugging
            use_mmap: If True, use memory mapping to load data (saves RAM)
            bone: If True, compute bone vectors instead of joint positions
            vel: If True, compute velocity vectors (frame differences)
            
        Example:
            >>> # Basic usage
            >>> dataset = DatasetFeeder(
            ...     data_path='ntu_data.npz',
            ...     split='train',
            ...     window_size=64,
            ...     random_rot=True
            ... )
            >>> data, label, index = dataset[0]
            >>> print(data.shape)  # (3, 64, 25, 2)
        """
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        
        # Initialize data attributes
        self.data: np.ndarray
        self.label: np.ndarray 
        self.sample_name: List[str]
        self.mean_map: Optional[np.ndarray] = None
        self.std_map: Optional[np.ndarray] = None
        
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self) -> None:
        """
        Load skeleton data and labels from NPZ file.
        
        Loads the appropriate data split and reshapes from NTU format
        (N, T, V*M*C) to standard format (N, C, T, V, M) where:
        - N: Number of samples
        - C: Number of channels (typically 3 for x,y,z coordinates)  
        - T: Temporal frames
        - V: Number of joints (25 for NTU RGB+D)
        - M: Number of persons (typically 2 for NTU RGB+D)
        
        Raises:
            NotImplementedError: If split is not 'train' or 'test'
            
        Note:
            Labels are converted from one-hot encoding to class indices.
            Sample names are generated as '{split}_{index}'.
        """
        # data: N C V T M
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def get_mean_map(self) -> None:
        """
        Calculate dataset statistics for normalization.
        
        Computes mean and standard deviation across the temporal and person dimensions
        for each channel-joint combination. These statistics can be used for 
        data normalization during training.
        
        Sets:
            mean_map: Mean values with shape (C, 1, V, 1) 
            std_map: Standard deviation values with shape (C, 1, V, 1)
            
        Note:
            Statistics are computed across all samples, temporal frames, and persons
            but maintained separately for each channel and joint.
        """
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        
        Returns:
            Number of samples (sequences) in the dataset
        """
        return len(self.label)

    def __iter__(self) -> Iterator['DatasetFeeder']:
        """
        Return iterator over the dataset.
        
        Returns:
            Iterator object (self)
        """
        return self

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int, int]:
        """
        Get a single sample from the dataset with preprocessing applied.
        
        Applies the following preprocessing pipeline:
        1. Load raw skeleton data for the given index
        2. Determine valid frame count (non-zero frames)
        3. Apply temporal cropping and resizing
        4. Apply random 3D rotation (if enabled)
        5. Compute bone vectors (if enabled) 
        6. Compute velocity vectors (if enabled)
        
        Args:
            index: Sample index to retrieve
            
        Returns:
            Tuple containing:
            - data_numpy: Preprocessed skeleton data with shape (C, T, V, M)
            - label: Action class label (integer)
            - index: Sample index (for tracking)
            
        Example:
            >>> dataset = DatasetFeeder('data.npz', bone=True, vel=True)
            >>> data, label, idx = dataset[0]
            >>> print(f"Data shape: {data.shape}, Label: {label}")
        """
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        
        data_numpy = data_utils.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        
        if self.random_rot:
            data_numpy = data_utils.random_rot(data_numpy)
            
        if self.bone:
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
            
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return data_numpy, label, index

    def top_k(self, score: np.ndarray, top_k: int) -> float:
        """
        Calculate top-k accuracy given prediction scores.
        
        Computes the percentage of samples where the true label appears
        in the top-k highest scoring predictions.
        
        Args:
            score: Prediction scores with shape (N, num_classes) where N is
                  number of samples and num_classes is number of action classes
            top_k: Number of top predictions to consider (e.g., 1 for top-1, 5 for top-5)
            
        Returns:
            Top-k accuracy as a float between 0.0 and 1.0
            
        Example:
            >>> dataset = DatasetFeeder('data.npz', split='test')
            >>> predictions = model.predict(test_data)  # Shape: (N, num_classes)
            >>> top1_acc = dataset.top_k(predictions, top_k=1)
            >>> top5_acc = dataset.top_k(predictions, top_k=5)
            >>> print(f"Top-1: {top1_acc:.3f}, Top-5: {top5_acc:.3f}")
        """
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name: str) -> Any:
    """
    Dynamically import a class or module from a string path.
    
    This utility function enables dynamic importing of classes based on 
    string module paths, which is useful for configuration-driven model
    instantiation and plugin systems.
    
    Args:
        name: Dot-separated module path (e.g., 'torch.nn.Linear', 'models.gcn.GCN')
        
    Returns:
        The imported class or module object
        
    Raises:
        ImportError: If the module cannot be imported
        AttributeError: If the class/attribute doesn't exist in the module
        
    Example:
        >>> # Import a PyTorch class
        >>> LinearLayer = import_class('torch.nn.Linear')
        >>> layer = LinearLayer(128, 64)
        
        >>> # Import custom model class
        >>> ModelClass = import_class('models.my_model.MyGCN')
        >>> model = ModelClass(num_classes=60)
    """
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod