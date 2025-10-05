from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.X, self.y = parse_mnist(image_filename, label_filename)
        self.transforms = transforms
        self.X = self.X.reshape(-1, 28, 28, 1)  # Reshape to (N, H, W, C)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        image = self.X[index]
        label = self.y[index]
        if self.transforms:
            for transform in self.transforms:
                image = transform(image)
        return image, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.X)
        ### END YOUR SOLUTION

def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    # Read image file
    with gzip.open(image_filename, 'rb') as f:
        # Read header
        magic_num = struct.unpack('>I', f.read(4))[0]  # MSB first (big endian)
        num_images = struct.unpack('>I', f.read(4))[0]
        num_rows = struct.unpack('>I', f.read(4))[0]
        num_cols = struct.unpack('>I', f.read(4))[0]
        
        # Read pixel data
        pixels = np.frombuffer(f.read(), dtype=np.uint8)
        # Reshape to (num_images, num_rows * num_cols) and normalize to [0, 1]
        X = pixels.reshape(num_images, num_rows * num_cols).astype(np.float32) / 255.0
    
    # Read label file
    with gzip.open(label_filename, 'rb') as f:
        # Read header
        magic_num = struct.unpack('>I', f.read(4))[0]  # MSB first (big endian)
        num_labels = struct.unpack('>I', f.read(4))[0]
        
        # Read label data
        y = np.frombuffer(f.read(), dtype=np.uint8)
    
    return X, y
    ### END YOUR CODE