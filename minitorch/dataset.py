import os
import struct
import numpy as np
import urllib.request
import gzip

def load_mnist(dataset="training", path="."):
    """
    Downloads and loads the MNIST dataset.
    Returns:
        X: (N, 784) numpy array of images
        Y: (N,) numpy array of labels
    """
    files = {
        'training': ('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz'),
        'testing': ('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')
    }
    
    os.makedirs(path, exist_ok=True)
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    
    images_file, labels_file = files[dataset]
    images_path = os.path.join(path, images_file)
    labels_path = os.path.join(path, labels_file)
    
    if not os.path.exists(images_path):
        print(f"Downloading {images_file}...")
        urllib.request.urlretrieve(base_url + images_file, images_path)
        
    if not os.path.exists(labels_path):
        print(f"Downloading {labels_file}...")
        urllib.request.urlretrieve(base_url + labels_file, labels_path)
        
    with gzip.open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)

    with gzip.open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 784)

    return images, labels
