import numpy as np
import os

import gzip
from matplotlib import pyplot as plt


DATA_PATH = 'FashionMNIST/raw'

class FashionMNIST:
  def __init__(self, root : str, train = True):
    super().__init__()
    self.path = os.path.join(root, DATA_PATH)
    self.img_size = 28
    self.train = train
    self.data, self.labels = self._load_data()
    self.classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
  
  def _load_data(self):
    if (self.train):
      with gzip.open(os.path.join(self.path, 'train-images-idx3-ubyte.gz'), 'r') as f:
        f.read(16)
        img_bytes = f.read()
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8).astype(np.float32)
        img_arr = img_arr.reshape(-1, 28, 28, 1)
      with gzip.open(os.path.join(self.path, 'train-labels-idx1-ubyte.gz'), 'r') as f:
        f.read(8)
        label_bytes = f.read()
        labels = np.frombuffer(label_bytes, dtype=np.uint8)
    else:
      with gzip.open(os.path.join(self.path, 't10k-images-idx3-ubyte.gz'), 'r') as f:
        f.read(16)
        img_bytes = f.read()
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8).astype(np.float32)
        img_arr = img_arr.reshape(-1, 28, 28, 1)
      with gzip.open(os.path.join(self.path, 't10k-labels-idx1-ubyte.gz'), 'r') as f:
        f.read(8)
        label_bytes = f.read()
        labels = np.frombuffer(label_bytes, dtype=np.uint8)
    return img_arr, labels
  
  def __len__(self):
    return len(self.data)

    
  def __getitem__(self, idx):
    return self.data[idx], self.labels[idx]
  
  def show_img(self, idx : int):
    data_idx = self[idx]
    img_arr, label = data_idx
    name = self.classes[label]
    plt.imshow(img_arr, cmap='gray')
    plt.title(name)
    plt.show()


