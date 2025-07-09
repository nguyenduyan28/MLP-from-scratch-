import numpy as np
from FashionMNIST import FashionMNIST
import timeit
from multiprocessing import Pool

np.set_printoptions(threshold=1)

class Dataset:
  def __init__(self, raw_dataset):
    self.data = raw_dataset
  def __len__(self):
    return len(self.data)
  def __getitem__(self, idx):
    img_arr, label = self.data[idx]
    return img_arr, label

class Subset:
  def __init__(self, img_list, label_list):
    self.x = img_list
    self.y = label_list
    self.data = list(zip(self.x, self.y))
  def __len__(self):
    return len(self.data)
  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]

def _init_ds(ds):
  global g_ds 
  g_ds = ds

def get_item(arg):
  idx = arg
  return g_ds[idx]

def random_split(ds, size):
  assert isinstance(size, list)
  assert np.sum(size) == len(ds)
  start_idx = 0
  list_ds_split = []
  for split_size in size:
    img_list, label_list = ds[start_idx : start_idx + split_size]
    ds_split = Subset(img_list, label_list)
    list_ds_split.append(ds_split)
  return (*list_ds_split,)

class DataLoader:
  def __init__(self, ds, batch_size, num_workers=1, shuffle=False):
    self.batch_size = batch_size
    self.ds = ds
    self.index = 0
    self.shuffle = shuffle
    self.num_workers = num_workers

  def __len__(self):
    return (len(self.ds) + self.batch_size - 1) // self.batch_size

  def __iter__(self):
    self.pool = None
    self.index = 0
    if self.shuffle:
      indices = np.arange(len(self.ds))
      np.random.shuffle(indices)
      self.ds.data = [self.ds.data[i] for i in indices]
    if self.num_workers > 1:
      self.pool = Pool(processes=self.num_workers, initializer=_init_ds, initargs=(self.ds,))
    return self

  def __next__(self):
    if self.index >= len(self.ds):
      self.index = 0
      raise StopIteration
    if self.num_workers > 1:
      batch_idx = np.arange(self.index, min(self.index + self.batch_size, len(self.ds)))
      result = self.pool.map(get_item, batch_idx)
      img_arr, labels = zip(*result)
      img_arr = np.stack(img_arr)
      labels = np.array(labels)
    else:
      batch_indices = range(self.index, self.index + self.batch_size)
      end = min(self.index + self.batch_size, len(self.ds))
      batch = [self.ds[i] for i in range(self.index, end)]
      img_arr, labels = zip(*batch)
      img_arr = np.stack(img_arr)
      labels = np.array(labels)
    self.index += self.batch_size
    return img_arr, labels