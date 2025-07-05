import numpy as np
from FashionMNIST import FashionMNIST
import timeit
from multiprocessing import Pool


class Dataset:
  def __init__(self, raw_dataset):
    self.data = raw_dataset
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    img_arr, label = self.data[idx]
    return img_arr, label


def get_item(arg):
  idx = arg
  return g_ds[idx]


def _init_ds(ds):
  global g_ds 
  g_ds = ds
  



class DataLoader:
  def __init__(self, ds, batch_size, num_workers = 1, shuffle = False):
    self.batch_size = batch_size
    self.ds = ds
    self.index = 0
    self.shuffle = shuffle
    self.num_workers = num_workers
  
  
  def __len__(self):
    return len(self.ds) 
  
  def __iter__(self):
    # DEBUG 
    self.pool = None
    if self.num_workers > 1:
      self.pool = Pool(processes = self.num_workers, initializer=_init_ds, initargs=(self.ds,))
      return self
    else : return self

  
  def __next__(self):
    #DEBUG
    if (self.num_workers > 1):
      '''
      Loader will divide the batch for each worker => Each worker will has batch_size / num_worker ds to load, make it faster (NOT NOW)
      '''
      if (self.index >= len(self.ds)):
        self.index = 0
        raise StopIteration
      batch_idx = np.arange(self.index, min(self.index + self.batch_size, len(self.ds)))
      # print(batch_idx)
      num_split = self.batch_size / self.num_workers
      batch_idx = np.split(batch_idx, num_split)
      self.index += min(self.batch_size, len(self.ds) - self.index)
      result = self.pool.map(get_item, batch_idx)
      img_arr, labels = zip(*result)
      img_arr = np.concatenate(img_arr)
      labels = np.concatenate(labels)
      # print(f"Current self index is : {self.index}")
      # print(f"Current batch_size is : {img_arr.shape}")
      return img_arr, labels
    else :
      
      if (self.index >= len(self.ds)):
        self.index = 0
        raise StopIteration
      if (self.index + self.batch_size >= len(self.ds)):
        remain_value = len(self) - self.index
        batch = self.ds[self.index : len(self)]
        self.index += remain_value
      else: 
        batch = self.ds[self.index : self.index + self.batch_size]
        self.index += self.batch_size
      return batch

  # def __iter__(self):
  #   self.pool = None
  #   if (self.num_workers > 1):
  #     self.pool = Pool(processes = self.num_workers, initializer=_init_ds, initargs=(self.ds,))
  #   return self

  # def __next__(self):
  #   if (self.index >= len(self.ds)):
  #     self.index = 0
  #     raise StopIteration
  #   batch_idx = np.arange(self.index, min(self.index + self.batch_size, len(self.ds)))
  #   print(batch_idx)
  #   num_split = self.batch_size / self.num_workers
  #   batch_idx = np.split(batch_idx, num_split)
  #   self.index += min(self.batch_size, len(self.ds) - self.index)
  #   result = self.pool.map(get_item, batch_idx)
  #   img_arr, labels = zip(*result)
  #   img_arr = np.concatenate(img_arr)
  #   labels = np.concatenate(labels)
  #   print(f"Current self index is : {self.index}")
  #   print(f"Current batch_size is : {img_arr.shape}")
  #   return img_arr, labels



    






def main():
  ds = Dataset(FashionMNIST(root = '.', train = True))
  train_loader = DataLoader(ds, batch_size=32, num_workers=1)
  t1 = timeit.default_timer()
  list_img = []
  for batch in train_loader:
    img_arr, label = batch
    print(label, label.shape)

  t2 = timeit.default_timer()
  print(f"{t2 - t1:.2f}")
  
if __name__ == '__main__':
  main()