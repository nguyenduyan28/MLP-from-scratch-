import numpy as np
from FashionMNIST import FashionMNIST
import timeit
from multiprocessing import Pool

# For debug print not all
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
  assert isinstance(size, list), "The size must be in list, each element represent for len of each data for splitting"
  assert np.sum(size) == len(ds), "The input length must equal to dataset length"
  start_idx = 0
  list_ds_split = []
  for split_size in size:
    img_list, label_list = ds[start_idx : start_idx + split_size]
    ds_split = Subset(img_list, label_list)
    list_ds_split.append(ds_split)

  return (*list_ds_split,)
    


  
  



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


    






def main():
  train_val_mnist = FashionMNIST(root = '.', train=True)
  train_size = int(0.8 * len(train_val_mnist))
  val_size = int(0.2 * len(train_val_mnist))
  test_mnist = FashionMNIST(root = '.', train = False)


  train_val_ds = Dataset(train_val_mnist)

  train_ds, val_ds = random_split(train_val_ds, [train_size, val_size])

  test_ds = Dataset(test_mnist)


  



  train_loader = DataLoader(train_ds, batch_size=32, num_workers=1)
  test_loader = DataLoader(test_ds, batch_size = 32, num_workers=1)
  val_loader = DataLoader(val_ds, batch_size = 32, num_workers=1)

  t1 = timeit.default_timer()
  for batch in train_loader:
    img_arr, label = batch
    print(batch)
    break
  t2 = timeit.default_timer()
  print(f"{t2 - t1:.2f}")
  
if __name__ == '__main__':
  main()