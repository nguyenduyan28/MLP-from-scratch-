import torch 
from torchvision.datasets import FashionMNIST

from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms.transforms import ToTensor, ToPILImage
from torchvision.transforms.functional import to_pil_image
from torch.optim import SGD, Adam
from tqdm import tqdm
from torch import nn


def show_img(img_arr):
  img =  to_pil_image(img_arr) 
  img.show()



class Fashion_MNIST_Dataset(Dataset):
  def __init__(self, train = True, download = True):
    self.data = FashionMNIST(root = '.', train = train, download = download, transform=ToTensor())
  def __len__(self):
    return len(self.data)
  def __getitem__(self, idx):
    img_arr, label = self.data[idx]
    return img_arr, label


train_val_ds = Fashion_MNIST_Dataset(train=True, download=False) # C, H, W

test_ds = Fashion_MNIST_Dataset(train=False, download=True)
train_len, val_len = int(len(train_val_ds) * 0.8), int(len(train_val_ds) * 0.2)
train_ds, val_ds =  random_split(train_val_ds, [train_len, val_len])

train_loader = DataLoader(train_ds, batch_size = 32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size = 32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size = 32, shuffle=True)


img_size = int(*train_ds[0][0].flatten().size())
num_of_classes = len(test_ds.data.classes)
print(img_size, num_of_classes)

class MLP(nn.Module):
  def __init__(self, image_size, num_of_classes):
    super().__init__()
    self.fc1 = nn.Linear(image_size, 512) # 28 * 28
    self.a1 = nn.ReLU()
    self.fc2 = nn.Linear(512, 256) # 28 * 28
    self.a2 = nn.ReLU()
    self.fc3 = nn.Linear(256, num_of_classes)
  
  def forward(self, x : torch.Tensor):
    x = x.reshape(x.shape[0], -1)
    x = self.fc1(x)
    x = self.a1(x)
    x = self.fc2(x)
    x = self.a2(x)
    x = self.fc3(x)
    return x

model = MLP(image_size = img_size, num_of_classes= num_of_classes)


loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(params=model.parameters(), lr = 0.01)
num_epoch = 20





def train():
  loop = tqdm(train_loader, desc = 'Training')
  for idx, (img, label) in (enumerate(loop)):
    output = model(img)
    loss = loss_fn(output, label)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (idx % 100 == 0):
      loop.set_postfix({"batch" : idx,"loss" : loss.item()})


def validation(idx):
  total_loss = 0
  size = len(val_loader.dataset)
  num_batches = len(val_loader)
  acc = 0
  with torch.no_grad():
    loop = tqdm(val_loader, desc = 'Validation')
    for img, label in loop:
      output = model(img)
      loss = loss_fn(output, label)
      acc += (torch.argmax(output, dim = 1) == label).type(torch.float32).sum()
      total_loss += loss.item()
  print(f"Epoch {idx + 1}, Total loss is:  {total_loss / num_batches:2f}, Accuracy is : {acc / size :2f}")
  

def test():
  total_loss = 0
  size = len(test_loader.dataset)
  num_batches = len(test_loader)
  acc = 0
  with torch.no_grad():
    loop = tqdm(test_loader, desc = 'Validation')
    for img, label in loop:
      output = model(img)
      loss = loss_fn(output, label)
      acc += (torch.argmax(output, dim = 1) == label).type(torch.float32).sum()
      total_loss += loss.item()
  print(f"Total loss is:  {total_loss / num_batches:2f}, Accuracy is : {acc / size :2f}")
  

# for idx in (range(num_epoch)):
#   train()
#   validation(idx)
# test()


print(model.parameters)

