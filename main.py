
from nn import *
from tqdm import tqdm
from data import Dataset, DataLoader, random_split
from FashionMNIST import FashionMNIST

class MLP(Module):
  def __init__(self, image_size, num_of_classes):
    super().__init__()
    self.fc1 = Linear(image_size, 512) # 28 * 28
    self.a1 = ReLU()
    self.fc2 = Linear(512, 256) # 28 * 28
    self.a2 = ReLU()
    self.fc3 = Linear(256, num_of_classes)
  

  
  def forward(self, x : np.array ):
    x = x.reshape(x.shape[0], -1)
    x = self.fc1(x)
    x = self.a1(x)
    x = self.fc2(x)
    x = self.a2(x)
    x = self.fc3(x)
    return x
  
  def __call__(self, X):
    return self.forward(X)






# loss_fn = nn.CrossEntropyLoss()
# optimizer = Adam(params=model.parameters(), lr = 0.01)
num_epoch = 20

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




# def train():
#   loop = tqdm(train_loader, desc = 'Training')
#   for idx, (img, label) in (enumerate(loop)):
#     output = model(img)
    # loss = loss_fn(output, label)
    # loss.backward()
    # optimizer.step()
    # optimizer.zero_grad()
    # if (idx % 100 == 0):
    #   loop.set_postfix({"batch" : idx,"loss" : loss.item()})


# def validation(idx):
#   total_loss = 0
#   size = len(val_loader.dataset)
#   num_batches = len(val_loader)
#   acc = 0
#   loop = tqdm(val_loader, desc = 'Validation')
#   for img, label in loop:
#     output = model(img)
#     loss = loss_fn(output, label)
#     acc += (torch.argmax(output, dim = 1) == label).type(torch.float32).sum()
#     total_loss += loss.item()
#   print(f"Epoch {idx + 1}, Total loss is:  {total_loss / num_batches:2f}, Accuracy is : {acc / size :2f}")
  

# def test():
#   total_loss = 0
#   size = len(test_loader.dataset)
#   num_batches = len(test_loader)
#   acc = 0
#   with torch.no_grad():
#     loop = tqdm(test_loader, desc = 'Validation')
#     for img, label in loop:
#       output = model(img)
#       loss = loss_fn(output, label)
#       acc += (torch.argmax(output, dim = 1) == label).type(torch.float32).sum()
#       total_loss += loss.item()
#   print(f"Total loss is:  {total_loss / num_batches:2f}, Accuracy is : {acc / size :2f}")
  

# for idx in (range(num_epoch)):
#   train()
#   validation(idx)
# test()


def main():
  img_size = 28 * 28
  model = MLP(image_size = img_size, num_of_classes=10)
  for batch in train_loader:
    img_arr, label = batch 
    output = model(img_arr)
    print(output.shape)
    break


if __name__ == '__main__':
  main()
