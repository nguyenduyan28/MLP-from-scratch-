
from nn import *
from tqdm import tqdm
from data import Dataset, DataLoader, random_split
from FashionMNIST import FashionMNIST
from loss import CrossEntropyLoss

class MLP(Module):
  def __init__(self, image_size, num_of_classes):
    super().__init__()
    self.fc1 = Linear(image_size, 512) # 28 * 28
    self.a1 = ReLU()
    self.fc2 = Linear(512, 256) # 28 * 28
    self.a2 = ReLU()
    self.fc3 = Linear(256, 128)
    self.a3 = ReLU()
    self.fc4 = Linear(128, num_of_classes)
  

  
  def forward(self, x : np.array ):
    x = x.reshape(x.shape[0], -1)
    x = self.fc1(x)
    x = self.a1(x)
    x = self.fc2(x)
    x = self.a2(x)
    x = self.fc3(x)
    x = self.a3(x)
    x = self.fc4(x)
  
    return x
  
  def __call__(self, X):
    return self.forward(X)

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




def main():
  img_size = 28 * 28
  model = MLP(image_size=img_size, num_of_classes=10)
  loss_fn = CrossEntropyLoss()
  lr = 0.01
  num_epochs = 10

  for epoch in range(num_epochs):
    total_loss = 0
    total_correct = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    for img_arr, label in loop:
      output = model(img_arr)
      loss = loss_fn(output, label)

      # Backward pass
      grad = loss_fn.backward()
      grad = model.fc4.backward(grad)
      grad = model.a3.backward(grad)
      grad = model.fc3.backward(grad)
      grad = model.a2.backward(grad)
      grad = model.fc2.backward(grad)
      grad = model.a1.backward(grad)
      grad = model.fc1.backward(grad)

      #SGD
      for param, g in model.value_parameters():
        param -= lr * g
      total_loss += loss
      preds = np.argmax(output, axis=1)
      total_correct += np.sum(preds == label)
      acc = total_correct / ((loop.n + 1) * train_loader.batch_size)
      loop.set_postfix({"loss": loss, "acc": f"{acc:.4f}"})

    val_loss = 0
    val_correct = 0
    val_total = 0
    vloop = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
    for img_arr, label in vloop:
      output = model(img_arr)
      loss = loss_fn(output, label)
      val_loss += loss
      preds = np.argmax(output, axis=1)
      val_correct += np.sum(preds == label)
      val_total += len(label)
    val_acc = val_correct / val_total
    print(f"Validation - Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_acc:.4f}")

  test_loss = 0
  test_correct = 0
  test_total = 0
  tloop = tqdm(test_loader, desc="Test")
  for img_arr, label in tloop:
    output = model(img_arr)
    loss = loss_fn(output, label)
    test_loss += loss
    preds = np.argmax(output, axis=1)
    test_correct += np.sum(preds == label)
    test_total += len(label)
  test_acc = test_correct / test_total
  print(f"Test - Loss: {test_loss/len(test_loader):.4f}, Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
  main()
