import numpy as np



class Module:
  def __init__(self):
    self._parameters = []
    self.data_parameters = []
  def __setattr__(self, name, value):
    nn_list  = [Linear, ReLU]
    if isinstance(value, (*nn_list,)):
      self._parameters.append({name: value})
      self.data_parameters.append({name: value})  
    super().__setattr__(name, value)
    
  @property
  def parameters(self):
    lines = [f"Parameters for {type(self).__name__}("] 
    for item in self._parameters:
      for name, value in item.items():
        lines.append(f"  ({name}): {value}")
    lines.append(")")
    return '\n'.join(lines)
  
  def value_parameters(self):
    list_params = []
    for k in self.data_parameters:
      for name, layer in k.items():
        if hasattr(layer, 'params'):
          list_params.append((layer.params, layer.grad_w))
        if hasattr(layer, 'bias'):
          list_params.append((layer.bias, layer.grad_b))
    return list_params

  



  






class Linear:
  def __init__(self, in_features, out_features):
    '''
    X @ W :
    1. in_features --> X.shape[0]
    2. out_features --> user select
      W.shape = in_features + 1, out_features
    we need the xavier init 
    '''
    self.params = np.random.randn(in_features, out_features) * np.sqrt(2 / in_features)
    self.bias = np.zeros((1, out_features))  
    self.grad_w = np.zeros_like(self.params)
    self.grad_b = np.zeros_like(self.bias)
  
  def __repr__(self):
    return f"Linear(in_features={self.params.shape[0]}, out_features = {self.params.shape[1]})"

  def forward(self, X):
    self.X = X
    return (X @ self.params) + self.bias
  
  def __call__(self, X):
    return self.forward(X)
  

  def backward(self, grad_output):
    self.grad_w = self.X.T @ grad_output / self.X.shape[0]  # dL/dW
    self.grad_b = np.mean(grad_output, axis=0, keepdims=True)  # dL/db
    grad_input = grad_output @ self.params.T  # dL/dX

    return grad_input
  

  def add_ones(self, X):
    b = np.ones((X.shape[0], 1))
    return np.concatenate((X, b), axis = 1)


class ReLU:
  def __init__(self):
    pass
  def forward(self, X):
    self.mask = (X > 0)
    return X * self.mask
  
  def backward(self, grad_output):
    return grad_output * self.mask


  def __call__(self, X):
    return self.forward(X)
  def __repr__(self):
    return "ReLU()"


  


