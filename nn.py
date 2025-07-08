import numpy as np



class Module:
  _parameters = []
  data_parameters = []
  def __init__(self):
    pass
  def __setattr__(self, name, value):
    nn_list  = [Linear, ReLU]
    if isinstance(value, (*nn_list,)):
      self._parameters.append({name : value})
      if hasattr(value, 'params') and hasattr(value, 'bias'):
        self.data_parameters.append({name : (value.params, value.bias)} )
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
      for name, value in k.items():
        params, bias = value
        list_params.extend((params, bias))
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
    self.params = np.random.randn(in_features, out_features) / np.sqrt(1 / (in_features))
    self.bias = np.random.randn(1 , out_features) / np.sqrt(1 / in_features)
  
  def __repr__(self):
    return f"Linear(in_features={self.params.shape[0]}, out_features = {self.params.shape[1]})"

  def forward(self, X):
    return (X @ self.params) + self.bias
  
  def __call__(self, X):
    return self.forward(X)
  

  def add_ones(self, X):
    b = np.ones((X.shape[0], 1))
    return np.concatenate((X, b), axis = 1)


class ReLU:
  def __init__(self):
    pass
  def forward(self, X):
    self.params = np.array(X > 0)
    return X * self.params


  def __call__(self, X):
    return self.forward(X)
  def __repr__(self):
    return "ReLU()"


  


