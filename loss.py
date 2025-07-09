import numpy as np


class CrossEntropyLoss:
  def __init__(self):
    self.softmax = None
    self.y = None

  def forward(self, F, y):
    """
    F: (N, C) – logits
    y: (N,)   – ground truth labels
    """
    F_shifted = F - np.max(F, axis=1, keepdims=True)
    exp_scores = np.exp(F_shifted)
    self.softmax = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # (N, C)
    self.y = y

    log_probs = -np.log(self.softmax[np.arange(F.shape[0]), y] + 1e-9)
    return np.mean(log_probs)

  def backward(self):
    """
    Returns the gradient of loss w.r.t. logits F: ∂L/∂F
    """
    N = self.softmax.shape[0]
    grad = self.softmax.copy()
    grad[np.arange(N), self.y] -= 1
    grad /= N
    return grad

  def __call__(self, F, y):
    return self.forward(F, y)