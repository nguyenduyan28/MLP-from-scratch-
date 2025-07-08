import numpy as np


class CrossEntropyLoss:
  """
  Computes the average cross-entropy loss over a batch of examples.

  Parameters:
  -----------
  F : np.ndarray, shape (N, C)
    The raw score (logit) matrix where N is the number of examples
    and C is the number of classes. Each row corresponds to the unnormalized
    class scores for a single input example.
      
  y : np.ndarray, shape (N,)
    Ground truth labels for each example in the batch. Each entry is an integer
    representing the correct class index (0 <= y[i] < C).

  Returns:
  --------
  float
    The average cross-entropy loss over the batch.

  Notes:
  ------
  - Applies the numerical stability trick by shifting logits before computing softmax.
  - The loss is computed as:
    L_i = -f_{y_i} + log(sum_j exp(f_j))
  - The final returned loss is the average across all examples.
  """
  def __init__(self):
    pass

  def forward(self, F, y):
    F = F - np.max(F, axis = 1, keepdims=True)
    correct_class_score = F[np.arange(F.shape[0]), y]
    log_sum_other = np.log(np.sum(np.exp(F), axis=1, keepdims=True))
    return (-1 / F.shape[0]) * np.sum(correct_class_score[:, np.newaxis] - log_sum_other) 
  
  def __call__(self, F, y):
    return self.forward(F, y)
