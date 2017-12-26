import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  loss = 0.0
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for a in xrange(num_train):
    prob_yi = X[a].dot(W) 
    prob_yi =prob_yi- np.max(prob_yi) 
    prob_yi = np.exp(prob_yi)
    norm=np.sum(prob_yi)
    prob_yi /= norm 
    temp_loss=np.log(prob_yi[y[a]])
    loss += -temp_loss 
    temp_grad = prob_yi
    temp_grad[y[a]]=temp_grad[y[a]]-1
    for b in range(num_classes):
        dW[:, b] += temp_grad[b] * X[a].transpose()
  loss /= num_train
  loss += 0.5*reg*np.sum(W * W)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  num_train = X.shape[0]
  score = X.dot(W)
  score=score-np.max(score, axis=1).reshape(score.shape[0], 1)
  score = np.exp(score)
  score=score/np.sum(score, axis=1).reshape(score.shape[0], 1)
  loss_1=(-1.0 / num_train)
  loss_2=np.sum(np.log(score[np.arange(num_train), y])) + 0.5*reg*np.sum(W * W)
  loss = loss_1*loss_2
  score[np.arange(num_train), y] -= 1
  dW = X.transpose().dot(score) * (1.0 / num_train) + reg * W

  return loss, dW

