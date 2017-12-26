import numpy as np

from asgn2.layers import *
from asgn2.fast_layers import *
from asgn2.layer_utils import *

class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, conv_stride=1, pool_stride=2, pool_height=2, pool_width=2):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    
    p3 = (filter_size - 1)
    self.conv_param = {'stride': conv_stride, 'p3': p3 / 2}
    
    
    self.pool_param = {'pool_height': pool_height, 'pool_width': pool_width, 'stride': pool_stride}
    
    
    w = (W + p3 - filter_size) / conv_stride + 1
    h = (H + p3 - filter_size) / conv_stride + 1
    
    wpool = (w - pool_width) / pool_stride + 1
    hpool = (h - pool_height) / pool_stride + 1
    
    self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = weight_scale * np.random.randn(num_filters*wpool*hpool, hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    
    conv_param = self.conv_param
    pool_param = self.pool_param
    
    conv_out, conv_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    
    rel_out, rel_cache = affine_relu_forward(conv_out, W2, b2)
    scores, last_cache = affine_forward(rel_out, W3, b3)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dout = softmax_loss(scores, y)
    loss += 0.5*self.reg*(np.sum(W1 ** 2) + np.sum(W2**2) + np.sum(W3 ** 2))
    
    score_l, w3_s, b3_s = affine_backward(dout, last_cache)
    relu_l, w2_s, b2_s = affine_relu_backward(score_l, rel_cache)
    dl_dX, w1_s, b1_s = conv_relu_pool_backward(relu_l, conv_cache)
    
    w1_s += self.reg * W1
    w2_s += self.reg * W2
    w3_s += self.reg * W3
    
    grads['W1'] = w1_s
    grads['W2'] = w2_s
    grads['W3'] = w3_s
    grads['b1'] = b1_s
    grads['b2'] = b2_s
    grads['b3'] = b3_s
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


pass