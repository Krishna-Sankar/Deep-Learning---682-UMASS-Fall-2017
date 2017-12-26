#Modified version of cnn.py to make it extra credit
import numpy as np

from asgn2.layers import *
from asgn2.fast_layers import *
from asgn2.layer_utils import *

lu_pr = False

sanity_relu = True 

prev_mask = {}

def conv_batchnorm_relu_pool_forward(x, w, b, conv_param, pool_param, gamma, beta, bn_param):
  global prev_mask
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  bn, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
  if lu_pr:
    s, relu_cache = leaky_relu_forward(bn)
  else:
    s, relu_cache = relu_forward(bn)
  if sanity_relu:
    mask = relu_cache[1]
    try:
      prev_mask[w.size] = prev_mask[w.size] & mask
      print "% dead in conv:", (np.sum(prev_mask[w.size]) / mask.size)
    except:
      prev_mask[w.size] = mask
      
      
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, bn_cache, relu_cache, pool_cache)
  return out, cache
  
def conv_batchnorm_relu_pool_backward(dout, cache):
  conv_cache, bn_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  if lu_pr:
    da = leaky_relu_backward(ds, relu_cache)
  else:
    da = relu_backward(ds, relu_cache)
  dbn, dgamma, dbeta = spatial_batchnorm_backward(da, bn_cache)
  dx, dw, db = conv_backward_fast(dbn, conv_cache)
  return dx, dw, db, dgamma, dbeta

def affine_relu_forward(x, w, b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db

def affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_param):
  global prev_mask
  a, fc_cache = affine_forward(x, w, b)
  bn, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
  rel, relu_cache = relu_forward(bn)
  cache = (bn_cache, fc_cache, relu_cache)
  return bn, cache
  
def affine_batchnorm_relu_backward(dout, cache):
  bn_cache, fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dl_dbn, dgamma, dbeta = batchnorm_backward(da, bn_cache)
  dl_dout, dl_dW, dl_db = affine_backward(dl_dbn, fc_cache)
  return dl_dout, dl_dW, dl_db, dgamma, dbeta

def affine_batchnorm_relu_dropout_forward(x, w, b, gamma, beta, bn_param, drop_param):
  out1, cache = affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_param)
  out, drop_cache = dropout_forward(out1, drop_param)
  cache += (drop_cache, )
  return out, cache
  
def affine_batchnorm_relu_dropout_backward(dout, cache):
  drop_cache = cache[-1]
  dl_ddrop = dropout_backward(dout, drop_cache)
  return affine_batchnorm_relu_backward(dl_ddrop, cache[:-1])

def affine_relu_dropout_forward(x, w, b, drop_param):
  out1, cache = affine_relu_forward(x, w, b)
  out, drop_cache = dropout_forward(out1, drop_param)
  cache += (drop_cache, )
  return out, cache
  
def affine_relu_dropout_backward(dout, cache):
  drop_cache = cache[-1]
  dl_ddrop = dropout_backward(dout, drop_cache)
  return affine_relu_backward(dl_ddrop, cache[:-1])
  
class ConvNetModel1(object):
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=[32, ], filter_sizes=[7,],
               hidden_dims=[100, ], num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, conv_strides=[1, ], pool_sizes=[2, ], pool_strides=[2, ],
              use_batchnorm=False,
              use_prelu=False,
              dropout=0):
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_dropout = False
    if dropout > 0:
      self.use_dropout = True
      self.dropout_param = {'mode': 'train', 'p':dropout}
    global lu_pr
    if use_prelu:
        lu_pr = True
    assert len(filter_sizes) == len(pool_sizes) == len(pool_strides) == len(num_filters)
    C, H, W = input_dim
    self.num_conv_layers = len(filter_sizes)
    self.num_affine_layers = len(hidden_dims)
    
#   
    self.conv_params = []
    self.pool_params = []
    for i in range(self.num_conv_layers):    
      p3 = (filter_sizes[i] - 1)
      self.conv_params.append({'stride': conv_strides[i], 'p3': p3 / 2})
  
      self.pool_params.append({'pool_height': pool_sizes[i], 'pool_width': pool_sizes[i], 'stride': pool_strides[i]})
    
  
    for i in range(self.num_conv_layers):
      self.params['W' + str(i + 1)] = weight_scale * np.random.randn(num_filters[i], C, filter_sizes[i], filter_sizes[i])
      print 'W' + str(i + 1), self.params['W' + str(i + 1)].shape
      C = num_filters[i]
      
      self.params['b' + str(i + 1)] = np.zeros(num_filters[i])
      if self.use_batchnorm:
          self.params['gamma' + str(i + 1)] = np.ones(num_filters[i])
          self.params['beta' + str(i + 1)] = np.zeros(num_filters[i])
      
  
      w = (W + p3 - filter_sizes[i]) / conv_strides[i] + 1
      h = (H + p3 - filter_sizes[i]) / conv_strides[i] + 1
      wpool = (w - pool_sizes[i]) / pool_strides[i] + 1
      hpool = (h - pool_sizes[i]) / pool_strides[i] + 1
      H, W = (hpool, wpool)
          
        
    inp = num_filters[-1]*wpool*hpool
    for i in range(self.num_affine_layers):
      self.params['W' + str(i + self.num_conv_layers + 1)] = weight_scale * np.random.randn(inp, hidden_dims[i])
      self.params['b' + str(i + self.num_conv_layers + 1)] = np.zeros(hidden_dims[i])
      if self.use_batchnorm:
        self.params['gamma' + str(i + 1 + self.num_conv_layers)] = np.ones(hidden_dims[i])
        self.params['beta' + str(i + 1 + self.num_conv_layers)] = np.zeros(hidden_dims[i])
      inp = hidden_dims[i]
    
    
        
    self.params['W' + str(self.num_affine_layers + self.num_conv_layers + 1)] = weight_scale * np.random.randn(hidden_dims[-1], num_classes)
    
    self.params['b' + str(self.num_affine_layers + self.num_conv_layers + 1)] = np.zeros(num_classes)
    
    
    self.conv_bn = []
    self.affine_bn = []
    if self.use_batchnorm:
      self.conv_bn = [{'mode': 'train'} for i in xrange((self.num_conv_layers))]
      self.affine_bn = [{'mode': 'train'} for i in xrange((self.num_affine_layers))]
      
    
      
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

  
    

    scores = None
    
    conv_outs = [X]
    conv_caches = []
    for i in range(self.num_conv_layers):
      conv_param = self.conv_params[i]
      pool_param = self.pool_params[i]
      if self.use_batchnorm:
        gamma = self.params['gamma' + str(i + 1)]
        beta = self.params['beta' + str(i + 1)]
        bn_param = self.conv_bn[i]
        out, cache = conv_batchnorm_relu_pool_forward(conv_outs[i],
                                                      self.params['W' + str(i + 1)], 
                                                      self.params['b' + str(i + 1)], 
                                                      conv_param, 
                                                      pool_param, 
                                                      gamma, 
                                                      beta, 
                                                      bn_param)
      else:
        out, cache = conv_relu_pool_forward(conv_outs[i],
                                            self.params['W' + str(i + 1)],
                                            self.params['b' + str(i + 1)],
                                            conv_param, 
                                            pool_param)
      conv_outs.append(out)
      conv_caches.append(cache)
      
    rel_outs = [conv_outs[-1]]
    rel_caches = []
    for i in range(self.num_affine_layers):
      if self.use_batchnorm:
        gamma = self.params['gamma' + str(i + 1 + self.num_conv_layers)]
        beta = self.params['beta' + str(i + 1 + self.num_conv_layers)]
        bn_param = self.affine_bn[i]
        if self.use_dropout:
          out, cache = affine_batchnorm_relu_dropout_forward(rel_outs[i], 
                                        self.params['W' + str(i + 1 + self.num_conv_layers)], 
                                        self.params['b' + str(i + 1 + self.num_conv_layers)],
                                        gamma, 
                                        beta, 
                                        bn_param,
                                        self.dropout_param)
        else:
          out, cache = affine_batchnorm_relu_forward(rel_outs[i], 
                                        self.params['W' + str(i + 1 + self.num_conv_layers)], 
                                        self.params['b' + str(i + 1 + self.num_conv_layers)],
                                        gamma, 
                                        beta, 
                                        bn_param)
      elif self.use_dropout:
        out, cache = affine_relu_dropout_forward(rel_outs[i], 
                                      self.params['W' + str(i + 1 + self.num_conv_layers)], 
                                      self.params['b' + str(i + 1 + self.num_conv_layers)], self.dropout_param)
      else:
        out, cache = affine_relu_forward(rel_outs[i], 
                                      self.params['W' + str(i + 1 + self.num_conv_layers)], 
                                      self.params['b' + str(i + 1 + self.num_conv_layers)])
      rel_outs.append(out)
      rel_caches.append(cache)
    
    scores, last_cache = affine_forward(rel_outs[-1],
                                        self.params['W' + str(self.num_conv_layers + self.num_affine_layers + 1)],
                                        self.params['b' + str(self.num_conv_layers + self.num_affine_layers + 1)])
    
    if y is None:
      return scores

    loss, grads = 0, {}
    
    loss, dout = softmax_loss(scores, y)
    
    for i in range(self.num_conv_layers + self.num_affine_layers + 1):
      loss += 0.5*self.reg*np.sum(self.params['W' + str(i + 1)] ** 2)
    
    if loss > 3:
      assert False 
      
    dl_dscores, dl_dW, dl_db = affine_backward(dout, last_cache)
    weight_name = 'W' + str(self.num_affine_layers + self.num_conv_layers + 1)
    grads[weight_name] = dl_dW + self.reg * self.params[weight_name]
    grads['b' + str(self.num_affine_layers + self.num_conv_layers + 1)] = dl_db
    
    dl_dout = dl_dscores
    for i in range(self.num_affine_layers - 1, -1, -1):
      if self.use_batchnorm:
        if self.use_dropout:
          dx, dw, db, dgamma, dbeta = affine_batchnorm_relu_dropout_backward(dl_dout, rel_caches[i])
        else:
          dx, dw, db, dgamma, dbeta = affine_batchnorm_relu_backward(dl_dout, rel_caches[i])
        grads['gamma' + str(i + 1 + self.num_conv_layers)] = dgamma
        grads['beta' + str(i + 1 + self.num_conv_layers)] = dbeta
      elif self.use_dropout:
        dx, dw, db = affine_relu_dropout_backward(dl_dout, rel_caches[i])
      else:
        dx, dw, db = affine_relu_backward(dl_dout, rel_caches[i])
      grads['W' + str(i + 1 + self.num_conv_layers)] = dw + self.reg * self.params['W' + str(i + 1 + self.num_conv_layers)]
      grads['b' + str(i + 1 + self.num_conv_layers)] = db
      dl_dout = dx
    
    dl_dout = dl_dout
    for i in range(self.num_conv_layers - 1, -1, -1):
      if self.use_batchnorm:
        dx, dw, db, dgamma, dbeta = conv_batchnorm_relu_pool_backward(dl_dout, conv_caches[i])
        grads['gamma' + str(i + 1)] = dgamma
        grads['beta' + str(i + 1)] = dbeta
      else:
        dx, dw, db = conv_relu_pool_backward(dl_dout, conv_caches[i])
      grads['W' + str(i + 1)] = dw + self.reg * self.params['W' + str(i + 1)]
      grads['b' + str(i + 1)] = db
      dl_dout = dx
    
    dl_dX = dl_dout
    
    
    return loss, grads