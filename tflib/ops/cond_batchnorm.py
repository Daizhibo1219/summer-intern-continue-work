import tflib as lib

import numpy as np
import tensorflow as tf
#加入load_offset,load_scale,load_mark
def Batchnorm(name, axes, inputs,load_offset,load_scale,load_mark=False, is_training=None, stats_iter=None,
              update_moving_stats=True, fused=True, labels=None, n_labels=None):
    """conditional batchnorm (dumoulin et al 2016) for BCHW conv filtermaps"""
    if axes != [0,2,3]:
        raise Exception('unsupported')
    mean, var = tf.nn.moments(inputs, axes, keep_dims=True)
    shape = mean.get_shape().as_list() # shape is [1,n,1,1]
    if load_mark:
        offset_m = lib.param(name + '.offset', load_offset)
        scale_m = lib.param(name + '.scale', load_scale)
    else:
        offset_m = lib.param(name+'.offset', np.zeros([n_labels,shape[1]], dtype='float32'))
        scale_m = lib.param(name+'.scale', np.ones([n_labels,shape[1]], dtype='float32'))
    offset = tf.nn.embedding_lookup(offset_m, labels)
    scale = tf.nn.embedding_lookup(scale_m, labels)
    result = tf.nn.batch_normalization(inputs, mean, var, offset[:,:,None,None], scale[:,:,None,None], 1e-5)
    return result