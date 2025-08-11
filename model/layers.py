import os
os.environ['TF_USE_LEGACY_KERAS'] = "1"
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# -----------------------------------------------------------
# Basic blocks: ResidualBlock, ResidualBlockUpsample
# -----------------------------------------------------------
class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, activation=tf.nn.leaky_relu):
        super().__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size, padding='same', use_bias=True)
        self.act = activation
        self.conv2 = layers.Conv2D(filters, kernel_size, padding='same', use_bias=True)
    def call(self, x):
        y = self.conv1(x)
        y = self.act(y)
        y = self.conv2(y)
        return x + y

class ResidualBlockUpsample(layers.Layer):
    def __init__(self, filters, up=2, kernel_size=3, activation=tf.nn.leaky_relu):
        super().__init__()
        self.up = layers.UpSampling2D(size=(up, up), interpolation='nearest')
        self.conv1 = layers.Conv2D(filters, kernel_size, padding='same', use_bias=True)
        self.act = activation
        self.conv2 = layers.Conv2D(filters, kernel_size, padding='same', use_bias=True)
        self.conv_skip = layers.Conv2D(filters, 1, padding='same', use_bias=True)
    def call(self, x):
        s = self.up(x)
        skip = self.conv_skip(s)
        y = self.conv1(s)
        y = self.act(y)
        y = self.conv2(y)
        return skip + y

# -----------------------------------------------------------
# MaskedConv2D for CTXm (mask 'A' for causal raster-scan)
# -----------------------------------------------------------
class MaskedConv2D(layers.Layer):
    def __init__(self, filters, kernel_size=5, mask_type='A', strides=1, padding='same', use_bias=True):
        super().__init__()
        assert kernel_size % 2 == 1
        assert mask_type in ('A', 'B', None)
        self.filters = filters
        self.k = kernel_size
        self.mask_type = mask_type
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
    def build(self, input_shape):
        in_ch = int(input_shape[-1])
        k = self.k
        self.kernel = self.add_weight('kernel', shape=(k,k,in_ch,self.filters), initializer='glorot_uniform', trainable=True)
        if self.use_bias:
            self.bias = self.add_weight('bias', shape=(self.filters,), initializer='zeros', trainable=True)
        else:
            self.bias = None
        if self.mask_type is None:
            self.mask = tf.ones(self.kernel.shape, dtype=tf.float32)
        else:
            mask = np.ones(self.kernel.shape, dtype=np.float32)
            center_h = k//2; center_w = k//2
            mask[center_h, center_w+1:, :, :] = 0.
            mask[center_h+1:, :, :, :] = 0.
            if self.mask_type == 'A':
                mask[center_h, center_w, :, :] = 0.
            self.mask = tf.constant(mask, dtype=tf.float32)
    def call(self, x):
        k = self.kernel * self.mask
        y = tf.nn.conv2d(x, k, strides=[1,self.strides,self.strides,1], padding=self.padding.upper())
        if self.use_bias:
            y = tf.nn.bias_add(y, self.bias)
        return y


# -----------------------------------------------------------
# Entropy Parameter (EP) block: three Conv(3) -> output 2*Lj (means & scales)
# -----------------------------------------------------------
class EntropyParameter(layers.Layer):
    def __init__(self, Lj, hidden=None):
        super().__init__()
        self.Lj = Lj
        if hidden is None:
            hidden = max(4*Lj, 128)
        self.conv0 = layers.Conv2D(hidden, 3, padding='same', activation=tf.nn.leaky_relu)
        self.conv1 = layers.Conv2D(hidden, 3, padding='same', activation=tf.nn.leaky_relu)
        self.conv_out = layers.Conv2D(2*Lj, 3, padding='same', activation=None)
    def call(self, x):
        y = self.conv0(x); y = self.conv1(y); out = self.conv_out(y)
        means, scales_raw = tf.split(out, 2, axis=-1)
        scales = tf.nn.softplus(scales_raw) + 1e-6
        return means, scales