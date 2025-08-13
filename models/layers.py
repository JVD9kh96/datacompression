import os
os.environ['TF_USE_LEGACY_KERAS'] = "1"
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, initializers
try:
    import tensorflow_compression as tfc
    TF_COMPRESSION_AVAILABLE = True
except Exception:
    TF_COMPRESSION_AVAILABLE = False

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



# fallback simple (approximate) IGDN if tfc not available (not ideal)
class _FallbackInverseGDN(layers.Layer):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.built = False
    def build(self, input_shape):
        ch = int(input_shape[-1])
        # simple per-channel params for fallback
        self.beta = self.add_weight('beta', shape=(ch,), initializer=tf.keras.initializers.Constant(1.0), trainable=True)
        self.gamma = self.add_weight('gamma', shape=(ch,), initializer=tf.keras.initializers.Zeros(), trainable=True)
        self.built = True
    def call(self, x):
        # diagonal approximate IGDN: x * sqrt(beta + gamma * x^2)
        # Note: this is a crude diagonal simplification of the full GDN matrix form.
        sq = tf.square(x)
        # broadcast channel params over spatial dims
        val = tf.sqrt(tf.nn.relu(self.beta + self.gamma * sq) + self.eps)
        return x * val

class ResidualBlockUpsample(layers.Layer):
    """
    RB with upsample as in the paper:
      main branch: upsample -> Conv(3) -> LeakyReLU -> Conv(3) -> IGDN
      skip branch:  upsample -> Conv(3)
      output = skip + main_branch_out
    """
    def __init__(self, filters, up=2, kernel_size=3, activation=tf.nn.leaky_relu, use_bias=True, name=None):
        super().__init__(name=name)
        self.up_factor = up
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.use_bias = use_bias

        # Up-sampling layer (nearest as in earlier code)
        self.upsample = layers.UpSampling2D(size=(up, up), interpolation='nearest')

        # main path convs
        self.conv1 = layers.Conv2D(filters, kernel_size, padding='same', use_bias=use_bias)
        self.conv2 = layers.Conv2D(filters, kernel_size, padding='same', use_bias=use_bias)

        # skip conv (paper shows Conv(3) on skip)
        self.conv_skip = layers.Conv2D(filters, kernel_size, padding='same', use_bias=use_bias)

        # activation (LeakyReLU)
        self.act = layers.Activation(self.activation)

        # IGDN layer (use tfc if available, else fallback)
        if TF_COMPRESSION_AVAILABLE:
            # tensorflow_compression exposes a GDN layer. Use inverse=True for IGDN
            try:
                self.igdn = tfc.layers.GDN(inverse=True)   # TF compression API
            except Exception:
                # Older tfc API sometimes had tfc.GDN(..., inverse=True)
                try:
                    self.igdn = tfc.GDN(inverse=True)
                except Exception:
                    # fallback to our approximate IGDN
                    self.igdn = _FallbackInverseGDN()
        else:
            self.igdn = _FallbackInverseGDN()

    def call(self, x):
        # upsample input once
        s = self.upsample(x)                 # spatially upsampled input

        # main branch
        y = self.conv1(s)
        y = self.act(y)
        y = self.conv2(y)
        y = self.igdn(y)                     # inverse GDN applied here

        # skip branch
        skip = self.conv_skip(s)

        # residual add
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
    def __init__(self, Lj, hidden=None, scale_bias_init=-4.0):
        super().__init__()
        self.Lj = Lj
        if hidden is None:
            hidden = max(4*Lj, 128)

        self.conv0 = layers.Conv2D(
            hidden, 3, padding='same',
            activation=tf.nn.leaky_relu,
            kernel_initializer=initializers.HeNormal()
        )
        self.conv1 = layers.Conv2D(
            hidden, 3, padding='same',
            activation=tf.nn.leaky_relu,
            kernel_initializer=initializers.HeNormal()
        )

        def bias_init(shape, dtype=None):
            b = tf.zeros(shape, dtype=dtype or tf.float32)
            if len(shape) == 1 and shape[0] == 2*self.Lj:
                b = tf.concat(
                    [tf.zeros((self.Lj,), dtype=b.dtype),
                     tf.fill((self.Lj,), tf.constant(scale_bias_init, dtype=b.dtype))],
                    axis=0
                )
            return b

        self.conv_out = layers.Conv2D(
            2 * Lj, 3, padding='same', activation=None,
            kernel_initializer=initializers.RandomNormal(0.0, 1e-3),
            bias_initializer=bias_init
        )

    def call(self, x):
        y = self.conv0(x)
        y = self.conv1(y)
        out = self.conv_out(y)
        means, scales_raw = tf.split(out, 2, axis=-1)

        # dtype-safe epsilon
        eps = tf.constant(1e-6, dtype=means.dtype)
        scales = tf.nn.softplus(scales_raw) + eps

        # Practical clamp to avoid near-delta distributions
        scales = tf.clip_by_value(scales, clip_value_min=1e-3, clip_value_max=1e3)

        return means, scales
