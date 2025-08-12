import os
os.environ['TF_USE_LEGACY_KERAS'] = "1"
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

# Try import tensorflow_compression
try:
    from tensorflow_compression.entropy_models import\
             ContinuousBatchedEntropyModel as EntropyBottleneck
    _HAS_TFC = True
except Exception:
    _HAS_TFC = False
    from compression import EntropyBottleneck
    raise RuntimeError("Please install tensorflow-compression (tfc). E.g. pip install tensorflow-compression")

from models.layers import ResidualBlock, ResidualBlockUpsample,\
                         MaskedConv2D, EntropyParameter

from models.utils import add_uniform_noise, ste_round,\
                        discrete_gaussian_likelihood,\
                        bits_from_likelihoods



# -----------------------------------------------------------
# Analysis (g_a) and Synthesis (g_s) transforms (backbone)
# - Ensure spatial downsample by factor 16 approx (4 blocks x2)
# - Output latent channels C (paper uses C=192)
# -----------------------------------------------------------
class AnalysisTransform(Model):
    def __init__(self, C=192):
        super().__init__()
        self.conv0 = layers.Conv2D(128, 5, strides=2, padding='same', activation=tf.nn.leaky_relu)  # /2
        self.rb1 = ResidualBlock(128)
        self.down1 = layers.Conv2D(192, 3, strides=2, padding='same', activation=tf.nn.leaky_relu)    # /4
        self.rb2 = ResidualBlock(192)
        self.down2 = layers.Conv2D(192, 3, strides=2, padding='same', activation=tf.nn.leaky_relu)    # /8
        self.rb3 = ResidualBlock(192)
        self.down3 = layers.Conv2D(192, 3, strides=2, padding='same', activation=tf.nn.leaky_relu)    # /16
        self.conv_out = layers.Conv2D(C, 3, strides=1, padding='same', activation=None)
    def call(self, x):
        x = self.conv0(x)
        x = self.rb1(x)
        x = self.down1(x); x = self.rb2(x)
        x = self.down2(x); x = self.rb3(x)
        x = self.down3(x)
        y = self.conv_out(x)
        return y

class SynthesisTransform(Model):
    def __init__(self, C=192):
        super().__init__()
        self.up1 = ResidualBlockUpsample(192, up=2)
        self.up2 = ResidualBlockUpsample(192, up=2)
        self.up3 = ResidualBlockUpsample(128, up=2)
        # last up to match original resolution (we had /16 -> 3 ups = /2*2*2= /16)
        self.conv_out = layers.Conv2D(3, 5, padding='same', activation='sigmoid')
    def call(self, y_hat):
        x = self.up1(y_hat)
        x = self.up2(x)
        x = self.up3(x)
        x = self.conv_out(x)
        return x

# -----------------------------------------------------------
# Hyper networks: hyper-analysis (h_a) and hyper-synthesis (h_s)
# hyper-synthesis outputs H with 2*C channels
# -----------------------------------------------------------
class HyperAnalysis(Model):
    def __init__(self, C=192):
        super().__init__()
        self.conv1 = layers.Conv2D(C, 3, strides=1, padding='same', activation=tf.nn.leaky_relu)
        self.conv2 = layers.Conv2D(C, 3, strides=2, padding='same', activation=tf.nn.leaky_relu)
        self.conv3 = layers.Conv2D(C, 3, strides=1, padding='same', activation=tf.nn.leaky_relu)
        self.conv4 = layers.Conv2D(C, 3, strides=2, padding='same', activation=None)  # z channels = C
    def call(self, y):
        x = self.conv1(y)
        x = self.conv2(x)
        x = self.conv3(x)
        z = self.conv4(x)
        return z

class HyperSynthesis(Model):
    def __init__(self, C=192):
        super().__init__()
        self.up1 = layers.Conv2DTranspose(C, 3, strides=2, padding='same', activation=tf.nn.leaky_relu)
        self.conv1 = layers.Conv2D(C, 3, padding='same', activation=tf.nn.leaky_relu)
        self.conv_out = layers.Conv2D(2*C, 3, padding='same', activation=None)  # H: 2*C channels
    def call(self, z_hat):
        x = self.up1(z_hat)
        x = self.conv1(x)
        H = self.conv_out(x)
        return H



# -----------------------------------------------------------
# LST (latent-space transform) - map base sub-latent Y1 -> detector feature F~^(l)
# Paper used residual blocks + RB with upsample; for YOLO conv13 target:
# - If Y has spatial size N x M and base Y1 is N x M x 128,
# - They map to 2N x 2M x 256 for YOLO conv13 (so upsample x2 + conv to 256)
# We'll implement LST with up_factors = [2,1,1,1] with final out_channels=256
# -----------------------------------------------------------
class LatentSpaceTransform(Model):
    def __init__(self, out_channels=256, up_factors=(2,1,1,1)):
        super().__init__()
        self.blocks = []
        for up in up_factors:
            if up > 1:
                self.blocks.append(ResidualBlockUpsample(out_channels, up=up))
            else:
                self.blocks.append(ResidualBlock(out_channels))
        self.conv_out = layers.Conv2D(out_channels, 3, padding='same', activation=None)
    def call(self, y_sub):
        x = y_sub
        for b in self.blocks:
            x = b(x)
        x = self.conv_out(x)
        return x


# -----------------------------------------------------------
# The MultiTaskCodec model wiring everything together (training forward)
# - splits y into Y1 (base) and Y2 (enhancement)
# - computes CTX_j = masked_conv(y_slice), Hj from H, calls EPj -> means/scales
# - calculates discrete-likelihoods per-slice and total R
# - returns x_hat reconstruction as well
# -----------------------------------------------------------
class MultiTaskCodec(Model):
    def __init__(self, C=192, base_channels=128):
        super().__init__()
        self.C = C
        self.base_channels = base_channels
        self.analysis = AnalysisTransform(C=C)
        self.synthesis = SynthesisTransform(C=C)
        self.ha = HyperAnalysis(C=C)
        self.hs = HyperSynthesis(C=C)
        # sub-latent split: base L1, rest L2
        self.slice_sizes = [base_channels, C - base_channels]
        self.ctxs = [MaskedConv2D(filters=2*L, kernel_size=5, mask_type='A') for L in self.slice_sizes]
        self.eps = [EntropyParameter(L) for L in self.slice_sizes]
        # z entropy bottleneck (tfc)
        self.entropy_bottleneck = EntropyBottleneck()
    def call(self, x, training=True):
        # Analysis
        y = self.analysis(x)  # [B, ny, nx, C]
        if training:
            y_tilde = add_uniform_noise(y)
        else:
            y_tilde = ste_round(y)
        # Hyper
        z = self.ha(y)
        if training:
            z_tilde = add_uniform_noise(z)
        else:
            z_tilde = ste_round(z)
        # Hyper compression: training-time we still compute z likelihood via entropy_bottleneck
        if training:
            z_likelihoods = None
        # Hyper synthesis -> H (2*C channels)
        H = self.hs(z_tilde)
        # split H into Hj for each sub-latent (each Hj has 2*Lj channels per paper)
        Hj_list = []
        ch = 0
        for Lj in self.slice_sizes:
            Hj = H[..., 2*ch: 2*(ch+Lj)]
            Hj_list.append(Hj)
            ch += Lj
        # compute CTX and EP per slice
        start = 0
        y_like_slices = []
        means_list = []
        scales_list = []
        for j, Lj in enumerate(self.slice_sizes):
            y_slice = y_tilde[..., start:start+Lj]  # B,H,W,Lj
            ctx = self.ctxs[j](y_slice)             # B,H,W,2*Lj
            Hj = Hj_list[j]                         # B,H,W,2*Lj
            concat = tf.concat([ctx, Hj], axis=-1)
            means, scales = self.eps[j](concat)     # each B,H,W,Lj
            means_list.append(means); scales_list.append(scales)
            like = discrete_gaussian_likelihood(y_slice, means, scales)
            y_like_slices.append(like)
            start += Lj
        y_likelihoods = tf.concat(y_like_slices, axis=-1)
        # z likelihood via tfc (training: use entropy_bottleneck...)
        # We compute z_likelihoods for rate term using tfc API (call with training flag)
        _, z_likelihoods = self.entropy_bottleneck(z, training=training)
        # Reconstruct x_hat
        x_hat = self.synthesis(y_tilde if training else ste_round(y))
        return {
            'x_hat': x_hat, 'y': y, 'y_tilde': y_tilde,
            'z': z, 'z_tilde': z_tilde, 'H': H,
            'y_likelihoods': y_likelihoods, 'z_likelihoods': z_likelihoods,
            'means_list': means_list, 'scales_list': scales_list,
            'slice_sizes': self.slice_sizes
        }
