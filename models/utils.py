import os
os.environ['TF_USE_LEGACY_KERAS'] = "1"
import tensorflow as tf 
import math
import scipy.special as _scipyerf  # used in numpy-form pmf (encoder side)
import numpy as np


# -----------------------------------------------------------
# Quantization surrogate and discrete gaussian likelihood (for training)
# -----------------------------------------------------------
def add_uniform_noise(x):
    return x + tf.random.uniform(tf.shape(x), -0.5, 0.5)

def ste_round(x):
    return tf.round(x)  # in forward pass when using quantized inference

def expand_medians(medians, x):
    """Reshape medians (shape [C] or [channels, ...]) to broadcast to tensor x.
    Works for typical 4D NHWC tensors (B,H,W,C) and also for other ranks if statically known.
    """
    # medians: tensor shape (C,)
    rank = x.shape.rank
    if rank is None:
        # fallback to 4D NHWC if dynamic rank unavailable
        rank = 4
    # produce shape [1, 1, ..., C] with (rank-1) leading ones
    target_shape = [1] * (rank - 1) + [medians.shape[0]]
    return tf.reshape(medians, target_shape)

def discrete_gaussian_likelihood(x, means, scales, eps=1e-12):
    sqrt2 = math.sqrt(2.0)
    upper = 0.5 * (1.0 + tf.math.erf((x + 0.5 - means) / (scales * sqrt2 + eps)))
    lower = 0.5 * (1.0 + tf.math.erf((x - 0.5 - means) / (scales * sqrt2 + eps)))
    p = upper - lower
    p = tf.clip_by_value(p, 1e-12, 1.0)
    return p

def bits_from_likelihoods(likelihoods):
    nats = -tf.reduce_sum(tf.math.log(likelihoods + 1e-12))
    bits = nats / tf.math.log(2.0)
    return bits

def build_pmf_numpy(mu, sigma, sample_range):
    """
    mu, sigma: scalars or 1D arrays (for mixture you'd weight them)
    sample_range: numpy array of integer symbol centers, e.g. np.arange(0, 2*minmax+1)
    returns pmf (numpy array) over sample_range
    """
    # pmf_k = Phi((k+0.5 - mu)/sigma) - Phi((k-0.5 - mu)/sigma)
    sqrt2 = math.sqrt(2.0)
    upper = 0.5 * (1.0 + _scipyerf.erf((sample_range + 0.5 - mu) / (sigma * sqrt2)))
    lower = 0.5 * (1.0 + _scipyerf.erf((sample_range - 0.5 - mu) / (sigma * sqrt2)))
    pmf = upper - lower
    pmf = np.clip(pmf, 1e-12, None)
    pmf = pmf / pmf.sum()
    return pmf