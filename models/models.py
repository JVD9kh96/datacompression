import os
os.environ['TF_USE_LEGACY_KERAS'] = "1"
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from models.layers import _FallbackInverseGDN
from models.utils import expand_medians
# Try import tensorflow_compression
try:
    from tensorflow_compression.entropy_models import\
             ContinuousBatchedEntropyModel as EntropyBottleneck
    _HAS_TFC = True
except Exception:
    _HAS_TFC = False
    from compression import EntropyBottleneck
    # raise RuntimeError("Please install tensorflow-compression (tfc). E.g. pip install tensorflow-compression")

try:
    import tensorflow_compression as tfc
    _HAS_TFC = True
    def make_igdn(name=None):
        # tfc.GDN supports inverse=True to create IGDN
        return tfc.GDN(name=name, inverse=True)
except Exception:
    _HAS_TFC = False
    def make_igdn(name=None):
        return _FallbackInverseGDN()  # use the fallback you provided


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


class SynthesisTransform(Model):
    """
    Synthesis (decoder) transform matching the TF1 reference (no attention).
    - Four up-sampling blocks (i = 0..3)
    - For i < 3:
        - residual sub-block (two convs + residual)
        - shortcut: Conv2D(4*F, 1x1) -> depth_to_space(2)
        - main:       Conv2D(4*F, 3x3) -> depth_to_space(2) -> IGDN(inverse=True)
        - output = shortcut + main
    - For i == 3 (final block):
        - residual sub-block
        - Conv2D(12, 3x3) -> depth_to_space(2) -> (optional sigmoid) => RGB
    """
    def __init__(self, C=192, final_activation=None, name="synthesis_transform"):
        """
        num_filters: F in the above description (paper uses F=num_filters)
        final_activation: None or 'sigmoid' (if you want outputs in [0,1])
        """
        super().__init__(name=name)
        num_filters = C
        self.F = num_filters
        self.final_activation = final_activation

        # residual convs for each block: two conv layers per block (like the reference)
        # We'll create them as lists to keep Keras friendly
        self.res_conv0 = [layers.Conv2D(self.F, 3, padding='same', use_bias=True, name=f"res{i}_c0")
                          for i in range(4)]
        self.res_conv1 = [layers.Conv2D(self.F, 3, padding='same', use_bias=True, name=f"res{i}_c1")
                          for i in range(4)]
        # main up-convs and shortcut convs for first 3 blocks
        self.main_conv_up = [layers.Conv2D(self.F * 4, 3, padding='same', use_bias=True, name=f"main_up_{i}")
                             for i in range(3)]
        self.shortcut_conv = [layers.Conv2D(self.F * 4, 1, padding='same', use_bias=True, name=f"sc_{i}")
                              for i in range(3)]

        # IGDN (inverse) layers for the main path after depth_to_space
        self.igdn = [make_igdn(name=f"igdn_{i}") for i in range(3)]

        # last conv producing 12 channels (-> depth_to_space -> RGB)
        self.last_conv_12 = layers.Conv2D(12, 3, padding='same', use_bias=True, name="last_conv_12")
        # optional final activation
        if final_activation == 'sigmoid':
            self.final_act = layers.Activation('sigmoid')
        else:
            self.final_act = None

        # small leaky relu activation used inside blocks (to mimic TF1)
        self.leaky = tf.nn.leaky_relu

    def call(self, y_hat, training=False):
        """
        y_hat: [B, Hc, Wc, C] the latent (quantized during inference)
        returns: reconstructed image tensor [B, H, W, 3]
        """
        x = y_hat

        for i in range(4):
            # residual sub-block: two convs with leaky relu, then skip-add
            t = self.res_conv0[i](x)
            t = self.leaky(t)
            t = self.res_conv1[i](t)
            x = x + t

            if i < 3:
                # shortcut path (1x1 conv to 4*F channels then depth_to_space)
                sc = self.shortcut_conv[i](x)
                sc = tf.nn.depth_to_space(sc, block_size=2)

                # main path: conv(3x3 -> 4*F) -> depth_to_space -> IGDN(inverse)
                main = self.main_conv_up[i](x)
                main = tf.nn.depth_to_space(main, block_size=2)
                main = self.igdn[i](main)

                x = main + sc
            else:
                # final block: conv -> depth_to_space -> rgb
                main = self.last_conv_12(x)
                main = tf.nn.depth_to_space(main, block_size=2)
                x = main

        if self.final_act is not None:
            x = self.final_act(x)
        return x


# -----------------------------------------------------------
# Hyper networks: hyper-analysis (h_a) and hyper-synthesis (h_s)
# hyper-synthesis outputs H with 2*C channels
# -----------------------------------------------------------
class HyperAnalysis(Model):
    def __init__(self, C=192):
        super().__init__()
        self.conv1 = layers.Conv2D(C, 3, strides=1, padding='same', activation=tf.nn.leaky_relu)
        self.conv2 = layers.Conv2D(C, 3, strides=1, padding='same', activation=tf.nn.leaky_relu)
        self.conv3 = layers.Conv2D(C, 3, strides=2, padding='same', activation=tf.nn.leaky_relu)
        self.conv4 = layers.Conv2D(C, 3, strides=1, padding='same', activation=tf.nn.leaky_relu)
        self.conv5 = layers.Conv2D(C, 3, strides=2, padding='same', activation=None)  # z channels = C
    def call(self, y):
        x = self.conv1(y)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        z = self.conv5(x)
        return z

class HyperSynthesis(Model):
    def __init__(self, C=192):
        super().__init__()
        self.up1 = layers.Conv2DTranspose(C, 3, strides=2, padding='same', activation=tf.nn.leaky_relu)
        self.up2 = layers.Conv2DTranspose(int(C*1.5), 3, strides=2, padding='same', activation=tf.nn.leaky_relu)
        self.conv1 = layers.Conv2D(C, 3, padding='same', activation=tf.nn.leaky_relu)
        self.conv2 = layers.Conv2D(int(C*1.5), 3, padding='same', activation=tf.nn.leaky_relu)
        self.conv_out = layers.Conv2D(2*C, 3, padding='same', activation=None)  # H: 2*C channels
    def call(self, z_hat):
        x = self.conv1(z_hat)
        x = self.up1(x)
        x = self.conv2(x)
        x = self.up2(x)
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
    def __init__(self, out_channels=128, up_factors=(2,1,1,1)):
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
class MultiTaskCodec(tf.keras.Model):
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
        # NOTE: create ctxs once; they will be built upon first call.
        self.ctxs = [MaskedConv2D(filters=2*L, kernel_size=5, mask_type='A') for L in self.slice_sizes]
        self.eps = [EntropyParameter(L) for L in self.slice_sizes]
        # z entropy bottleneck (tfc)
        self.entropy_bottleneck = EntropyBottleneck()

    def call(self, x, training=True):
        # Analysis
        y = self.analysis(x)  # [B, ny, nx, C]

        if training:
            # training-time surrogate: additive uniform noise
            y_tilde = add_uniform_noise(y)
        else:
            # For debugging you used y_tilde = add_uniform_noise(y) — that's OK as a surrogate;
            # below we'll override that with true sequential decoding for exact bpp.
            y_tilde = add_uniform_noise(y)

        # compute z and entropy bottleneck outputs
        z = self.ha(y)
        z_tilde, z_likelihoods = self.entropy_bottleneck(z, training=training)

        # Build z_hat for hyper-synthesis (STE rounding behavior at training time; exact quantization at inference)
        z_medians = self.entropy_bottleneck._get_medians()   # shape (C,)
        z_offset = expand_medians(z_medians, z)              # shape [1,1,1,C] for NHWC
        z_tmp = z - z_offset
        z_hat = z_tmp + tf.stop_gradient(tf.round(z_tmp) - z_tmp) + z_offset

        # H produced by hyper-synthesis (use z_hat as in PyTorch)
        H = self.hs(z_hat)

        # split H into Hj slices (each Hj has 2*Lj channels)
        Hj_list = []
        ch = 0
        for Lj in self.slice_sizes:
            Hj = H[..., 2*ch: 2*(ch + Lj)]
            Hj_list.append(Hj)
            ch += Lj

        # -----------------------------
        # Training-time path (fast, surrogate)
        # -----------------------------
        if training:
            # Use y_tilde for ctx computation and likelihoods exactly as in training loop.
            start = 0
            y_like_slices = []
            means_list = []
            scales_list = []
            for j, Lj in enumerate(self.slice_sizes):
                # Compute ctx from the **full** noisy latent y_tilde — ensures ctxs built with in_ch=C
                ctx = self.ctxs[j](y_tilde)             # shape [B,Hy,Wy,2*Lj] (filters=2*Lj)
                Hj = Hj_list[j]                         # [B,Hy,Wy,2*Lj]
                concat = tf.concat([ctx, Hj], axis=-1)  # [B,Hy,Wy, 4*Lj]
                means, scales = self.eps[j](concat)     # each [B,Hy,Wy,Lj]
                means_list.append(means)
                scales_list.append(scales)

                y_slice = y_tilde[..., start:start+Lj]
                like = discrete_gaussian_likelihood(y_slice, means, scales)
                y_like_slices.append(like)
                start += Lj

            y_likelihoods = tf.concat(y_like_slices, axis=-1)  # [B,Hy,Wy,C]
            x_hat = self.synthesis(y_tilde)

            return {
                'x_hat': x_hat,
                'y': y,
                'y_tilde': y_tilde,
                'y_likelihoods': y_likelihoods,
                'z': z,
                'z_tilde': z_tilde,
                'z_hat': z_hat,
                'z_likelihoods': z_likelihoods,
                'H': H,
                'means_list': means_list,
                'scales_list': scales_list,
                'slice_sizes': self.slice_sizes
            }

        # -----------------------------
        # Inference-time path: sequential per-slice, per-pixel decoding for correct bpp
        # This is correct but slow (pure Python loops). Use for validation / ground-truth bpp.
        # -----------------------------
        # Prepare containers
        B = tf.shape(y)[0]
        Hy = int(y.shape[1])  # small latent spatial size (hopefully static)
        Wy = int(y.shape[2])
        y_hat = tf.zeros_like(y)          # decoded (dequantized) latent, will fill slice-by-slice
        y_likelihood_slices = []
        means_list = []
        scales_list = []

        # Precompute spatial masks for raster order once (numpy -> tf.constant). A mask for each (i,k):
        # mask_pos[i,k] has 1 for positions strictly before (i,k) in row-major ordering, else 0.
        # We'll broadcast that to channels as needed.
        masks = []
        for i in range(Hy):
            for k in range(Wy):
                m = np.zeros((Hy, Wy), dtype=np.float32)
                if i > 0:
                    m[:i, :] = 1.0
                if k > 0:
                    m[i, :k] = 1.0
                masks.append(tf.constant(m[:, :, None], dtype=tf.float32))  # shape (Hy,Wy,1)

        # Sequentially decode each slice
        start = 0
        for j, Lj in enumerate(self.slice_sizes):
            Hj = Hj_list[j]  # [B,Hy,Wy,2*Lj]

            # per-slice temporary storage
            like_slice = tf.zeros((B, Hy, Wy, Lj), dtype=y.dtype)
            means_slice = tf.zeros((B, Hy, Wy, Lj), dtype=y.dtype)
            scales_slice = tf.zeros((B, Hy, Wy, Lj), dtype=y.dtype)

            # We'll fill the slice in raster order
            for idx, mask2d in enumerate(masks):
                # mask2d shape [Hy,Wy,1] with 1's for previously decoded pixels
                # Build partial y_hat where current-slice positions not-yet-decoded are zeroed.
                # Full partial_y has shape [B,Hy,Wy,C]
                # We'll zero only the *current slice positions* that are not decoded yet,
                # while keeping previously-decoded positions intact.
                # Prepare channel-wise mask to apply only to current slice channels.
                mask3c = tf.concat([
                    tf.ones((1,1,1, start), dtype=tf.float32),                         # previous slices keep
                    tf.cast(mask2d[None, ...], dtype=tf.float32),                      # current slice: mask
                    tf.zeros((1,1,1, self.C - start - Lj), dtype=tf.float32)           # future slices zero
                ], axis=-1)  # shape [1,Hy,Wy,C] (broadcastable)

                # apply mask: partial_y = y_hat * mask3c
                partial_y = y_hat * mask3c  # shape [B,Hy,Wy,C]

                # Compute ctx from partial_y using ctxs[j] (ctx conv was built with input_ch=C)
                ctx = self.ctxs[j](partial_y)  # [B,Hy,Wy, 2*Lj]
                concat = tf.concat([ctx, Hj], axis=-1)
                means, scales = self.eps[j](concat)  # [B,Hy,Wy,Lj]

                # extract the position (i,k) being decoded
                i = idx // Wy
                k = idx % Wy

                # take the scalar values at that position across batch and channels
                # y_pixel: [B, Lj], means_pixel, scales_pixel shapes [B,Lj]
                y_pixel = tf.reshape(y[:, i:i+1, k:k+1, start:start+Lj], (B, Lj))
                means_pixel = tf.reshape(means[:, i:i+1, k:k+1, :], (B, Lj))
                scales_pixel = tf.reshape(scales[:, i:i+1, k:k+1, :], (B, Lj))

                # quantize pixel symbols (round(y - mean))
                symbols = tf.round(y_pixel - means_pixel)  # integer symbols [B, Lj]
                # dequantize back to float
                y_hat_pixel = symbols + means_pixel       # [B, Lj]

                # compute discrete Gaussian probability for these pixel values (use vectorized function)
                # reshape to [B,1,1,Lj] so discrete_gaussian_likelihood matches [B,Hy,Wy,Lj]
                y_hat_pixel_reshaped = tf.reshape(y_hat_pixel, (B, 1, 1, Lj))
                means_pixel_reshaped = tf.reshape(means_pixel, (B, 1, 1, Lj))
                scales_pixel_reshaped = tf.reshape(scales_pixel, (B, 1, 1, Lj))

                p_pixel = discrete_gaussian_likelihood(y_hat_pixel_reshaped, means_pixel_reshaped, scales_pixel_reshaped)
                p_pixel = tf.reshape(p_pixel, (B, Lj))  # [B, Lj]

                # write the decoded pixel into y_hat at (i,k) for channels start:start+Lj
                # Build update mask for assignment
                # Create a tensor same shape as y_hat with zeros except the pixel position for current channels
                update = tf.zeros_like(y_hat)
                # assign at spatial pos (i,k) for channels start:start+Lj
                # build slicing: update[:, i:i+1, k:k+1, start:start+Lj] = y_hat_pixel_reshaped
                update = tf.tensor_scatter_nd_update(
                    update,
                    indices=tf.constant([[b, i, k, 0] for b in range(int(B))], dtype=tf.int32),
                    updates=tf.reshape(y_hat_pixel, (-1, Lj))
                )
                # tensor_scatter_nd_update above is a simplistic idea — if it fails due to shape mismatch,
                # fallback to create a full slice assignment via concatenation (slower).
                # For clarity and cross-version robustness we'll instead assign via concatenation:

                # Build per-batch slice and place it into y_hat via slicing & concat (clear and robust)
                # first convert y_hat to numpy? No — do slicing with tf.concat
                left = y_hat[..., :start]
                mid = y_hat[..., start:start+Lj]
                right = y_hat[..., start+Lj:]

                # mid is [B,Hy,Wy,Lj]; replace mid[:, i, k, :] with y_hat_pixel value
                # create a mask for mid
                mid_mask = tf.zeros((B, Hy, Wy, Lj), dtype=mid.dtype)
                mid_mask = tf.tensor_scatter_nd_update(
                    mid_mask,
                    indices=tf.constant([[b, i, k, 0] for b in range(int(B))], dtype=tf.int32),
                    updates=tf.reshape(y_hat_pixel, (-1, Lj))
                )
                # mid_new = mid * (1 - pos_mask) + mid_mask
                pos_mask = tf.zeros((Hy, Wy), dtype=tf.float32)
                pos_mask = tf.tensor_scatter_nd_update(pos_mask, indices=tf.constant([[i,k]]), updates=tf.constant([1.0]))
                pos_mask = pos_mask[None, :, :, None]  # [1,Hy,Wy,1]
                pos_mask = tf.cast(pos_mask, dtype=mid.dtype)
                mid_new = mid * (1.0 - pos_mask) + mid_mask

                y_hat = tf.concat([left, mid_new, right], axis=-1)

                # record probabilities, means, scales for that pixel into full tensors
                # assign to like_slice, means_slice, scales_slice at (i,k)
                like_slice = tf.tensor_scatter_nd_update(
                    like_slice,
                    indices=tf.constant([[b, i, k, 0] for b in range(int(B))], dtype=tf.int32),
                    updates=tf.reshape(p_pixel, (-1, Lj))
                )
                means_slice = tf.tensor_scatter_nd_update(
                    means_slice,
                    indices=tf.constant([[b, i, k, 0] for b in range(int(B))], dtype=tf.int32),
                    updates=tf.reshape(means_pixel, (-1, Lj))
                )
                scales_slice = tf.tensor_scatter_nd_update(
                    scales_slice,
                    indices=tf.constant([[b, i, k, 0] for b in range(int(B))], dtype=tf.int32),
                    updates=tf.reshape(scales_pixel, (-1, Lj))
                )

            # end raster scan for slice j
            y_likelihood_slices.append(like_slice)
            means_list.append(means_slice)
            scales_list.append(scales_slice)
            start += Lj

        # end all slices
        y_likelihoods = tf.concat(y_likelihood_slices, axis=-1)  # [B,Hy,Wy,C]
        x_hat = self.synthesis(y_hat)

        return {
            'x_hat': x_hat,
            'y': y,
            'y_tilde': None,               # not used at inference
            'y_likelihoods': y_likelihoods,
            'z': z,
            'z_tilde': z_tilde,
            'z_hat': z_hat,
            'z_likelihoods': z_likelihoods,
            'H': H,
            'means_list': means_list,
            'scales_list': scales_list,
            'slice_sizes': self.slice_sizes
        }

    


