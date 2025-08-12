import os 
os.environ['TF_USE_LEGACY_KERAS'] = "1"
import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np


class EntropyModel(Layer):
  """Base class (minimal)"""

  def __init__(self, tail_mass=2 ** -8, likelihood_bound=1e-9,
               range_coder_precision=16, **kwargs):
    super().__init__(**kwargs)
    self._tail_mass = float(tail_mass)
    if not 0 < self.tail_mass < 1:
      raise ValueError("`tail_mass` must be between 0 and 1, got {}.".format(self.tail_mass))
    self._likelihood_bound = float(likelihood_bound)
    self._range_coder_precision = int(range_coder_precision)

  @property
  def tail_mass(self):
    return self._tail_mass

  @property
  def likelihood_bound(self):
    return self._likelihood_bound

  @property
  def range_coder_precision(self):
    return self._range_coder_precision

  def _quantize(self, inputs, mode):
    raise NotImplementedError

  def _dequantize(self, inputs, mode):
    raise NotImplementedError

  def _likelihood(self, inputs):
    raise NotImplementedError

  def call(self, inputs, training=False):
    inputs = tf.convert_to_tensor(inputs, dtype=self.dtype or tf.float32)
    if inputs.dtype.is_integer:
      raise ValueError("{} can't take integer inputs.".format(type(self).__name__))

    outputs = self._quantize(inputs, "noise" if training else "dequantize")
    assert outputs.dtype == inputs.dtype
    likelihood = self._likelihood(outputs)
    if self.likelihood_bound > 0:
      likelihood_bound = tf.constant(self.likelihood_bound, dtype=self.dtype or tf.float32)
      likelihood = tf.maximum(likelihood, likelihood_bound)
    return outputs, likelihood


class EntropyBottleneck(EntropyModel):
  """TF2-eager-compatible EntropyBottleneck for training-time use.

  Implements identical math to tensorflow_compression. It computes:
   - auxiliary loss (quantiles),
   - medians for quantization,
   - discrete likelihoods for the quantized/perturbed values.

  NOTE: compress()/decompress() (binary coder) are NOT implemented here.
  """

  def __init__(self, init_scale=10, filters=(3, 3, 3),
               data_format="channels_last", **kwargs):
    super().__init__(**kwargs)
    self._init_scale = float(init_scale)
    self._filters = tuple(int(f) for f in filters)
    self._data_format = str(data_format)
    self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)

    if self.data_format not in ("channels_first", "channels_last"):
      raise ValueError("Unknown data format: '{}'.".format(self.data_format))

  @property
  def init_scale(self):
    return self._init_scale

  @property
  def filters(self):
    return self._filters

  @property
  def data_format(self):
    return self._data_format

  def _channel_axis(self, ndim):
    return {"channels_first": 1, "channels_last": ndim - 1}[self.data_format]

  def _get_input_dims(self):
    ndim = self.input_spec.ndim
    channel_axis = self._channel_axis(ndim)
    channels = self.input_spec.axes[channel_axis]
    # Tuple of slices to expand a vector across input dims (vector runs across channels)
    input_slices = ndim * [None]
    input_slices[channel_axis] = slice(None)
    return ndim, channel_axis, channels, tuple(input_slices)

  def _logits_cumulative(self, inputs, stop_gradient):
    # inputs: shape (channels, filters_prev, N) (the original used (channels,1,batch))
    logits = inputs
    for i in range(len(self.filters) + 1):
      matrix = self._matrices[i]
      if stop_gradient:
        matrix = tf.stop_gradient(matrix)
      # matrix: (channels, filters[i+1], filters[i])
      # logits: (channels, filters[i], N)
      logits = tf.linalg.matmul(matrix, logits)

      bias = self._biases[i]
      if stop_gradient:
        bias = tf.stop_gradient(bias)
      logits = logits + bias  # broadcasting on last dim

      if i < len(self._factors):
        factor = self._factors[i]
        if stop_gradient:
          factor = tf.stop_gradient(factor)
        logits = logits + factor * tf.math.tanh(logits)
    return logits

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    channel_axis = self._channel_axis(input_shape.ndims)
    channels = input_shape.as_list()[channel_axis]
    if channels is None:
      raise ValueError("The channel dimension of the inputs must be defined.")
    self.input_spec = tf.keras.layers.InputSpec(
        ndim=input_shape.ndims, axes={channel_axis: channels})
    filters = (1,) + self.filters + (1,)
    scale = self.init_scale ** (1 / (len(self.filters) + 1))

    # Create variables: matrices, biases, factors
    self._matrices = []
    self._biases = []
    self._factors = []
    for i in range(len(self.filters) + 1):
      # initializer constant same as tfc
      init = np.log(np.expm1(1 / scale / filters[i + 1]))
      mat_shape = (channels, filters[i + 1], filters[i])
      matrix = self.add_weight(
          name=f"matrix_{i}",
          shape=mat_shape,
          dtype=self.dtype or tf.float32,
          initializer=tf.initializers.Constant(init),
      )
      # apply softplus as in original
      matrix = tf.nn.softplus(matrix)
      self._matrices.append(matrix)

      bias = self.add_weight(
          name=f"bias_{i}",
          shape=(channels, filters[i + 1], 1),
          dtype=self.dtype or tf.float32,
          initializer=tf.keras.initializers.RandomUniform(-.5, .5),
      )
      self._biases.append(bias)

      if i < len(self.filters):
        factor = self.add_weight(
            name=f"factor_{i}",
            shape=(channels, filters[i + 1], 1),
            dtype=self.dtype or tf.float32,
            initializer=tf.keras.initializers.Zeros(),
        )
        factor = tf.math.tanh(factor)
        self._factors.append(factor)

    # Quantiles variable and auxiliary loss (same targets as original)
    target = np.log(2 / self.tail_mass - 1)
    target = tf.constant([-target, 0.0, target], dtype=self.dtype or tf.float32)

    def quantiles_initializer(shape, dtype=None):
      assert tuple(shape[1:]) == (1, 3)
      init = tf.constant([[[-self.init_scale, 0.0, self.init_scale]]], dtype=dtype)
      return tf.tile(init, (shape[0], 1, 1))

    quantiles = self.add_weight(
        name="quantiles",
        shape=(channels, 1, 3),
        dtype=self.dtype or tf.float32,
        initializer=quantiles_initializer,
    )
    logits = self._logits_cumulative(quantiles, stop_gradient=True)
    loss = tf.reduce_sum(tf.abs(logits - target))
    self.add_loss(loss)

    # medians for quantize/dequantize
    medians = quantiles[:, 0, 1]
    self._medians = tf.stop_gradient(medians)

    # minima/maxima for PMF sampling ranges (kept for parity / internal bookkeeping)
    minima = medians - quantiles[:, 0, 0]
    minima = tf.cast(tf.math.ceil(minima), tf.int32)
    minima = tf.math.maximum(minima, 0)
    maxima = quantiles[:, 0, 2] - medians
    maxima = tf.cast(tf.math.ceil(maxima), tf.int32)
    maxima = tf.math.maximum(maxima, 0)

    # offsets / pmf start / pmf length (not used for training likelihoods directly here,
    # but computed to stay close to original implementation)
    self._offset = -minima
    pmf_start = medians - tf.cast(minima, self.dtype or tf.float32)
    pmf_length = maxima + minima + 1

    # sample densities for bookkeeping (not stored as CDF var since no range coder here)
    max_length = tf.reduce_max(pmf_length)
    samples = tf.range(tf.cast(max_length, self.dtype or tf.float32), dtype=self.dtype or tf.float32)
    samples = samples[None, :] + pmf_start[:, None]

    half = tf.constant(0.5, dtype=self.dtype or tf.float32)
    lower = self._logits_cumulative(samples - half, stop_gradient=True)
    upper = self._logits_cumulative(samples + half, stop_gradient=True)
    sign = -tf.math.sign(tf.math.add_n([lower, upper]))
    pmf = tf.abs(tf.math.sigmoid(sign * upper) - tf.math.sigmoid(sign * lower))
    pmf = pmf[:, 0, :]

    # tail mass (kept to match semantics)
    tail_mass = (tf.math.sigmoid(lower[:, 0:1, :1]) if False else
                 tf.math.sigmoid(lower[:, 0, :1]))  # keep shape consistent
    # (we don't need to use tail_mass further in this TF2 training-only variant)

    super().build(input_shape)

  def _quantize(self, inputs, mode):
    half = tf.constant(0.5, dtype=self.dtype or tf.float32)
    _, _, _, input_slices = self._get_input_dims()

    if mode == "noise":
      noise = tf.random.uniform(tf.shape(inputs), -half, half, dtype=self.dtype or tf.float32)
      return inputs + noise

    medians = self._medians[input_slices]
    outputs = tf.floor(inputs + (half - medians))

    if mode == "dequantize":
      outputs = tf.cast(outputs, self.dtype or tf.float32)
      return outputs + medians
    else:
      assert mode == "symbols", mode
      outputs = tf.cast(outputs, tf.int32)
      return outputs

  def _dequantize(self, inputs, mode="dequantize"):
    _, _, _, input_slices = self._get_input_dims()
    medians = self._medians[input_slices]
    outputs = tf.cast(inputs, self.dtype or tf.float32)
    return outputs + medians

  def _likelihood(self, inputs):
    ndim, channel_axis, _, _ = self._get_input_dims()
    half = tf.constant(0.5, dtype=self.dtype or tf.float32)

    # Move channel axis to front and collapse others => (channels, 1, -1)
    order = list(range(ndim))
    order.pop(channel_axis)
    order.insert(0, channel_axis)
    x = tf.transpose(inputs, perm=order)
    shape = tf.shape(x)
    x = tf.reshape(x, (shape[0], 1, -1))  # (channels, 1, batch_flat)

    lower = self._logits_cumulative(x - half, stop_gradient=False)
    upper = self._logits_cumulative(x + half, stop_gradient=False)
    sign = -tf.math.sign(tf.math.add_n([lower, upper]))
    sign = tf.stop_gradient(sign)
    likelihood = tf.abs(tf.math.sigmoid(sign * upper) - tf.math.sigmoid(sign * lower))

    # Reshape back to original
    order_back = list(range(1, ndim))
    order_back.insert(channel_axis, 0)
    likelihood = tf.reshape(likelihood, shape)
    likelihood = tf.transpose(likelihood, perm=order_back)
    return likelihood

  # compress/decompress are intentionally omitted for training-only use.
  def compress(self, inputs):
    raise NotImplementedError("Compression (range coder) not implemented in pure-TF training-only EntropyBottleneck.")

  def decompress(self, strings, **kwargs):
    raise NotImplementedError("Decompression (range coder) not implemented in pure-TF training-only EntropyBottleneck.")
