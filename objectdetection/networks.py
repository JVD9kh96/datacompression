import os
os.environ['TF_USE_LEGACY_KERAS'] = "1"
import tensorflow as tf
import numpy as np
from objectdetection.utils import fixed_padding

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-5
_LEAKY_RELU = 0.1
_ANCHORS = [(10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)]
_MODEL_SIZE = (None, None)

_MINIMAL_YOLO_CHECKPOINT_DIR = "objectdetection/ckeckpoints/minial_yolo.weights.h5"
_YOLO_CHECKPOINT_DIR = "objectdetection/ckeckpoints/yolo.weights.h5"
_BNPLUSLRELU_CHECKPOINT_DIR = "objectdetection/ckeckpoints/V13.weights.h5"


class YoloV3(tf.keras.Model):
    def __init__(self, n_classes=80, model_size=(None, None),
                 max_output_size=10, iou_threshold=0.5,
                 confidence_threshold=0.5, data_format=None, name='yolo_v3_model'):
        super().__init__(name=name)
        if not data_format:
            data_format = 'channels_first' if tf.test.is_built_with_cuda() else 'channels_last'

        self.n_classes = n_classes
        self.model_size = model_size
        self.max_output_size = max_output_size
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.data_format = data_format

        # store created layers and variable order for inspection/debug
        self.layer_map = {}
        self._vars_ordered = []
        self.build((None, _MODEL_SIZE[0], _MODEL_SIZE[1], 3))
        self.load_weights(_YOLO_CHECKPOINT_DIR)

    def leaky(self, inputs):
        return tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    def _apply_conv(self, x, name_prefix, training):
        conv = self.layer_map[name_prefix + '_conv']
        if any(s > 1 for s in conv.strides):
            kernel_size = conv.kernel_size[0] if isinstance(conv.kernel_size, (list, tuple)) else conv.kernel_size
            x = fixed_padding(x, kernel_size, self.data_format)
        x = conv(x)
        bn_key = name_prefix + '_bn'
        if bn_key in self.layer_map:
            bn = self.layer_map[bn_key]
            x = bn(x, training=training)
        x = self.leaky(x)
        return x

    def _is_int(self, x):
        return isinstance(x, int) and x is not None

    def _conv_output_shape(self, input_shape, out_filters, strides):
        if self.data_format == 'channels_first':
            _, C, H, W = input_shape
            H_out = H // strides if self._is_int(H) else None
            W_out = W // strides if self._is_int(W) else None
            return (None, out_filters, H_out, W_out)
        else:
            _, H, W, C = input_shape
            H_out = H // strides if self._is_int(H) else None
            W_out = W // strides if self._is_int(W) else None
            return (None, H_out, W_out, out_filters)

    def _create_conv_bn_build(self, name_prefix, filters, kernel_size, strides=1, use_bias=False, input_shape=None):
        data_format_layer = 'channels_first' if self.data_format == 'channels_first' else 'channels_last'
        padding = 'valid' if strides > 1 else 'same'
        conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                      strides=strides, padding=padding,
                                      use_bias=use_bias, data_format=data_format_layer,
                                      kernel_initializer='glorot_uniform',
                                      name=f'{name_prefix}_conv')
        self.layer_map[f'{name_prefix}_conv'] = conv

        if input_shape is None:
            raise ValueError("input_shape must be provided for building layers.")
        conv.build(input_shape)

        if not use_bias:
            bn = tf.keras.layers.BatchNormalization(axis=1 if self.data_format == 'channels_first' else 3,
                                                    momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
                                                    name=f'{name_prefix}_bn')
            self.layer_map[f'{name_prefix}_bn'] = bn
            out_shape = self._conv_output_shape(input_shape, filters, strides)
            bn.build(out_shape)

            # Store variables for inspection (conv then bn vars)
            if hasattr(conv, 'kernel') and conv.kernel is not None:
                self._vars_ordered.append(conv.kernel)
            if hasattr(bn, 'beta') and bn.beta is not None:
                self._vars_ordered.append(bn.beta)
            if hasattr(bn, 'gamma') and bn.gamma is not None:
                self._vars_ordered.append(bn.gamma)
            if hasattr(bn, 'moving_mean') and bn.moving_mean is not None:
                self._vars_ordered.append(bn.moving_mean)
            if hasattr(bn, 'moving_variance') and bn.moving_variance is not None:
                self._vars_ordered.append(bn.moving_variance)
        else:
            if hasattr(conv, 'kernel') and conv.kernel is not None:
                self._vars_ordered.append(conv.kernel)
            if hasattr(conv, 'bias') and conv.bias is not None:
                self._vars_ordered.append(conv.bias)

        return conv, self._conv_output_shape(input_shape, filters, strides)

    def build(self, input_shape):
        # derive H,W
        if self.data_format == 'channels_first':
            if len(input_shape) == 4 and input_shape[1] != 3 and input_shape[3] == 3:
                H, W = input_shape[1], input_shape[2]
            else:
                H, W = input_shape[2], input_shape[3]
        else:
            H, W = input_shape[1], input_shape[2]

        if self.data_format == 'channels_first':
            cur_shape = (None, 3, H, W)
        else:
            cur_shape = (None, H, W, 3)

        # Darknet53
        _, cur_shape = self._create_conv_bn_build('d_conv0', 32, 3, strides=1, use_bias=False, input_shape=cur_shape)
        _, cur_shape = self._create_conv_bn_build('d_conv1', 64, 3, strides=2, use_bias=False, input_shape=cur_shape)

        _, tmp_shape = self._create_conv_bn_build('d_res1_conv1', 32, 1, strides=1, use_bias=False, input_shape=cur_shape)
        _, cur_shape = self._create_conv_bn_build('d_res1_conv2', 64, 3, strides=1, use_bias=False, input_shape=tmp_shape)

        _, cur_shape = self._create_conv_bn_build('d_conv2', 128, 3, strides=2, use_bias=False, input_shape=cur_shape)
        for i in range(2):
            _, tmp_shape = self._create_conv_bn_build(f'd_res2_{i}_conv1', 64, 1, strides=1, use_bias=False, input_shape=cur_shape)
            _, cur_shape = self._create_conv_bn_build(f'd_res2_{i}_conv2', 128, 3, strides=1, use_bias=False, input_shape=tmp_shape)

        _, cur_shape = self._create_conv_bn_build('d_conv3', 256, 3, strides=2, use_bias=False, input_shape=cur_shape)
        for i in range(8):
            _, tmp_shape = self._create_conv_bn_build(f'd_res3_{i}_conv1', 128, 1, strides=1, use_bias=False, input_shape=cur_shape)
            _, cur_shape = self._create_conv_bn_build(f'd_res3_{i}_conv2', 256, 3, strides=1, use_bias=False, input_shape=tmp_shape)

        route1_shape = cur_shape

        _, cur_shape = self._create_conv_bn_build('d_conv4', 512, 3, strides=2, use_bias=False, input_shape=cur_shape)
        for i in range(8):
            _, tmp_shape = self._create_conv_bn_build(f'd_res4_{i}_conv1', 256, 1, strides=1, use_bias=False, input_shape=cur_shape)
            _, cur_shape = self._create_conv_bn_build(f'd_res4_{i}_conv2', 512, 3, strides=1, use_bias=False, input_shape=tmp_shape)

        route2_shape = cur_shape

        _, cur_shape = self._create_conv_bn_build('d_conv5', 1024, 3, strides=2, use_bias=False, input_shape=cur_shape)
        for i in range(4):
            _, tmp_shape = self._create_conv_bn_build(f'd_res5_{i}_conv1', 512, 1, strides=1, use_bias=False, input_shape=cur_shape)
            _, cur_shape = self._create_conv_bn_build(f'd_res5_{i}_conv2', 1024, 3, strides=1, use_bias=False, input_shape=tmp_shape)

        # YOLO head 1: c1..c6 but we must capture route_shape = output after c5 (before c6)
        prev_shape = cur_shape
        # c1..c5
        _, prev_shape = self._create_conv_bn_build('yolo1_c1', 512, 1, strides=1, use_bias=False, input_shape=prev_shape)
        _, prev_shape = self._create_conv_bn_build('yolo1_c2', 1024, 3, strides=1, use_bias=False, input_shape=prev_shape)
        _, prev_shape = self._create_conv_bn_build('yolo1_c3', 512, 1, strides=1, use_bias=False, input_shape=prev_shape)
        _, prev_shape = self._create_conv_bn_build('yolo1_c4', 1024, 3, strides=1, use_bias=False, input_shape=prev_shape)
        _, route_yolo1_shape = self._create_conv_bn_build('yolo1_c5', 512, 1, strides=1, use_bias=False, input_shape=prev_shape)
        # c6 (final conv of block)
        _, prev_shape = self._create_conv_bn_build('yolo1_c6', 1024, 3, strides=1, use_bias=False, input_shape=route_yolo1_shape)

        n_anchors = len(_ANCHORS[6:9])
        detect_filters = n_anchors * (5 + self.n_classes)
        # detection conv uses prev_shape (output after c6)
        _, detect1_shape = self._create_conv_bn_build('yolo_out1', detect_filters, 1, strides=1, use_bias=True, input_shape=prev_shape)

        # upsample path 1 uses route_yolo1_shape (route), NOT detect1_shape
        _, reduced_shape = self._create_conv_bn_build('yolo_up1_conv', 256, 1, strides=1, use_bias=False, input_shape=route_yolo1_shape)

        # Build yolo2 block: concatenation of upsampled(reduced) with route2 => build c1..c6 on that concatenated shape.
        if self.data_format == 'channels_first':
            route2_C = route2_shape[1]
            concat_channels = 256 + route2_C
            concat_shape = (None, concat_channels, route2_shape[2], route2_shape[3])
        else:
            route2_C = route2_shape[3]
            concat_channels = 256 + route2_C
            concat_shape = (None, route2_shape[1], route2_shape[2], concat_channels)

        prev_shape = concat_shape
        # yolo2 c1..c5, capture route shape
        _, prev_shape = self._create_conv_bn_build('yolo2_c1', 256, 1, strides=1, use_bias=False, input_shape=prev_shape)
        _, prev_shape = self._create_conv_bn_build('yolo2_c2', 512, 3, strides=1, use_bias=False, input_shape=prev_shape)
        _, prev_shape = self._create_conv_bn_build('yolo2_c3', 256, 1, strides=1, use_bias=False, input_shape=prev_shape)
        _, prev_shape = self._create_conv_bn_build('yolo2_c4', 512, 3, strides=1, use_bias=False, input_shape=prev_shape)
        _, route_yolo2_shape = self._create_conv_bn_build('yolo2_c5', 256, 1, strides=1, use_bias=False, input_shape=prev_shape)
        # c6
        _, prev_shape = self._create_conv_bn_build('yolo2_c6', 512, 3, strides=1, use_bias=False, input_shape=route_yolo2_shape)

        n_anchors = len(_ANCHORS[3:6])
        detect_filters = n_anchors * (5 + self.n_classes)
        _, detect2_shape = self._create_conv_bn_build('yolo_out2', detect_filters, 1, strides=1, use_bias=True, input_shape=prev_shape)

        # upsample path 2 uses route_yolo2_shape (route), NOT detect2_shape
        _, reduced_shape = self._create_conv_bn_build('yolo_up2_conv', 128, 1, strides=1, use_bias=False, input_shape=route_yolo2_shape)

        # build yolo3 block: concat reduced with route1
        if self.data_format == 'channels_first':
            route1_C = route1_shape[1]
            concat_channels = 128 + route1_C
            concat_shape = (None, concat_channels, route1_shape[2], route1_shape[3])
        else:
            route1_C = route1_shape[3]
            concat_channels = 128 + route1_C
            concat_shape = (None, route1_shape[1], route1_shape[2], concat_channels)

        prev_shape = concat_shape
        _, prev_shape = self._create_conv_bn_build('yolo3_c1', 128, 1, strides=1, use_bias=False, input_shape=prev_shape)
        _, prev_shape = self._create_conv_bn_build('yolo3_c2', 256, 3, strides=1, use_bias=False, input_shape=prev_shape)
        _, prev_shape = self._create_conv_bn_build('yolo3_c3', 128, 1, strides=1, use_bias=False, input_shape=prev_shape)
        _, prev_shape = self._create_conv_bn_build('yolo3_c4', 256, 3, strides=1, use_bias=False, input_shape=prev_shape)
        _, prev_shape = self._create_conv_bn_build('yolo3_c5', 128, 1, strides=1, use_bias=False, input_shape=prev_shape)
        _, prev_shape = self._create_conv_bn_build('yolo3_c6', 256, 3, strides=1, use_bias=False, input_shape=prev_shape)

        n_anchors = len(_ANCHORS[0:3])
        detect_filters = n_anchors * (5 + self.n_classes)
        _, detect3_shape = self._create_conv_bn_build('yolo_out3', detect_filters, 1, strides=1, use_bias=True, input_shape=prev_shape)

        super().build(input_shape)

    def yolo_layer(self, inputs, n_classes, anchors, img_size, out_name):
        conv = self.layer_map[out_name + '_conv']
        x = inputs
        if any(s > 1 for s in conv.strides):
            k = conv.kernel_size[0]
            x = fixed_padding(x, k, self.data_format)
        x = conv(x)

        if self.data_format == 'channels_first':
            grid_h = tf.shape(x)[2]; grid_w = tf.shape(x)[3]
            x = tf.transpose(x, [0, 2, 3, 1])
        else:
            grid_h = tf.shape(x)[1]; grid_w = tf.shape(x)[2]

        n_anchors = len(anchors)
        x = tf.reshape(x, [-1, n_anchors * grid_h * grid_w, 5 + n_classes])
        strides = (img_size[0] // tf.cast(grid_h, tf.int32), img_size[1] // tf.cast(grid_w, tf.int32))
        box_centers, box_shapes, confidence, classes = tf.split(x, [2, 2, 1, n_classes], axis=-1)

        x_range = tf.range(tf.cast(grid_h, tf.float32), dtype=tf.float32)
        y_range = tf.range(tf.cast(grid_w, tf.float32), dtype=tf.float32)
        x_offset, y_offset = tf.meshgrid(x_range, y_range)
        x_offset = tf.reshape(x_offset, (-1, 1)); y_offset = tf.reshape(y_offset, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        x_y_offset = tf.tile(x_y_offset, [1, n_anchors])
        x_y_offset = tf.reshape(x_y_offset, [1, -1, 2])

        box_centers = tf.nn.sigmoid(box_centers)
        box_centers = (box_centers + x_y_offset) * tf.cast(strides[0], tf.float32)

        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), (1, -1, 2))
        multiples = tf.stack([grid_h * grid_w, 1, 1])
        anchors_tensor = tf.tile(anchors_tensor, multiples)
        anchors_tensor = tf.reshape(anchors_tensor, [1, -1, 2])

        box_shapes = tf.exp(box_shapes) * anchors_tensor
        confidence = tf.sigmoid(confidence)
        classes = tf.sigmoid(classes)
        out = tf.concat([box_centers, box_shapes, confidence, classes], axis=-1)
        return out

    def upsample(self, inputs, out_shape):
        if self.data_format == 'channels_first':
            inputs = tf.transpose(inputs, [0, 2, 3, 1])
            new_h = out_shape[3]; new_w = out_shape[2]
        else:
            new_h = out_shape[1]; new_w = out_shape[2]
        x = tf.image.resize(inputs, (new_h, new_w), method='nearest')
        if self.data_format == 'channels_first':
            x = tf.transpose(x, [0, 3, 1, 2])
        return x

    @staticmethod
    def build_boxes(inputs):
        center_x, center_y, width, height, confidence, classes = tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)
        top_left_x = center_x - width / 2.0
        top_left_y = center_y - height / 2.0
        bottom_right_x = center_x + width / 2.0
        bottom_right_y = center_y + height / 2.0
        boxes = tf.concat([top_left_x, top_left_y, bottom_right_x, bottom_right_y, confidence, classes], axis=-1)
        return boxes

    @staticmethod
    def non_max_suppression_numpy(inputs, n_classes, max_output_size, iou_threshold, confidence_threshold):
        batch = np.array(inputs)
        batch_out = []
        for sample in batch:
            mask = sample[:, 4] > confidence_threshold
            sample = sample[mask]
            boxes_dict = {}
            if sample.shape[0] == 0:
                for cls in range(n_classes):
                    boxes_dict[cls] = np.array([])
                batch_out.append(boxes_dict)
                continue
            classes = np.argmax(sample[:, 5:], axis=-1).astype(np.int32)
            for cls in range(n_classes):
                cls_mask = classes == cls
                cls_boxes = sample[cls_mask]
                if cls_boxes.shape[0] == 0:
                    boxes_dict[cls] = np.array([])
                    continue
                boxes_coords = cls_boxes[:, :4]
                scores = cls_boxes[:, 4]
                selected_idx = tf.image.non_max_suppression(boxes=boxes_coords.astype(np.float32),
                                                            scores=scores.astype(np.float32),
                                                            max_output_size=max_output_size,
                                                            iou_threshold=iou_threshold).numpy()
                selected = cls_boxes[selected_idx][:, :5]
                boxes_dict[cls] = selected
            batch_out.append(boxes_dict)
        return batch_out

    def call(self, inputs, training=False):
        x = tf.cast(inputs, tf.float32)
        if self.data_format == 'channels_first':
            x = tf.transpose(x, [0, 3, 1, 2])
        x = x / 255.0

        x = self._apply_conv(x, 'd_conv0', training)
        x = self._apply_conv(x, 'd_conv1', training)

        shortcut = x
        x = self._apply_conv(x, 'd_res1_conv1', training)
        x = self._apply_conv(x, 'd_res1_conv2', training)
        x = x + shortcut

        x = self._apply_conv(x, 'd_conv2', training)
        for i in range(2):
            shortcut = x
            x = self._apply_conv(x, f'd_res2_{i}_conv1', training)
            x = self._apply_conv(x, f'd_res2_{i}_conv2', training)
            x = x + shortcut

        x = self._apply_conv(x, 'd_conv3', training)
        for i in range(8):
            shortcut = x
            x = self._apply_conv(x, f'd_res3_{i}_conv1', training)
            x = self._apply_conv(x, f'd_res3_{i}_conv2', training)
            x = x + shortcut
        route1 = x

        x = self._apply_conv(x, 'd_conv4', training)
        for i in range(8):
            shortcut = x
            x = self._apply_conv(x, f'd_res4_{i}_conv1', training)
            x = self._apply_conv(x, f'd_res4_{i}_conv2', training)
            x = x + shortcut
        route2 = x

        x = self._apply_conv(x, 'd_conv5', training)
        for i in range(4):
            shortcut = x
            x = self._apply_conv(x, f'd_res5_{i}_conv1', training)
            x = self._apply_conv(x, f'd_res5_{i}_conv2', training)
            x = x + shortcut

        # yolo1 block apply convs
        for idx in ['c1', 'c2', 'c3', 'c4', 'c5', 'c6']:
            x = self._apply_conv(x, f'yolo1_{idx}', training)
            if idx == 'c5':
                route = x  # route is the output after c5
        detect1 = self.yolo_layer(x, n_classes=self.n_classes, anchors=_ANCHORS[6:9], img_size=self.model_size, out_name='yolo_out1')

        # upsample1 (apply conv to route, not detect1 result)
        x = self._apply_conv(route, 'yolo_up1_conv', training)
        upsample_size = tf.shape(route2)
        x = self.upsample(x, out_shape=upsample_size)
        axis = 1 if self.data_format == 'channels_first' else 3
        x = tf.concat([x, route2], axis=axis)
        for idx in ['c1', 'c2', 'c3', 'c4', 'c5', 'c6']:
            x = self._apply_conv(x, f'yolo2_{idx}', training)
            if idx == 'c5':
                route = x
        detect2 = self.yolo_layer(x, n_classes=self.n_classes, anchors=_ANCHORS[3:6], img_size=self.model_size, out_name='yolo_out2')

        # upsample2 (apply conv to route from yolo2)
        x = self._apply_conv(route, 'yolo_up2_conv', training)
        upsample_size = tf.shape(route1)
        x = self.upsample(x, out_shape=upsample_size)
        x = tf.concat([x, route1], axis=axis)
        for idx in ['c1', 'c2', 'c3', 'c4', 'c5', 'c6']:
            x = self._apply_conv(x, f'yolo3_{idx}', training)
        detect3 = self.yolo_layer(x, n_classes=self.n_classes, anchors=_ANCHORS[0:3], img_size=self.model_size, out_name='yolo_out3')

        detections = tf.concat([detect1, detect2, detect3], axis=1)
        boxes = self.build_boxes(detections)
        boxes_np = boxes.numpy()
        boxes_dicts = self.non_max_suppression_numpy(boxes_np, self.n_classes,
                                                     self.max_output_size, self.iou_threshold,
                                                     self.confidence_threshold)
        return boxes_dicts

    def get_ordered_vars(self):
        return self._vars_ordered


class MinimalYoloV3(tf.keras.Model):
    """
    Minimal implementation of the YOLO network. This network aims to output 
    the output of 13th conv layer in the backbone, required for training stage. 
    """
    def __init__(self,  model_size=(None, None), data_format=None, name='minimal_yolo_v3_model'):
        super().__init__(name=name)
        if not data_format:
            data_format = 'channels_first' if tf.test.is_built_with_cuda() else 'channels_last'

        self.model_size = model_size

        self.data_format = data_format

        # store created layers and variable order for inspection/debug
        self.layer_map = {}
        self._vars_ordered = []
        self.build((None, _MODEL_SIZE[0], _MODEL_SIZE[1], 3))
        self.load_weights(_MINIMAL_YOLO_CHECKPOINT_DIR)



    def leaky(self, inputs):
        return tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    def _apply_conv(self, x, name_prefix, training):
        conv = self.layer_map[name_prefix + '_conv']
        if any(s > 1 for s in conv.strides):
            kernel_size = conv.kernel_size[0] if isinstance(conv.kernel_size, (list, tuple)) else conv.kernel_size
            x = fixed_padding(x, kernel_size, self.data_format)
        x = conv(x)
        bn_key = name_prefix + '_bn'
        if bn_key in self.layer_map:
            bn = self.layer_map[bn_key]
            x = bn(x, training=training)
        x = self.leaky(x)
        return x

    def _is_int(self, x):
        return isinstance(x, int) and x is not None

    def _conv_output_shape(self, input_shape, out_filters, strides):
        if self.data_format == 'channels_first':
            _, C, H, W = input_shape
            H_out = H // strides if self._is_int(H) else None
            W_out = W // strides if self._is_int(W) else None
            return (None, out_filters, H_out, W_out)
        else:
            _, H, W, C = input_shape
            H_out = H // strides if self._is_int(H) else None
            W_out = W // strides if self._is_int(W) else None
            return (None, H_out, W_out, out_filters)

    def _create_conv_bn_build(self, name_prefix, filters, kernel_size, strides=1, use_bias=False, input_shape=None):
        data_format_layer = 'channels_first' if self.data_format == 'channels_first' else 'channels_last'
        padding = 'valid' if strides > 1 else 'same'
        conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                      strides=strides, padding=padding,
                                      use_bias=use_bias, data_format=data_format_layer,
                                      kernel_initializer='glorot_uniform',
                                      name=f'{name_prefix}_conv')
        self.layer_map[f'{name_prefix}_conv'] = conv

        if input_shape is None:
            raise ValueError("input_shape must be provided for building layers.")
        conv.build(input_shape)

        if not use_bias:
            bn = tf.keras.layers.BatchNormalization(axis=1 if self.data_format == 'channels_first' else 3,
                                                    momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
                                                    name=f'{name_prefix}_bn')
            self.layer_map[f'{name_prefix}_bn'] = bn
            out_shape = self._conv_output_shape(input_shape, filters, strides)
            bn.build(out_shape)

            # Store variables for inspection (conv then bn vars)
            if hasattr(conv, 'kernel') and conv.kernel is not None:
                self._vars_ordered.append(conv.kernel)
            if hasattr(bn, 'beta') and bn.beta is not None:
                self._vars_ordered.append(bn.beta)
            if hasattr(bn, 'gamma') and bn.gamma is not None:
                self._vars_ordered.append(bn.gamma)
            if hasattr(bn, 'moving_mean') and bn.moving_mean is not None:
                self._vars_ordered.append(bn.moving_mean)
            if hasattr(bn, 'moving_variance') and bn.moving_variance is not None:
                self._vars_ordered.append(bn.moving_variance)
        else:
            if hasattr(conv, 'kernel') and conv.kernel is not None:
                self._vars_ordered.append(conv.kernel)
            if hasattr(conv, 'bias') and conv.bias is not None:
                self._vars_ordered.append(conv.bias)

        return conv, self._conv_output_shape(input_shape, filters, strides)

    def build(self, input_shape):
        # derive H,W
        if self.data_format == 'channels_first':
            if len(input_shape) == 4 and input_shape[1] != 3 and input_shape[3] == 3:
                H, W = input_shape[1], input_shape[2]
            else:
                H, W = input_shape[2], input_shape[3]
        else:
            H, W = input_shape[1], input_shape[2]

        if self.data_format == 'channels_first':
            cur_shape = (None, 3, H, W)
        else:
            cur_shape = (None, H, W, 3)

        # Darknet53
        _, cur_shape = self._create_conv_bn_build('d_conv0', 32, 3, strides=1, use_bias=False, input_shape=cur_shape)
        _, cur_shape = self._create_conv_bn_build('d_conv1', 64, 3, strides=2, use_bias=False, input_shape=cur_shape)

        _, tmp_shape = self._create_conv_bn_build('d_res1_conv1', 32, 1, strides=1, use_bias=False, input_shape=cur_shape)
        _, cur_shape = self._create_conv_bn_build('d_res1_conv2', 64, 3, strides=1, use_bias=False, input_shape=tmp_shape)

        _, cur_shape = self._create_conv_bn_build('d_conv2', 128, 3, strides=2, use_bias=False, input_shape=cur_shape)
        for i in range(2):
            _, tmp_shape = self._create_conv_bn_build(f'd_res2_{i}_conv1', 64, 1, strides=1, use_bias=False, input_shape=cur_shape)
            _, cur_shape = self._create_conv_bn_build(f'd_res2_{i}_conv2', 128, 3, strides=1, use_bias=False, input_shape=tmp_shape)

        _, cur_shape = self._create_conv_bn_build('d_conv3', 256, 3, strides=2, use_bias=False, input_shape=cur_shape)
        
        _, tmp_shape = self._create_conv_bn_build(f'd_res3_0_conv1', 128, 1, strides=1, use_bias=False, input_shape=cur_shape)
        _, cur_shape = self._create_conv_bn_build(f'd_res3_0_conv2', 256, 3, strides=1, use_bias=False, input_shape=tmp_shape)

        _, tmp_shape = self._create_conv_bn_build(f'd_res3_1_conv1', 128, 1, strides=1, use_bias=False, input_shape=cur_shape)
        
        route1_shape = cur_shape

        super().build(input_shape)


    def upsample(self, inputs, out_shape):
        if self.data_format == 'channels_first':
            inputs = tf.transpose(inputs, [0, 2, 3, 1])
            new_h = out_shape[3]; new_w = out_shape[2]
        else:
            new_h = out_shape[1]; new_w = out_shape[2]
        x = tf.image.resize(inputs, (new_h, new_w), method='nearest')
        if self.data_format == 'channels_first':
            x = tf.transpose(x, [0, 3, 1, 2])
        return x


    def call(self, inputs, training=False):
        x = tf.cast(inputs, tf.float32)
        if self.data_format == 'channels_first':
            x = tf.transpose(x, [0, 3, 1, 2])
        x = x / 255.0

        x = self._apply_conv(x, 'd_conv0', training)
        x = self._apply_conv(x, 'd_conv1', training)

        shortcut = x
        x = self._apply_conv(x, 'd_res1_conv1', training)
        x = self._apply_conv(x, 'd_res1_conv2', training)
        x = x + shortcut

        x = self._apply_conv(x, 'd_conv2', training)
        
        shortcut = x
        x = self._apply_conv(x, f'd_res2_0_conv1', training)
        x = self._apply_conv(x, f'd_res2_0_conv2', training)
        x = x + shortcut

        shortcut = x
        x = self._apply_conv(x, f'd_res2_1_conv1', training)
        x = self._apply_conv(x, f'd_res2_1_conv2', training)
        x = x + shortcut

        x = self._apply_conv(x, 'd_conv3', training)
        shortcut = x
        x = self._apply_conv(x, f'd_res3_0_conv1', training)
        x = self._apply_conv(x, f'd_res3_0_conv2', training)
        x = x + shortcut
        shortcut = x
        x = self._apply_conv(x, f'd_res3_1_conv1', training)
        
        return x

    def get_ordered_vars(self):
        return self._vars_ordered



class BNPlusLReLU(tf.keras.Model):
    """This model aims to apply bachtnormalization and leaky_relu based on 
       the 13th conv layer batchnormalization and leaky relu for distortion loss 
       calculation.
    """
    def __init__(self, *args, **kwargs):
        super(BNPlusLReLU, self).__init__(**kwargs)
        self.build(input_shape=[None, None, None, 128])
        

    def build(self, input_shape):
        self.norm_act = tf.keras.models.Sequential([tf.keras.layers.BatchNormalization(input_shape=(None, None, 128),
                                                                                         momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON),
                                  tf.keras.layers.Lambda(lambda x: tf.nn.leaky_relu(x, alpha=_LEAKY_RELU))
                                 ])
        self.norm_act.load_weights(_BNPLUSLRELU_CHECKPOINT_DIR)
    def call(self, x, training=False):
        return self.norm_act(x, training=training)

