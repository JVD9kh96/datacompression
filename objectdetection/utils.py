import os
os.environ['TF_USE_LEGACY_KERAS'] = "1"
import tensorflow as tf 
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display
from seaborn import color_palette


def fixed_padding(inputs, kernel_size, data_format='channels_last'):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    if data_format == 'channels_first':
        paddings = [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]]
    else:
        paddings = [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]
    return tf.pad(inputs, paddings)




def load_darknet_weights_by_layer_order(model, weights_file: str, verbose: bool = True):
    with open(weights_file, 'rb') as f:
        _ = np.fromfile(f, dtype=np.int32, count=5)
        weights = np.fromfile(f, dtype=np.float32)
    total = weights.size
    ptr = 0

    def _read(n):
        nonlocal ptr
        if ptr + n > total:
            raise ValueError(f"Attempt to read {n} floats but only {total - ptr} remain")
        vals = weights[ptr:ptr + n]
        ptr += n
        return vals

    # conv order = same as build()
    conv_names = []
    conv_names += ['d_conv0', 'd_conv1']
    conv_names += ['d_res1_conv1', 'd_res1_conv2']
    conv_names += ['d_conv2']
    for i in range(2):
        conv_names += [f'd_res2_{i}_conv1', f'd_res2_{i}_conv2']
    conv_names += ['d_conv3']
    for i in range(8):
        conv_names += [f'd_res3_{i}_conv1', f'd_res3_{i}_conv2']
    conv_names += ['d_conv4']
    for i in range(8):
        conv_names += [f'd_res4_{i}_conv1', f'd_res4_{i}_conv2']
    conv_names += ['d_conv5']
    for i in range(4):
        conv_names += [f'd_res5_{i}_conv1', f'd_res5_{i}_conv2']
    conv_names += [f'yolo1_c{i}' for i in ['1','2','3','4','5','6']]
    conv_names += ['yolo_out1']
    conv_names += ['yolo_up1_conv']
    conv_names += [f'yolo2_c{i}' for i in ['1','2','3','4','5','6']]
    conv_names += ['yolo_out2']
    conv_names += ['yolo_up2_conv']
    conv_names += [f'yolo3_c{i}' for i in ['1','2','3','4','5','6']]
    conv_names += ['yolo_out3']

    layer_map = model.layer_map
    assigned = 0
    for conv_base in conv_names:
        conv_key = conv_base + '_conv'
        if conv_key not in layer_map:
            raise KeyError(f"{conv_key} not found in model.layer_map.")
        conv_layer = layer_map[conv_key]

        bn_key = conv_base + '_bn'
        has_bn = bn_key in layer_map
        use_bias = getattr(conv_layer, 'use_bias', False)

        if has_bn:
            bn = layer_map[bn_key]
            # Darknet order: beta, gamma, mean, variance
            for attr, name in [('beta','beta'), ('gamma','gamma'), ('moving_mean','mean'), ('moving_variance','var')]:
                var = getattr(bn, attr)
                n = int(np.prod(var.shape.as_list()))
                vals = _read(n).reshape(var.shape.as_list())
                var.assign(vals)
                assigned += n
            if verbose:
                print(f"Loaded BN -> {bn_key}: beta{bn.beta.shape} gamma{bn.gamma.shape} mean{bn.moving_mean.shape} var{bn.moving_variance.shape}")
        elif use_bias:
            bias = conv_layer.bias
            n = int(np.prod(bias.shape.as_list()))
            vals = _read(n).reshape(bias.shape.as_list())
            bias.assign(vals)
            assigned += n
            if verbose:
                print(f"Loaded bias -> {conv_key}: {bias.shape}")

        kernel = conv_layer.kernel
        kshape = kernel.shape.as_list()
        n_k = int(np.prod(kshape))
        kvals = _read(n_k)
        try:
            kvals = kvals.reshape((kshape[3], kshape[2], kshape[0], kshape[1]))
            kvals = np.transpose(kvals, (2, 3, 1, 0))
        except Exception:
            kvals = kvals.reshape(kshape)
        kernel.assign(kvals.reshape(kshape))
        assigned += n_k
        if verbose:
            print(f"Loaded kernel -> {conv_key}: {kernel.shape}")

    if ptr != total:
        print(f"Warning: not all weights consumed. consumed: {ptr} total: {total} leftover: {total - ptr}")
    else:
        print(f"Finished assigning weights. consumed: {ptr} total: {total}")

    return ptr, total


def load_images(img_names, model_size):
    imgs = []
    for img_name in img_names:
        img = Image.open(img_name).resize(model_size)
        arr = np.array(img, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)
        imgs.append(arr)
    return np.concatenate(imgs, axis=0)


def load_class_names(file_name):
    with open(file_name, 'r') as f:
        return f.read().splitlines()
def draw_boxes(img_names, boxes_dicts, class_names, model_size):
    """Draw detected boxes robustly across Pillow versions."""
    colors = ((np.array(color_palette("hls", 80)) * 255)).astype(np.uint8)

    def _text_size(draw, text, font, img=None):
        """Return (w,h) for text using several fallbacks to support many Pillow versions."""
        try:
            if hasattr(draw, "textbbox"):
                bbox = draw.textbbox((0, 0), text, font=font)
                return (bbox[2] - bbox[0], bbox[3] - bbox[1])
        except Exception:
            pass
        try:
            if hasattr(font, "getbbox"):
                bbox = font.getbbox(text)
                return (bbox[2] - bbox[0], bbox[3] - bbox[1])
        except Exception:
            pass
        try:
            if hasattr(font, "getmask"):
                mask = font.getmask(text)
                return mask.size
        except Exception:
            pass
        try:
            if hasattr(draw, "textsize"):
                return draw.textsize(text, font=font)
        except Exception:
            pass
        try:
            if hasattr(font, "getsize"):
                return font.getsize(text)
        except Exception:
            pass
        # Last-resort approximation
        fontsize = getattr(font, "size", None)
        if fontsize is None and img is not None:
            fontsize = (img.size[0] + img.size[1]) // 100
        fontsize = fontsize or 12
        return (int(len(text) * fontsize * 0.6), int(fontsize))

    for num, img_name, boxes_dict in zip(range(len(img_names)), img_names, boxes_dicts):
        img = Image.open(img_name).convert("RGB")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                font='/kaggle/input/data-for-yolo-v3-kernel/futur.ttf',
                size=(img.size[0] + img.size[1]) // 100
            )
        except Exception:
            font = ImageFont.load_default()

        resize_factor = (img.size[0] / model_size[0], img.size[1] / model_size[1])

        for cls in range(len(class_names)):
            boxes = boxes_dict.get(cls, np.array([]))
            if boxes is None:
                continue
            boxes = np.array(boxes)
            if boxes.size == 0:
                continue

            color = colors[cls % len(colors)]
            color_tuple = tuple(int(c) for c in color)

            for box in boxes:
                # box format: [x_center, y_center, width, height, conf] per your pipeline
                # but your code earlier used xy = box[:4] where xy were top-left, bottom-right
                # Keep same behavior: assume box[:4] are top_left_x, top_left_y, bottom_right_x, bottom_right_y
                xy, confidence = box[:4], float(box[4])
                # scale to original image
                xy_scaled = [float(xy[i]) * resize_factor[i % 2] for i in range(4)]
                x0, y0, x1, y1 = xy_scaled[0], xy_scaled[1], xy_scaled[2], xy_scaled[3]

                # thickness drawing without mutating coords
                thickness = max(1, (img.size[0] + img.size[1]) // 200)
                for t in range(thickness):
                    offs = t
                    rect = (
                        int(round(x0 + offs)),
                        int(round(y0 + offs)),
                        int(round(x1 - offs)),
                        int(round(y1 - offs))
                    )
                    draw.rectangle(rect, outline=color_tuple)

                # prepare text and draw background
                text = '{} {:.1f}%'.format(class_names[cls], confidence * 100.0)
                text_w, text_h = _text_size(draw, text, font, img=img)
                tx0 = int(round(x0))
                ty0 = int(round(y0))
                background_rect = [tx0, ty0 - text_h, tx0 + text_w, ty0]
                draw.rectangle(background_rect, fill=color_tuple)
                draw.text((tx0, ty0 - text_h), text, fill='black', font=font)

        display(img)
