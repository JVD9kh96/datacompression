import os
os.environ['TF_USE_LEGACY_KERAS'] = "1"
import glob
import random
import tensorflow as tf
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.AUTOTUNE

# -------------------------
# Helpers
# -------------------------
def decode_and_convert(contents, channels=0):
    """
    Decode bytes -> float32 [0,1]. Use channels=0 to preserve whatever channels are present,
    we'll normalize afterwards in ensure_3_channels().
    """
    # decode_image returns a rank-3 tensor [H,W,C] (C may be 1,3,4,...)
    img = tf.image.decode_image(contents, channels=channels, expand_animations=False)
    # decode_image can return uint8; convert to float32 in [0,1]
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def ensure_3_channels(image):
    """
    Robustly ensure image has 3 channels (RGB), returns float32 tensor [H,W,3].
    Rules:
      - If channels == 1: convert grayscale -> RGB (tile)
      - If channels >= 3: take first 3 channels (drop alpha/extra channels)
    This avoids the tf.cond branch-shape mismatch you saw.
    """
    image = tf.convert_to_tensor(image)
    # Ensure rank is 3 (H,W,C). If not, try to expand dims or raise.
    rank = tf.rank(image)
    # If image somehow has no channel dim (rank==2), add channel dim.
    def _expand_if_needed():
        return tf.cond(
            tf.equal(rank, 2),
            lambda: tf.expand_dims(image, axis=-1),
            lambda: image,
        )
    image = _expand_if_needed()

    c = tf.shape(image)[-1]

    def gray_to_rgb():
        # grayscale_to_rgb expects shape [...,1] and returns [...,3]
        return tf.image.grayscale_to_rgb(image)

    def take_first3():
        # If image already has >=3 channels, slice to first 3 channels.
        return image[..., :3]

    # If c == 1 -> convert; else (c >= 2) take first 3 channels (works also when c==3).
    return tf.cond(tf.equal(c, 1), gray_to_rgb, take_first3)


def load_image_file(path):
    contents = tf.io.read_file(path)
    return decode_and_convert(contents, channels=3)  # request 3 channels; if grayscale, decoder will expand


def random_crop_or_pad_to_size(image, target_h, target_w):
    """
    If image is smaller -> pad (reflect) then random crop to target.
    If bigger -> random crop to target.
    """
    # get current size
    shape = tf.shape(image)
    h = shape[0]; w = shape[1]

    pad_h = tf.maximum(0, target_h - h)
    pad_w = tf.maximum(0, target_w - w)
    # reflect pad if needed
    def _pad():
        # tf.image.pad_to_bounding_box pads with zeros; use reflect via tf.pad
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        paddings = [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
        return tf.pad(image, paddings, mode='REFLECT')
    image_padded = tf.cond(tf.logical_or(pad_h > 0, pad_w > 0), _pad, lambda: image)

    # Now random crop (if larger or equal)
    cropped = tf.image.random_crop(image_padded, size=[target_h, target_w, 3])
    return cropped

def preprocess_train_image(image, patch_size=256, seed=None, augment=True):
    """
    image: float32 [H,W,3] in [0,1]
    returns: float32 [patch_size,patch_size,3]
    """
    img = ensure_3_channels(image)
    img = random_crop_or_pad_to_size(img, patch_size, patch_size)
    if augment:
        img = tf.image.random_flip_left_right(img, seed=seed)
    return img

# -------------------------
# CLIC (TFDS) pipeline
# -------------------------
def make_clic_train_dataset(batch_size=16, patch_size=256, shuffle_buffer=256, cache=False, seed=None, as_supervised=False):
    """
    Stage 1 dataset from TFDS 'clic' train split.
    - Random 256x256 crops as in paper
    - batch_size default 16
    - uses the official 'train' split (1,633 images per TFDS metadata)
    """
    # Load TFDS CLIC dataset (as provided by TFDS)
    ds = tfds.load("clic", split="train", shuffle_files=True, as_supervised=as_supervised)
    # Depending on tfds version, elements are dict {'image': ...} or (image,) if supervised
    def _extract_image(x):
        if as_supervised:
            # x is image tensor
            img = x
        else:
            img = x["image"]
        img = tf.image.convert_image_dtype(img, tf.float32)  # to [0,1]
        return img

    ds = ds.map(lambda x: _extract_image(x), num_parallel_calls=AUTOTUNE)
    if cache:
        ds = ds.cache()
    ds = ds.shuffle(shuffle_buffer, seed=seed)
    ds = ds.map(lambda img: preprocess_train_image(img, patch_size=patch_size, seed=seed),
                num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(AUTOTUNE)
    return ds

def make_clic_validation_dataset(batch_size=8, target_size=256, as_supervised=False):
    """
    Validation dataset from TFDS 'clic' validation split. Use central crop/resizing for eval.
    """
    ds = tfds.load("clic", split="validation", as_supervised=as_supervised)
    def _extract_image(x):
        if as_supervised:
            img = x
        else:
            img = x["image"]
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img
    def _prep_eval(img):
        img = ensure_3_channels(img)
        # center crop or pad then resize to target_size (paper used 256 patches for training; evaluation may use full images)
        # We'll center-crop if larger, otherwise pad then center-crop:
        img = tf.image.resize_with_crop_or_pad(img, target_size, target_size)
        return img
    ds = ds.map(lambda x: _extract_image(x), num_parallel_calls=AUTOTUNE)
    ds = ds.map(_prep_eval, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds

# -------------------------
# Vimeo (Kaggle) pipeline
# -------------------------
def build_vimeo_image_list_from_sep(base_dir, sep_trainlist_path, sequence_dir_name="sequence"):
    """
    Scan sep_trainlist and convert to a flat list of image file paths.
    - base_dir: root of the dataset that contains the 'sequence' folder.
    - sep_trainlist_path: file path to sep_trainlist.txt which contains lines like "00081/0001"
    - sequence_dir_name: typically "sequence" in the Kaggle layout
    Returns: list of absolute image file paths (PNG/JPG)
    NOTE: this is done in Python (not inside tf.data) to produce a list of file paths
    """
    img_paths = []
    base_seq_dir = os.path.join(base_dir, sequence_dir_name)
    if not os.path.isdir(base_seq_dir):
        raise FileNotFoundError(f"Sequence directory not found: {base_seq_dir}")
    with open(sep_trainlist_path, "r") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    for rel in lines:
        # rel example: "00081/0001"
        seq_path = os.path.join(base_seq_dir, rel)
        if not os.path.isdir(seq_path):
            # If the repo uses different sub-structure, try joining differently:
            # try base_seq_dir/rel/  or base_seq_dir/rel/images
            # We'll skip missing unless you want an exception
            continue
        # gather all image files in that directory
        # pattern match common image extensions:
        found = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            found.extend(glob.glob(os.path.join(seq_path, ext)))
        if not found:
            continue
        # append each frame file
        # If you prefer to sample only one frame per sequence per epoch, you could append seq_path instead.
        img_paths.extend(found)
    if len(img_paths) == 0:
        raise RuntimeError("No image files found from sep_trainlist. Check paths.")
    random.shuffle(img_paths)
    return img_paths

def make_vimeo_train_dataset_from_filelist(img_file_list, batch_size=16, patch_size=256, shuffle_buffer=256, augment=True, seed=None):
    """
    Create tf.data.Dataset from a list of image file paths (python list).
    Each element is loaded and randomly cropped to patch_size (256) and batched.
    """
    # Convert to TF dataset of strings (paths)
    ds = tf.data.Dataset.from_tensor_slices(img_file_list)
    ds = ds.shuffle(shuffle_buffer, seed=seed)
    def _load_and_preprocess(path):
        img = load_image_file(path)
        img = preprocess_train_image(img, patch_size=patch_size, seed=seed, augment=augment)
        return img
    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(AUTOTUNE)
    return ds

def make_vimeo_train_dataset(base_dir, sep_trainlist_path, batch_size=16, patch_size=256, shuffle_buffer=10000, augment=True, seed=None):
    """
    One-shot helper that parses sep_trainlist and returns a tf.data.Dataset for training.
    """
    img_list = build_vimeo_image_list_from_sep(base_dir, sep_trainlist_path, sequence_dir_name="sequence")
    return make_vimeo_train_dataset_from_filelist(img_list, batch_size=batch_size, patch_size=patch_size, shuffle_buffer=shuffle_buffer, augment=augment, seed=seed)

# -------------------------
# Utilities
# -------------------------
def count_steps(dataset, steps_per_epoch=None):
    """
    If dataset is a finite batched dataset, attempt to compute number of steps per epoch.
    If steps_per_epoch provided, returns it; else returns None (unknown).
    """
    if steps_per_epoch is not None:
        return steps_per_epoch
    return None
