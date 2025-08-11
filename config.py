
# Dataset paths
CLIC_TFDS_NAME = "clic"          # using TFDS
VIMEO_ROOT = "/kaggle/input/vimeo-90k-9"   
VIMEO_SEP_TRAINLIST = "/kaggle/input/vimeo-90k-9/sep_trainlist.txt"

# Training schedule
GLOBAL_BATCH_SIZE = 16           # total batch over all GPUs (paper used 16)
PATCH_SIZE = 256                 # training crop size
YOLO_INPUT_SIZE = 512            # size used to compute YOLO features (paper used 512 for YOLO)
EPOCHS_STAGE1 = 400              # paper: stage1
EPOCHS_STAGE2 = 300              # continue stage2
STEPS_PER_EPOCH_STAGE1 = None    # if None, computed from dataset size (CLIC train size ~1633)
STEPS_PER_EPOCH_STAGE2 = None    # set if you know number of frames / batch

# Learning rates / optimizer
INITIAL_LR = 1e-4
POLY_END_LR = 1e-6  # final LR for polynomial decay in stage2
POLY_POWER = 1.0

# Rate-distortion weighting (paper used multiple lambda values for different qualities).
# We'll use one lambda here; you can override per-quality later.
LAMBDA_RD = 0.01
GAMMA_TASK = 0.006   # weight for YOLO feature MSE steering into base latent

# Checkpoint / logging
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"
SAVE_EVERY_N_EPOCHS = 5

# Model settings
LATENT_C = 192
BASE_CHANNELS = 128
LST_OUT_CHANNELS = 256
LST_UP_FACTORS = (2, 1, 1, 1)

# Misc
SEED = 42
PREFETCH_AHEAD = tf.data.AUTOTUNE if 'tf' in globals() else None
NUM_WORKERS = 8

# Dataset cardinalities (TFDS reported)
CLIC_TRAIN_SIZE = 1633    # TFDS metadata
CLIC_VALID_SIZE = 102
