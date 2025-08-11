import os
os.environ['TF_USE_LEGACY_KERAS'] = "1"
import math
import numpy as np
import tensorflow as tf
import tqdm 
from model import MultiTaskCodec
from objectdetection.networks import  MinimalYoloV3 as YOLOv3, BNPlusLReLU
from dataset import make_vimeo_train_dataset,\
                               make_clic_train_dataset,\
                               make_clic_validation_dataset
from metrics import compute_rate_metrics_from_model_outputs,\
                            estimate_mi_kde

yolo3 = YOLOv3() #takes an input in range [0, 255] and outputs the 13th conv layer when called (F_1^{13} from paper)
bn_and_lrelu = BNPlusLReLU() # The output feature map from latent space transform (LST) requires normalization and activation. This layer does that (V13 operation from paper)

for layer in yolo3.layers:
    layer.trainable = False 
for layer in bn_and_lrelu.layers:
    layer.trainable = False 



