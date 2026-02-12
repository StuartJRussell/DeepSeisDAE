
# Import modules
import tensorflow as tf
import keras.backend as K
from tensorflow.python.ops import math_ops


def BCE2(y_true, y_pred):
    """
    This is just the usual mean binary cross-entropy of the predicted and target 
    mask functions for the signal and the noise.
    """
    
    # Calculate the BCE
    return K.mean(K.binary_crossentropy(y_true, y_pred, from_logits=False), axis=-1)
