"""
Utiiity functions for tensorflow
"""
import tensorflow as tf

def gpu_fix():
    """
    temporary bug fix for laptop RTX architectures
        - https://github.com/tensorflow/tensorflow/issues/24496
    """
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices("GPU")
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)