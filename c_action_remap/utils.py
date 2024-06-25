import tensorflow as tf
import numpy as np

def get_loss_map(action_label, action_mapped):
    
    epsilon = tf.keras.backend.epsilon()
    action_mapped = tf.clip_by_value(action_mapped, epsilon, 1)

    cross_entropy = -tf.reduce_sum(action_label * tf.math.log(action_mapped), axis=-1)

    return cross_entropy