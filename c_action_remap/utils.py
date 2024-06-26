import tensorflow as tf
import numpy as np

def get_loss_map(action_label, action_mapped):
    
    epsilon = tf.keras.backend.epsilon()
    action_mapped = tf.clip_by_value(action_mapped, epsilon, 1)

    cross_entropy = -tf.reduce_sum(action_label * tf.math.log(action_mapped), axis=-1)

    return cross_entropy

def gpu_limit(GB) :
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("########################################")
    print('{} GPU(s) is(are) available'.format(len(gpus)))
    print("########################################")
    # set the only one GPU and memory limit
    memory_limit = 1024 * GB
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
            print("Use only one GPU{} limited {}MB memory".format(gpus[0], memory_limit))
        except RuntimeError as e:
            print(e)
    else:
        print('GPU is not available')