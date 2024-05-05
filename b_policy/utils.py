import numpy as np
import tensorflow as tf

def get_loss_min(delta_s, delta_s_hat):
    tile_delta_s = tf.reshape(tf.tile(delta_s, [1, delta_s_hat.shape[1]]), delta_s_hat.shape)

    loss_min = tf.norm(tf.subtract(tile_delta_s, delta_s_hat), axis=-1)
    loss_min = tf.reduce_mean(tf.reduce_min(loss_min, axis=1))

    return loss_min

def get_loss_exp(delta_s, z_p, delta_s_hat):
    expect_delta_s = tf.multiply(delta_s_hat, tf.expand_dims(z_p, axis=-1))
    expect_delta_s = tf.reduce_mean(expect_delta_s, axis=1)

    loss_exp = tf.subtract(delta_s, expect_delta_s)
    loss_exp = tf.reduce_mean(tf.norm(loss_exp, axis=-1))

    return loss_exp

def get_metric(delta_s, z_p, delta_s_hat):
    max_z_idx = tf.argmax(z_p, axis=-1)
    max_z_idx = tf.expand_dims(tf.one_hot(max_z_idx, z_p.shape[-1]), axis=-1)

    max_delta_s_hat = tf.reduce_sum(tf.multiply(delta_s_hat, max_z_idx), axis=1)
    
    metric = tf.subtract(delta_s, max_delta_s_hat)
    metric = tf.reduce_mean(tf.norm(metric, axis=-1))

    return metric

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