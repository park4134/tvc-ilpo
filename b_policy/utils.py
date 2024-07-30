import numpy as np
import tensorflow as tf

def get_loss_min(delta_s, delta_s_hat):
    tile_delta_s = tf.tile(tf.expand_dims(delta_s, axis=1), [1, delta_s_hat.shape[1], 1]) # (batch, n_state) >> (batch, n_z, n_state)

    loss_min = tf.square(tf.subtract(tile_delta_s, delta_s_hat)) # (batch, n_z, n_state)
    loss_min = tf.reduce_min(tf.reduce_mean(loss_min, axis=-1), axis=-1) # (batch, )
    loss_min = tf.reduce_mean(loss_min)

    return loss_min

def get_loss_exp(delta_s, z_p, delta_s_hat):
    expect_delta_s = tf.multiply(delta_s_hat, tf.expand_dims(z_p, axis=-1))
    expect_delta_s = tf.reduce_sum(expect_delta_s, axis=1) # (batch, n_state)

    loss_exp = tf.subtract(delta_s, expect_delta_s) # (batch, n_state)
    loss_exp = tf.reduce_mean(tf.square(loss_exp)) # (batch, )
    loss_exp = tf.reduce_mean(loss_exp)

    return loss_exp

def get_metric(delta_s, z_p, delta_s_hat):
    max_z_idx = tf.argmax(z_p, axis=-1)
    max_z_idx = tf.expand_dims(tf.one_hot(max_z_idx, z_p.shape[-1]), axis=-1)

    max_delta_s_hat = tf.reduce_sum(tf.multiply(delta_s_hat, max_z_idx), axis=1)
    
    metric = tf.subtract(delta_s, max_delta_s_hat)
    metric = tf.reduce_mean(tf.norm(metric, axis=-1))

    relative_metric = get_relative_metric(delta_s, max_delta_s_hat)
    # relative_metric = tf.reduce_mean(tf.divide(max_delta_s_hat, tf.add(delta_s, 1e-06)))

    return metric, np.array(relative_metric)

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

def get_relative_metric(delta_s, delta_s_hat):
    # delta_s : (length, n_state)
    # delta_s_hat : (length, n_state)
    # print(delta_s.shape, delta_s_hat.shape)

    zero_order = tf.zeros_like(delta_s)
    
    # calculate MSE
    mse_delta_s = tf.reduce_mean(tf.square(delta_s - delta_s_hat), axis=0)
    mse_zero_order = tf.reduce_mean(tf.square(delta_s - zero_order), axis=0)

    # calculate Relative Metircs
    rm = mse_delta_s / mse_zero_order

    # print(rm.shape)
    
    # rm : (n_state)
    return rm

def get_test_metric(delta_s, z_p, delta_s_hat):
    max_z_idx = tf.argmax(z_p, axis=-1)
    max_z_idx = tf.expand_dims(tf.one_hot(max_z_idx, z_p.shape[-1]), axis=-1)

    max_delta_s_hat = tf.reduce_sum(tf.multiply(delta_s_hat, max_z_idx), axis=1)
    
    metric = tf.subtract(delta_s, max_delta_s_hat)
    metric = tf.norm(metric, axis=-1)

    zero_order = tf.zeros_like(delta_s)
    
    mse_delta_s = tf.square(delta_s - max_delta_s_hat)
    mse_zero_order = tf.square(delta_s - zero_order) + 1e-06

    rm = mse_delta_s / mse_zero_order

    return np.array(metric), np.array(rm)