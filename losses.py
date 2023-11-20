import tensorflow as tf
#from utils import tonemap, hdr_to_ldr
import numpy as np
import math

def PSNR(img1, img2):
    return tf.image.psnr(img1, img2, max_val=1.0)
"""
def PSNR_T(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.image.psnr(tonemap(y_true), tonemap(y_pred), max_val=1.0)

def MSE_TM(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.keras.losses.MSE(tonemap(y_true), tonemap(y_pred))

def MAE_TM(y_true, y_pred):
    return tf.keras.losses.MAE(tonemap(y_true), tonemap(y_pred))

def PSNR_L(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return PSNR(y_true, y_pred)
"""
def MS_SSIM(y_true, y_pred):
    #y_true = tonemap(y_true)
    #y_pred = tonemap(y_pred)
    #y_true = tf.image.rgb_to_grayscale(y_true)
    #y_pred = tf.image.rgb_to_grayscale(y_pred)
    return 1.0 - tf.image.ssim(y_true, y_pred, 1.0)

def MSE(y_true, y_pred):
    return tf.keras.losses.MSE(y_true, y_pred)

def MAE(y_true, y_pred):
    return tf.keras.losses.MAE(y_true, y_pred)

def GRAD(y_true, y_pred):
    y_true = tf.image.rgb_to_grayscale(y_true[:, :, :, 6:9])
    y_pred = tf.image.rgb_to_grayscale(y_pred)
    grad_x_true, grad_y_true = tf.image.image_gradients(y_true)
    grad_x_pred, grad_y_pred = tf.image.image_gradients(y_pred)
    return tf.reduce_mean(tf.square(grad_x_pred - grad_x_true) + tf.square(grad_y_pred - grad_y_true))

def EDGE(y_true, y_pred):
    loss_edge = tf.reduce_mean(
        tf.square(tf.abs(y_pred[:, 1:, 1:, :] - y_pred[:, :-1, 1:, :]) - tf.abs(y_true[:, 1:, 1:, :] - y_true[:, :-1, 1:, :]))
        + tf.square(tf.abs(y_pred[:, 1:, 1:, :] - y_pred[:, 1:, :-1, :]) - tf.abs(y_true[:, 1:, 1:, :] - y_true[:, 1:, :-1, :])))
    return loss_edge