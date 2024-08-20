from PIL import Image
import cv2
import random
import math
import numpy as np


def uint8_to_float(img_orig):
    """
        Convert a uint8 image into float with range [0, 1]
    """
    return img_orig.astype(np.float32) / 255.


def float_to_int(img_float):
    """
        Convert a float image with range [0, 1] into uint 8 with range [0 255]
    """
    return (img_float * 255).round().astype(np.int16)


def float_to_uint8(img_float):
    return (img_float * 255).round().astype(np.uint8)


def watermark_np_to_str(watermark_np):
    """
        Convert a watermark in np format into a str to display.
    """
    return "".join([str(i) for i in watermark_np.tolist()])


def watermark_str_to_numpy(watermark_str):
    result = [int(i) for i in watermark_str]
    return np.asarray(result)


def compute_bitwise_acc(watermark_gt, watermark_decoded):
    """
        Compute the bitwise acc., both inputs in ndarray.
    """
    return np.mean(watermark_gt == watermark_decoded)


def bgr2rgb(img_bgr):
    """
        img_bgr.shape = (xx, xx, 3)
    """
    img_rgb = np.stack(
        [img_bgr[:, :, 2], img_bgr[:, :, 1], img_bgr[:, :, 0]], axis=2
    )
    return img_rgb


def rgb2bgr(img_rgb):
    """
        img_bgr.shape = (xx, xx, 3)
    """
    return bgr2rgb(img_rgb)


def save_image_bgr(img_np, path):
    cv2.imwrite(path, img_np)


def save_image_rgb(img_np, path):
    img_np = rgb2bgr(img_np)
    save_image_bgr(img_np, path)


def bytearray_to_bits(x):
    """Convert bytearray to a list of bits"""
    result = []
    for i in x:
        bits = bin(i)[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result

