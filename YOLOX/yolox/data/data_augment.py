#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import math
import random

import cv2
import torch
import numpy as np
import albumentations as A

from yolox.utils import xyxy2cxcywh, cxcywh2xyxy
from yolox.data.custom_augmentations import Moire, LCDScreenPattern


def flatten_raw_bbox(raw_bbox, mean=False):
    is_np = isinstance(raw_bbox, np.ndarray)
    if not len(raw_bbox):
        return np.zeros((0, 4)) if is_np else torch.zeros((0, 4))
    a = []
    counts = raw_bbox[:, 0] / 4
    for i, c in enumerate(counts):
        c = int(c)
        if mean:
            a.append(raw_bbox[i, 1:1+c*4].reshape(c, 4).mean(0)[None])
        else:
            a.append(raw_bbox[i, 1:1+c*4].reshape(c, 4))
    if is_np:
        return np.concatenate(a).copy()
    else:
        return torch.cat(a).clone()

def restore_raw_bbox(raw_bbox, flatten):
    is_np = isinstance(raw_bbox, np.ndarray)
    if is_np:
        ret = np.zeros_like(raw_bbox)
    else:
        ret = torch.zeros_like(raw_bbox)
    counts = raw_bbox[:, 0] / 4
    ret[:, 0] = raw_bbox[:, 0]
    curr_j = 0
    for i, c in enumerate(counts):
        c = int(c)
        ret[i, 1:1+c*4] = flatten[curr_j: curr_j+c].reshape(-1)
        curr_j += c
    return ret

def augment_hsv(img, hgain=5, sgain=30, vgain=30):
    hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
    hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
    hsv_augs = hsv_augs.astype(np.int16)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

    # img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

    cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def get_aug_params(value, center=0):
    if isinstance(value, float):
        return random.uniform(center - value, center + value)
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
             or single float values. Got {}".format(value)
        )


def get_affine_matrix(
    target_size,
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    twidth, theight = target_size

    # Rotation and Scale
    angle = get_aug_params(degrees)
    scale = get_aug_params(scales, center=1.0)

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

    M = np.ones([2, 3])
    # Shear
    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]

    # Translation
    translation_x = get_aug_params(translate) * twidth  # x translation (pixels)
    translation_y = get_aug_params(translate) * theight  # y translation (pixels)

    M[0, 2] = translation_x
    M[1, 2] = translation_y

    return M, scale


def apply_affine_to_bboxes(targets, target_size, M, scale):
    num_gts = len(targets)

    # warp corner points
    twidth, theight = target_size
    corner_points = np.ones((4 * num_gts, 3))
    corner_points[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        4 * num_gts, 2
    )  # x1y1, x2y2, x1y2, x2y1
    corner_points = corner_points @ M.T  # apply affine transform
    corner_points = corner_points.reshape(num_gts, 8)

    # create new boxes
    corner_xs = corner_points[:, 0::2]
    corner_ys = corner_points[:, 1::2]
    new_bboxes = (
        np.concatenate(
            (corner_xs.min(1), corner_ys.min(1), corner_xs.max(1), corner_ys.max(1))
        )
        .reshape(4, num_gts)
        .T
    )

    # clip boxes
    new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clip(0, twidth)
    new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clip(0, theight)

    targets[:, :4] = new_bboxes

    return targets


def random_affine(
    img,
    targets=(),
    target_size=(640, 640),
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
    uncertain_offset=None
):
    M, scale = get_affine_matrix(target_size, degrees, translate, scales, shear)

    img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(114, 114, 114))

    # Transform label coordinates
    if len(targets) > 0:
        targets = apply_affine_to_bboxes(targets, target_size, M, scale)
        if uncertain_offset is not None:
            flatten = flatten_raw_bbox(targets[:, uncertain_offset:])
            flatten = apply_affine_to_bboxes(flatten, target_size, M, scale)
            targets[:, uncertain_offset:] = restore_raw_bbox(targets[:, uncertain_offset:], flatten)

    return img, targets


def _mirror(image, boxes, prob=0.5, flip_ud=False, r_boxes=None):
    height, width, _ = image.shape
    if random.random() < prob:
        image = image[:, ::-1]
        boxes[:, 0::2] = width - boxes[:, 2::-2]
        if r_boxes is not None:
            flatten = flatten_raw_bbox(r_boxes)
            flatten[:, 0::2] = width - flatten[:, 2::-2]
            r_boxes = restore_raw_bbox(r_boxes, flatten)
    if flip_ud and random.random() < prob:
        image = image[::-1, ...]
        boxes[:, 1::2] = height - boxes[:, 3::-2]
        if r_boxes is not None:
            flatten = flatten_raw_bbox(r_boxes)
            flatten[:, 1::2] = height - flatten[:, 3::-2]
            r_boxes = restore_raw_bbox(r_boxes, flatten)
    return image, boxes, r_boxes

def _rotate90(image, boxes, factor, r_boxes=None):
    height, width, _ = image.shape
    def rotatebox(boxes, factor):
        x_min, y_min, x_max, y_max = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        if factor == 3:  # ROT90_270_FACTOR
            boxes = np.stack([y_min, width - x_max, y_max, width - x_min], axis=-1)
        elif factor == 2:  # ROT90_180_FACTOR:
            boxes = np.stack([width - x_max, height - y_max, width - x_min, height - y_min], axis=-1)
        elif factor == 1:  # ROT90_90_FACTOR:
            boxes = np.stack([height - y_max, x_min, height - y_min, x_max], axis=-1)
        return boxes
    if factor == 1:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif factor == 2:  # ROT90_180_FACTOR:
        image = cv2.rotate(image, cv2.ROTATE_180_CLOCKWISE)
    elif factor == 3:  # ROT90_270_FACTOR:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    boxes = rotatebox(boxes, factor)
    if r_boxes is not None:
        flatten = flatten_raw_bbox(r_boxes)
        flatten = rotatebox(flatten, factor)
        r_boxes = restore_raw_bbox(r_boxes, flatten)
    return image, boxes, r_boxes

def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def get_raw_bbox_first_idx(t):
    for i in range(t.shape[-1]):
        if i > 4: # skip bbox
            if t[0, i] > 1:
                return i
    raise ValueError('invalid target')

class TrainTransform:
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0, uncertain=False,):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.uncertain = uncertain
        self.rotate90_prob = 0.5
        self.album_aug = A.Compose([A.AdvancedBlur(p=0.5, blur_limit=(5, 15)),
                                    A.GaussNoise(p=0.5, var_limit=(10., 128.), noise_scale_factor=.75),
                                    A.ISONoise(p=0.5, intensity=(0.2, 1.0))])
        self.custom_aug = [LCDScreenPattern(p=0.5), Moire(p=0.5)]

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :4].copy()
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, targets.shape[-1]), dtype=np.float32)
            if random.random() < self.hsv_prob:
                augment_hsv(image, vgain=100)
            image = self.album_aug(image=image)['image']
            for c_aug in self.custom_aug:
                image = c_aug(image=image)['image']
            if random.random() < self.rotate90_prob:
                image, boxes, r_boxes = _rotate90(image, boxes, random.choice([1, 3]))
            if random.random() < 0.5: #hflip
                image = image[:, ::-1]
            if random.random() < 0.5: #vflip
                image = image[::-1, ...]
            image, r_o = preproc(image, input_dim)
            return image, targets
        r_boxes = None
        if self.uncertain:
            raw_bbox_first_idx = get_raw_bbox_first_idx(targets)
            labels = targets[:, 4:raw_bbox_first_idx].copy()
            r_boxes = targets[:, raw_bbox_first_idx:].copy()
        else:
            labels = targets[:, 4].copy()

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        if self.uncertain:
            labels_o = targets_o[:, 4:raw_bbox_first_idx]
            r_boxes_o = targets_o[:, raw_bbox_first_idx:]
        else:
            labels_o = targets_o[:, 4]
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)

        if random.random() < self.hsv_prob:
            augment_hsv(image, vgain=100)
        image = self.album_aug(image=image)['image']
        for c_aug in self.custom_aug:
            image = c_aug(image=image)['image']

        image_t, boxes, r_boxes = _mirror(image, boxes, self.flip_prob, True, r_boxes)
        if random.random() < self.rotate90_prob:
            image_t, boxes, r_boxes = _rotate90(image_t, boxes, random.choice([1, 3]), r_boxes)

        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim)
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        if self.uncertain:
            r_boxes *= r_
            r_boxes_t = r_boxes[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o
            if self.uncertain:
                r_boxes_o *= r_o
                r_boxes_t = r_boxes_o

        if self.uncertain:
            labels_t = np.hstack((labels_t, r_boxes_t))

        # check
        # if self.uncertain:
        #     flatten = flatten_raw_bbox(r_boxes_t, mean=True)
        #     np.testing.assert_allclose(cxcywh2xyxy(boxes_t.copy()), flatten, rtol=1e-5, atol=2)

        targets_t = np.hstack((labels_t, boxes_t))
        padded_labels = np.zeros((self.max_labels, targets.shape[-1]))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image_t, padded_labels


class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1), legacy=False):
        self.swap = swap
        self.legacy = legacy

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return img, np.zeros((1, 5))
