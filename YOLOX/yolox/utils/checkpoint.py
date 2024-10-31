#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
import os
import shutil
from loguru import logger

import torch


def load_ckpt(model, ckpt):
    model_state_dict = model.state_dict()
    load_dict = {}
    for key_model, v in model_state_dict.items():
        if key_model not in ckpt:
            logger.warning(
                "{} is not in the ckpt. Please double check and see if this is desired.".format(
                    key_model
                )
            )
            continue
        v_ckpt = ckpt[key_model]
        if v.shape < v_ckpt.shape:
            logger.warning(
                "Shape of {} in checkpoint is {}, while shape of {} in model is {}. Attempting to truncate".format(
                    key_model, v_ckpt.shape, key_model, v.shape
                )
            )
            v_ckpt = v_ckpt[:v.shape[0]]
        elif v.shape != v_ckpt.shape:
            logger.warning(
                "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                    key_model, v_ckpt.shape, key_model, v.shape
                )
            )
            continue
        load_dict[key_model] = v_ckpt

    model.load_state_dict(load_dict, strict=False)
    with torch.no_grad():
        for i in range(len(model.head.reg_preds)):
            w_name = f'head.reg_preds.{i}.weight'
            b_name = f'head.reg_preds.{i}.bias'
            if w_name not in load_dict.keys() and w_name in ckpt:
                w = ckpt[w_name]
                model.head.reg_preds[i].weight[:w.shape[0]].copy_(w)
                logger.info('forced loaded {}'.format(w_name))
            if b_name not in load_dict.keys() and b_name in ckpt:
                w = ckpt[b_name]
                model.head.reg_preds[i].bias[:w.shape[0]].copy_(w)
                logger.info('forced loaded {}'.format(b_name))
    return model


def save_checkpoint(state, is_best, save_dir, model_name=""):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, model_name + "_ckpt.pth")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_dir, "best_ckpt.pth")
        shutil.copyfile(filename, best_filename)
