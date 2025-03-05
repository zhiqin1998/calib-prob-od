#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.num_classes = 14
        self.data_dir = '<path to vbdcxr/processed_coco_unc_ind>'

        self.warmup_epochs = 1
        self.max_epoch = 30
        self.eval_interval = 1
        # save history checkpoint or not.
        # If set to False, yolox will only save latest and best ckpt.
        self.save_history_ckpt = False

        self.basic_lr_per_img = 0.001 / 32
        self.no_aug_epochs = 5

        self.uncertain = True
        self.bbox_unc_loss = 'nll_na'  # nll, dmm (direct moment matching)
        self.clamp_log_var = 7.
        self.bbox_unc_weight = 1. # 0.1
