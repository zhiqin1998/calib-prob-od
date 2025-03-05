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
        self.data_dir = '../data/vbdcxr'

        self.max_epoch = 100
        self.warmup_epochs = 5
        self.eval_interval = 1
        # save history checkpoint or not.
        # If set to False, yolox will only save latest and best ckpt.
        self.save_history_ckpt = False

        self.basic_lr_per_img = 0.001 / 32
        self.no_aug_epochs = 25

        self.uncertain = True
        self.bbox_unc_loss = 'dmm'
        self.clamp_log_var = 7.
        self.bbox_unc_weight = 0.1  # 0.1
