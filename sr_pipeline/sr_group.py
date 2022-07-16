# -*- encoding: utf-8 -*-
'''
@File    :   sr_group.py
@Time    :   2022/04/02 01:17:21
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from SwissArmyTransformer.resources import auto_create
from .direct_sr import DirectSuperResolution
from .iterative_sr import IterativeSuperResolution

class SRGroup:
    def __init__(self, args, home_path=None,):
        dsr_path = auto_create('cogview2-dsr', path=home_path)
        itersr_path = auto_create('cogview2-itersr', path=home_path)
        dsr = DirectSuperResolution(args, dsr_path)
        itersr = IterativeSuperResolution(args, itersr_path, shared_transformer=dsr.model.transformer)
        self.dsr = dsr
        self.itersr = itersr

    def sr_base(self, img_tokens, txt_tokens):
        assert img_tokens.shape[-1] == 400 and len(img_tokens.shape) == 2
        batch_size = img_tokens.shape[0]
        txt_len = txt_tokens.shape[-1]
        if len(txt_tokens.shape) == 1:
            txt_tokens = txt_tokens.unsqueeze(0).expand(batch_size, txt_len)
        sred_tokens = self.dsr(txt_tokens, img_tokens)
        iter_tokens = self.itersr(txt_tokens, sred_tokens[:, -3600:].clone())
        return iter_tokens[-batch_size:]
    
    # def sr_patch(self, img_tokens, txt_tokens):
    #     assert img_tokens.shape[-1] == 3600 and len(img_tokens.shape) == 2
    #     batch_size = img_tokens.shape[0] * 9
    #     txt_len = txt_tokens.shape[-1]
    #     if len(txt_tokens.shape) == 1:
    #         txt_tokens = txt_tokens.unsqueeze(0).expand(batch_size, txt_len)
    #     img_tokens = img_tokens.view(img_tokens.shape[0], 3, 20, 3, 20).permute(0, 1, 3, 2, 4).reshape(batch_size, 400)
    #     iter_tokens = self.sr_base(img_tokens, txt_tokens)
    #     return iter_tokens