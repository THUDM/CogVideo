# -*- encoding: utf-8 -*-
'''
@File    :   iterative_sr.py
@Time    :   2022/03/02 15:57:45
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random

# here put the import lib
import os
import sys
import math
import random
from PIL import ImageEnhance, Image

import torch
import argparse
from torchvision import transforms

from SwissArmyTransformer.training.model_io import load_checkpoint
from SwissArmyTransformer import get_args
from .itersr_sampling import filling_sequence_itersr, IterativeEntfilterStrategy
from SwissArmyTransformer.generation.utils import timed_name, save_multiple_images, generate_continually

from .itersr_model import ItersrModel

from icetk import icetk as tokenizer

class IterativeSuperResolution:
    def __init__(self, args, path, max_bz=4, shared_transformer=None):
        args.load = path
        args.kernel_size = 5
        args.kernel_size2 = 5
        args.new_sequence_length = 4624
        args.layout = [16,3616]
        
        model = ItersrModel(args, transformer=shared_transformer)
        if args.fp16:
            model = model.half()
        
        load_checkpoint(model, args) # on cpu
        model.eval()
        self.model = model.cuda()

        # save cpu weights
        self.saved_weights = dict((k,v.cpu()) 
            for k, v in model.named_parameters()
            if 'transformer' in k
        )
        
        invalid_slices = [slice(tokenizer.num_image_tokens, None)]
    
        self.strategy = IterativeEntfilterStrategy(invalid_slices,
            temperature=args.temp_all_itersr, topk=args.topk_itersr)
        self.max_bz = max_bz

    def _restore_transformer_from_cpu(self, non_blocking=False):
        for k, v in self.model.named_parameters():
            if k in self.saved_weights:
                v.copy_(self.saved_weights[k])
        
    def __call__(self, text_tokens, image_tokens, enhance=False, input_mask=None):
        if len(text_tokens.shape) == 1:
            text_tokens.unsqueeze_(0)
        text_tokens = text_tokens.clone()[..., :16]
        if len(image_tokens.shape) == 1:
            image_tokens.unsqueeze_(0)
        if enhance:
            new_image_tokens = []
            for big_img in image_tokens:
                decoded = tokenizer.decode(image_ids=big_img).squeeze(0)
                ndarr = decoded.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                image_pil_raw = ImageEnhance.Sharpness(Image.fromarray(ndarr))
                big_img2 = tokenizer.encode(image_pil=image_pil_raw.enhance(1.5), image_size=480).view(-1)
                new_image_tokens.append(big_img2)
            image_tokens = torch.stack(new_image_tokens)
        print('Converting Itersr model...')
        self._restore_transformer_from_cpu()
        model = self.model
        print('iterative super-resolution...')
        output_list = []
        for tim in range(max(text_tokens.shape[0] // self.max_bz, 1)): 
                big_img = image_tokens[tim*self.max_bz:(tim+1)*self.max_bz]
                text_seq = text_tokens[tim*self.max_bz:(tim+1)*self.max_bz]
                mask_raw = torch.tensor(
                    [
                        -1, 0, 1, 2, 3, 4,
                        0, -1, 2, -1, -2, 5,
                        1, -2, 3, 4, 5, 6,
                        2, 3, 4, 5, -1, 1,
                        3, -1, -2, 0, -1, 2,
                        4, 5, 6, 1, 3, -2
                    ]
                ).view(1, 6, 1, 6).expand(10, 6, 10, 6).reshape(-1).contiguous()

                topks = [60, 40, 40, 40, 20, 20, 10]

                for mask_ratio in range(1, 7):  
                    self.strategy.topk = topks[mask_ratio]
                    mask = (mask_raw.to(big_img.device) >= mask_ratio)
                    if input_mask is not None:
                        mask = mask & input_mask
                    big_img.masked_fill_(mask, tokenizer['<start_of_image>'])
                    seq1 = big_img
                    output1 = filling_sequence_itersr(model, text_seq, seq1, 
                        warmup_steps=1, block_hw=(1, 0),
                        strategy=self.strategy
                        )
                    big_img = output1
                    print(f'Iter {mask_ratio} times.')
                output_list.append(output1.clone())
        return torch.cat(output_list, dim=0)