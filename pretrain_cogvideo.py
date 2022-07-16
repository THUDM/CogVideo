# -*- encoding: utf-8 -*-
'''
@File    :   pretrain_cogvideo.py
@Time    :   2021/10/06 00:58:32
@Author  :   Wenyi Hong
@Contact :   hwy22@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch
import argparse
import numpy as np
from icetk import icetk as tokenizer
tokenizer.add_special_tokens(['<start_of_image>', '<start_of_english>', '<start_of_chinese>'])

from models.cogvideo_model import CogVideoModel
from SwissArmyTransformer import mpu, get_args
from SwissArmyTransformer.training.deepspeed_training import training_main
from SwissArmyTransformer.data_utils import BinaryDataset

def get_masks_and_position_ids_video(data, attention_mask_totxt=None, args=None):
    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()
    assert attention_mask_totxt is not None
    layout = args.layout
    assert seq_length == layout[-1]
    n_pads = layout[0] - attention_mask_totxt.sum(dim=-1).long()
    frame_len = layout[1]-layout[0]
    position_ids = torch.zeros(batch_size, layout[2], dtype=torch.long,
                                device=data.device)
    for i in range(batch_size):
        torch.arange(layout[0] - n_pads[i], out=position_ids[i, n_pads[i]:layout[0]], 
            dtype=torch.long, device=data.device)
        torch.arange(512, 512+layout[2]-layout[0], 
            out=position_ids[i, layout[0]:], dtype=torch.long, device=data.device)
    return position_ids


def get_batch(data_iterator, args, timers):
    # Items and their type.
    keys = ['text', 'loss_mask', 'attention_mask_totxt']
    datatype = torch.int64

    # Broadcast data.
    timers('data loader').start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    timers('data loader').stop()

    data_b = mpu.broadcast_data(keys, data, datatype)
    # Unpack.
    tokens_ = data_b['text'].long()
    loss_mask = data_b['loss_mask'].float()
    attention_mask_totxt = data_b['attention_mask_totxt'].float()

    labels = tokens_[:, 1:].clone().contiguous()
    loss_mask = loss_mask[:, 1:].contiguous()
    tokens = tokens_[:, :-1].clone().contiguous()
    
    for idx in range(args.layout[0], args.layout[2], 400):
        tokens[:, idx] = tokenizer['<start_of_image>']
    # Get the masks and postition ids.
    position_ids = get_masks_and_position_ids_video(
        tokens,
        attention_mask_totxt=attention_mask_totxt,
        args=args
        )
    attention_mask_totxt = attention_mask_totxt.unsqueeze(1).unsqueeze(1)
    # Convert
    if args.fp16:
        attention_mask_totxt = attention_mask_totxt.half()
    return tokens, labels, loss_mask, attention_mask_totxt, position_ids


def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    tokens, labels, loss_mask, attention_mask_totxt, position_ids = get_batch(
        data_iterator, args, timers)
    timers('batch generator').stop()

    # Forward model.
    logits, *mems = model(tokens, position_ids, attention_mask_totxt)
    # ======= hyper params =======#
    perframe_len = 400
    text_len=64
    frame_num = 5
    logits_img_tokens = logits[:, text_len:, :tokenizer.num_image_tokens].float().contiguous()
    losses = mpu.vocab_parallel_cross_entropy(logits_img_tokens, labels[:, text_len:])
    # scaling loss mask
    loss_mask = loss_mask[:, text_len:].reshape(-1)  

    losses_1d = losses.reshape(-1) * loss_mask
    loss = torch.sum(losses_1d) / loss_mask.sum()
    # =====================   Log partial losses   ======================== #
    log_loss_dict = {}
    bs = losses.shape[0]

    if args.cogvideo_stage == 1:
        for i in range(frame_num):
            log_loss_dict[f'AR_f{i}_loss'] = losses[:, i*perframe_len:(i+1)*perframe_len].contiguous().reshape(-1).detach().sum() / max((perframe_len*bs), 1) 
    else:
        for i in range(1, frame_num-1):
            log_loss_dict[f'ITP_f{i}_loss'] = losses[:, i*perframe_len:(i+1)*perframe_len].contiguous().reshape(-1).detach().sum() / max((perframe_len*bs), 1) 
 
    # ===================== END OF BLOCK ======================= #
    return loss, log_loss_dict
    

def create_dataset_function(path, args):
    dataset_layout = [64, 464, 2064]
    input_layout = [64, 464, 2064]
    # frame_num = 6
    # frame_interval = 2 # DEBUG!!!
    def process_fn(row):
        row = row.astype(np.int64)
        text = row[:dataset_layout[0]]
        frames = row[dataset_layout[0]:]
        
        if text[0] == tokenizer['<pad>']:
            text = text[1:] # due to our way of data processing
        if args.cogvideo_stage == 1:
            text, loss_mask, frames = make_text_video_generation(text, frames)
        else: 
            text, loss_mask, frames = mask_video_frame_interpolation(text, frames)
            
        n_pad = input_layout[0] - len(text)
        parts = [
            np.array([tokenizer['<pad>']] * n_pad, dtype=np.int64),
            text,
            np.array([tokenizer['<start_of_image>']], dtype=np.int64),
            frames,
        ]
        ret = np.concatenate(parts, axis=0)
        
        attention_mask_totxt = np.array([0] * n_pad + [1] * (input_layout[0]-n_pad))
        return {'text': ret, 
            'loss_mask':  loss_mask,
            'attention_mask_totxt': attention_mask_totxt,
            }
    return BinaryDataset(path, process_fn, length_per_sample=dataset_layout[-1])

def make_text_video_generation(text, frames):
    input_layout = [64, 464, 2064]
    text = text[text!= tokenizer['<pad>']][:input_layout[0]] # dataset format: 1.0秒<n>{text}<pad><pad> ... 
    loss_mask = np.array([0] * (input_layout[1]+1) + [1] * (input_layout[2] - input_layout[1])) # 按照input的，之后loss_mask会左移一位
    return text, loss_mask, frames

def mask_video_frame_interpolation(text, frames):
    input_layout = [64, 464, 2064]
    frame_len = input_layout[1]-input_layout[0]
    # text format: <pad> 1.0秒 <n> {text} <pad> <pad>
    text = text[text!= tokenizer['<pad>']][:input_layout[0]]
    loss_mask = np.array([0] * (input_layout[1]+1) 
                        + [1] * (input_layout[1]-input_layout[0])
                        + [0] * (input_layout[1]-input_layout[0])
                        + [1] * (input_layout[1]-input_layout[0])
                        + [0] * (input_layout[1]-input_layout[0]) )# 按照input的，之后loss_mask会左移一位
        
    return text, loss_mask, frames
    


if __name__ == '__main__':    
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--txt-loss-scale', type=float, default=1)    
    CogVideoModel.add_model_specific_args(py_parser)
    
    known, args_list = py_parser.parse_known_args()
    
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    
    args.layout = [int(x) for x in args.layout.split(',')]
    
    training_main(args, model_cls=CogVideoModel, forward_step_function=forward_step, create_dataset_function=create_dataset_function)
