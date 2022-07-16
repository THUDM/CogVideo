# -*- encoding: utf-8 -*-
'''
@File    :   cuda2d_sampling.py
@Time    :   2021/10/09 00:46:04
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
from cv2 import reduce
import torch

import torch
import torch.nn.functional as F
import numpy as  np

def top_k_logits_(logits, top_k=0, filter_value=-float('Inf')):
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = filter_value     
    return logits

class IterativeEntfilterStrategy:
    def __init__(self, invalid_slices=[], temperature=1., topk=6):
        self.invalid_slices = invalid_slices
        self.temperature = temperature
        self.topk = topk        
        self.cluster_labels = torch.tensor(np.load('cluster_label2.npy'), device='cuda', dtype=torch.long)


    def forward(self, logits_, tokens, temperature=None, entfilter=None, filter_topk=5, temperature2=None):
        # In interative strategy, logits are of shape [batch_size, seq_length, hidden_size]
        if temperature is None:
            temperature = self.temperature 
            
        logits = logits_.float() / temperature
        for invalid_slice in self.invalid_slices:
            logits[..., invalid_slice] = -float('Inf')
        logits = logits.view(-1, logits.shape[-1])
        
        rprobs = F.softmax(logits.float(), dim=-1)
        c = self.cluster_labels.expand(*rprobs.shape)
        cprobs = torch.zeros(logits.shape[0], 500, device=logits.device).scatter_add_(1, c, rprobs)
    
        best_scores, best_clusters = cprobs.topk(self.topk)
        bz = logits.shape[0]
        best_scores = best_scores / best_scores.sum(dim=-1, keepdim=True)
        sampled_ids = torch.multinomial(best_scores, num_samples=1)
        selected_clusters = torch.gather(best_clusters, dim=1, index=sampled_ids)
        selected_mask = (self.cluster_labels.unsqueeze(0).expand(bz, -1) != selected_clusters) # cluster_labels [1, 20000] \in [0,500)
        logits[selected_mask] = -65504
        # for i in range(bz):
        #     selected_cluster = best_clusters[i][torch.multinomial(best_scores[i] / best_scores[i].sum(), num_samples=1)]
        #     logits[i, self.cluster_labels != selected_cluster] = -65504
            
        # logits = top_k_logits(logits, self.topk, self.top_p)
        probs = F.softmax(logits.float()/0.6, dim=-1)  # float is essetial, due to a bug in Pytorch
        pred = torch.multinomial(probs, num_samples=1).view(*logits_.shape[:2])
        
        assert tokens.shape[1] == pred.shape[1] + 1
        tokens = torch.cat((tokens[:, :1], pred), dim=1)
        return tokens

def filling_sequence_dsr(
        model, 
        seq0,
        seq1, 
        warmup_steps=3,
        block_hw=(4, 4),
        strategy=IterativeEntfilterStrategy(topk=10),
        ):
    '''
        seq: [PAD]... [ROI1] text ... [BOI1] {layout[0]} 1024 {layout[1]} [EOI1]
            4095 {layout[2]} final_token.
        Attention:
        The sampling temperature are changing, temporally we hard code them here.
        The temperature in the strategy is not used.
    '''
    assert hasattr(model, 'layout')
    layout = model.layout
    assert len(seq0.shape) == 2 and len(seq1.shape) == 2 \
        and seq0.shape[0] == seq1.shape[0]
    assert len(layout) == 3
    assert seq1.shape[1] == layout[-1] - layout[-2] + 1
    assert (seq1 >= 0).all() and (seq0 >= 0).all()
    device = seq0.device
    # concat and pad sequences
    batch_size = seq0.shape[0]
    n_pad = layout[1] - seq0.shape[1]
    assert n_pad > 0, "You should truncate long input before filling."
    seq = torch.cat((
        torch.tensor([0]*n_pad, device=device, dtype=seq0.dtype)
            .unsqueeze(0).expand(batch_size, n_pad),
        seq0, seq1), dim=1) # [b, layout[-1]+1]
    assert seq.shape[1] == layout[-1] + 1

    # build initial tokens, attention_mask, and position_ids
    tokens = seq.clone()
    attention_mask = torch.ones(layout[1], layout[1]).to(device)
    attention_mask[:layout[0], layout[0]:] = 0
    attention_mask[n_pad:, :n_pad] = 0
    attention_mask = attention_mask.type_as(next(model.parameters())) # if fp16
    position_ids = torch.cat((
        torch.zeros(n_pad, dtype=torch.long),
        torch.arange(0, layout[0] - n_pad), 
        torch.arange(513, 513 + layout[1] - layout[0]),
        torch.arange(1024, 1024+layout[2]-layout[1]))).to(device)
    log_attention_weights = torch.zeros(layout[1], layout[1], 
            device=device).type_as(next(model.parameters()))
    log_attention_weights[layout[0]:, n_pad:layout[0]] = 0.

    # prepare for interation
    unfixed = (tokens < 0) # just init an all-False tensor
    unfixed[:, -layout[-1] + layout[-2]:] = True
    
    ll, rr = block_hw
    edge_len = int(math.sqrt(layout[-1] - layout[-2]) + 1e-4)
    num_steps = warmup_steps + ll - 1 + rr
    # interative refining
    
    # unfixed[..., -(layout[-1] - layout[-2]):].view(
    #     batch_size, edge_len//ll, ll, edge_len//rr, rr)[:, :, :, :, -1] = False
    
    
    ret = []
    ret.append(tokens[:, layout[-2]+1:].clone())
    for step_cnt in range(1, num_steps+1):
        if step_cnt <= warmup_steps:
            logits, *_dump = model(tokens[:,:-1], position_ids, attention_mask, log_attention_weights=log_attention_weights)
            real_temp = 1.
            new_tokens = strategy.forward(logits, tokens, real_temp)
            tokens[unfixed] = new_tokens[unfixed]
        else:
            logits, *_dump = model(tokens[:,:-1], position_ids, attention_mask, log_attention_weights=log_attention_weights)
            real_temp = 1.
            new_tokens = strategy.forward(
                logits, tokens, real_temp,
                entfilter=1.3,
                filter_topk=5,
                temperature2=0.6
            )
            # tokens[unfixed] = new_tokens[unfixed]
            # fixed tokens (update unfixed)
            unfixed2 = (tokens > 10000000)
            for x in range(min(ll, step_cnt - warmup_steps)):
                y = step_cnt - warmup_steps - x - 1
                if y < rr:
                    unfixed[..., -(layout[-1] - layout[-2]):].view(
                        batch_size, edge_len//ll, ll, edge_len//rr, rr)[:, :, x, :, y] = False
                    unfixed2[..., -(layout[-1] - layout[-2]):].view(
                        batch_size, edge_len//ll, ll, edge_len//rr, rr)[:, :, x, :, y] = True
            tokens[unfixed2] = new_tokens[unfixed2]
                
        ret.append(tokens[:, layout[-2]+1:].clone())

    return ret
