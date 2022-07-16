# -*- encoding: utf-8 -*-
'''
@File    :   itersr_sampling.py
@Time    :   2022/03/03 14:24:28
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
from icetk import icetk as tokenizer

def top_k_logits_(logits, top_k=0, filter_value=-float('Inf')):
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = filter_value     
    return logits

# class IterativeEntfilterStrategy:
#     def __init__(self, invalid_slices=[], temperature=1., topk=10):
#         self.invalid_slices = invalid_slices
#         self.temperature = temperature
#         self.topk = topk        
#         self.cluster_labels = torch.tensor(np.load('cluster_label.npy'), device='cuda', dtype=torch.long)


#     def forward(self, logits_, tokens, temperature=None, entfilter=None, filter_topk=5, temperature2=None):
#         # In interative strategy, logits are of shape [batch_size, seq_length, hidden_size]
#         if temperature is None:
#             temperature = self.temperature 
            
#         logits = logits_.float() / temperature
#         for invalid_slice in self.invalid_slices:
#             logits[..., invalid_slice] = -float('Inf')
#         logits = logits.view(-1, logits.shape[-1])
        
#         rprobs = F.softmax(logits.float(), dim=-1)
#         c = self.cluster_labels.expand(*rprobs.shape)
#         cprobs = torch.zeros(logits.shape[0], 500, device=logits.device).scatter_add_(1, c, rprobs)
    
#         best_scores, best_clusters = cprobs.topk(self.topk)
#         bz = logits.shape[0]
#         best_scores = best_scores / best_scores.sum(dim=-1, keepdim=True)
#         sampled_ids = torch.multinomial(best_scores, num_samples=1)
#         selected_clusters = torch.gather(best_clusters, dim=1, index=sampled_ids)
#         selected_mask = (self.cluster_labels.unsqueeze(0).expand(bz, -1) != selected_clusters) # cluster_labels [1, 20000] \in [0,500)
#         logits[selected_mask] = -65504
#         # for i in range(bz):
#         #     selected_cluster = best_clusters[i][torch.multinomial(best_scores[i] / best_scores[i].sum(), num_samples=1)]
#         #     logits[i, self.cluster_labels != selected_cluster] = -65504
            
#         # logits = top_k_logits(logits, self.topk, self.top_p)
#         probs = F.softmax(logits.float(), dim=-1)  # float is essetial, due to a bug in Pytorch
#         pred = torch.multinomial(probs, num_samples=1).view(*logits_.shape[:2])
        
#         assert tokens.shape[1] == pred.shape[1]
#         tokens = pred
#         return tokens

class IterativeEntfilterStrategy:
    def __init__(self, invalid_slices=[], temperature=1., topk=10):
        self.invalid_slices = invalid_slices
        self.temperature = temperature
        self.topk = topk

    def forward(self, logits, tokens, temperature=None, entfilter=None, filter_topk=5, temperature2=None):
        # In interative strategy, logits are of shape [batch_size, seq_length, hidden_size]
        if temperature is None:
            temperature = self.temperature 
        # check entropy filter
        # if entfilter is not None:
        #     assert temperature2 is not None
        #     topraw = (torch.topk(logits, filter_topk, dim=-1)[0]).softmax(dim=-1)
        #     ent = -(topraw * topraw.log()).sum(dim=-1) # [batch_size, seq_length]
        #     temperature = torch.tensor([[[temperature - temperature2]]], device=logits.device).expand(*logits.shape[:2], 1) * (ent > entfilter).unsqueeze(-1) + temperature2
            
        logits = logits.float() / temperature
        for invalid_slice in self.invalid_slices:
            logits[..., invalid_slice] = -float('Inf')
        
        # debiased topk
        # probs = F.softmax(logits, dim=-1)
        # tk_value, tk_idx = torch.topk(probs, self.topk, dim=-1)
        # pred = torch.multinomial(probs.view(-1, logits.shape[-1]), num_samples=1).view(*logits.shape[:2], 1)
        # edge_idx = tk_idx[:, :, -1:]
        # edge_value = tk_value[:, :, -1:]
        # edge_mask = probs.gather(dim=-1, index=pred) < edge_value
        # pred[edge_mask] = edge_idx[edge_mask] # replace outliers as the "filter_topk"-th token
        # pred.squeeze_(-1) # [batch_size, seq_length]
        
        top_k_logits_(logits, self.topk)
        probs = F.softmax(logits, dim=-1)
        pred = torch.multinomial(probs.view(-1, logits.shape[-1]), num_samples=1).view(*logits.shape[:2], 1)
        pred.squeeze_(-1)
        
        assert tokens.shape[1] == pred.shape[1]
        tokens = pred
        return tokens

def filling_sequence_itersr(
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
    
    device = seq0.device
    # concat and pad sequences
    batch_size = seq0.shape[0]
    n_pad = layout[0] - seq0.shape[1]
    assert n_pad >= 0, "You should truncate long input before filling."
    seq = torch.cat((
        torch.tensor([0]*n_pad, device=device, dtype=seq0.dtype)
            .unsqueeze(0).expand(batch_size, n_pad),
        seq0, seq1), dim=1) # [b, layout[-1]+1]
    assert seq.shape[1] == layout[-1]

    # build initial tokens, attention_mask, and position_ids
    tokens = seq.clone()
    attention_mask = torch.ones(layout[0]).to(device)
    attention_mask[:n_pad] = 0
    attention_mask = attention_mask.unsqueeze(0).type_as(next(model.parameters())) # if fp16
    position_ids = torch.cat((
        torch.zeros(n_pad, dtype=torch.long),
        torch.arange(0, layout[0] - n_pad), 
        torch.arange(1024, 1024+layout[1]-layout[0]))).to(device)
    log_attention_weights = torch.zeros(layout[0], device=device).type_as(next(model.parameters()))
    log_attention_weights[n_pad:layout[0]] = 0.
    log_attention_weights = log_attention_weights.unsqueeze(0)

    # prepare for interation
    unfixed = (tokens == tokenizer['<start_of_image>']) 
    ll, rr = block_hw
    edge_len = int(math.sqrt(layout[-1] - layout[-2]) + 1e-4)
    num_steps = 1
    # interative refining
    
    # unfixed[..., -(layout[-1] - layout[-2]):].view(
    #     batch_size, edge_len//ll, ll, edge_len//rr, rr)[:, :, :, :, -1] = False
    
    
    ret = []
    # ret.append(tokens[:, layout[-2]:-1].clone())
    for step_cnt in range(1, num_steps+1):
        logits, *_dump = model(tokens, position_ids, attention_mask, log_attention_weights=log_attention_weights)
        real_temp = 1.
        new_tokens = strategy.forward(logits, tokens, real_temp)
        tokens[unfixed] = new_tokens[unfixed]
                
        ret.append(tokens[:, layout[-2]:].clone())
    return torch.cat(ret, dim=0)