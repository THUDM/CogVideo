# -*- encoding: utf-8 -*-
'''
@File    :   itersr_model.py
@Time    :   2021/10/02 01:36:32
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch
import torch.nn.functional as F


from SwissArmyTransformer.model.base_model import BaseModel, BaseMixin

from SwissArmyTransformer.mpu.utils import sqrt
from deepspeed.runtime.activation_checkpointing.checkpointing import get_cuda_rng_tracker
from SwissArmyTransformer.mpu import ColumnParallelLinear, RowParallelLinear
from SwissArmyTransformer.model.transformer import unscaled_init_method, split_tensor_along_last_dim

class PositionEmbeddingMixin(BaseMixin):
    def __init__(self, additional_sequence_length, hidden_size,
                 init_method_std=0.02, reinit_slice=slice(512, 512+400)
                 ):
        super(PositionEmbeddingMixin, self).__init__()
        self.reinit_slice = reinit_slice
        self.position_embeddings = torch.nn.Embedding(additional_sequence_length, hidden_size)
        torch.nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=init_method_std)

    def reinit(self, parent_model=None):
        old_weights = self.transformer.position_embeddings.weight.data[self.reinit_slice]
        old_len, hidden_size = old_weights.shape
        assert hidden_size == self.position_embeddings.weight.shape[-1]
        old_edge, new_edge = sqrt(old_len), sqrt(self.position_embeddings.weight.shape[-2])
        assert new_edge % old_edge == 0
        self.position_embeddings.weight.data.view(new_edge // old_edge, old_edge, new_edge // old_edge, old_edge, hidden_size).copy_(old_weights.view(1, old_edge, 1, old_edge, hidden_size))

class ItersrModel(BaseModel):
    def __init__(self, args, transformer=None):
        super().__init__(args, transformer=transformer)
        self.original_sequence_length = args.max_sequence_length
        additional_seqlen = args.new_sequence_length - args.max_sequence_length
        self.add_mixin('extra_position_embedding', PositionEmbeddingMixin(
            additional_seqlen, args.hidden_size
        ))
        # self.add_mixin('attention_plus', AttentionMixin(
        #     num_layers=args.num_layers,
        #     hidden_size=args.hidden_size
        # ))
        self.layout = args.layout
        # [PAD]... [ROI1] text ... [BOI1] {layout[0]} 1024 {layout[1]} [EOI1] 4095 {layout[2]}
        self.kernel_size = args.kernel_size
        self.kernel_size2 = args.kernel_size2
        self.log_attention_weights = None
    
    def position_embedding_forward(self, position_ids, **kw_args):
        position = position_ids[..., :self.layout[0]]
        position_plus = position_ids[..., self.layout[0]:] - self.original_sequence_length
        position_embeddings = torch.cat(
                (
                    self.transformer.position_embeddings(position),
                    self.get_mixin('extra_position_embedding').position_embeddings(position_plus)
                ),
                dim=-2
            )
        return position_embeddings
        
    def attention_forward(self, hidden_states, mask, 
                        layer_id=None, log_attention_weights=None, **kw_args):
        attn_module = self.transformer.layers[layer_id].attention
        # base model qkv
        mixed_raw_layer = attn_module.query_key_value(hidden_states)
        q0, k0, v0 = split_tensor_along_last_dim(mixed_raw_layer[:, :self.layout[0]], 3)
        # cuda2d model qkv
        q1, k1, v1 = split_tensor_along_last_dim(mixed_raw_layer[:, self.layout[0]:], 3)
        
        dropout_fn = attn_module.attention_dropout if self.training else None

        # cuda2d attention
        context_layer = sparse_attention_2d_text(
                q0, k0, v0,
                q1, k1, v1,
                mask,
                n_head=attn_module.num_attention_heads_per_partition,
                text_len=self.layout[0],
                kernel_size=self.kernel_size,
                attention_dropout=dropout_fn,
                log_attention_weights=log_attention_weights,
            )

        output = attn_module.dense(context_layer)
        
        return output
    
    def final_forward(self, logits, **kwargs):
        logits_parallel = logits
        logits_parallel = torch.nn.functional.linear(logits_parallel, self.transformer.word_embeddings.weight[:20000]).float()
        # logits_parallel = torch.nn.functional.linear(logits_parallel, self.transformer.word_embeddings.weight[:20000])
        return logits_parallel
    
    # def disable_untrainable_params(self):
    #     self.transformer.requires_grad_(False)
    
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('Cuda2dModel', 'cuda2d model configurations')
        group.add_argument("--kernel-size", type=int, default=5)
        group.add_argument("--kernel-size2", type=int, default=5)
        group.add_argument("--layout", type=str, default='16,3616')
        group.add_argument("--new-sequence-length", type=int, default=4096)
        return parser

def sparse_attention_2d_text(q0, k0, v0, q1, k1, v1, attention_mask, n_head, text_len, kernel_size=9,  attention_dropout=None, log_attention_weights = None, **kwargs):
    '''
    q0, k0, v0: [batch_size, 16, hidden_size]
    q1, k1, v1: [batch_size, 3600, hidden_size]
    n_head: int
    attention_mask: [batch_size, 16]
    '''
    from SwissArmyTransformer.ops.local_attention_function import f_similar, f_weighting
    b, s0, h0 = q0.shape
    b, s1, h1 = q1.shape
    h, l1 = h0 // n_head, sqrt(s1)
    assert attention_mask.shape[-1] == s0, f"Mask Shape: {attention_mask.shape}"

    q0 = q0.reshape(b, s0, n_head, h).permute(0, 2, 1, 3)
    v0 = v0.reshape(b, s0, n_head, h).permute(0, 2, 1, 3)
    k0T = k0.reshape(b, s0, n_head, h).permute(0, 2, 3, 1)
    
    # standard attention for level 0
    attention_scores = torch.matmul(q0 / math.sqrt(q0.shape[-1]), k0T)
    
    attention_scores = torch.mul(attention_scores, attention_mask) - \
                    10000.0 * (1.0 - attention_mask)
    
    attention_probs0 = F.softmax(attention_scores, dim=-1)
    
    # local attention for level 1
    q1 = (q1.view(b, s1, n_head, h1 // n_head).permute(0, 2, 3, 1) / math.sqrt(h1//n_head)).contiguous().view(b*n_head, h1//n_head, l1, l1)
    k1 = k1.view(b, s1, n_head, h1 // n_head).permute(0, 2, 3, 1).contiguous().view(b*n_head, h1//n_head, l1, l1)
    v1 = v1.view(b, s1, n_head, h1 // n_head).permute(0, 2, 3, 1).contiguous().view(b*n_head, h1//n_head, l1, l1)
    scores_1_to_1 = f_similar(q1, k1, kernel_size*2-1, kernel_size, False)    

    # cross attention
    scores_1_to_0 = torch.matmul(q1.view(b, n_head, h, s1).transpose(-1, -2), k0T)
    if log_attention_weights is not None:
        scores_1_to_0 += log_attention_weights
    scores_1_to_0 = torch.mul(scores_1_to_0, attention_mask) - \
                    10000.0 * (1.0 - attention_mask)
    scores_1 = torch.cat(
        (
            scores_1_to_0.view(b*n_head, s1, s0),
            scores_1_to_1.view(b*n_head, -1, scores_1_to_1.shape[3])
        ),
        dim=-1)
    attention_probs1 = F.softmax(scores_1, dim=-1)
    
    if attention_dropout is not None:
        with get_cuda_rng_tracker().fork():
            attention_probs1 = attention_dropout(attention_probs1)
        
    # weighting for level 0
    context0 = torch.matmul(attention_probs0, v0) # [b, n_head, s0, h]
    # weighting for level 1
    probs_1_to_1 = attention_probs1[:, :, -scores_1_to_1.shape[3]:].view_as(scores_1_to_1)
    context1_to_1 = f_weighting(v1, probs_1_to_1.contiguous(), kernel_size*2-1, kernel_size, False)

    context1 = context1_to_1.view(b, n_head, h, l1**2)
    # weighting for cross attention
    probs_1_to_0 = attention_probs1[:, :, :scores_1_to_0.shape[3]].view(b, n_head, -1, scores_1_to_0.shape[3])
    
    context1_to_0 = torch.matmul(probs_1_to_0, v0)
    context1 = context1.transpose(-1, -2) + context1_to_0
    
    output = torch.cat((context0, context1), dim=2).transpose(1, 2).reshape(b, s0+s1, h0)

    return output

def sparse_attention_2d_notext(q0, k0, v0, q1, k1, v1, attention_mask, n_head, text_len, kernel_size=9,  attention_dropout=None, log_attention_weights = None, **kwargs):
    '''
    q0, k0, v0: [batch_size, 16, hidden_size]
    q1, k1, v1: [batch_size, 3600, hidden_size]
    n_head: int
    attention_mask: [batch_size, 16]
    '''
    from SwissArmyTransformer.mpu.local_attention_function import f_similar, f_weighting
    b, s0, h0 = q0.shape
    b, s1, h1 = q1.shape
    h, l1 = h0 // n_head, sqrt(s1)
    assert len(attention_mask.shape) == 4 and attention_mask.shape[-1] == s0, f"Mask Shape: {attention_mask.shape}"

    q0 = q0.reshape(b, s0, n_head, h).permute(0, 2, 1, 3)
    v0 = v0.reshape(b, s0, n_head, h).permute(0, 2, 1, 3)
    k0T = k0.reshape(b, s0, n_head, h).permute(0, 2, 3, 1)
    
    # standard attention for level 0
    attention_scores = torch.matmul(q0 / math.sqrt(q0.shape[-1]), k0T)
    
    attention_scores = torch.mul(attention_scores, attention_mask) - \
                    10000.0 * (1.0 - attention_mask)
    
    attention_probs0 = F.softmax(attention_scores, dim=-1)
    
    # local attention for level 1
    q1 = (q1.view(b, s1, n_head, h1 // n_head).permute(0, 2, 3, 1) / math.sqrt(h1//n_head)).contiguous().view(b*n_head, h1//n_head, l1, l1)
    k1 = k1.view(b, s1, n_head, h1 // n_head).permute(0, 2, 3, 1).contiguous().view(b*n_head, h1//n_head, l1, l1)
    v1 = v1.view(b, s1, n_head, h1 // n_head).permute(0, 2, 3, 1).contiguous().view(b*n_head, h1//n_head, l1, l1)
    scores_1_to_1 = f_similar(q1, k1, kernel_size*2-1, kernel_size, False)    

    attention_probs1 = F.softmax(scores_1_to_1, dim=-1)
    
    if attention_dropout is not None:
        with get_cuda_rng_tracker().fork():
            attention_probs1 = attention_dropout(attention_probs1)
        
    # weighting for level 0
    context0 = torch.matmul(attention_probs0, v0) # [b, n_head, s0, h]
    # weighting for level 1
    probs_1_to_1 = attention_probs1
    context1_to_1 = f_weighting(v1, probs_1_to_1.contiguous(), kernel_size*2-1, kernel_size, False)

    context1 = context1_to_1.view(b, n_head, h, l1**2)
    # weighting for cross attention    
    context1 = context1.transpose(-1, -2)
    
    output = torch.cat((context0, context1), dim=2).transpose(1, 2).reshape(b, s0+s1, h0)

    return output