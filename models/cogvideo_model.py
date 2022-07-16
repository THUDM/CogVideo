# -*- encoding: utf-8 -*-
'''
@File    :   cogvideo_model.py
@Time    :   2022/07/11 16:12:05
@Author  :   Wenyi Hong 
@Version :   1.0
@Contact :   hwy22@mails.tsinghua.edu.cn
'''

# here put the import lib

import torch
from SwissArmyTransformer.model.base_model import BaseModel, BaseMixin

from SwissArmyTransformer.mpu.utils import split_tensor_along_last_dim
from SwissArmyTransformer.model.transformer import unscaled_init_method
from SwissArmyTransformer.mpu import ColumnParallelLinear, RowParallelLinear
import torch.nn.functional as F
from deepspeed.runtime.activation_checkpointing.checkpointing import get_cuda_rng_tracker
import math

class PositionEmbeddingMixin(BaseMixin):
    def __init__(self, additional_sequence_length, hidden_size,
                 init_method_std=0.02, reinit_slice=slice(512, 912), 
                 ):
        super(PositionEmbeddingMixin, self).__init__()
        self.reinit_slice = reinit_slice
        self.position_embeddings = torch.nn.Embedding(additional_sequence_length, hidden_size)
        torch.nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=init_method_std)

    def reinit(self, parent_model=None):
        old_weights = self.transformer.position_embeddings.weight.data[self.reinit_slice]
        old_len, hidden_size = old_weights.shape
        assert hidden_size == self.position_embeddings.weight.shape[-1]
        self.position_embeddings.weight.data.view(-1, old_len, hidden_size).copy_(old_weights)
        
def window_partition(x, window_size):
    """
    Args:
        x: (B, framenum, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, frame_num, window_size, window_size, C)
    """
    B, framenum, H, W, C = x.shape
    x = x.view(B, framenum, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(-1, framenum, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, frame_num, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, frame_num, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    framenum = windows.shape[1]
    x = windows.view(B, H // window_size, W // window_size, framenum, window_size, window_size, -1)
    x = x.permute(0, 3, 1, 4, 2, 5, 6).contiguous().view(B, framenum, H, W, -1)
    return x

class WindowAttentionMixin(BaseMixin):
    def __init__(self, num_layers,
                hidden_size, 
                frame_resolution,
                window_size,
                shift_size,
                n_head,
                frame_num, 
                init_method=unscaled_init_method(0.02),
                output_layer_init_method=unscaled_init_method(0.02),
        ):
        super(WindowAttentionMixin, self).__init__()
        self.num_layers = num_layers # replace attention in the LAST n layers
        self.query_key_value = torch.nn.ModuleList(
            [ColumnParallelLinear(hidden_size, 3*hidden_size,stride=3,
                gather_output=False,init_method=init_method)
                for layer_id in range(num_layers)
            ])
        self.dense = torch.nn.ModuleList(
            [RowParallelLinear(
                hidden_size,
                hidden_size,
                input_is_parallel=True,
                init_method=output_layer_init_method,
                bias=True,
                module=self,
                name="dense",
                ) 
                for layer_id in range(num_layers)
            ])

        self.n_head = n_head
        self.window_size = window_size
        self.frame_resolution = frame_resolution
        self.frame_len = frame_resolution * frame_resolution
        assert frame_resolution % window_size == 0
        assert 0 < shift_size < window_size
        nW = (self.frame_resolution // self.window_size) ** 2
        ws_squre = self.window_size * self.window_size
        
        # odd non-shift, even shift
        img_mask = torch.zeros((1, 1, frame_resolution, frame_resolution, 1))
        h_slices = (slice(0, -shift_size),
                    slice(-shift_size, None))
        w_slices = (slice(0, -shift_size),
                    slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, :, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)  # nW, 1, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        sub_attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) #[nW, self.window_size * self.window_size, self.window_size * self.window_size]
        sub_attn_mask = sub_attn_mask.masked_fill(sub_attn_mask != 0, float(0.0)).masked_fill(sub_attn_mask == 0, float(1.00))
        attn_mask = sub_attn_mask.repeat(1, frame_num, frame_num)
        
        self.attn_mask_sequential = attn_mask.clone().tril()
        self.causal_mask_sequential = torch.ones(1, ws_squre*frame_num, ws_squre*frame_num).tril()
        
        self.causal_mask_interp = torch.ones(1, ws_squre*frame_num, ws_squre*frame_num)
        self.attn_mask_interp = attn_mask.clone()

        # bi-dir 
        for bi_idx in range(0, frame_num, 2):
            for uni_idx in range(1, frame_num, 2):
                self.attn_mask_interp[:, bi_idx*ws_squre:(bi_idx+1)*ws_squre, uni_idx*ws_squre:(uni_idx+1)*ws_squre] = 0
                self.causal_mask_interp[:, bi_idx*ws_squre:(bi_idx+1)*ws_squre, uni_idx*ws_squre:(uni_idx+1)*ws_squre] = 0
        # uni-dir
        for uni_idx in range(1, frame_num, 2):
            self.attn_mask_interp[:, ws_squre*uni_idx:ws_squre*(uni_idx+1), ws_squre*uni_idx:ws_squre*(uni_idx+1)].tril_()
            self.causal_mask_interp[:, ws_squre*uni_idx:ws_squre*(uni_idx+1), ws_squre*uni_idx:ws_squre*(uni_idx+1)].tril_()
            for uni_idx2 in range(uni_idx+2, frame_num, 2):
                self.attn_mask_interp[:, ws_squre*uni_idx:ws_squre*(uni_idx+1), ws_squre*uni_idx2:ws_squre*(uni_idx2+1)] = 0
                self.causal_mask_interp[:, ws_squre*uni_idx:ws_squre*(uni_idx+1), ws_squre*uni_idx2:ws_squre*(uni_idx2+1)] = 0

        # expand dim
        self.attn_mask_sequential = self.attn_mask_sequential[None, None, :, None]
        self.attn_mask_interp = self.attn_mask_interp[None, None, :, None]
        self.causal_mask_sequential = self.causal_mask_sequential[None, None, :, None]
        self.causal_mask_interp = self.causal_mask_interp[None, None, :, None]
                
        self.shift_sizes = [0, shift_size]
        # self.register_buffer("attn_mask", attn_mask)
        # self.register_buffer("causal_mask", causal_mask)
        self.mask_initialized = False
        
        self.attn_distribution = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(hidden_size))
            for _ in range(num_layers)
        ])
        
    def reinit(self, *pre_mixins):
        start_layer = len(self.transformer.layers) - self.num_layers
        assert start_layer >= 0
        for layer_id in range(self.num_layers):
            old_attention = self.transformer.layers[start_layer + layer_id].attention
            self.query_key_value[layer_id].weight.data.copy_(old_attention.query_key_value.weight.data)
            self.query_key_value[layer_id].bias.data.copy_(old_attention.query_key_value.bias.data)
            
    def attention_extra(self, frame_hidden_state, layer_id, attn_dropout, text_hidden_state=None, 
                       text_attn_mask=None, mode_sequential=True):
        # pb relax 
        swin_pb_relax = True
        alpha = 16
        
        # frame_hidden_state [batchsize, frame_num*frame_size, n_head*hiddensize_perhead]
        if not self.mask_initialized:
            self.attn_mask_sequential = self.attn_mask_sequential.to(device=frame_hidden_state.device, dtype=frame_hidden_state.dtype)
            self.causal_mask_sequential = self.causal_mask_sequential.to(device=frame_hidden_state.device, dtype=frame_hidden_state.dtype)
            self.attn_mask_interp = self.attn_mask_interp.to(device=frame_hidden_state.device, dtype=frame_hidden_state.dtype)
            self.causal_mask_interp = self.causal_mask_interp.to(device=frame_hidden_state.device, dtype=frame_hidden_state.dtype)
            self.mask_initialized = True
        b0, s1, h0 = frame_hidden_state.shape 
        h = h0 // self.n_head
        frame_len = self.frame_resolution * self.frame_resolution
        frame_num = s1 // frame_len
        assert frame_num*frame_len == s1
        wind_square = self.window_size * self.window_size
        nW = frame_len // wind_square
        bswin = b0 * nW
        
        causal_mask = self.causal_mask_sequential if mode_sequential else self.causal_mask_interp
        attn_mask = self.attn_mask_sequential if mode_sequential else self.attn_mask_interp
        if text_hidden_state is not None:
            s0 = text_hidden_state.shape[1]
            qkv_text = self.query_key_value[layer_id](text_hidden_state).reshape(b0, s0, 3, self.n_head, h).permute(2, 0, 3, 1, 4) #[3, b0, n_head, s0, h]
            q_text, k_text, v_text = qkv_text[0], qkv_text[1], qkv_text[2]
            
        # shift
        frame_hidden_state = frame_hidden_state.reshape(b0, frame_num, self.frame_resolution, self.frame_resolution, h0)
        if self.shift_sizes[layer_id%2] > 0:
            frame_hidden_state = torch.roll(frame_hidden_state, shifts=(-self.shift_sizes[layer_id%2], -self.shift_sizes[layer_id%2]), dims=(2,3))
        # window partition    
        frame_hidden_state = window_partition(frame_hidden_state, self.window_size).reshape(bswin, frame_num*wind_square, h0)
        qkv = self.query_key_value[layer_id](frame_hidden_state).reshape(bswin, frame_num*wind_square, 3, self.n_head, h)\
                .permute(2, 0, 3, 1, 4) #[3, bswin, n_head, frame_num*wind_size*wind_size, h]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # pb-relax
        if swin_pb_relax:
            attn = torch.matmul(q / (math.sqrt(h)*alpha), k.transpose(-1, -2))
        else: 
            attn = torch.matmul(q / math.sqrt(h), k.transpose(-1, -2))

        if self.shift_sizes[layer_id%2] > 0:
            # attn = attn.view(bswin // nW, nW, self.n_head, frame_num*wind_square, frame_num*wind_square) + self.attn_mask.unsqueeze(1).unsqueeze(0)
            attn = torch.mul(attn.view(bswin // nW, nW, self.n_head, frame_num*wind_square, frame_num*wind_square), attn_mask)\
                 - 10000.0 * (1.0 - attn_mask)
            attn = attn.view(bswin, self.n_head, frame_num*wind_square, frame_num*wind_square)
        else:
            attn = torch.mul(attn.view(bswin // nW, nW, self.n_head, frame_num*wind_square, frame_num*wind_square), causal_mask)\
                 - 10000.0 * (1.0 - causal_mask)
            attn = attn.view(bswin, self.n_head, frame_num*wind_square, frame_num*wind_square)
        if swin_pb_relax:
            swin_pb_relax_const = torch.max(attn.reshape(bswin, self.n_head, -1), dim=-1, keepdim=True)[0].detach().unsqueeze(-1)
            attn = (attn - swin_pb_relax_const)*alpha
            
        if text_hidden_state is None:
            attn = F.softmax(attn, dim=-1)
            if attn_dropout is not None:
                with get_cuda_rng_tracker().fork():
                    attn = attn_dropout(attn)
            context_swin = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(bswin, frame_num, self.window_size, self.window_size, h0)
        else:
            assert text_attn_mask is not None
            text_attn_mask = text_attn_mask.unsqueeze(2).unsqueeze(2)
            # pb-relax
            if swin_pb_relax:
                attn_frame2text = torch.matmul(q.reshape(b0, -1, self.n_head, frame_num*wind_square, h) / (math.sqrt(h)*alpha), k_text.unsqueeze(1).transpose(-1, -2))
                attn_frame2text = (attn_frame2text-swin_pb_relax_const.reshape(b0, -1, self.n_head, 1, 1))*alpha
            else:
                attn_frame2text = torch.matmul(q.reshape(b0, -1, self.n_head, frame_num*wind_square, h) / math.sqrt(h), k_text.unsqueeze(1).transpose(-1, -2))

            attn_frame2text = torch.mul(text_attn_mask, attn_frame2text) - 10000.0 * (1.0 - text_attn_mask)
            attn_frame2text = attn_frame2text.reshape(bswin, self.n_head, frame_num*wind_square, s0)
            attn = torch.cat((attn, attn_frame2text), dim=-1)
            attn = F.softmax(attn, dim=-1)
            
            if attn_dropout is not None:
                with get_cuda_rng_tracker().fork():
                    attn = attn_dropout(attn)
                    
            context_swin = (torch.matmul(attn[..., :-s0], v) + 
                            torch.matmul(attn[..., -s0:].reshape(b0, -1, self.n_head,frame_num*wind_square, s0), v_text.unsqueeze(1))\
                                .reshape(bswin, self.n_head, frame_num*wind_square, h))\
                .permute(0, 2, 1, 3).reshape(bswin, frame_num, self.window_size, self.window_size, h0)
                
        context_swin = window_reverse(context_swin, self.window_size, self.frame_resolution, self.frame_resolution)
        # reverse cycle shift
        if self.shift_sizes[layer_id%2] > 0:
            context_swin = torch.roll(context_swin, shifts=(self.shift_sizes[layer_id%2], self.shift_sizes[layer_id%2]), dims=(2,3))
        context_swin = context_swin.reshape(b0, s1, h0)

        return context_swin
    

class FullAttentionMixin(BaseMixin):
    def __init__(self, num_layers,
                hidden_size, 
                frame_resolution,
                n_head,
                frame_num, 
                init_method=unscaled_init_method(0.02),
                output_layer_init_method=unscaled_init_method(0.02),
        ):
        super(FullAttentionMixin, self).__init__()
        self.num_layers = num_layers # replace attention in the LAST n layers
        self.query_key_value = torch.nn.ModuleList(
            [ColumnParallelLinear(hidden_size, 3*hidden_size,stride=3,
                gather_output=False,init_method=init_method) 
                for layer_id in range(num_layers)
            ])
        self.dense = torch.nn.ModuleList(
            [RowParallelLinear(
                hidden_size,
                hidden_size,
                input_is_parallel=True,
                init_method=output_layer_init_method,
                bias=True,
                module=self,
                name="dense",)
                for layer_id in range(num_layers)
            ])

        self.n_head = n_head
        self.frame_resolution = frame_resolution
        self.frame_len = frame_resolution * frame_resolution
        self.causal_mask = torch.ones(1, 1, self.frame_len*frame_num, self.frame_len*frame_num).tril()
        
        self.mask_initialized = False
        
        self.attn_distribution = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(hidden_size))
            for _ in range(num_layers)
        ])
        
    def reinit(self, *pre_mixins):
        start_layer = len(self.transformer.layers) - self.num_layers
        assert start_layer >= 0
        for layer_id in range(self.num_layers):
            base_attention = self.transformer.layers[start_layer + layer_id].attention
            self.query_key_value[layer_id].weight.data.copy_(base_attention.query_key_value.weight.data)
            self.query_key_value[layer_id].bias.data.copy_(base_attention.query_key_value.bias.data)
    
    def attention_extra(self, frame_hidden_state, layer_id, attn_dropout, text_hidden_state=None, 
                       text_attn_mask=None, mode_sequential=False):
        # pb relax
        # frame_hidden_state [batchsize, frame_num*frame_size, n_head*hiddensize_perhead]
        assert mode_sequential == True # only 
        swin_pb_relax = True
        alpha = 16
        
        if not self.mask_initialized:
            self.causal_mask = self.causal_mask.to(device=frame_hidden_state.device, dtype=frame_hidden_state.dtype)
            self.mask_initialized = True
        b0, s1, h0 = frame_hidden_state.shape 
        h = h0 // self.n_head
        frame_len = self.frame_resolution * self.frame_resolution
        frame_num = s1 // frame_len
        assert frame_num*frame_len == s1
            
        qkv = self.query_key_value[layer_id](frame_hidden_state).reshape(b0, s1, 3, self.n_head, h)\
                .permute(2, 0, 3, 1, 4) #[3, b0, n_head, s1, h]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # frames-to-frames 
        if swin_pb_relax:
            attn = torch.matmul(q / (math.sqrt(h)*alpha), k.transpose(-1, -2))
        else: 
            attn = torch.matmul(q / math.sqrt(h), k.transpose(-1, -2))
        attn = torch.mul(attn, self.causal_mask) - 10000.0 * (1.0 - self.causal_mask)
        if swin_pb_relax:
            swin_pb_relax_const = torch.max(attn.reshape(b0, self.n_head, -1), dim=-1, keepdim=True)[0].detach().unsqueeze(-1)
            attn = (attn - swin_pb_relax_const)*alpha

        if text_hidden_state is None:
            attn = F.softmax(attn, dim=-1)
            if attn_dropout is not None:
                with get_cuda_rng_tracker().fork():
                    attn = attn_dropout(attn)
            context_swin = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(b0, s1, h0)
        else:
            # frame-to-text
            assert text_attn_mask is not None
            s0 = text_hidden_state.shape[1]
            qkv_text = self.query_key_value[layer_id](text_hidden_state).reshape(b0, s0, 3, self.n_head, h).permute(2, 0, 3, 1, 4) #[3, b0, n_head, s0, h]
            q_text, k_text, v_text = qkv_text[0], qkv_text[1], qkv_text[2]
            text_attn_mask = text_attn_mask.unsqueeze(2)
            if swin_pb_relax:
                attn_frame2text = torch.matmul(q.reshape(b0, self.n_head, s1, h) / (math.sqrt(h)*alpha), k_text.transpose(-1, -2))
                attn_frame2text = (attn_frame2text-swin_pb_relax_const.reshape(b0, self.n_head, 1, 1))*alpha
            else:
                attn_frame2text = torch.matmul(q.reshape(b0, self.n_head, s1, h) / math.sqrt(h), k_text.transpose(-1, -2))
            attn_frame2text = torch.mul(text_attn_mask, attn_frame2text) - 10000.0 * (1.0 - text_attn_mask)
            attn_frame2text = attn_frame2text.reshape(b0, self.n_head, s1, s0)
            
            attn = torch.cat((attn, attn_frame2text), dim=-1)
            attn = F.softmax(attn, dim=-1)
            
            if attn_dropout is not None:
                with get_cuda_rng_tracker().fork():
                    attn = attn_dropout(attn)
                    
            context_frame = (torch.matmul(attn[..., :-s0], v) + 
                            torch.matmul(attn[..., -s0:].reshape(b0, self.n_head,s1, s0), v_text))\
                .permute(0, 2, 1, 3).reshape(b0, s1, h0)
                
        return context_frame
        

def attention_localframe_and_text(q0, k0, v0, attention_mask_totxt, attention_mask_local, 
                             n_head, text_len, frame_len, frame_num, attention_dropout=None, layer_id=0, **kwargs):
    b, s0, h0 = q0.shape
    s1 = s0 - text_len
    h = h0 // n_head
    assert q0.shape[1] == v0.shape[1] == k0.shape[1] == text_len+frame_len*frame_num
    # attention_mask_totxt [b, 1, 1, text_len]
    # attention_mask_local [1, 1, frame_num, frame_len, frame_len]
    # attention_mask: [1, 1, text_len+frame_len, text_len+frame_len]

    q0 = q0.reshape(b, s0, n_head, h).permute(0, 2, 1, 3)
    v0 = v0.reshape(b, s0, n_head, h).permute(0, 2, 1, 3)
    k0 = k0.reshape(b, s0, n_head, h).permute(0, 2, 1, 3)
    k0T = k0.transpose(-1, -2)

    # score: any2text
    score_any2text = torch.matmul(q0 / math.sqrt(q0.shape[-1]), k0T[..., :text_len])
    score_any2text_part1 = torch.mul(score_any2text[..., :text_len, :], attention_mask_totxt) \
        - 10000.0 * (1.0 - attention_mask_totxt)
    score_any2text_part2 = torch.mul(score_any2text[..., text_len:, :], attention_mask_totxt) - \
                                     10000.0 * (1.0 - attention_mask_totxt)
    
    # score: frame local
    q0_frame = q0[:, :, text_len:].reshape(b, n_head, frame_num, frame_len, h)
    v0_frame = v0[:, :, text_len:].reshape(b, n_head, frame_num, frame_len, h)
    k0T_frame = k0[:, :, text_len:].reshape(b, n_head, frame_num, frame_len, h).transpose(-1, -2)
    score_frame_local0 = torch.matmul(q0_frame / math.sqrt(q0_frame.shape[-1]), k0T_frame)
    score_frame_local0 = torch.mul(score_frame_local0, attention_mask_local) \
        - 10000.0 * (1.0 - attention_mask_local)
    
    # context for frame
    score_frame_all = torch.cat((score_any2text_part2, 
                                 score_frame_local0.view(b, n_head, s1, frame_len)), dim=-1)
    attention_probs_frame = F.softmax(score_frame_all, dim=-1)
    
    if attention_dropout is not None:
        with get_cuda_rng_tracker().fork():
            attention_probs_frame = attention_dropout(attention_probs_frame)
            
    context_frame2text = torch.matmul(attention_probs_frame[..., :text_len], v0[..., :text_len, :]) # [b, n_head, s1, h]
    context_frame_local0 = torch.matmul(attention_probs_frame[..., text_len:text_len+frame_len].\
        view(b, n_head, frame_num, frame_len, frame_len), v0_frame).view(b, n_head, s1, h)
    context_frame = (context_frame2text + context_frame_local0).transpose(1, 2).reshape(b, s1, h0)

    # context for text
    attention_probs_text = F.softmax(score_any2text_part1, dim=-1)
    if attention_dropout is not None:
        with get_cuda_rng_tracker().fork():
            attention_probs_text = attention_dropout(attention_probs_text)
    context_text2text = torch.matmul(attention_probs_text, v0[..., :text_len, :])
    context_text2text = context_text2text.transpose(1, 2).reshape(b, text_len, h0)
    
    return context_text2text, context_frame
    
    
class CogVideoModel(BaseModel):
    def __init__(self, args, transformer=None, parallel_output=True):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output)
        self.stage = args.cogvideo_stage # 1 or 2
        self.mode_sequential = True if self.stage==1 else False
        self.layout = args.layout # [64, 64+400, 64+5*400]
        self.n_head = args.num_attention_heads
        frame_resolution = int(math.sqrt(self.layout[1]-self.layout[0]))
        frame_num = (args.layout[2]-args.layout[0])//(args.layout[1]-args.layout[0])
        frame_len = self.layout[1]-self.layout[0]
        
        self.add_mixin('extra_position_embedding', PositionEmbeddingMixin(
            args.additional_seqlen, args.hidden_size
        ))
        
        if args.window_size == -1:
            # full attention
            assert self.stage == 1
            self.add_mixin('attention_plus', FullAttentionMixin(
                num_layers=args.num_layers,
                hidden_size=args.hidden_size,
                frame_resolution=frame_resolution,
                n_head=args.num_attention_heads,
                frame_num=frame_num,
            ))
        else:
            self.add_mixin('attention_plus', WindowAttentionMixin(
                num_layers=args.num_layers,
                hidden_size=args.hidden_size,
                frame_resolution=frame_resolution,
                window_size=args.window_size,
                shift_size=args.window_size//2,
                n_head=args.num_attention_heads,
                frame_num=frame_num,
            ))
        # attention_mask_local
        self.attention_mask_local_sequential = torch.ones(1, 1, frame_num, frame_len, frame_len).tril().unsqueeze(0)
        self.attention_mask_local_interp = torch.ones(1, 1, frame_num, frame_len, frame_len)

        for idx in range(1, frame_num, 2):
            self.attention_mask_local_interp[:, :, idx:idx+1].tril_()
        self.attention_mask_local_interp = self.attention_mask_local_interp.unsqueeze(0)
        self.mask_initialized = False
        
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('CogVideoModel', 'CogVideo model configurations')
        group.add_argument("--layout", type=str, default='64, 464, 2064', help='text_len, textlen+frame_len, textlen+frame_len*frame_num')
        group.add_argument("--window-size", type=int, default=10, help="swin attention's window size in temperal channel, -1 represents full attention")
        group.add_argument("--additional-seqlen", type=int, default=2000)
        group.add_argument("--cogvideo-stage", type=int, default=1, choices=[1,2])
        return parser
    
    def disable_untrainable_params(self):
        self.transformer.requires_grad_(False)
        
    def position_embedding_forward(self, position_ids, **kw_args):
        position = position_ids[..., :(64+400)]
        position_plus = position_ids[..., (64+400):]
        position_embeddings = torch.cat(
            (
                self.transformer.position_embeddings(position),
                self.get_mixin('extra_position_embedding').position_embeddings(position_plus-(512+400))
            ),
            dim=-2
        )
        return position_embeddings
        
    def attention_forward(self, hidden_states, mask, layer_id, **kw_args):
        # mask.shape=[bs, 1, 1, 64]
        if not self.mask_initialized:
            self.attention_mask_local_sequential = self.attention_mask_local_sequential.to(device=hidden_states.device, dtype=hidden_states.dtype)
            self.attention_mask_local_interp = self.attention_mask_local_interp.to(device=hidden_states.device, dtype=hidden_states.dtype)
            self.mask_initialized = True
        
        attn_module = self.transformer.layers[layer_id].attention
        hidden_size = hidden_states.shape[-1]
        bs = hidden_states.shape[0]
        
        # base model qkv
        mixed_raw_layer = attn_module.query_key_value(hidden_states)
        q0, k0, v0 = split_tensor_along_last_dim(mixed_raw_layer, 3)
        dropout_fn = self.transformer.layers[layer_id].attention.attention_dropout if self.training else None

        attention_mask_local = self.attention_mask_local_sequential if self.mode_sequential else self.attention_mask_local_interp
        context_text, context_frame_local_text = attention_localframe_and_text(
                q0, k0, v0,
                attention_mask_totxt=mask,
                attention_mask_local=attention_mask_local,
                n_head=attn_module.num_attention_heads_per_partition,
                text_len=self.layout[0],
                frame_len=self.layout[1]-self.layout[0],
                frame_num=(self.layout[2]-self.layout[0])//(self.layout[1]-self.layout[0]),
                attention_dropout=dropout_fn, 
                layer_id=layer_id,
            )

        context_frame_swin = self.get_mixin('attention_plus').attention_extra(
            hidden_states[:, self.layout[0]:], layer_id, dropout_fn, 
            text_hidden_state=hidden_states[:, :self.layout[0]], 
            text_attn_mask=mask[..., 0, :],
            mode_sequential=self.mode_sequential)
            
        attn_distrib = torch.sigmoid(self.get_mixin('attention_plus').attn_distribution[layer_id])
        attn_distrib = attn_distrib.unsqueeze(0).unsqueeze(0)
        
        output_text = attn_module.dense(context_text)
        output_frame = torch.mul(attn_module.dense(context_frame_local_text), attn_distrib)\
            +torch.mul(self.get_mixin('attention_plus').dense[layer_id](context_frame_swin), 1-attn_distrib)
        output = torch.cat((output_text, output_frame), dim=-2)

        return output