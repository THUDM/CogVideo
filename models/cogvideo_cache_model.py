# -*- encoding: utf-8 -*-
'''
@File    :   cogvideo_cache_model.py
@Time    :   2022/07/15 11:22:19
@Author  :   Wenyi Hong 
@Version :   1.0
@Contact :   hwy22@mails.tsinghua.edu.cn
'''

# here put the import lib

from multiprocessing import context
from tkinter import E
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
                time_dim_attend_length=0
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
                name="dense")
                for layer_id in range(num_layers)
            ])

        self.n_head = n_head
        self.window_size = window_size
        self.frame_resolution = frame_resolution
        self.frame_len = frame_resolution * frame_resolution
        self.time_dim_attend_length = time_dim_attend_length
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
        attn_mask = attn_mask.tril()
        
        causal_mask = torch.ones(ws_squre*frame_num, ws_squre*frame_num)
        causal_mask = causal_mask.tril()

        self.shift_sizes = [0, shift_size]
        self.attn_mask = attn_mask
        self.causal_mask = causal_mask
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
            
    def attention_extra_NAR_inference(self, frame_hidden_state, layer_id, attn_dropout=None, memkv_text=None, stage=1):
        # frame_hidden_state [batchsize, frame_num*frame_size, n_head*hiddensize_perhead]
        if not self.mask_initialized:
            self.attn_mask = self.attn_mask.to(device=frame_hidden_state.device, dtype=frame_hidden_state.dtype)
            self.causal_mask = self.causal_mask.to(device=frame_hidden_state.device, dtype=frame_hidden_state.dtype)
            self.mask_initialized = True
        b0, s1, h0 = frame_hidden_state.shape 
        h = h0 // self.n_head
        frame_len = self.frame_resolution * self.frame_resolution
        frame_num = s1 // frame_len
        if stage == 2:
            assert frame_num == 3
        assert frame_num*frame_len == s1
        wind_square = self.window_size * self.window_size
        nW = frame_len // wind_square
        bswin = b0 * nW
        
        if memkv_text is not None:
            s0 = memkv_text.shape[-2]
            k_text = memkv_text[..., :h0].expand(b0, -1, -1).reshape(b0, s0, self.n_head, h).permute(0, 2, 1, 3)
            v_text = memkv_text[..., h0:].expand(b0, -1, -1).reshape(b0, s0, self.n_head, h).permute(0, 2, 1, 3)
        
        # shift
        frame_hidden_state = frame_hidden_state.reshape(b0, frame_num, self.frame_resolution, self.frame_resolution, h0)
        if self.shift_sizes[layer_id%2] > 0:
            frame_hidden_state = torch.roll(frame_hidden_state, shifts=(-self.shift_sizes[layer_id%2], -self.shift_sizes[layer_id%2]), dims=(2,3))
        # window partition    
        frame_hidden_state = window_partition(frame_hidden_state, self.window_size).reshape(bswin, frame_num*wind_square, h0)
        qkv = self.query_key_value[layer_id](frame_hidden_state).reshape(bswin, frame_num*wind_square, 3, self.n_head, h)\
                .permute(2, 0, 3, 1, 4) #[3, bswin, n_head, frame_num*wind_size*wind_size, h]
        q, k, v = qkv[0], qkv[1], qkv[2] 
        attn = torch.matmul(q / math.sqrt(h), k.transpose(-1, -2))
        
        if stage == 1:
            if self.shift_sizes[layer_id%2] > 0:
                attn = torch.mul(attn.view(bswin // nW, nW, self.n_head, frame_num*wind_square, frame_num*wind_square),
                                self.attn_mask[:,:frame_num*wind_square, :frame_num*wind_square].unsqueeze(1).unsqueeze(0))\
                    - 10000.0 * (1.0 - self.attn_mask[:,:frame_num*wind_square, :frame_num*wind_square].unsqueeze(1).unsqueeze(0))
                attn = attn.view(bswin, self.n_head, frame_num*wind_square, frame_num*wind_square)
            else:
                attn = torch.mul(attn, self.causal_mask[:frame_num*wind_square, :frame_num*wind_square].unsqueeze(0).unsqueeze(0))\
                    - 10000.0 * (1.0 - self.causal_mask[:frame_num*wind_square, :frame_num*wind_square].unsqueeze(0).unsqueeze(0))
        
        if memkv_text is None:
            attn = F.softmax(attn, dim=-1)
            if attn_dropout is not None:
                with get_cuda_rng_tracker().fork():
                    attn = attn_dropout(attn)
            context_swin = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(bswin, frame_num, self.window_size, self.window_size, h0)
        else:
            attn_frame2text = torch.matmul(q.reshape(b0, -1, self.n_head, frame_num*wind_square, h) / math.sqrt(h), k_text.unsqueeze(1).transpose(-1, -2))
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
        ret_context = context_swin.reshape(b0, s1, h0)
    
        # for mem
        memk = k.permute(0, 2, 1, 3).reshape(bswin, frame_num, self.window_size, self.window_size, h0)
        memv = v.permute(0, 2, 1, 3).reshape(bswin, frame_num, self.window_size, self.window_size, h0)
        memk = window_reverse(memk, self.window_size, self.frame_resolution, self.frame_resolution)
        memv = window_reverse(memv, self.window_size, self.frame_resolution, self.frame_resolution)
        if self.shift_sizes[layer_id%2] > 0:
            memk = torch.roll(memk, shifts=(self.shift_sizes[layer_id%2], self.shift_sizes[layer_id%2]), dims=(2,3))
            memv = torch.roll(memv, shifts=(self.shift_sizes[layer_id%2], self.shift_sizes[layer_id%2]), dims=(2,3))
        memk, memv = memk.reshape(b0, s1, h0), memv.reshape(b0, s1, h0)
                
        ret_mem = torch.cat((memk, memv), dim=-1)
        return ret_context, ret_mem
        
    def attention_extra_AR_inference(self, frame_hidden_state, memkv, pos, layer_id, log_text_attention_weights=0, attn_dropout=None, memkv_text=None, stage=1):
        # frame_hidden_state [batchsize, 1, n_head*hiddensize_perhead]
        # memkv [batchsize, pos, hidden_size*2] (include frames only)
        # if memkv_text is not None: will attend to text
        # pos: token's pos
        b0, sin, h0 = frame_hidden_state.shape 
        h = h0 // self.n_head
        assert sin == 1
        this_qkv = self.query_key_value[layer_id](frame_hidden_state)
        thisq, thisk, thisv = this_qkv[..., :h0], this_qkv[..., h0:2*h0], this_qkv[..., 2*h0:]
        s1 = memkv.shape[1] if memkv is not None else 0
        frame_len = self.frame_resolution * self.frame_resolution
        frame_num_before = s1 // frame_len
        
        
        if memkv is not None:
            pos_inframe = pos - frame_num_before * frame_len
            
            xpos = pos_inframe // self.frame_resolution # pos = xpos*self.frame_resolution + ypos
            ypos = pos_inframe % self.frame_resolution
            # [start, end)
            if self.shift_sizes[layer_id%2] > 0:
                xstart = ((xpos+self.shift_sizes[layer_id%2]) // self.window_size) * self.window_size - self.shift_sizes[layer_id%2]
                ystart = ((ypos+self.shift_sizes[layer_id%2]) // self.window_size) * self.window_size - self.shift_sizes[layer_id%2]
                xend = xstart + self.window_size
                yend = ystart + self.window_size
                xstart, ystart = max(0, xstart), max(0, ystart)
                xend, yend = min(xend, self.frame_resolution), min(yend, self.frame_resolution)
            else:
                xstart = (xpos // self.window_size) * self.window_size
                ystart = (ypos // self.window_size) * self.window_size
                xend, yend = xstart + self.window_size, ystart+self.window_size
            
            # select index 
            selected_index = list()
            if frame_num_before > 0:
                # frames before
                frame_attended_start = max(0, frame_num_before-self.time_dim_attend_length+1) if self.time_dim_attend_length > 0 else 0
                for x in range(xstart, xend):
                    for y in range(ystart, yend):
                        selected_index.append(x*self.frame_resolution+y+frame_len*frame_attended_start)
                cnt_per_frame = len(selected_index)
                for _ in range((frame_num_before-frame_attended_start-1)*cnt_per_frame):
                    selected_index.append(selected_index[-cnt_per_frame]+frame_len)
                    
            # the last frame
            for x in range(xstart, xend):
                for y in range(ystart, yend):
                    tmppos = x*self.frame_resolution+y + frame_num_before * frame_len
                    if tmppos < pos:
                        selected_index.append(tmppos)
                    else:
                        break
            cnt_all = len(selected_index)+1
            selected_index = torch.tensor(selected_index, device=memkv.device)
            used_memkv = torch.index_select(memkv, 1, selected_index)
            used_k, used_v = used_memkv[..., :h0], used_memkv[..., h0:]
            used_k = torch.cat((used_k.expand(thisk.shape[0], -1, -1), thisk), dim=-2)
            used_v = torch.cat((used_v.expand(thisv.shape[0], -1, -1), thisv), dim=-2)
            if memkv_text is not None:
                cnt_all += memkv_text.shape[-2]
                used_k = torch.cat((memkv_text[..., :h0].expand(thisk.shape[0], -1, -1), used_k), dim=-2)
                used_v = torch.cat((memkv_text[..., h0:].expand(thisv.shape[0], -1, -1), used_v), dim=-2)
            used_k = used_k.reshape(b0, cnt_all, self.n_head, h).permute(0, 2, 1, 3)
            used_v = used_v.reshape(b0, cnt_all, self.n_head, h).permute(0, 2, 1, 3)
        else: 
            used_k = thisk
            used_v = thisv
            
            if memkv_text is not None:
                used_k = torch.cat((memkv_text[..., :h0].expand(thisk.shape[0], -1, -1), used_k), dim=-2)
                used_v = torch.cat((memkv_text[..., h0:].expand(thisv.shape[0], -1, -1), used_v), dim=-2)
                used_k = used_k.reshape(b0, 1+memkv_text.shape[-2], self.n_head, h).permute(0, 2, 1, 3)
                used_v = used_v.reshape(b0, 1+memkv_text.shape[-2], self.n_head, h).permute(0, 2, 1, 3)
            else:
                used_k = used_k.reshape(b0, 1, self.n_head, h).permute(0, 2, 1, 3)
                used_v = used_v.reshape(b0, 1, self.n_head, h).permute(0, 2, 1, 3)
        
        thisq = thisq.reshape(b0, 1, self.n_head, h).permute(0, 2, 1, 3) # [b0, n_head, 1, h]
        attn = torch.matmul(thisq / math.sqrt(h), used_k.transpose(-1, -2))
        if memkv_text is not None:
            attn[..., :memkv_text.shape[-2]] += log_text_attention_weights
        attn = F.softmax(attn, dim=-1)
        context_swin = torch.matmul(attn, used_v).permute(0, 2, 1, 3).reshape(b0, 1, h0)

        return context_swin, this_qkv[..., h0:]
        
class FullAttentionMixin(BaseMixin):
    def __init__(self, num_layers,
                hidden_size, 
                frame_resolution,
                n_head,
                frame_num, 
                init_method=unscaled_init_method(0.02),
                output_layer_init_method=unscaled_init_method(0.02),
                **kwargs,
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
                name="dense")
                for layer_id in range(num_layers)
            ])

        self.n_head = n_head
        self.frame_resolution = frame_resolution
        self.frame_len = frame_resolution * frame_resolution
        
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
            
            
    def attention_extra_NAR_inference(self, frame_hidden_state, layer_id, attn_dropout=None, memkv_text=None, stage=1):
        # frame_hidden_state [batchsize, frame_num*frame_size, n_head*hiddensize_perhead]
        assert stage == 1
        
        b0, s1, h0 = frame_hidden_state.shape 
        h = h0 // self.n_head
        frame_len = self.frame_resolution * self.frame_resolution
        frame_num = s1 // frame_len
        assert frame_num*frame_len == s1

        if memkv_text is not None:
            s0 = memkv_text.shape[-2]
            k_text = memkv_text[..., :h0].expand(b0, -1, -1).reshape(b0, s0, self.n_head, h).permute(0, 2, 1, 3)
            v_text = memkv_text[..., h0:].expand(b0, -1, -1).reshape(b0, s0, self.n_head, h).permute(0, 2, 1, 3)
        qkv = self.query_key_value[layer_id](frame_hidden_state).reshape(b0, s1, 3, self.n_head, h)\
                .permute(2, 0, 3, 1, 4) #[3, b0, n_head, s1, h]
        q, k, v = qkv[0], qkv[1], qkv[2] 
        attn = torch.matmul(q / math.sqrt(h), k.transpose(-1, -2))
        attn = attn - 10000.0 * (1.0-torch.ones(b0, self.n_head, s1, s1, device=attn.device, dtype=attn.dtype).tril())
                
        if memkv_text is None:
            attn = F.softmax(attn, dim=-1)
            if attn_dropout is not None:
                with get_cuda_rng_tracker().fork():
                    attn = attn_dropout(attn)
            context_swin = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(b0, s1, h0)
        else:
            attn_frame2text = torch.matmul(q / math.sqrt(h), k_text.transpose(-1, -2)) #[b0, s1, s0]
            attn = torch.cat((attn, attn_frame2text), dim=-1)
            attn = F.softmax(attn, dim=-1)
            if attn_dropout is not None:
                with get_cuda_rng_tracker().fork():
                    attn = attn_dropout(attn)
            context_swin = (torch.matmul(attn[..., :-s0], v) + torch.matmul(attn[..., -s0:], v_text))\
                .permute(0, 2, 1, 3).reshape(b0, s1, h0)
                    
        # for mem
        memk = k.permute(0, 2, 1, 3).reshape(b0, s1, h0)
        memv = v.permute(0, 2, 1, 3).reshape(b0, s1, h0)
        ret_mem = torch.cat((memk, memv), dim=-1)
        
        return context_swin, ret_mem
        
    def attention_extra_AR_inference(self, frame_hidden_state, memkv, pos, layer_id, log_text_attention_weights=0, attn_dropout=None, memkv_text=None, stage=1):
        # pos: current token's pos
        b0, sin, h0 = frame_hidden_state.shape 
        h = h0 // self.n_head
        assert sin == 1
        assert stage == 1
        
        this_qkv = self.query_key_value[layer_id](frame_hidden_state)
        thisq, thisk, thisv = this_qkv[..., :h0], this_qkv[..., h0:2*h0], this_qkv[..., 2*h0:]
        
        if memkv is not None:
            used_k, used_v = memkv[..., :h0], memkv[..., h0:]
            used_k = torch.cat((used_k.expand(thisk.shape[0], -1, -1), thisk), dim=-2)
            used_v = torch.cat((used_v.expand(thisv.shape[0], -1, -1), thisv), dim=-2)
        else: 
            used_k, used_v = thisk, thisv
            
        if memkv_text is not None:
            used_k = torch.cat((memkv_text[..., :h0].expand(thisk.shape[0], -1, -1), used_k), dim=-2)
            used_v = torch.cat((memkv_text[..., h0:].expand(thisv.shape[0], -1, -1), used_v), dim=-2)
        
        used_k = used_k.reshape(b0, -1, self.n_head, h).permute(0, 2, 1, 3)
        used_v = used_v.reshape(b0, -1, self.n_head, h).permute(0, 2, 1, 3)
        thisq = thisq.reshape(b0, 1, self.n_head, h).permute(0, 2, 1, 3) # [b0, n_head, 1, h]
        attn = torch.matmul(thisq / math.sqrt(h), used_k.transpose(-1, -2))
        if memkv_text is not None:
            attn[..., :memkv_text.shape[-2]] += log_text_attention_weights
        attn = F.softmax(attn, dim=-1)
        
        context_swin = torch.matmul(attn, used_v).permute(0, 2, 1, 3).reshape(b0, 1, h0)

        return context_swin, this_qkv[..., h0:]
        

def attention_localframe_and_text_NAR(q0, k0, v0, attention_mask, 
                             n_head, text_len, frame_len, frame_num, 
                             attention_dropout=None, log_text_attention_weights=0, stage=1, **kwargs):
    b, s0, h0 = q0.shape
    s1 = s0 - text_len
    h = h0 // n_head
    assert q0.shape[1] == v0.shape[1] == k0.shape[1] == text_len+frame_len*frame_num
    # attention_mask.shape [4, b or 1, 1, text_len+frame_len, text_len+frame_len]
    if stage == 2:
        assert frame_num == 3

    q0 = q0.reshape(b, s0, n_head, h).permute(0, 2, 1, 3)
    v0 = v0.reshape(b, s0, n_head, h).permute(0, 2, 1, 3)
    k0 = k0.reshape(b, s0, n_head, h).permute(0, 2, 1, 3)
    k0T = k0.transpose(-1, -2)

    score_any2text = torch.matmul(q0 / math.sqrt(q0.shape[-1]), k0T[..., :text_len])
    score_any2text += log_text_attention_weights
    score_any2text_part1 = torch.mul(score_any2text[..., :text_len, :], attention_mask[..., :text_len, :text_len]) \
        - 10000.0 * (1.0 - attention_mask[..., :text_len, :text_len])
    # context for text
    attention_probs_text = F.softmax(score_any2text_part1, dim=-1)
    if attention_dropout is not None:
        with get_cuda_rng_tracker().fork():
            attention_probs_text = attention_dropout(attention_probs_text)
    context_text2text = torch.matmul(attention_probs_text, v0[..., :text_len, :])
    context_text2text = context_text2text.transpose(1, 2).reshape(b, text_len, h0)
    
    if frame_num > 0:
        score_any2text_part2 = score_any2text[..., text_len:, :]
        
        # score: frame local
        q0_frame = q0[:, :, text_len:].reshape(b, n_head, frame_num, frame_len, h)
        v0_frame = v0[:, :, text_len:].reshape(b, n_head, frame_num, frame_len, h)
        k0T_frame = k0[:, :, text_len:].reshape(b, n_head, frame_num, frame_len, h).transpose(-1, -2)
        score_frame_local0 = torch.matmul(q0_frame / math.sqrt(q0_frame.shape[-1]), k0T_frame)
        if stage == 1:
            score_frame_local0 = torch.mul(score_frame_local0, attention_mask[..., text_len:, text_len:].unsqueeze(1)) \
                - 10000.0 * (1.0 - attention_mask[..., text_len:, text_len:].unsqueeze(1))
        
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
    else: 
        context_frame = None

    return context_text2text, context_frame
    
def attention_localframe_and_text_AR(q0, k0, v0, n_head, text_len, frame_len, frame_num, 
                                        attention_dropout=None, log_text_attention_weights=0, layer_id=None, limited_spatial_channel_mem=False, stage=1, **kwargs):
    # limited_spatial_channel_mem=True means: mems in spatial channel is consisted of {mem_text, mem_current_frame}
    b, s0, h0 = k0.shape
    frame_num_before = (s0-text_len-1) // frame_len # frame_num == frame_num_before or frame_num == frame_num_before+1
    h = h0 // n_head
    assert q0.shape[1] == 1
    assert v0.shape[1] == k0.shape[1]

    q0 = q0.reshape(b, 1, n_head, h).permute(0, 2, 1, 3)
    v0 = v0.reshape(b, s0, n_head, h).permute(0, 2, 1, 3)
    k0T = k0.reshape(b, s0, n_head, h).permute(0, 2, 3, 1)
    
    if limited_spatial_channel_mem:
        assert frame_num_before == 0
        assert stage == 1 # not implemented for stage-2 yet
        score = torch.matmul(q0 / math.sqrt(q0.shape[-1]), k0T)
        score[..., :text_len] += log_text_attention_weights
        attention_probs_frame = F.softmax(score, dim=-1)
        context_frame = torch.matmul(attention_probs_frame, v0).transpose(1, 2).reshape(b, 1, h0)
        
    else:
        score_token2text = torch.matmul(q0 / math.sqrt(q0.shape[-1]), k0T[..., :text_len])
        score_token2text += log_text_attention_weights
        score_frame_local0 = torch.matmul(q0 / math.sqrt(q0.shape[-1]), k0T[..., text_len+frame_num_before*frame_len:])
        score_frame_all = torch.cat((score_token2text, 
                                    score_frame_local0), dim=-1)
        attention_probs_frame = F.softmax(score_frame_all, dim=-1)
        
        context_token2text = torch.matmul(attention_probs_frame[..., :text_len], v0[..., :text_len, :]) # [b, n_head, s1, h]
        context_frame_local0 = torch.matmul(attention_probs_frame[..., text_len:], \
            v0[:, :, text_len+frame_num_before*frame_len:, :])
        context_frame = (context_token2text + context_frame_local0).transpose(1, 2).reshape(b, 1, h0)
    
    return context_frame

    
class CogVideoCacheModel(BaseModel):
    def __init__(self, args, transformer=None, parallel_output=True, window_size=None, cogvideo_stage=None):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output)
        self.layout = args.layout # [64, 64+1024, 64+6*1024]
        self.stage = cogvideo_stage if cogvideo_stage is not None else args.cogvideo_stage # 1 or 2
        self.n_head = args.num_attention_heads
        self.window_size = window_size if window_size is not None else args.window_size
        
        frame_resolution = int(math.sqrt(self.layout[1]-self.layout[0]))
        self.add_mixin('extra_position_embedding', PositionEmbeddingMixin(
            args.additional_seqlen, args.hidden_size
        ))
        
        if self.stage == 1:
            self.add_mixin('attention_plus', FullAttentionMixin(
                num_layers=args.num_layers,
                hidden_size=args.hidden_size,
                frame_resolution=frame_resolution,
                n_head=args.num_attention_heads,
                frame_num=(args.layout[2]-args.layout[0])//(args.layout[1]-args.layout[0]),
            ))
        else: 
            self.add_mixin('attention_plus', WindowAttentionMixin(
                num_layers=args.num_layers,
                hidden_size=args.hidden_size,
                frame_resolution=frame_resolution,
                window_size=self.window_size,
                shift_size=self.window_size//2,
                n_head=args.num_attention_heads,
                frame_num=(args.layout[2]-args.layout[0])//(args.layout[1]-args.layout[0]),
            ))

        
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('VideoSwinLocalModel', 'video swin local model configurations')
        group.add_argument("--layout", type=str, default='64, 464, 2064') 
        group.add_argument("--window-size", type=int, default=10) # 优先级在直接参数赋值之后
        group.add_argument("--additional-seqlen", type=int, default=2000)
        group.add_argument("--cogvideo-stage", type=int, default=1, choices=[1,2])  # 优先级在直接参数赋值之后
        return parser
    
    def disable_untrainable_params(self):
        pass
    
    def position_embedding_forward(self, position_ids, **kw_args):
        if position_ids.shape[-1] > 1:
            if self.stage == 1:
                if position_ids[0,-1] >= (512+400):
                    frame_num = position_ids.shape[-1] // 400
                    position_embeddings = torch.cat(
                    (
                        self.transformer.position_embeddings(position_ids[..., :-400*(frame_num-1)]),
                        self.get_mixin('extra_position_embedding').position_embeddings(position_ids[..., -400*(frame_num-1):]-(512+400))
                    ),
                    dim=-2
                )
                else:
                    position_embeddings = self.transformer.position_embeddings(position_ids)
            else:
                # given 3, interpolate 2
                position_embeddings = torch.cat(
                    (
                        self.transformer.position_embeddings(position_ids[..., :-800]),
                        self.get_mixin('extra_position_embedding').position_embeddings(position_ids[..., -800:]-(512+400))
                    ),
                    dim=-2
                )
        else:
            if position_ids[0, 0] >= (512+400):
                position_embeddings = self.get_mixin('extra_position_embedding').position_embeddings(position_ids-(512+400))
            else:
                position_embeddings = self.transformer.position_embeddings(position_ids)
        return position_embeddings
        
    def attention_forward(self, hidden_states, mask, layer_id, mems=None, log_text_attention_weights=0, text_len=0, frame_len=0, counter=0, enforce_no_swin=False, limited_spatial_channel_mem=False, **kw_args):
        attn_module = self.transformer.layers[layer_id].attention
        hidden_size = hidden_states.shape[-1]
        
        # base model qkv
        if mems is None:
            mixed_raw_layer = attn_module.query_key_value(hidden_states)
            q0, k0, v0 = split_tensor_along_last_dim(mixed_raw_layer, 3)
            assert (q0.shape[1]-text_len) % frame_len == 0
            memkv0 = torch.cat((k0, v0), dim=-1)
            context_text, context_frame_local_text = attention_localframe_and_text_NAR(
                    q0, k0, v0,
                    mask,
                    n_head=attn_module.num_attention_heads_per_partition,
                    text_len=text_len,
                    frame_len=frame_len,
                    frame_num=(q0.shape[1]-text_len)//frame_len,
                    log_text_attention_weights=log_text_attention_weights,
                    stage=self.stage
                )

            # change: self.swin_attend_to_text默认为True:
            memkv1_text = self.get_mixin('attention_plus').query_key_value[layer_id](hidden_states[..., :text_len, :])[..., hidden_size:]
            output_text = attn_module.dense(context_text)
                
            if (q0.shape[1]-text_len)//frame_len > 0:
                assert (q0.shape[1]-text_len) % frame_len == 0
                context_frame_swin, memkv1_frame = self.get_mixin('attention_plus').attention_extra_NAR_inference(
                    hidden_states[:,text_len:], layer_id, memkv_text=memkv1_text, stage=self.stage)
                if not enforce_no_swin:
                    attn_distrib = torch.sigmoid(self.get_mixin('attention_plus').attn_distribution[layer_id])
                    attn_distrib = attn_distrib.unsqueeze(0).unsqueeze(0)
                    output_frame = torch.mul(attn_module.dense(context_frame_local_text), attn_distrib)\
                        +torch.mul(self.get_mixin('attention_plus').dense[layer_id](context_frame_swin), 1-attn_distrib)
                else:
                    output_frame = attn_module.dense(context_frame_local_text[..., :frame_len, :])
                output = torch.cat((output_text, output_frame), dim=-2)
                memkv1 = torch.cat((memkv1_text, memkv1_frame), dim=-2) if memkv1_text is not None else memkv1_frame
            else: 
                output = output_text
                memkv1 = memkv1_text
            kw_args['output_this_layer']['mem_kv'] = (memkv0, memkv1)
            
            
        else:
            mixed_raw_layer = attn_module.query_key_value(hidden_states)
            q0, k0, v0 = split_tensor_along_last_dim(mixed_raw_layer, 3)
            new_memkv0 = torch.cat((k0, v0), dim=-1)
            old_k0, old_v0 = mems[0][layer_id][..., :hidden_size], mems[0][layer_id][..., hidden_size:] 
            
            context_frame_local_text = attention_localframe_and_text_AR(
                    q0, 
                    torch.cat((old_k0.expand(k0.shape[0], -1, -1), k0), dim=-2),
                    torch.cat((old_v0.expand(v0.shape[0], -1, -1), v0), dim=-2),
                    n_head=attn_module.num_attention_heads_per_partition,
                    text_len=text_len,
                    frame_len=frame_len,
                    frame_num=None,
                    log_text_attention_weights=log_text_attention_weights,
                    layer_id=layer_id,
                    limited_spatial_channel_mem=limited_spatial_channel_mem,
                )
            
            old_memkv1 = mems[1][layer_id] if mems[1] is not None else None
            
            context_frame_swin, new_memkv1 = self.get_mixin('attention_plus').attention_extra_AR_inference(hidden_states, 
                                                                                                            old_memkv1[..., text_len:, :] if old_memkv1.shape[-2]>text_len else None,
                                                                                                            counter-text_len,
                                                                                                            layer_id, 
                                                                                                            memkv_text=old_memkv1[..., :text_len, :],
                                                                                                            log_text_attention_weights=log_text_attention_weights)
            if not enforce_no_swin:
                attn_distrib = torch.sigmoid(self.get_mixin('attention_plus').attn_distribution[layer_id])
                attn_distrib = attn_distrib.unsqueeze(0).unsqueeze(0)
                output = torch.mul(attn_module.dense(context_frame_local_text), attn_distrib)\
                    +torch.mul(self.get_mixin('attention_plus').dense[layer_id](context_frame_swin), 1-attn_distrib)
            else: 
                output = attn_module.dense(context_frame_local_text)
                
            kw_args['output_this_layer']['mem_kv'] = (new_memkv0, new_memkv1)
            
        return output