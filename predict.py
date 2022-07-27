import os
from random import randint
import subprocess
import tempfile
import random
import typing
from PIL import Image, UnidentifiedImageError
from deep_translator import GoogleTranslator
from cog import BasePredictor, Input, Path

import torch
import time
import logging,sys
import stat
from torchvision.utils import save_image
from icetk import icetk as tokenizer
import torch.distributed as dist
tokenizer.add_special_tokens(['<start_of_image>', '<start_of_english>', '<start_of_chinese>'])

from SwissArmyTransformer import get_args
from SwissArmyTransformer.data_utils import BinaryDataset, make_loaders
from SwissArmyTransformer.generation.sampling_strategies import BaseStrategy
from SwissArmyTransformer.generation.utils import timed_name, save_multiple_images, generate_continually
from SwissArmyTransformer.resources import auto_create

from models.cogvideo_cache_model import CogVideoCacheModel
from coglm_strategy import CoglmStrategy




def get_masks_and_position_ids_stage1(data, textlen, framelen):
    # Extract batch size and sequence length.
    tokens = data
    seq_length = len(data[0])
    # Attention mask (lower triangular).
    attention_mask = torch.ones((1, textlen+framelen, textlen+framelen), device=data.device)
    attention_mask[:, :textlen, textlen:] = 0
    attention_mask[:, textlen:, textlen:].tril_()
    attention_mask.unsqueeze_(1)
    # Unaligned version
    position_ids = torch.zeros(seq_length, dtype=torch.long,
                                device=data.device)
    torch.arange(textlen, out=position_ids[:textlen], 
        dtype=torch.long, device=data.device)
    torch.arange(512, 512+seq_length-textlen, out=position_ids[textlen:], 
        dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0)

    return tokens, attention_mask, position_ids

def get_masks_and_position_ids_stage2(data, textlen, framelen):
    # Extract batch size and sequence length.
    tokens = data
    seq_length = len(data[0])

    # Attention mask (lower triangular).
    attention_mask = torch.ones((1, textlen+framelen, textlen+framelen), device=data.device)
    attention_mask[:, :textlen, textlen:] = 0
    attention_mask[:, textlen:, textlen:].tril_()
    attention_mask.unsqueeze_(1)
    
    # Unaligned version
    position_ids = torch.zeros(seq_length, dtype=torch.long,
                                device=data.device)
    torch.arange(textlen, out=position_ids[:textlen], 
        dtype=torch.long, device=data.device)
    frame_num = (seq_length-textlen)//framelen
    assert frame_num == 5
    torch.arange(512, 512+framelen, out=position_ids[textlen:textlen+framelen], 
            dtype=torch.long, device=data.device)
    torch.arange(512+framelen*2, 512+framelen*3, out=position_ids[textlen+framelen:textlen+framelen*2], 
            dtype=torch.long, device=data.device)
    torch.arange(512+framelen*(frame_num-1), 512+framelen*frame_num, out=position_ids[textlen+framelen*2:textlen+framelen*3], 
            dtype=torch.long, device=data.device)
    torch.arange(512+framelen*1, 512+framelen*2, out=position_ids[textlen+framelen*3:textlen+framelen*4], 
            dtype=torch.long, device=data.device)
    torch.arange(512+framelen*3, 512+framelen*4, out=position_ids[textlen+framelen*4:textlen+framelen*5], 
            dtype=torch.long, device=data.device)

    position_ids = position_ids.unsqueeze(0)

    return tokens, attention_mask, position_ids

def my_update_mems(hiddens, mems_buffers, mems_indexs, limited_spatial_channel_mem, text_len, frame_len):
    if hiddens is None:
        return None, mems_indexs
    mem_num = len(hiddens)
    ret_mem = []
    with torch.no_grad():
        for id in range(mem_num):
            if hiddens[id][0] is None:
                ret_mem.append(None)
            else: 
                if id == 0 and limited_spatial_channel_mem and mems_indexs[id]+hiddens[0][0].shape[1] >= text_len+frame_len:
                    if mems_indexs[id] == 0:
                        for layer, hidden in enumerate(hiddens[id]):
                            mems_buffers[id][layer, :, :text_len] = hidden.expand(mems_buffers[id].shape[1], -1, -1)[:, :text_len]
                    new_mem_len_part2 = (mems_indexs[id]+hiddens[0][0].shape[1]-text_len)%frame_len 
                    if new_mem_len_part2 > 0:
                        for layer, hidden in enumerate(hiddens[id]):
                            mems_buffers[id][layer, :, text_len:text_len+new_mem_len_part2] = hidden.expand(mems_buffers[id].shape[1], -1, -1)[:, -new_mem_len_part2:]
                    mems_indexs[id] = text_len+new_mem_len_part2
                else:
                    for layer, hidden in enumerate(hiddens[id]):
                        mems_buffers[id][layer, :, mems_indexs[id]:mems_indexs[id]+hidden.shape[1]] = hidden.expand(mems_buffers[id].shape[1], -1, -1)
                    mems_indexs[id] += hidden.shape[1]
                ret_mem.append(mems_buffers[id][:, :, :mems_indexs[id]])
    return ret_mem, mems_indexs


def my_save_multiple_images(imgs, path, subdir, debug=True):
    # imgs: list of tensor images
    if debug:
        imgs = torch.cat(imgs, dim=0)
        logging.debug("\nSave to: ", path, flush=True)
        save_image(imgs, path, normalize=True)
    else:
        logging.debug("\nSave to: ", path, flush=True)
        single_frame_path = os.path.join(path, subdir)
        os.makedirs(single_frame_path, exist_ok=True)
        for i in range(len(imgs)):
            save_image(imgs[i], os.path.join(single_frame_path, f'{str(i).rjust(4,"0")}.jpg'), normalize=True)
            os.chmod(os.path.join(single_frame_path,f'{str(i).rjust(4,"0")}.jpg'), stat.S_IRWXO+stat.S_IRWXG+stat.S_IRWXU)
        save_image(torch.cat(imgs, dim=0), os.path.join(single_frame_path,f'frame_concat.jpg'), normalize=True)
        os.chmod(os.path.join(single_frame_path,f'frame_concat.jpg'), stat.S_IRWXO+stat.S_IRWXG+stat.S_IRWXU)
        
def calc_next_tokens_frame_begin_id(text_len, frame_len, total_len):
    # The fisrt token's position id of the frame that the next token belongs to; 
    if total_len < text_len:
        return None
    return (total_len-text_len)//frame_len * frame_len + text_len

def my_filling_sequence(
        model, 
        args,
        seq, 
        batch_size,
        get_masks_and_position_ids,
        text_len,
        frame_len,
        strategy=BaseStrategy(),
        strategy2=BaseStrategy(),
        mems=None,
        log_text_attention_weights=0, # default to 0: no artificial change
        mode_stage1=True,
        enforce_no_swin=False,
        guider_seq=None,
        guider_text_len=0,
        guidance_alpha=1,
        limited_spatial_channel_mem=False, # 空间通道的存储限制在本帧内
        **kw_args
        ):
    '''
        seq: [2, 3, 5, ..., -1(to be generated), -1, ...]
        mems: [num_layers, batch_size, len_mems(index), mem_hidden_size]
            cache, should be first mems.shape[1] parts of context_tokens.
            mems are the first-level citizens here, but we don't assume what is memorized.
            input mems are used when multi-phase generation.
    '''
    if guider_seq is not None:
        logging.debug("Using Guidance In Inference")
    if limited_spatial_channel_mem:
        logging.debug("Limit spatial-channel's mem to current frame")
    assert len(seq.shape) == 2

    # building the initial tokens, attention_mask, and position_ids
    actual_context_length = 0
    
    while seq[-1][actual_context_length] >= 0: # the last seq has least given tokens
        actual_context_length += 1 # [0, context_length-1] are given
    assert actual_context_length > 0
    current_frame_num = (actual_context_length-text_len) // frame_len
    assert current_frame_num >= 0
    context_length = text_len + current_frame_num * frame_len
    
    tokens, attention_mask, position_ids = get_masks_and_position_ids(seq, text_len, frame_len)
    tokens = tokens[..., :context_length]
    input_tokens = tokens.clone()
    
    if guider_seq is not None:
        guider_index_delta = text_len - guider_text_len
        guider_tokens, guider_attention_mask, guider_position_ids = get_masks_and_position_ids(guider_seq, guider_text_len, frame_len)
        guider_tokens = guider_tokens[..., :context_length-guider_index_delta]
        guider_input_tokens = guider_tokens.clone()
        
    for fid in range(current_frame_num):
        input_tokens[:, text_len+400*fid] = tokenizer['<start_of_image>']
        if guider_seq is not None:
            guider_input_tokens[:, guider_text_len+400*fid] = tokenizer['<start_of_image>']

    attention_mask = attention_mask.type_as(next(model.parameters())) # if fp16
    # initialize generation
    counter = context_length - 1 # Last fixed index is ``counter'' 
    index = 0 # Next forward starting index, also the length of cache.
    mems_buffers_on_GPU = False
    mems_indexs = [0, 0]
    mems_len = [(400+74) if limited_spatial_channel_mem else 5*400+74, 5*400+74]
    mems_buffers = [torch.zeros(args.num_layers, batch_size, mem_len, args.hidden_size*2, dtype=next(model.parameters()).dtype)
                        for mem_len in mems_len]

    
    if guider_seq is not None: 
        guider_attention_mask = guider_attention_mask.type_as(next(model.parameters())) # if fp16
        guider_mems_buffers = [torch.zeros(args.num_layers, batch_size, mem_len, args.hidden_size*2, dtype=next(model.parameters()).dtype)
                        for mem_len in mems_len]
        guider_mems_indexs = [0, 0]
        guider_mems = None
    
    torch.cuda.empty_cache()
    # step-by-step generation
    while counter < len(seq[0]) - 1:
        # we have generated counter+1 tokens
        # Now, we want to generate seq[counter + 1],
        # token[:, index: counter+1] needs forwarding.
        if index == 0:
            group_size = 2 if (input_tokens.shape[0] == batch_size and not mode_stage1) else batch_size
            
            logits_all = None
            for batch_idx in range(0, input_tokens.shape[0], group_size):
                logits, *output_per_layers = model(
                    input_tokens[batch_idx:batch_idx+group_size, index:],
                    position_ids[..., index: counter+1],
                    attention_mask, # TODO memlen
                    mems=mems,
                    text_len=text_len,
                    frame_len=frame_len,
                    counter=counter,
                    log_text_attention_weights=log_text_attention_weights,
                    enforce_no_swin=enforce_no_swin,
                    **kw_args
                )
                logits_all = torch.cat((logits_all, logits), dim=0) if logits_all is not None else logits
                mem_kv01 = [[o['mem_kv'][0] for o in output_per_layers], [o['mem_kv'][1] for o in output_per_layers]]
                next_tokens_frame_begin_id = calc_next_tokens_frame_begin_id(text_len, frame_len, mem_kv01[0][0].shape[1])
                for id, mem_kv in enumerate(mem_kv01):
                    for layer, mem_kv_perlayer in enumerate(mem_kv):
                        if limited_spatial_channel_mem and id == 0:
                            mems_buffers[id][layer, batch_idx:batch_idx+group_size, :text_len] = mem_kv_perlayer.expand(min(group_size, input_tokens.shape[0]-batch_idx), -1, -1)[:, :text_len]
                            mems_buffers[id][layer, batch_idx:batch_idx+group_size, text_len:text_len+mem_kv_perlayer.shape[1]-next_tokens_frame_begin_id] =\
                                mem_kv_perlayer.expand(min(group_size, input_tokens.shape[0]-batch_idx), -1, -1)[:, next_tokens_frame_begin_id:]
                        else:
                            mems_buffers[id][layer, batch_idx:batch_idx+group_size, :mem_kv_perlayer.shape[1]] = mem_kv_perlayer.expand(min(group_size, input_tokens.shape[0]-batch_idx), -1, -1)
                mems_indexs[0], mems_indexs[1] = mem_kv01[0][0].shape[1], mem_kv01[1][0].shape[1]
                if limited_spatial_channel_mem:
                    mems_indexs[0] -= (next_tokens_frame_begin_id - text_len)

            mems = [mems_buffers[id][:, :, :mems_indexs[id]] for id in range(2)]
            logits = logits_all
            
            # Guider 
            if guider_seq is not None:
                guider_logits_all = None
                for batch_idx in range(0, guider_input_tokens.shape[0], group_size):
                    guider_logits, *guider_output_per_layers = model(
                        guider_input_tokens[batch_idx:batch_idx+group_size, max(index-guider_index_delta, 0):],
                        guider_position_ids[..., max(index-guider_index_delta, 0): counter+1-guider_index_delta],
                        guider_attention_mask,
                        mems=guider_mems,
                        text_len=guider_text_len,
                        frame_len=frame_len,
                        counter=counter-guider_index_delta,
                        log_text_attention_weights=log_text_attention_weights,
                        enforce_no_swin=enforce_no_swin,
                        **kw_args
                    )
                    guider_logits_all = torch.cat((guider_logits_all, guider_logits), dim=0) if guider_logits_all is not None else guider_logits
                    guider_mem_kv01 = [[o['mem_kv'][0] for o in guider_output_per_layers], [o['mem_kv'][1] for o in guider_output_per_layers]]
                    for id, guider_mem_kv in enumerate(guider_mem_kv01):
                        for layer, guider_mem_kv_perlayer in enumerate(guider_mem_kv):
                            if limited_spatial_channel_mem and id == 0:
                                guider_mems_buffers[id][layer, batch_idx:batch_idx+group_size, :guider_text_len] = guider_mem_kv_perlayer.expand(min(group_size, input_tokens.shape[0]-batch_idx), -1, -1)[:, :guider_text_len]
                                guider_next_tokens_frame_begin_id = calc_next_tokens_frame_begin_id(guider_text_len, frame_len, guider_mem_kv_perlayer.shape[1])
                                guider_mems_buffers[id][layer, batch_idx:batch_idx+group_size, guider_text_len:guider_text_len+guider_mem_kv_perlayer.shape[1]-guider_next_tokens_frame_begin_id] =\
                                    guider_mem_kv_perlayer.expand(min(group_size, input_tokens.shape[0]-batch_idx), -1, -1)[:, guider_next_tokens_frame_begin_id:]
                            else:
                                guider_mems_buffers[id][layer, batch_idx:batch_idx+group_size, :guider_mem_kv_perlayer.shape[1]] = guider_mem_kv_perlayer.expand(min(group_size, input_tokens.shape[0]-batch_idx), -1, -1)
                    guider_mems_indexs[0], guider_mems_indexs[1] = guider_mem_kv01[0][0].shape[1], guider_mem_kv01[1][0].shape[1]
                    if limited_spatial_channel_mem:
                        guider_mems_indexs[0] -= (guider_next_tokens_frame_begin_id-guider_text_len)
                guider_mems = [guider_mems_buffers[id][:, :, :guider_mems_indexs[id]] for id in range(2)]
                guider_logits = guider_logits_all
        else:
            if not mems_buffers_on_GPU:
                if not mode_stage1:
                    torch.cuda.empty_cache()
                    for idx, mem in enumerate(mems):
                        mems[idx] = mem.to(next(model.parameters()).device)
                    if guider_seq is not None:
                        for idx, mem in enumerate(guider_mems):
                            guider_mems[idx] = mem.to(next(model.parameters()).device) 
                else: 
                    torch.cuda.empty_cache()
                    for idx, mem_buffer in enumerate(mems_buffers):
                        mems_buffers[idx] = mem_buffer.to(next(model.parameters()).device)
                    mems = [mems_buffers[id][:, :, :mems_indexs[id]] for id in range(2)]
                    if guider_seq is not None:
                        for idx, guider_mem_buffer in enumerate(guider_mems_buffers):
                            guider_mems_buffers[idx] = guider_mem_buffer.to(next(model.parameters()).device)
                        guider_mems = [guider_mems_buffers[id][:, :, :guider_mems_indexs[id]] for id in range(2)]
                    mems_buffers_on_GPU = True
                    
            logits, *output_per_layers = model(
                input_tokens[:, index:],
                position_ids[..., index: counter+1],
                attention_mask, # TODO memlen
                mems=mems,
                text_len=text_len,
                frame_len=frame_len,
                counter=counter,
                log_text_attention_weights=log_text_attention_weights,
                enforce_no_swin=enforce_no_swin,
                limited_spatial_channel_mem=limited_spatial_channel_mem,
                **kw_args
            )
            mem_kv0, mem_kv1 = [o['mem_kv'][0] for o in output_per_layers], [o['mem_kv'][1] for o in output_per_layers]
            
            if guider_seq is not None:
                guider_logits, *guider_output_per_layers = model(
                    guider_input_tokens[:, max(index-guider_index_delta, 0):],
                    guider_position_ids[..., max(index-guider_index_delta, 0): counter+1-guider_index_delta],
                    guider_attention_mask,
                    mems=guider_mems,
                    text_len=guider_text_len,
                    frame_len=frame_len,
                    counter=counter-guider_index_delta,
                    log_text_attention_weights=0,
                    enforce_no_swin=enforce_no_swin,
                    limited_spatial_channel_mem=limited_spatial_channel_mem,
                    **kw_args
                )
                guider_mem_kv0, guider_mem_kv1 = [o['mem_kv'][0] for o in guider_output_per_layers], [o['mem_kv'][1] for o in guider_output_per_layers]
            
            if not mems_buffers_on_GPU:
                torch.cuda.empty_cache()
                for idx, mem_buffer in enumerate(mems_buffers):
                    mems_buffers[idx] = mem_buffer.to(next(model.parameters()).device)
                if guider_seq is not None:
                    for idx, guider_mem_buffer in enumerate(guider_mems_buffers):
                        guider_mems_buffers[idx] = guider_mem_buffer.to(next(model.parameters()).device)
                mems_buffers_on_GPU = True

            mems, mems_indexs = my_update_mems([mem_kv0, mem_kv1], mems_buffers, mems_indexs, limited_spatial_channel_mem, text_len, frame_len) 
            if guider_seq is not None: 
                guider_mems, guider_mems_indexs = my_update_mems([guider_mem_kv0, guider_mem_kv1], guider_mems_buffers, guider_mems_indexs, limited_spatial_channel_mem, guider_text_len, frame_len)

       
        counter += 1 
        index = counter        

        logits = logits[:, -1].expand(batch_size, -1) # [batch size, vocab size]
        tokens = tokens.expand(batch_size, -1)
        if guider_seq is not None:
            guider_logits = guider_logits[:, -1].expand(batch_size, -1)
            guider_tokens = guider_tokens.expand(batch_size, -1)
        
        if seq[-1][counter].item() < 0:
            # sampling
            guided_logits = guider_logits+(logits-guider_logits)*guidance_alpha if guider_seq is not None else logits
            if mode_stage1 and counter < text_len + 400:
                tokens, mems = strategy.forward(guided_logits, tokens, mems)
            else:
                tokens, mems = strategy2.forward(guided_logits, tokens, mems)
            if guider_seq is not None:
                guider_tokens = torch.cat((guider_tokens, tokens[:, -1:]), dim=1)
                
            if seq[0][counter].item() >= 0:
                for si in range(seq.shape[0]):
                    if seq[si][counter].item() >= 0:
                        tokens[si, -1] = seq[si, counter]
                        if guider_seq is not None:
                            guider_tokens[si, -1] = guider_seq[si, counter-guider_index_delta]
                    
        else:
            tokens = torch.cat((tokens, seq[:, counter:counter+1].clone().expand(tokens.shape[0], 1).to(device=tokens.device, dtype=tokens.dtype)), dim=1)
            if guider_seq is not None: 
                guider_tokens = torch.cat((guider_tokens, 
                                           guider_seq[:, counter-guider_index_delta:counter+1-guider_index_delta]
                                           .clone().expand(guider_tokens.shape[0], 1).to(device=guider_tokens.device, dtype=guider_tokens.dtype)), dim=1)
        
        input_tokens = tokens.clone()
        if guider_seq is not None:
            guider_input_tokens = guider_tokens.clone()
        if (index-text_len-1)//400 < (input_tokens.shape[-1]-text_len-1)//400:
            boi_idx = ((index-text_len-1)//400 +1)*400+text_len
            while boi_idx < input_tokens.shape[-1]:
                input_tokens[:, boi_idx] = tokenizer['<start_of_image>']
                if guider_seq is not None:
                    guider_input_tokens[:, boi_idx-guider_index_delta] = tokenizer['<start_of_image>']
                boi_idx += 400
        
        if strategy.is_done:
            break
    return strategy.finalize(tokens, mems)

class InferenceModel_Sequential(CogVideoCacheModel):
    def __init__(self, args, transformer=None, parallel_output=True):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output, window_size=-1, cogvideo_stage=1)
    # TODO: check it 
    
    def final_forward(self, logits, **kwargs):
        logits_parallel = logits
        logits_parallel = torch.nn.functional.linear(logits_parallel.float(), self.transformer.word_embeddings.weight[:20000].float())
        return logits_parallel
    
class InferenceModel_Interpolate(CogVideoCacheModel):
    def __init__(self, args, transformer=None, parallel_output=True):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output, window_size=10, cogvideo_stage=2)
    # TODO: check it 

    def final_forward(self, logits, **kwargs):
        logits_parallel = logits
        logits_parallel = torch.nn.functional.linear(logits_parallel.float(), self.transformer.word_embeddings.weight[:20000].float())
        return logits_parallel

class Predictor(BasePredictor):
    def setup(self):
        logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')        
        os.environ["SAT_HOME"] = "/sharefs/cogview-new"
        args = get_args([
            "--batch-size", "1",                                               
            "--tokenizer-type", "fake", 
            "--mode", "inference", 
            "--distributed-backend", "nccl",
            "--model-parallel-size", "1",
            "--fp16",
            "--sandwich-ln",            
            "--temperature", "1.05",                        
            "--max-inference-batch-size", "8"])
        args.layout = [64,464,2064]
        args.window_size = 10
        args.additional_seqlen = 2000
        args.cogvideo_stage = 1
        args.do_train = False
        args.parallel_size = 1
        args.guidance_alpha = 3.0
        args.generate_frame_num = 5
        args.coglm_temperature = 0.89
        args.coglm_temperature2 = 0.89
        args.generate_frame_num = 5
        args.stage1_max_inference_batch_size = -1
        args.max_inference_batch_size = 8
        args.top_k = 12
        args.use_guidance_stage1 = True
        args.use_guidance_stage2 = False
        args.both_stages = True           
        args.device = torch.device("cuda")
        self.image_prompt = None
        
        self.translator = GoogleTranslator(source="en", target="zh-CN")        
        self.model_stage1, args = InferenceModel_Sequential.from_pretrained(args, "cogvideo-stage1")
        self.model_stage1.eval()
        self.model_stage1 = self.model_stage1.cpu()
        self.model_stage2, args = InferenceModel_Interpolate.from_pretrained(args, "cogvideo-stage2")
        self.model_stage2.eval()
        self.model_stage2 = self.model_stage2.cpu()
        
        # enable dsr if model exists
        if os.path.exists("/sharefs/cogview-new/cogview2-dsr"):
            subprocess.check_output("python setup.py develop", cwd="/src/Image-Local-Attention", shell=True)
            sys.path.append('./Image-Local-Attention')
            from sr_pipeline import DirectSuperResolution 
            dsr_path = auto_create('cogview2-dsr', path=None)
            self.dsr = DirectSuperResolution(args, dsr_path,
                                        max_bz=12, onCUDA=False)
        else:
            self.dsr = None
        
        self.args = args                
        torch.cuda.empty_cache()
        
        
        self.generate_frame_num = 5
        

    @torch.no_grad()
    def predict(
        self,
        prompt: str = Input(description="Prompt"),
        seed: int = Input(description="Seed (-1 to use a random seed)", default=-1, le=(100000), ge=-1),
        translate: bool = Input(
            description="Translate prompt from English to Simplified Chinese (required if not entering Chinese text)",
            default=True,
        ),
        both_stages: bool = Input(
            description="Run both stages (uncheck to run more quickly and output only a few frames)", default=True
        ),
        use_guidance: bool = Input(description="Use stage 1 guidance (recommended)", default=True),
        image_prompt: Path = Input(description="Starting image (optional, prompt has little effect when used)", default=None)
    ) -> typing.Iterator[Path]:
        if translate:
            prompt = self.translator.translate(prompt.strip())

        if seed == -1:
            seed = randint(0, 100000)

        self.args.seed = seed
        self.args.use_guidance_stage1 = use_guidance        
        self.prompt = prompt
        self.image_prompt = None
        if os.path.exists(str(image_prompt)):
            try:
                image = Image.open(str(image_prompt)).convert("RGBA")
                # Remove alpha channel if present
                bg = Image.new("RGBA", image.size, (255, 255, 255))
                image = Image.alpha_composite(bg, image).convert("RGB")
                imagefile = f'{tempfile.mkdtemp()}/input.png'
                image.save(imagefile, format="png")
                self.image_prompt = imagefile
            except (FileNotFoundError, UnidentifiedImageError):
                logging.debug("Bad image prompt; ignoring")  # Is there a better way to input images?            
        self.args.both_stages = both_stages

        for file in self.run():
            yield Path(file)       
        torch.cuda.empty_cache()
        return         
    @torch.no_grad()
    def run(self):
        torch.manual_seed(self.args.seed)
        random.seed(self.args.seed)

        invalid_slices = [slice(tokenizer.num_image_tokens, None)]
        strategy_cogview2 = CoglmStrategy(invalid_slices, 
            temperature=1.0, top_k=16)
        strategy_cogvideo = CoglmStrategy(invalid_slices, 
            temperature=self.args.temperature, top_k=self.args.top_k,
            temperature2=self.args.coglm_temperature2)

        workdir = tempfile.mkdtemp()
        os.makedirs(f"{workdir}/output/stage1", exist_ok=True)
        os.makedirs(f"{workdir}/output/stage2", exist_ok=True)

        move_start_time = time.time()
        logging.debug("moving stage 2 model to cpu")
        self.model_stage2 = self.model_stage2.cpu()
        torch.cuda.empty_cache()
        logging.debug("moving stage 1 model to cuda")
        self.model_stage1 = self.model_stage1.cuda()
        logging.debug("moving in model1 takes time: {:.2f}".format(time.time()-move_start_time))
                       
        process_start_time = time.time()
        args = self.args
        use_guide = args.use_guidance_stage1        
        batch_size = 1
        seq_text = self.prompt
        video_raw_text = self.prompt
        duration=4.0
        video_guidance_text="视频"
        image_text_suffix=" 高清摄影"
        outputdir=f'{workdir}/output/stage1'
        mbz = args.stage1_max_inference_batch_size if args.stage1_max_inference_batch_size > 0 else args.max_inference_batch_size
        assert batch_size < mbz or batch_size % mbz == 0
        frame_len = 400
        
        # generate the first frame:
        enc_text = tokenizer.encode(seq_text+image_text_suffix)
        seq_1st = enc_text + [tokenizer['<start_of_image>']] + [-1]*400 # IV!!  # test local!!! # test randboi!!!
        logging.info("[Generating First Frame with CogView2]Raw text: {:s}".format(tokenizer.decode(enc_text)))
        text_len_1st = len(seq_1st) - frame_len*1 - 1

        seq_1st = torch.cuda.LongTensor(seq_1st, device=args.device).unsqueeze(0)
        if self.image_prompt is None:
            output_list_1st = []
            for tim in range(max(batch_size // mbz, 1)):
                start_time = time.time()
                output_list_1st.append(
                    my_filling_sequence(self.model_stage1, args,seq_1st.clone(),
                        batch_size=min(batch_size, mbz),
                        get_masks_and_position_ids=get_masks_and_position_ids_stage1,
                        text_len=text_len_1st, 
                        frame_len=frame_len,
                        strategy=strategy_cogview2,
                        strategy2=strategy_cogvideo,
                        log_text_attention_weights=1.4,
                        enforce_no_swin=True,
                        mode_stage1=True,
                        )[0]
                    )
                logging.info("[First Frame]Taken time {:.2f}\n".format(time.time() - start_time))
            output_tokens_1st = torch.cat(output_list_1st, dim=0)
            given_tokens = output_tokens_1st[:, text_len_1st+1:text_len_1st+401].unsqueeze(1) # given_tokens.shape: [bs, frame_num, 400]        
        else:
            given_tokens = tokenizer.encode(image_path=self.image_prompt, image_size=160).repeat(batch_size, 1).unsqueeze(1)
        
        # generate subsequent frames:
        total_frames = self.generate_frame_num
        enc_duration = tokenizer.encode(str(float(duration))+"秒")
        if use_guide:
            video_raw_text = video_raw_text + " 视频"
        enc_text_video = tokenizer.encode(video_raw_text)
        seq = enc_duration + [tokenizer['<n>']] + enc_text_video + [tokenizer['<start_of_image>']] + [-1]*400*self.generate_frame_num
        guider_seq = enc_duration + [tokenizer['<n>']] + tokenizer.encode(video_guidance_text) + [tokenizer['<start_of_image>']] + [-1]*400*self.generate_frame_num
        logging.info("[Stage1: Generating Subsequent Frames, Frame Rate {:.1f}]\nraw text: {:s}".format(4/duration, tokenizer.decode(enc_text_video)))

        text_len = len(seq) - frame_len*self.generate_frame_num - 1
        guider_text_len = len(guider_seq) - frame_len*self.generate_frame_num - 1
        seq = torch.cuda.LongTensor(seq, device=args.device).unsqueeze(0).repeat(batch_size, 1)
        guider_seq = torch.cuda.LongTensor(guider_seq, device=args.device).unsqueeze(0).repeat(batch_size, 1)

        for given_frame_id in range(given_tokens.shape[1]):
            seq[:, text_len+1+given_frame_id*400: text_len+1+(given_frame_id+1)*400] = given_tokens[:, given_frame_id]
            guider_seq[:, guider_text_len+1+given_frame_id*400:guider_text_len+1+(given_frame_id+1)*400] = given_tokens[:, given_frame_id]
        output_list = []
        
        if use_guide:
            video_log_text_attention_weights = 0
        else: 
            guider_seq = None
            video_log_text_attention_weights = 1.4
            
        for tim in range(max(batch_size // mbz, 1)):
            start_time = time.time()
            input_seq = seq[:min(batch_size, mbz)].clone() if tim == 0 else seq[mbz*tim:mbz*(tim+1)].clone()
            guider_seq2 = (guider_seq[:min(batch_size, mbz)].clone() if tim == 0 else guider_seq[mbz*tim:mbz*(tim+1)].clone()) if guider_seq is not None else None
            output_list.append(
                my_filling_sequence(self.model_stage1, args,input_seq,
                    batch_size=min(batch_size, mbz),
                    get_masks_and_position_ids=get_masks_and_position_ids_stage1,
                    text_len=text_len, frame_len=frame_len,
                    strategy=strategy_cogview2,
                    strategy2=strategy_cogvideo,
                    log_text_attention_weights=video_log_text_attention_weights,
                    guider_seq=guider_seq2,
                    guider_text_len=guider_text_len,
                    guidance_alpha=args.guidance_alpha,
                    limited_spatial_channel_mem=True,
                    mode_stage1=True,
                    )[0]
                )

        output_tokens = torch.cat(output_list, dim=0)[:, 1+text_len:]
                
        # decoding
        imgs, sred_imgs, txts = [], [], []
        for seq in output_tokens:
            decoded_imgs = [torch.nn.functional.interpolate(tokenizer.decode(image_ids=seq.tolist()[i*400: (i+1)*400]), size=(480, 480)) for i in range(total_frames)]
            imgs.append(decoded_imgs) # only the last image (target)

        assert len(imgs) == batch_size
        save_tokens = output_tokens[:, :+total_frames*400].reshape(-1, total_frames, 400).cpu()
        if outputdir is not None:
            for clip_i in range(len(imgs)):
                # os.makedirs(output_dir_full_paths[clip_i], exist_ok=True)
                my_save_multiple_images(imgs[clip_i], outputdir, subdir=f"frames/{clip_i}", debug=False)
                out_filename = f'{outputdir}/{clip_i}.gif'
                subprocess.check_output(f"gifmaker -i '{outputdir}'/frames/'{clip_i}'/0*.jpg -o '{out_filename}' -d 0.25", shell=True)
                yield out_filename
            torch.save(save_tokens, os.path.join(outputdir, 'frame_tokens.pt'))
        
        logging.info("CogVideo Stage1 completed. Taken time {:.2f}\n".format(time.time() - process_start_time))
            
        logging.debug("moving stage 1 model to cpu")
        self.model_stage1 = self.model_stage1.cpu()
        torch.cuda.empty_cache()        

        if not self.args.both_stages:
            logging.info("only stage 1 selected, exiting")        
            return
                
        gpu_rank=0
        gpu_parallel_size=1
        video_raw_text=self.prompt+" 视频"            
        duration=2.0
        video_guidance_text="视频"
        outputdir=f'{workdir}/output/stage2'
        parent_given_tokens = save_tokens
        stage2_starttime = time.time()        
        use_guidance = args.use_guidance_stage2        

        move_start_time = time.time()
        logging.debug("moving stage-2 model to cuda")
        self.model_stage2 = self.model_stage2.cuda()
        logging.debug("moving in stage-2 model takes time: {:.2f}".format(time.time()-move_start_time))
            
        try:
            sample_num_allgpu = parent_given_tokens.shape[0]
            sample_num = sample_num_allgpu // gpu_parallel_size
            assert sample_num * gpu_parallel_size == sample_num_allgpu
            parent_given_tokens = parent_given_tokens[gpu_rank*sample_num:(gpu_rank+1)*sample_num]
        except:
            logging.critical("No frame_tokens found in interpolation, skip")
            return False
        
        # CogVideo Stage2 Generation
        while duration >= 0.5: # TODO: You can change the boundary to change the frame rate
            parent_given_tokens_num = parent_given_tokens.shape[1]
            generate_batchsize_persample = (parent_given_tokens_num-1)//2
            generate_batchsize_total = generate_batchsize_persample * sample_num
            total_frames = self.generate_frame_num
            frame_len = 400
            enc_text = tokenizer.encode(seq_text)
            enc_duration = tokenizer.encode(str(float(duration))+"秒")
            seq = enc_duration + [tokenizer['<n>']] + enc_text + [tokenizer['<start_of_image>']] + [-1]*400*self.generate_frame_num
            text_len = len(seq) - frame_len*self.generate_frame_num - 1
            
            logging.info("[Stage2: Generating Frames, Frame Rate {:d}]\nraw text: {:s}".format(int(4/duration), tokenizer.decode(enc_text)))
            
            # generation
            seq = torch.cuda.LongTensor(seq, device=args.device).unsqueeze(0).repeat(generate_batchsize_total, 1)
            for sample_i in range(sample_num):
                for i in range(generate_batchsize_persample):
                    seq[sample_i*generate_batchsize_persample+i][text_len+1:text_len+1+400] = parent_given_tokens[sample_i][2*i]
                    seq[sample_i*generate_batchsize_persample+i][text_len+1+400:text_len+1+800] = parent_given_tokens[sample_i][2*i+1]
                    seq[sample_i*generate_batchsize_persample+i][text_len+1+800:text_len+1+1200] = parent_given_tokens[sample_i][2*i+2]
                
            if use_guidance:
                guider_seq = enc_duration + [tokenizer['<n>']] + tokenizer.encode(video_guidance_text) + [tokenizer['<start_of_image>']] + [-1]*400*self.generate_frame_num                 
                guider_text_len = len(guider_seq) - frame_len*self.generate_frame_num - 1
                guider_seq = torch.cuda.LongTensor(guider_seq, device=args.device).unsqueeze(0).repeat(generate_batchsize_total, 1)
                for sample_i in range(sample_num):
                    for i in range(generate_batchsize_persample):
                        guider_seq[sample_i*generate_batchsize_persample+i][text_len+1:text_len+1+400] = parent_given_tokens[sample_i][2*i]
                        guider_seq[sample_i*generate_batchsize_persample+i][text_len+1+400:text_len+1+800] = parent_given_tokens[sample_i][2*i+1]
                        guider_seq[sample_i*generate_batchsize_persample+i][text_len+1+800:text_len+1+1200] = parent_given_tokens[sample_i][2*i+2]
                video_log_text_attention_weights = 0
            else:
                guider_seq=None
                guider_text_len=0
                video_log_text_attention_weights = 1.4

            mbz = args.max_inference_batch_size

            assert generate_batchsize_total < mbz or generate_batchsize_total % mbz == 0
            output_list = []
            start_time = time.time()
            for tim in range(max(generate_batchsize_total // mbz, 1)):
                input_seq = seq[:min(generate_batchsize_total, mbz)].clone() if tim == 0 else seq[mbz*tim:mbz*(tim+1)].clone()
                guider_seq2 = (guider_seq[:min(generate_batchsize_total, mbz)].clone() if tim == 0 else guider_seq[mbz*tim:mbz*(tim+1)].clone()) if guider_seq is not None else None
                output_list.append(
                    my_filling_sequence(self.model_stage2, args, input_seq,
                        batch_size=min(generate_batchsize_total, mbz),
                        get_masks_and_position_ids=get_masks_and_position_ids_stage2,
                        text_len=text_len, frame_len=frame_len,
                        strategy=strategy_cogview2,
                        strategy2=strategy_cogvideo,
                        log_text_attention_weights=video_log_text_attention_weights,
                        mode_stage1=False,
                        guider_seq=guider_seq2,
                        guider_text_len=guider_text_len,
                        guidance_alpha=args.guidance_alpha,
                        limited_spatial_channel_mem=True,
                        )[0]
                    )
            logging.info("Duration {:.2f}, Taken time {:.2f}\n".format(duration, time.time() - start_time))

            output_tokens = torch.cat(output_list, dim=0)
            output_tokens = output_tokens[:, text_len+1:text_len+1+(total_frames)*400].reshape(sample_num, -1, 400*total_frames)
            output_tokens_merge = torch.cat((output_tokens[:, :, :1*400], 
                                            output_tokens[:, :, 400*3:4*400],
                                            output_tokens[:, :, 400*1:2*400],
                                            output_tokens[:, :, 400*4:(total_frames)*400]), dim=2).reshape(sample_num, -1, 400)     

            output_tokens_merge = torch.cat((output_tokens_merge, output_tokens[:, -1:, 400*2:3*400]), dim=1)
            duration /= 2
            parent_given_tokens = output_tokens_merge                        
        
        logging.info("CogVideo Stage2 completed. Taken time {:.2f}\n".format(time.time() - stage2_starttime))    

        enc_text = tokenizer.encode(seq_text)
        frame_num_per_sample = parent_given_tokens.shape[1]
        parent_given_tokens_2d = parent_given_tokens.reshape(-1, 400)        
            
        logging.debug("moving stage 2 model to cpu")
        self.model_stage2 = self.model_stage2.cpu()        
        torch.cuda.empty_cache()
            
        # use dsr if loaded
        if (self.dsr):
            # direct super-resolution by CogView2
            logging.info("[Direct super-resolution]")
            dsr_starttime = time.time()
            text_seq = torch.cuda.LongTensor(enc_text, device=args.device).unsqueeze(0).repeat(parent_given_tokens_2d.shape[0], 1)
            sred_tokens = self.dsr(text_seq, parent_given_tokens_2d)
            decoded_sr_videos = []
            
            for sample_i in range(sample_num):
                decoded_sr_imgs = []
                for frame_i in range(frame_num_per_sample):
                    decoded_sr_img = tokenizer.decode(image_ids=sred_tokens[frame_i+sample_i*frame_num_per_sample][-3600:])
                    decoded_sr_imgs.append(torch.nn.functional.interpolate(decoded_sr_img, size=(480, 480)))
                decoded_sr_videos.append(decoded_sr_imgs)

            for sample_i in range(sample_num):
                my_save_multiple_images(decoded_sr_videos[sample_i], outputdir,subdir=f"frames/{sample_i+sample_num*gpu_rank}", debug=False)
                output_file = f"{outputdir}/{sample_i+sample_num*gpu_rank}.gif"
                subprocess.check_output(f"gifmaker -i '{outputdir}'/frames/'{sample_i+sample_num*gpu_rank}'/0*.jpg -o '{output_file}' -d 0.125", shell=True)
                yield output_file
            
            logging.info("Direct super-resolution completed. Taken time {:.2f}\n".format(time.time() - dsr_starttime))
        else:            
            #imgs = [torch.nn.functional.interpolate(tokenizer.decode(image_ids=seq.tolist()), size=(480, 480)) for seq in output_tokens_merge]
            #os.makedirs(outputdir, exist_ok=True)
            #my_save_multiple_images(imgs, outputdir,subdir="frames", debug=False)            
            #os.system(f"gifmaker -i '{outputdir}'/frames/0*.jpg -o '{outputdir}/{str(float(duration))}_concat.gif' -d 0.2")
        
            
            output_tokens = torch.cat(output_list, dim=0)[:, 1+text_len:]
            decoded_videos = []

            for sample_i in range(sample_num):
                decoded_imgs = []
                for frame_i in range(frame_num_per_sample):
                    decoded_img = tokenizer.decode(image_ids=parent_given_tokens_2d[frame_i+sample_i*frame_num_per_sample][-3600:])
                    decoded_imgs.append(torch.nn.functional.interpolate(decoded_img, size=(480, 480)))
                decoded_videos.append(decoded_imgs)

            for sample_i in range(sample_num):
                my_save_multiple_images(decoded_videos[sample_i], outputdir,subdir=f"frames/{sample_i+sample_num*gpu_rank}", debug=False)
                output_file = f"{outputdir}/{sample_i+sample_num*gpu_rank}.gif"
                subprocess.check_output(f"gifmaker -i '{outputdir}'/frames/'{sample_i+sample_num*gpu_rank}'/0*.jpg -o '{output_file}' -d 0.125", shell=True)
                yield output_file
