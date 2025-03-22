from typing import Any, Dict
import torch
import argparse
from diffusers.loaders.lora_base import LoraBaseMixin
from diffusers.models.modeling_utils import load_state_dict


def get_state_dict(saved_dict: Dict[str, Any]) -> Dict[str, Any]:
    state_dict = saved_dict
    if "model" in saved_dict.keys():
        state_dict = state_dict["model"]
    if "module" in saved_dict.keys():
        state_dict = state_dict["module"]
    if "state_dict" in saved_dict.keys():
        state_dict = state_dict["state_dict"]
    return state_dict


LORA_KEYS_RENAME = {
    'attention.query_key_value.matrix_A.0': 'attn1.to_q.lora_A.weight',
    'attention.query_key_value.matrix_A.1': 'attn1.to_k.lora_A.weight',
    'attention.query_key_value.matrix_A.2': 'attn1.to_v.lora_A.weight',
    'attention.query_key_value.matrix_B.0': 'attn1.to_q.lora_B.weight',
    'attention.query_key_value.matrix_B.1': 'attn1.to_k.lora_B.weight',
    'attention.query_key_value.matrix_B.2': 'attn1.to_v.lora_B.weight',
    'attention.dense.matrix_A.0': 'attn1.to_out.0.lora_A.weight',
    'attention.dense.matrix_B.0': 'attn1.to_out.0.lora_B.weight',
}


PREFIX_KEY = "model.diffusion_model."
SAT_UNIT_KEY = "layers"
LORA_PREFIX_KEY = "transformer_blocks"


def export_lora_weight(ckpt_path, lora_save_directory):
    merge_original_state_dict = get_state_dict(torch.load(ckpt_path, map_location="cpu", mmap=True))

    lora_state_dict = {}
    for key in list(merge_original_state_dict.keys()):
        new_key = key[len(PREFIX_KEY) :]
        for special_key, lora_keys in LORA_KEYS_RENAME.items():
            if new_key.endswith(special_key):
                new_key = new_key.replace(special_key, lora_keys)
                new_key = new_key.replace(SAT_UNIT_KEY, LORA_PREFIX_KEY)

                lora_state_dict[new_key] = merge_original_state_dict[key]

    # final length should be 240
    if len(lora_state_dict) != 240:
        raise ValueError("lora_state_dict length is not 240")

    lora_state_dict.keys()

    LoraBaseMixin.write_lora_layers(
        state_dict=lora_state_dict,
        save_directory=lora_save_directory,
        is_main_process=True,
        weight_name=None,
        save_function=None,
        safe_serialization=True,
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sat_pt_path", type=str, required=True, help="Path to original sat transformer checkpoint"
    )
    parser.add_argument(
        "--lora_save_directory",
        type=str,
        required=True,
        help="Path where converted lora should be saved",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    export_lora_weight(args.sat_pt_path, args.lora_save_directory)
