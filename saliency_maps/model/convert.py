import dataclasses
import os
import re
import sys
import torch
import open_clip
import glob
from open_clip.hf_model import HFTextEncoder
from open_clip.model import CLIPVisionCfg
from transformers import CLIPVisionConfig, VisionTextDualEncoderConfig
from modeling_biomed_clip import BiomedCLIPModel


VISION_CONFIG_MAP = {
    "layers": "num_hidden_layers",
    "width": "hidden_size",
    "patch_size": "patch_size",
    "image_size": "image_size",
}
STATE_DICT_PATTERNS = [

    # Vision
    (r"visual\.head.proj.(\w+)", "visual_projection.{0}"),
    (r"visual\.trunk\.norm\.(\w+)", "vision_model.post_layernorm.{0}"),
    (r"visual\.trunk.patch_embed\.proj\.(\w+)", "vision_model.embeddings.patch_embedding.{0}"),
    (
        r"visual\.trunk\.blocks\.(\w+)\.norm2\.(\w+)",
        "vision_model.encoder.layers.{0}.layer_norm2.{1}",
    ),
    (
        r"visual\.trunk\.blocks\.(\w+)\.norm1\.(\w+)",
        "vision_model.encoder.layers.{0}.layer_norm1.{1}",
    ),
    (
        r"visual\.trunk\.blocks\.(\w+)\.attn\.proj\.(\w+)",
        "vision_model.encoder.layers.{0}.self_attn.out_proj.{1}",
    ),
    (
        r"visual\.trunk\.blocks\.(\w+)\.mlp\.fc1\.(\w+)",
        "vision_model.encoder.layers.{0}.mlp.fc1.{1}",
    ),
    (
        r"visual\.trunk\.blocks\.(\w+)\.mlp\.fc2\.(\w+)",
        "vision_model.encoder.layers.{0}.mlp.fc2.{1}",
    ),
    # Text
    (r"text\.transformer\.embeddings\.token_type_embeddings.(\w+)", "text_model.embeddings.token_type_embedding.{0}"),
    (r"text\.transformer\.embeddings\.word_embeddings.(\w+)", "text_model.embeddings.token_embedding.{0}"),
    (r"text\.transformer\.embeddings\.position_embeddings.(\w+)", "text_model.embeddings.position_embedding.{0}"),

    (r"text\.transformer\.embeddings\.LayerNorm\.(\w+)", "text_model.embeddings.layer_norm.{0}"),
    (r"text\.transformer\.encoder\.layer\.(\w+).attention.self.key.(\w+)", "text_model.encoder.layers.{0}.self_attn.k_proj.{1}"),
    (r"text\.transformer\.encoder\.layer\.(\w+).attention.self.query.(\w+)", "text_model.encoder.layers.{0}.self_attn.q_proj.{1}"),
    (r"text\.transformer\.encoder\.layer\.(\w+).attention.self.value.(\w+)", "text_model.encoder.layers.{0}.self_attn.v_proj.{1}"),
    (r"text\.transformer\.encoder\.layer\.(\w+).attention.output.dense.(\w+)", "text_model.encoder.layers.{0}.self_attn.out_proj.{1}"),
    (r"text\.transformer\.encoder\.layer\.(\w+).attention.output.LayerNorm.(\w+)", "text_model.encoder.layers.{0}.layer_norm1.{1}"),
    (r"text\.transformer\.encoder\.layer\.(\w+).intermediate.dense.(\w+)", "text_model.encoder.layers.{0}.mlp.fc1.{1}"),
    (r"text\.transformer\.encoder\.layer\.(\w+).output.LayerNorm.(\w+)", "text_model.encoder.layers.{0}.layer_norm2.{1}"),
    (r"text\.transformer\.encoder\.layer\.(\w+).output.dense.(\w+)", "text_model.encoder.layers.{0}.mlp.fc2.{1}"),
    (r"text\.proj\.0\.(\w+)", "text_projection.fc1.{0}"),
    (r"text\.proj\.2\.(\w+)", "text_projection.fc2.{0}"),
]


def convert_vision_config(config: CLIPVisionCfg):
    config = dataclasses.asdict(config)
    new_config = {
        "hidden_act": "gelu",
    }
    for key, value in config.items():
        if key in VISION_CONFIG_MAP:
            new_config[VISION_CONFIG_MAP[key]] = value
        elif key == "head_width":
            new_config["num_attention_heads"] = config["width"] // value
        elif key == "mlp_ratio":
            new_config["intermediate_size"] = int(config["width"] * value)
        elif not key.startswith("timm") and value:
            print(f"WARNING: Unknown key '{key}' in vision config.")

    return CLIPVisionConfig(**new_config)


def convert_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        found = False
        # special handling of vision attention blocks
        print(k)
        if match := re.match(r"visual\.trunk\.blocks\.(\w+)\.attn\.qkv\.(\w+)", k):
            # chunk weights into three
            chunks = v.chunk(3, dim=0)
            for proj_name, proj_v in zip(["q_proj", "k_proj", "v_proj"], chunks):
                new_k = f"vision_model.encoder.layers.{match.group(1)}.self_attn.{proj_name}.{match.group(2)}"
                print(k, "--->", new_k)
                new_state_dict[new_k] = proj_v
                found = True
        # transpose visual projection
        elif k == "visual.trunk.cls_token":
            new_k = "vision_model.embeddings.class_embedding"
            print(k, "--->", new_k)
            new_state_dict[new_k] = v.squeeze(0).squeeze(0)
            found = True
        elif k == "visual.trunk.pos_embed":
            new_k = "vision_model.embeddings.position_embedding.weight"
            print(k, "--->", new_k)
            new_state_dict[new_k] = v.squeeze(0)
            found = True
        else:
            for pattern, replacement in STATE_DICT_PATTERNS:
                if match := re.match(pattern, k):
                    new_k = replacement.format(*match.groups())
                    print(k, "--->", new_k)
                    new_state_dict[new_k] = v
                    found = True
                    break
        if not found:
            new_state_dict[k] = v

    return new_state_dict
from open_clip import create_model_from_pretrained
import json
if __name__ == "__main__":
    openclip_config = json.load(open("saliency_maps/model/config.json"))
    openclip_model, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    pt_files = glob.glob(f"saliency_maps/model/*.pt")
    if pt_files:
        # Load the first .pt file found
        state_dict = torch.load(pt_files[0])
        print(f"Loaded model: {pt_files[0]}")
    else:
        print("No .pt files found in the directory.")
    for key in list(state_dict.keys()):
        if(key.startswith("model.")):
            state_dict[key.replace('model.', '')] = state_dict.pop(key)
    openclip_model.load_state_dict(state_dict)
    for i in openclip_model.state_dict().keys():
        print(i)

    if not isinstance(openclip_model.text, HFTextEncoder):
        raise ValueError("Only HFTextEncoder is supported.")
    text_config = openclip_model.text.config
    state_dict = convert_state_dict(openclip_model.state_dict())
    from transformers import AutoModel, AutoProcessor, AutoTokenizer
    model = AutoModel.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
    torch.save(state_dict, "saliency_maps/model/pytorch_model.bin")
    