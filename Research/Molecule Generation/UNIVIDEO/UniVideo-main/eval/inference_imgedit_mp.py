import os
import torch
import torch.distributed as dist
import numpy as np
import yaml
import argparse
import json
from PIL import Image
from diffusers.utils import export_to_video
from diffusers.models.autoencoders.autoencoder_kl_hunyuan_video import AutoencoderKLHunyuanVideo
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformer_univideo_hunyuan_video import HunyuanVideoTransformer3DModel, TwoLayerMLP
from mllm_encoder import MLLMInContext, MLLMInContextConfig
from pipeline_univideo import UniVideoPipeline, UniVideoPipelineConfig
from utils import pad_image_pil_to_square, load_model
from torch.utils.data import DistributedSampler

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        default="configs/univideo_qwen2p5vl7b_hidden_hunyuanvideo.yaml",
        help="Path to yaml config file",
    )
    p.add_argument(
        "--metadata_path",
        type=str,
        required=True,
        help="Path to the basic_edit.json file",
    )
    p.add_argument(
        "--origin_img_root",
        type=str,
        required=True,
        help="Root directory containing the original images",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the generated images",
    )
    return p.parse_args()

def main():
    args = parse_args()

    # --- 1. DDP Initialization ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"Initialized Distributed Process Group. World Size: {world_size}")
    print(f"[Rank {rank}] Initialized on GPU {local_rank}")

    # Load config
    with open(args.config, "r") as f:
        raw = yaml.safe_load(f)

    if "mllm_config" not in raw:
        raise KeyError("Missing required config section: mllm_config")
    if "pipeline_config" not in raw:
        raise KeyError("Missing required config section: pipeline_config")

    mllm_config = MLLMInContextConfig(**raw["mllm_config"])
    pipe_cfg = UniVideoPipelineConfig(**raw["pipeline_config"])
    transformer_ckpt_path = raw.get("transformer_ckpt_path")
    mllm_encoder_ckpt_path = raw.get("mllm_encoder_ckpt", None)

    # Create MLLM encoder
    mllm_encoder = MLLMInContext(mllm_config)
    if mllm_encoder_ckpt_path is not None:
        if rank == 0: print(f"[INIT] loading mllm_encoder ckpt from {mllm_encoder_ckpt_path}")
        mllm_encoder = load_model(mllm_encoder, mllm_encoder_ckpt_path)
    mllm_encoder.requires_grad_(False)
    mllm_encoder.eval()

    # Load HunyuanVideo VAE
    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        pipe_cfg.hunyuan_model_id,
        subfolder="vae",
        low_cpu_mem_usage=True,
        device_map=None
    )
    vae.eval()

    # Load HunyuanVideo transformer
    qwenvl_txt_dim = 3584
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        pipe_cfg.hunyuan_model_id,
        subfolder="transformer",
        low_cpu_mem_usage=True, 
        device_map=None, 
        text_embed_dim=qwenvl_txt_dim 
    )
    transformer.qwen_project_in = TwoLayerMLP(qwenvl_txt_dim, qwenvl_txt_dim * 4, 4096)

    with torch.no_grad():
        torch.nn.init.ones_(transformer.qwen_project_in.ln.weight)
        for layer in transformer.qwen_project_in.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight, gain=1.0)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
    
    if rank == 0:
        print(f"[INIT] Reinitialized qwen_project_in ({qwenvl_txt_dim} -> {qwenvl_txt_dim * 4} -> 4096)")

    # Load ckpt
    def rename_func(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace("transformer.", "", 1) if k.startswith("transformer.") else k
            new_state_dict[new_k] = v
        return new_state_dict

    if isinstance(transformer_ckpt_path, str):
        if rank == 0: print(f"[INIT] loading ckpt from {transformer_ckpt_path}")
        transformer = load_model(transformer, transformer_ckpt_path, rename_func=rename_func)

    # Load scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        pipe_cfg.hunyuan_model_id,
        subfolder="scheduler"
    )

    # --- 2. Move Pipeline to GPU ---
    pipeline = UniVideoPipeline(
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        mllm_encoder=mllm_encoder,
        univideo_config=pipe_cfg
    ).to(device=device, dtype=torch.bfloat16)

    negative_prompt = "Bright tones, overexposed, oversharpening, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, walking backwards, computer-generated environment, weak dynamics, distorted and erratic motions, unstable framing and a disorganized composition."

    # --- 3. Data Loading ---
    # Load the full JSON dictionary
    if rank == 0:
        print(f"Loading metadata from {args.metadata_path}")
        
    with open(args.metadata_path, "r") as f:
        json_data = json.load(f)

    # Convert dictionary to a sorted list of items for deterministic splitting across GPUs
    # Each item is a tuple: (key, value_dict)
    data_items = sorted(list(json_data.items()), key=lambda x: x[0])
    
    # Create sampler
    sampler = DistributedSampler(
        dataset=data_items,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # --- 4. Main Processing Loop ---
    for i, global_idx in enumerate(sampler):
        # Extract the key (e.g., "1082") and the data dict
        img_key, item_data = data_items[global_idx]
        
        # Define output filename: {key}.png inside output_dir
        output_filename = f"{img_key}.png"
        output_path = os.path.join(args.output_dir, output_filename)

        # Resume Logic
        if os.path.exists(output_path):
            print(f"[Rank {rank}] Skipping {img_key} (Already exists)")
            continue

        # Construct full path to original image
        relative_path = item_data["id"]
        cond_image_path = os.path.join(args.origin_img_root, relative_path)
        
        if not os.path.exists(cond_image_path):
            print(f"[Rank {rank}] Warning: Source image not found for {img_key} at {cond_image_path}. Skipping.")
            continue

        prompt = item_data["prompt"]
        print(f"[Rank {rank}] Processing {img_key}: {prompt[:40]}...")

        # I2I Editing Pipeline Config
        pipeline_kwargs = dict(
            prompts=[prompt],
            negative_prompt=negative_prompt,
            cond_image_path=cond_image_path,
            height=1024,
            width=1024,
            num_frames=1,
            num_inference_steps=50,
            guidance_scale=7.0,
            image_guidance_scale=1.5,
            seed=42,
            timestep_shift=7.0,
            task="i2i_edit",
        )

        # Generate (single image)
        try:
            output = pipeline(**pipeline_kwargs)
            output = output.frames[0] 

            F, H, W, C = output.shape
            assert C == 3, f"Expected RGB, got C={C}"
            
            img = output[0] 
            
            if img.min() < 0:
                img = (img + 1.0) / 2.0
            img = (img * 255).clip(0, 255).astype(np.uint8)
            
            # Save directly as {key}.png
            Image.fromarray(img).save(output_path)
            
        except Exception as e:
            print(f"[Rank {rank}] Error processing {img_key}: {e}")

    print(f"[Rank {rank}] Finished.")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()