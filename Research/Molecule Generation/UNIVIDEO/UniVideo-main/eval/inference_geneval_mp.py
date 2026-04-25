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
        help="Path to the evaluation_metadata.jsonl file",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the generated images and metadata",
    )
    p.add_argument(
        "--num_images",
        type=int,
        default=4,
        help="Number of images to generate per prompt",
    )
    return p.parse_args()

def main():
    args = parse_args()

    # --- 1. DDP Initialization (Standard torchrun method) ---
    # torchrun automatically sets these environment variables
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

    # Create MLLM encoder from config
    mllm_encoder = MLLMInContext(mllm_config)

    # Load mllm_encoder checkpoint if provided
    if mllm_encoder_ckpt_path is not None:
        if rank == 0: print(f"[INIT] loading mllm_encoder ckpt from {mllm_encoder_ckpt_path}")
        mllm_encoder = load_model(mllm_encoder, mllm_encoder_ckpt_path)
    mllm_encoder.requires_grad_(False)
    mllm_encoder.eval()

    # Load HunyuanVideo VAE
    # Note: If you OOM on CPU RAM, set low_cpu_mem_usage=True (requires accelerate)
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

    # --- 2. Move Pipeline to Specific GPU ---
    # Crucial: Use 'device' variable which points to "cuda:local_rank"
    pipeline = UniVideoPipeline(
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        mllm_encoder=mllm_encoder,
        univideo_config=pipe_cfg
    ).to(device=device, dtype=torch.bfloat16)

    # Inference settings
    negative_prompt = "Bright tones, overexposed, oversharpening, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, walking backwards, computer-generated environment, weak dynamics, distorted and erratic motions, unstable framing and a disorganized composition."

    # Read the metadata file
    with open(args.metadata_path, "r") as f:
        metadata_lines = f.readlines()

    # --- 3. Distributed Sampler Setup ---
    # Using shuffle=False to ensure deterministic assignment for easier debugging
    sampler = DistributedSampler(
        dataset=metadata_lines,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    # --- 4. Main Processing Loop ---
    # Note: DistributedSampler yields INDICES, not items.
    for i, global_idx in enumerate(sampler):
        line = metadata_lines[global_idx]
        metadata = json.loads(line.strip())

        # Use global_idx for folder name to prevent overwrites
        folder_name = f"{args.output_dir}/{str(global_idx).zfill(5)}"
        
        # Resume Logic: Check if last image exists
        if os.path.exists(f"{folder_name}/samples/{str(args.num_images-1).zfill(4)}.png"):
            print(f"[Rank {rank}] Skipping prompt {global_idx} (Already completed)")
            continue

        os.makedirs(f"{folder_name}/samples", exist_ok=True)

        prompt = metadata["prompt"]
        print(f"[Rank {rank}] Processing prompt {global_idx}: {prompt[:40]}...")

        pipeline_kwargs = dict(
            prompts=[prompt],
            negative_prompt=negative_prompt,
            height=1024,
            width=1024,
            num_frames=1,
            num_inference_steps=50,
            guidance_scale=7.0,
            image_guidance_scale=1.0,
            seed=42,
            timestep_shift=7.0,
            task="t2i",
        )

        for img_idx in range(args.num_images):
            # Save metadata once per prompt
            if img_idx == 0:
                with open(f"{folder_name}/metadata.jsonl", "w") as meta_file:
                    json.dump(metadata, meta_file)

            output = pipeline(**pipeline_kwargs)
            output = output.frames[0] 

            F, H, W, C = output.shape
            assert C == 3, f"Expected RGB, got C={C}"
            assert F == 1
            img = output[0] 
            
            if img.min() < 0:
                img = (img + 1.0) / 2.0
            img = (img * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img).save(f"{folder_name}/samples/{str(img_idx).zfill(4)}.png")

    print(f"[Rank {rank}] Finished.")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()