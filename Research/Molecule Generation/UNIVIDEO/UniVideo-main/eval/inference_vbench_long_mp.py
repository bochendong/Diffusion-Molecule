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
        "--vbench_info_path",
        type=str,
        default="eval/vbench/VBench_full_info.json",
        help="Path to the VBench_full_info.json file",
    )
    p.add_argument(
        "--augmented_prompts_path",
        type=str,
        default="eval/vbench/all_dimension_longer.txt",
        help="Path to the all_dimension_longer.txt file",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the generated videos",
    )
    p.add_argument(
        "--num_videos",
        type=int,
        default=5,
        help="Number of videos per prompt (Standard=5, Debug=1)",
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

    # --- 2. Load and Align VBench Data (Augmented Version) ---
    if rank == 0:
        print(f"[INIT] Loading VBench Info from: {args.vbench_info_path}")
        print(f"[INIT] Loading Augmented Prompts from: {args.augmented_prompts_path}")

    # Load JSON (contains short prompts + dimensions)
    with open(args.vbench_info_path, "r") as f:
        vbench_data = json.load(f)

    # Load TXT (contains long prompts)
    with open(args.augmented_prompts_path, "r") as f:
        augmented_prompts = [line.strip() for line in f.readlines()]

    # Critical Safety Check: The lines must match exactly
    if len(vbench_data) != len(augmented_prompts):
        raise ValueError(f"CRITICAL ERROR: Length mismatch! JSON has {len(vbench_data)} items, "
                         f"TXT has {len(augmented_prompts)} lines. Files must align perfectly.")

    # Create aligned dataset list
    full_dataset = []
    for info, long_p in zip(vbench_data, augmented_prompts):
        full_dataset.append({
            "dimension": info['dimension'],
            "short_prompt": info['prompt_en'], # Used for FILENAME
            "long_prompt": long_p              # Used for GENERATION
        })

    # --- 3. Load Models ---
    with open(args.config, "r") as f:
        raw = yaml.safe_load(f)

    mllm_config = MLLMInContextConfig(**raw["mllm_config"])
    pipe_cfg = UniVideoPipelineConfig(**raw["pipeline_config"])
    transformer_ckpt_path = raw.get("transformer_ckpt_path")
    mllm_encoder_ckpt_path = raw.get("mllm_encoder_ckpt", None)

    mllm_encoder = MLLMInContext(mllm_config)
    if mllm_encoder_ckpt_path is not None:
        if rank == 0: print(f"[INIT] loading mllm_encoder ckpt from {mllm_encoder_ckpt_path}")
        mllm_encoder = load_model(mllm_encoder, mllm_encoder_ckpt_path)
    mllm_encoder.requires_grad_(False)
    mllm_encoder.eval()

    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        pipe_cfg.hunyuan_model_id,
        subfolder="vae",
        low_cpu_mem_usage=True,
    )
    vae.eval()

    qwenvl_txt_dim = 3584
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        pipe_cfg.hunyuan_model_id,
        subfolder="transformer",
        low_cpu_mem_usage=True, 
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

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        pipe_cfg.hunyuan_model_id,
        subfolder="scheduler"
    )

    pipeline = UniVideoPipeline(
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        mllm_encoder=mllm_encoder,
        univideo_config=pipe_cfg
    ).to(device=device, dtype=torch.bfloat16)

    negative_prompt = "Bright tones, overexposed, oversharpening, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, walking backwards, computer-generated environment, weak dynamics, distorted and erratic motions, unstable framing and a disorganized composition."

    # --- 4. Distributed Sampler ---
    sampler = DistributedSampler(
        dataset=full_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    # --- 5. Generation Loop ---
    for i, global_idx in enumerate(sampler):
        item = full_dataset[global_idx]
        
        dimension = item['dimension']

        if isinstance(dimension, list):
            # Take the first element if it's a list (e.g. ['human_action'] -> 'human_action')
            dimension = dimension[0]

        short_prompt = item['short_prompt'] 
        long_prompt = item['long_prompt']

        # Output Structure: output_dir / dimension / short_prompt-0.mp4
        save_dir = os.path.join(args.output_dir, dimension)
        os.makedirs(save_dir, exist_ok=True)

        print(f"[Rank {rank}] Processing ({dimension}): {long_prompt[:40]}...")

        for video_idx in range(args.num_videos):
            filename = f"{short_prompt}-{video_idx}.mp4"
            output_path = os.path.join(save_dir, filename)

            if os.path.exists(output_path):
                print(f"[Rank {rank}] Skipping {filename} (Exists)")
                continue

            pipeline_kwargs = dict(
                prompts=[long_prompt], # CRITICAL: Generating with LONG Prompt
                negative_prompt=negative_prompt,
                height=480,
                width=832,
                num_frames=81,
                num_inference_steps=50,
                guidance_scale=6.0,
                image_guidance_scale=1.0,
                seed=42 + video_idx,
                timestep_shift=7.0,
                task="t2v",
            )

            with torch.no_grad():
                output = pipeline(**pipeline_kwargs)
                video_frames = output.frames[0]

            export_to_video(video_frames, output_path, fps=16)
            print(f"[Rank {rank}] Saved: {output_path}")

    print(f"[Rank {rank}] Finished.")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()