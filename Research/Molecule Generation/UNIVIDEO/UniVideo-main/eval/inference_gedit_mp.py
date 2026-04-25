import os
import torch
import torch.distributed as dist
import cv2
import numpy as np
import yaml
import argparse
import json
import datetime
from PIL import Image
from datasets import load_dataset
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
    p.add_argument("--config", type=str, default="configs/univideo_qwen2p5vl7b_hidden_hunyuanvideo.yaml")
    p.add_argument("--output_dir", type=str, required=True, help="Root results directory")
    p.add_argument("--debug_limit", type=int, default=None)
    return p.parse_args()

def get_valid_resolution(h, w, multiple=16):
    """Rounds dimensions to nearest multiple of 16."""
    new_h = round(h / multiple) * multiple
    new_w = round(w / multiple) * multiple
    return new_h, new_w

def main():
    args = parse_args()

    # --- 1. DDP Initialization ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(minutes=120)  # Give Rank 0 two hours to download
    )
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"Initialized Distributed Process Group. World Size: {world_size}")

    # --- 3. Data Loading ---
    # Context manager to ensure only Rank 0 downloads first
    if rank == 0:
        print("Loading GEdit-Bench dataset (Rank 0 downloading)...")
        # This triggers the download and caches it locally
        load_dataset("stepfun-ai/GEdit-Bench", split="train")
    
    # CRITICAL: Make all other processes wait here until Rank 0 is finished
    dist.barrier()

    # Now that it is cached, all ranks (including 0) load it safely from disk
    if rank == 0:
        print("Dataset cached. Loading on all ranks...")
        
    dataset = load_dataset("stepfun-ai/GEdit-Bench", split="train")
    
    if args.debug_limit: 
        dataset = dataset.select(range(args.debug_limit))

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # --- 2. Load Models ---
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

    # --- 4. Setup Temp Directory for Images ---
    # We need a place to save the source images so the pipeline can read them as paths
    temp_source_dir = os.path.join(args.output_dir, "temp_sources")
    os.makedirs(temp_source_dir, exist_ok=True)

    # --- 5. Main Loop ---
    for i, global_idx in enumerate(sampler):
        item = dataset[global_idx]
        
        key = item['key']
        prompt = item['instruction']
        task_type = item['task_type']
        lang = item['instruction_language']
        input_image = item['input_image'] # PIL Image

        # Output Path
        save_dir = os.path.join(args.output_dir, "fullset", task_type, lang)
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, f"{key}.jpg")

        if os.path.exists(output_path):
            print(f"[Rank {rank}] Skipping {key} (Exists)")
            continue

        # --- Dynamic Resolution ---
        orig_w, orig_h = input_image.size
        run_h, run_w = get_valid_resolution(orig_h, orig_w)

        # --- CRITICAL FIX: Save Image to Disk for Pipeline ---
        temp_img_path = os.path.join(temp_source_dir, f"{key}_source.png")

        if not os.path.exists(temp_img_path):
            # Convert PIL -> Numpy (RGB)
            img_np = np.array(input_image.convert("RGB"))
            
            # Convert RGB -> BGR (because cv2 uses BGR)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # Save using cv2
            cv2.imwrite(temp_img_path, img_bgr)

        print(f"[Rank {rank}] Processing {key} | Res: {run_w}x{run_h} | Task: {task_type}")

        pipeline_kwargs = dict(
            prompts=[prompt],
            negative_prompt=negative_prompt,
            cond_image_path=temp_img_path, 
            height=run_h,
            width=run_w,
            num_frames=1,
            num_inference_steps=50,
            guidance_scale=7.0,
            image_guidance_scale=1.5,
            seed=42,
            timestep_shift=7.0,
            task="i2i_edit",
        )

        try:
            output = pipeline(**pipeline_kwargs)
            output_tensor = output.frames[0] # [1, H, W, C]
            img_tensor = output_tensor[0]    # [H, W, C]
            
            if isinstance(img_tensor, torch.Tensor):
                img_np = img_tensor.float().cpu().numpy()
            else:
                img_np = img_tensor

            if img_np.max() <= 1.0:
                 img_np = (img_np * 255).astype(np.uint8)
            else:
                 img_np = img_np.astype(np.uint8)

            Image.fromarray(img_np).save(output_path, quality=95)
            
        except Exception as e:
            print(f"[Rank {rank}] Error processing {key}: {e}")
        
        # Optional: Clean up temp file to save space?
        # Generally safer to keep them until run finishes, or just delete output_dir/temp_sources manually later.

    print(f"[Rank {rank}] Finished.")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()