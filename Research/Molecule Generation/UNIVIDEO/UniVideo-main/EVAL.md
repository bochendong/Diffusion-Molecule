
# GenEval

## Short prompt
```
cd eval
python -m torch.distributed.run \
    --nproc_per_node=8 \
    inference_geneval_mp.py \
    --config configs/univideo_qwen2p5vl7b_hidden_hunyuanvideo.yaml \
    --metadata_path geneval/evaluation_metadata.jsonl \
    --output_dir geneval/result_mp \
    --num_images 4
```
## Augmented prompt
```
cd eval
python -m torch.distributed.run \
    --nproc_per_node=8 \
    inference_geneval_mp.py \
    --config configs/univideo_qwen2p5vl7b_hidden_hunyuanvideo.yaml \
    --metadata_path geneval/evaluation_metadata_long.jsonl \
    --output_dir geneval/result_mp_long \
    --num_images 4
```


# ImagEdit
```
cd eval
python -m torch.distributed.run \
    --nproc_per_node=8 \
    inference_imgedit_mp.py \
    --config configs/univideo_qwen2p5vl7b_hidden_hunyuanvideo.yaml \
    --metadata_path imgedit/basic_edit.json \
	--origin_img_root imgedit/Benchmark/singleturn \
    --output_dir imgedit/result_mp
```

# Vbench

## Short prompt
```
cd eval
python -m torch.distributed.run \
    --nproc_per_node=8 \
    inference_vbench_mp.py \
    --config configs/univideo_qwen2p5vl7b_hidden_hunyuanvideo.yaml \
    --vbench_info_path vbench/VBench_full_info.json \
    --output_dir vbench/short/result \
    --num_videos 5
```

## Augmented prompt
```
cd eval
python -m torch.distributed.run \
    --nproc_per_node=8 \
    inference_vbench_long_mp.py \
    --config configs/univideo_qwen2p5vl7b_hidden_hunyuanvideo.yaml \
    --vbench_info_path vbench/VBench_full_info.json \
    --augmented_prompts_path all_dimension_longer.txt \
    --output_dir vbench/long/result \
    --num_videos 5
```