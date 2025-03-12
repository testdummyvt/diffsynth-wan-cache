import os
import time
from typing import Optional
import torch

from fbcache import fbcache_forward

from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":

    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [
            "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
            "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
            "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None)

    fb_cache_thr = 0.03
    #Add FB cache to the pipeline
    enable_fbcache = True
    pipe.dit.__class__.enable_fbcache = enable_fbcache
    pipe.dit.__class__.fb_cache_thr = fb_cache_thr
    pipe.dit.__class__.first_block_cache = None
    pipe.dit.__class__.last_block_cache = None
    pipe.dit.__class__.forward = fbcache_forward

    
    start_time = time.time()
    # Text-to-video
    video = pipe(
        prompt="纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        num_inference_steps=50,
        seed=0, tiled=True
    )

    end_time = time.time()
    execution_time = end_time - start_time

    save_video(video, f"./assets/Wan2.1-T2V-1.3B__fbcache_{fb_cache_thr:.2f}__{execution_time:.2f}.mp4", fps=15, quality=5)