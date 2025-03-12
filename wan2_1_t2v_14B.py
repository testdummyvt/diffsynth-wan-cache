import os
import time
from typing import Optional
import torch

from fbcache import fbcache_forward

from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData

os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == "__main__":
    # Load models
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [
            [
                "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors",
                "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors",
                "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors",
                "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors",
                "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors",
                "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors",
            ],
            "models/Wan-AI/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
            "models/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.float8_e4m3fn, # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
    )
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.

    fb_cache_thr = 0.06
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
        prompt="一名宇航员身穿太空服，面朝镜头骑着一匹机械马在火星表面驰骋。红色的荒凉地表延伸至远方，点缀着巨大的陨石坑和奇特的岩石结构。机械马的步伐稳健，扬起微弱的尘埃，展现出未来科技与原始探索的完美结合。宇航员手持操控装置，目光坚定，仿佛正在开辟人类的新疆域。背景是深邃的宇宙和蔚蓝的地球，画面既科幻又充满希望，让人不禁畅想未来的星际生活。",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        num_inference_steps=50,
        seed=0, tiled=True
    )

    end_time = time.time()
    execution_time = end_time - start_time

    save_video(video, f"./assets/Wan2.1-T2V-14B__fbcache_{fb_cache_thr:.2f}__{execution_time:.2f}.mp4", fps=15, quality=5)
