import os
import time
from typing import Optional
import torch

from fbcache import fbcache_forward

from diffsynth import ModelManager, WanVideoPipeline, save_video
from modelscope import dataset_snapshot_download
from PIL import Image

os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == "__main__":
    # Load models
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        ["/home/ubuntu/models/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
        torch_dtype=torch.float32, # Image Encoder is loaded with float32
    )
    model_manager.load_models(
        [
            [
                "/home/ubuntu/models/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00001-of-00007.safetensors",
                "/home/ubuntu/models/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00002-of-00007.safetensors",
                "/home/ubuntu/models/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00003-of-00007.safetensors",
                "/home/ubuntu/models/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00004-of-00007.safetensors",
                "/home/ubuntu/models/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00005-of-00007.safetensors",
                "/home/ubuntu/models/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00006-of-00007.safetensors",
                "/home/ubuntu/models/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00007-of-00007.safetensors",
            ],
            "/home/ubuntu/models/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth",
            "/home/ubuntu/models/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.float8_e4m3fn, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.

    # Download example image
    dataset_snapshot_download(
        dataset_id="DiffSynth-Studio/examples_in_diffsynth",
        local_dir="/home/ubuntu/",
        allow_file_pattern=f"data/examples/wan/input_image.jpg"
    )
    image = Image.open("/home/ubuntu/data/examples/wan/input_image.jpg")

    fb_cache_thr = 0.06
    #Add FB cache to the pipeline
    enable_fbcache = True
    pipe.dit.__class__.enable_fbcache = enable_fbcache
    pipe.dit.__class__.fb_cache_thr = fb_cache_thr
    pipe.dit.__class__.first_block_cache = None
    pipe.dit.__class__.last_block_cache = None
    pipe.dit.__class__.forward = fbcache_forward

    start_time = time.time()
    # Image-to-video
    video = pipe(
        prompt="一艘小船正勇敢地乘风破浪前行。蔚蓝的大海波涛汹涌，白色的浪花拍打着船身，但小船毫不畏惧，坚定地驶向远方。阳光洒在水面上，闪烁着金色的光芒，为这壮丽的场景增添了一抹温暖。镜头拉近，可以看到船上的旗帜迎风飘扬，象征着不屈的精神与冒险的勇气。这段画面充满力量，激励人心，展现了面对挑战时的无畏与执着。",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        input_image=image,
        num_inference_steps=50,
        seed=0, tiled=True
    )

    end_time = time.time()
    execution_time = end_time - start_time

    save_video(video, f"./assets/Wan2.1-T2V-14B__fbcache_{fb_cache_thr:.2f}__{execution_time:.2f}.mp4", fps=15, quality=5)
