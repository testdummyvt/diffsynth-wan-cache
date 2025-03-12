import os
import time
from typing import Optional
import torch
import torch.amp as amp

from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def fbcache_forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        x, (f, h, w) = self.patchify(x)
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        def check_current_cache_block_similarity(cache, current_block, threshold):
            mean_diff = torch.mean(torch.abs(cache - current_block))
            mean_x = torch.mean(torch.abs(cache))
            diff = mean_diff / mean_x
            print(diff.item())
            return diff.item() < threshold

        for b_id, block in enumerate(self.blocks):
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            else:
                # Check if fbcache is enabled and perform fbcache based forward pass
                if b_id == 0:
                    x = block(x, context, t_mod, freqs)
                    if self.enable_fbcache:
                        if self.first_block_cache is None:
                            self.first_block_cache = x
                        else:
                            apply_fbcache = check_current_cache_block_similarity(self.first_block_cache, x, self.fb_cache_thr)
                            if apply_fbcache:
                                print("Applying FB Cache")
                                return self.last_block_cache
                            else:
                                self.first_block_cache = x
                else:
                    x = block(x, context, t_mod, freqs)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        if self.enable_fbcache:
            self.last_block_cache = x
        return x

if __name__ == "__main__":
    # Load models
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [
            [
                "/home/ubuntu/models/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors",
                "/home/ubuntu/models/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors",
                "/home/ubuntu/models/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors",
                "/home/ubuntu/models/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors",
                "/home/ubuntu/models/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors",
                "/home/ubuntu/models/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors",
            ],
            "/home/ubuntu/models/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
            "/home/ubuntu/models/Wan2.1-T2V-14B/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.float8_e4m3fn, # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
    )
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.

    for fb_cache_thr in [0.0, 0.03, 0.04, 0.06]:
        #Add FB cache to the pipeline
        enable_fbcache = False
        if fb_cache_thr > 0:
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

# WAN1.3b model Test
# if __name__ == "__main__":

#     model_manager = ModelManager(device="cpu")
#     model_manager.load_models(
#         [
#             "/home/ubuntu/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
#             "/home/ubuntu/models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
#             "/home/ubuntu/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
#         ],
#         torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
#     )
#     pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
#     pipe.enable_vram_management(num_persistent_param_in_dit=None)

#     for fb_cache_thr in [0.0, 0.03, 0.04, 0.06]:
#         #Add FB cache to the pipeline
#         enable_fbcache = False
#         if fb_cache_thr > 0:
#             enable_fbcache = True
#         pipe.dit.__class__.enable_fbcache = enable_fbcache
#         pipe.dit.__class__.fb_cache_thr = fb_cache_thr
#         pipe.dit.__class__.first_block_cache = None
#         pipe.dit.__class__.last_block_cache = None
#         pipe.dit.__class__.forward = fbcache_forward

        
#         start_time = time.time()
#         # Text-to-video
#         video = pipe(
#             prompt="纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。",
#             negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#             num_inference_steps=50,
#             seed=0, tiled=True
#         )

#         end_time = time.time()
#         execution_time = end_time - start_time

#         save_video(video, f"./assets/Wan2.1-T2V-1.3B__fbcache_{fb_cache_thr:.2f}__{execution_time:.2f}.mp4", fps=15, quality=5)