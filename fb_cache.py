import os
from typing import Optional
import torch
import torch.amp as amp

from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData

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

    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [
            "/home/ubuntu/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
            "/home/ubuntu/models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
            "/home/ubuntu/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None)

    #Add FB cache to the pipeline
    pipe.dit.___class__.enable_fbcache = True
    pipe.dit.___class__.fb_cache_thr = 0.03
    pipe.dit.___class__.first_block_cache = None
    pipe.dit.___class__.last_block_cache = None
    pipe.dit.___class__.forward = fbcache_forward

    # Text-to-video
    video = pipe(
        prompt="纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        num_inference_steps=50,
        seed=0, tiled=True
    )
    save_video(video, "video1.mp4", fps=15, quality=5)