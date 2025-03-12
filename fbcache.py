from typing import Optional
import torch

from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d

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
        """
        Forward pass of the model with fbcache (First Block Cache) optimization.

        This function implements the forward pass of the diffusion model with an optional fbcache mechanism that caches the output of first block in training.

        Args:
            self: The model instance.
            x (torch.Tensor): Input tensor.
            timestep (torch.Tensor): Timestep tensor.
            context (torch.Tensor): Context tensor.
            clip_feature (Optional[torch.Tensor], optional): Clip feature tensor. Defaults to None.
            y (Optional[torch.Tensor], optional): Optional input tensor to concatenate with x. Defaults to None.
            use_gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
            use_gradient_checkpointing_offload (bool, optional): Whether to offload gradient checkpointing to CPU. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Output tensor.
        """
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
            # print(diff.item())
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
                    # First Block Cache Implementation
                    # first_block_cache is a cache for the output of the first block
                    # it checks the similarity between the cached output with the current one to see whether or not they're similar
                    # if they're similar it reuses last_block_cache, otherwise it overwrites first_block_cache with the current one
                    if self.enable_fbcache:
                        if self.first_block_cache is None:
                            self.first_block_cache = x
                        else:
                            apply_fbcache = check_current_cache_block_similarity(self.first_block_cache, x, self.fb_cache_thr)
                            if apply_fbcache:
                                # print("Applying FB Cache")
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