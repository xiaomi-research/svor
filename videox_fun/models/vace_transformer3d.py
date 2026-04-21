import torch
import torch.cuda.amp as amp
import torch.nn as nn
import types
from diffusers.configuration_utils import register_to_config
from diffusers.utils import is_torch_version
import math
from typing import Any, Dict

from .wan_transformer3d import WanTransformer3DModel, WanAttentionBlock, sinusoidal_embedding_1d
from ..dist import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
from ..dist.wan_xfuser import usp_attn_forward


class VaceWanAttentionBlock(WanAttentionBlock):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        block_id=0,
    ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, x, e, seq_lens, grid_sizes, freqs, context, context_lens, dtype=torch.float32):
        if self.block_id == 0:
            c = self.before_proj(c)
            c = c + x

        e = (self.modulation + e).chunk(6, dim=1)

        # self-attention
        temp_c = self.norm1(c) * (1 + e[1]) + e[0]
        temp_c = temp_c.to(dtype)

        y = self.self_attn(temp_c, seq_lens, grid_sizes, freqs, dtype)
        c = c + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            # cross-attention
            x = x + self.cross_attn(self.norm3(x), context, context_lens, dtype)

            # ffn function
            temp_x = self.norm2(x) * (1 + e[4]) + e[3]
            temp_x = temp_x.to(dtype)

            y = self.ffn(temp_x)
            x = x + y * e[5]
            return x

        c = cross_attn_ffn(c, context, context_lens, e)

        c_skip = self.after_proj(c)
        return c, c_skip


class BaseWanAttentionBlock(WanAttentionBlock):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        block_id=None,
    ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.block_id = block_id

    def forward(self, x, hints, context_scale=1.0, **kwargs):
        x = super().forward(x, **kwargs)
        if self.block_id is not None:
            x = x + hints[self.block_id] * context_scale
        return x


class VaceWanModel(WanTransformer3DModel):
    @register_to_config
    def __init__(
        self,
        vace_layers=None,
        vace_in_dim=None,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        in_channels=16,
        hidden_size=2048,
    ):
        super().__init__(
            model_type,
            patch_size,
            text_len,
            in_dim,
            dim,
            ffn_dim,
            freq_dim,
            text_dim,
            out_dim,
            num_heads,
            num_layers,
            window_size,
            qk_norm,
            cross_attn_norm,
            eps,
            in_channels,
            hidden_size,
        )

        self.vace_layers = [i for i in range(0, self.num_layers, 2)] if vace_layers is None else vace_layers
        self.vace_in_dim = self.in_dim if vace_in_dim is None else vace_in_dim

        assert 0 in self.vace_layers
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        # blocks
        self.blocks = nn.ModuleList(
            [
                BaseWanAttentionBlock(
                    "t2v_cross_attn",
                    self.dim,
                    self.ffn_dim,
                    self.num_heads,
                    self.window_size,
                    self.qk_norm,
                    self.cross_attn_norm,
                    self.eps,
                    block_id=self.vace_layers_mapping[i] if i in self.vace_layers else None,
                )
                for i in range(self.num_layers)
            ]
        )

        # vace blocks
        self.vace_blocks = nn.ModuleList(
            [
                VaceWanAttentionBlock(
                    "t2v_cross_attn",
                    self.dim,
                    self.ffn_dim,
                    self.num_heads,
                    self.window_size,
                    self.qk_norm,
                    self.cross_attn_norm,
                    self.eps,
                    block_id=i,
                )
                for i in self.vace_layers
            ]
        )

        # vace patch embeddings
        self.vace_patch_embedding = nn.Conv3d(
            self.vace_in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size
        )

    def forward_vace(
        self,
        x,
        vace_context,
        seq_len,
        kwargs,
    ):
        # embeddings
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in c])
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in c])

        # arguments
        new_kwargs = dict(x=x)
        new_kwargs.update(kwargs)

        # Context Parallel
        if self.sp_world_size > 1:
            c = torch.chunk(c, self.sp_world_size, dim=1)[self.sp_world_rank]

        hints = []
        for block in self.vace_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs, **kwargs):
                        return module(*inputs, **kwargs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                c, c_skip = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    c,
                    **new_kwargs,
                    **ckpt_kwargs,
                )
            else:
                c, c_skip = block(c, **new_kwargs)

            hints.append(c_skip)
        return hints, grid_sizes

    def unpatchify_vace(self, x, grid_sizes, c=None, with_patch=False):
        if c is None:
            c = self.out_dim
        else:
            c = c
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            if with_patch:
                u = u[: math.prod(v)].view(*v, *self.patch_size, c)
                u = torch.einsum("fhwpqrc->cfphqwr", u)
                u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            else:
                u = u[: math.prod(v)].view(*v, c)
                u = torch.einsum("fhwc->cfhw", u)
            out.append(u)
        return out

    def forward(
        self,
        x,
        t,
        vace_context,
        context,
        seq_len,
        vace_context_scale=1.0,
        clip_fea=None,
        y=None,
        cond_flag=True,
        hints_dict=None,
        return_hints=False,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # if self.model_type == 'i2v':
        #     assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        dtype = x.dtype
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # if y is not None:
        #     x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        if self.sp_world_size > 1:
            seq_len = int(math.ceil(seq_len / self.sp_world_size)) * self.sp_world_size
        assert seq_lens.max() <= seq_len
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            # assert e.dtype == torch.float32 and e0.dtype == torch.float32
            e0 = e0.to(dtype)
            e = e.to(dtype)

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        )

        if self.sp_world_size > 1:
            x = torch.chunk(x, self.sp_world_size, dim=1)[self.sp_world_rank]

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            dtype=dtype,
        )

        hints, hints_grid_size = self.forward_vace(x, vace_context, seq_len, kwargs)

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs, **kwargs):
                        return module(*inputs, **kwargs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    hints,
                    vace_context_scale,
                    **kwargs,
                    **ckpt_kwargs,
                )
            else:
                x = block(x, hints, vace_context_scale, **kwargs)

        if self.sp_world_size > 1:
            x = get_sp_group().all_gather(x, dim=1)

        # head
        x = self.head(x, e)
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        x = torch.stack(x)

        return x

    def enable_multi_gpus_inference(
        self,
    ):
        self.sp_world_size = get_sequence_parallel_world_size()
        self.sp_world_rank = get_sequence_parallel_rank()
        for block in self.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        for block in self.vace_blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
