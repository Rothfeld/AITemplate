#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
from typing import Optional, Tuple, Union

from aitemplate.compiler import ops
from aitemplate.frontend import nn

from .embeddings import TimestepEmbedding, Timesteps
from .unet_blocks import get_down_block, UNetMidBlock2DCrossAttn


class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        # conditioning_embedding_channels: int,
        # conditioning_channels: int = 3,
        # block_out_channels: Tuple[int] = (16, 32, 96, 256),
    ):
        super().__init__()
        """
        Note: This is different to diffusers ControlNetConditioningEmbedding
        Required Conv2dBiasFewChannels for the first layer, then Conv2dBias for the rest
        Could be changed back to a loop and use parameters though,
        but it ended up like this when debugging.
        """
        self.conv_in = nn.Conv2dBiasFewChannels(3, 16, 3, 1, 1)

        self.blocks = nn.ModuleList([])
        self.blocks.append(nn.Conv2dBias(16, 16, 3, 1, 1))
        self.blocks.append(nn.Conv2dBias(16, 32, 3, 2, 1))
        self.blocks.append(nn.Conv2dBias(32, 32, 3, 1, 1))
        self.blocks.append(nn.Conv2dBias(32, 96, 3, 2, 1))
        self.blocks.append(nn.Conv2dBias(96, 96, 3, 1, 1))
        self.blocks.append(nn.Conv2dBias(96, 256, 3, 2, 1))

        self.conv_out = nn.Conv2dBias(256, 320, 3, 1, 1)

    def forward(self, conditioning):
        """
        Padding required!
        """
        pad = ops.nhwc3to4()
        conditioning = pad(conditioning)
        embedding = self.conv_in(conditioning)
        embedding = ops.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = ops.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding


class ControlNetModel(nn.Module):
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int = 4,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 768,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        use_linear_projection: bool = False,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int]] = (16, 32, 96, 256),
        global_pool_conditions: bool = False,
    ):
        super().__init__()
        self.controlnet_conditioning_channel_order = (
            controlnet_conditioning_channel_order
        )
        self.global_pool_conditions = global_pool_conditions

        # input
        self.conv_in = nn.Conv2dBias(in_channels, block_out_channels[0], 3, 1, 1)

        # time
        time_embed_dim = block_out_channels[0] * 4

        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
        )

        # control net conditioning embedding
        self.controlnet_cond_embedding = ControlNetConditioningEmbedding()

        self.down_blocks = nn.ModuleList([])
        self.controlnet_down_blocks = nn.ModuleList([])

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]

        controlnet_block = nn.Conv2dBias(output_channel, output_channel, 1)
        controlnet_block = controlnet_block
        self.controlnet_down_blocks.append(controlnet_block)

        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                use_linear_projection=use_linear_projection,
            )
            self.down_blocks.append(down_block)

            for _ in range(layers_per_block):
                controlnet_block = nn.Conv2dBias(output_channel, output_channel, 1)
                controlnet_block = controlnet_block
                self.controlnet_down_blocks.append(controlnet_block)

            if not is_final_block:
                controlnet_block = nn.Conv2dBias(output_channel, output_channel, 1)
                controlnet_block = controlnet_block
                self.controlnet_down_blocks.append(controlnet_block)

        # mid
        mid_block_channel = block_out_channels[-1]

        controlnet_block = nn.Conv2dBias(mid_block_channel, mid_block_channel, 1)
        controlnet_block = controlnet_block
        self.controlnet_mid_block = controlnet_block

        self.mid_block = UNetMidBlock2DCrossAttn(
            in_channels=mid_block_channel,
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attention_head_dim[-1],
            resnet_groups=norm_num_groups,
            use_linear_projection=use_linear_projection,
            upcast_attention=upcast_attention,
        )

    def get_shape(self, sample):
        return [i._attrs["int_var"]._attrs["values"][0] for i in ops.size()(sample)]

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        controlnet_cond,
        conditioning_scale: float = 1.0,
    ) -> Tuple:
        t_emb = self.time_proj(timestep)
        emb = self.time_embedding(t_emb)

        # 2. pre-process
        sample = self.conv_in(sample)

        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
        controlnet_cond._attrs["shape"] = sample._attrs["shape"]
        sample = sample + controlnet_cond
        # 3. down
        down_block_res_samples = (sample,)  # up to but excluding last element
        sample, res_samples = self.down_blocks[0](
            hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states
        )
        down_block_res_samples += res_samples
        sample, res_samples = self.down_blocks[1](
            hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states
        )
        down_block_res_samples += res_samples
        sample, res_samples = self.down_blocks[2](
            hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states
        )
        down_block_res_samples += res_samples
        sample, res_samples = self.down_blocks[3](hidden_states=sample, temb=emb)
        down_block_res_samples += res_samples
        # return sample

        # 4. mid
        sample = self.mid_block(
            sample, emb, encoder_hidden_states=encoder_hidden_states
        )
        controlnet_down_block_res_samples = ()

        for down_block_res_sample, controlnet_block in zip(
            down_block_res_samples, self.controlnet_down_blocks
        ):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples = controlnet_down_block_res_samples + (
                down_block_res_sample,
            )

        down_block_res_samples = controlnet_down_block_res_samples
        mid_block_res_sample = self.controlnet_mid_block(sample)

        down_block_res_samples = [
            sample * conditioning_scale for sample in down_block_res_samples
        ]
        mid_block_res_sample = mid_block_res_sample * conditioning_scale

        return (
            down_block_res_samples[0],
            down_block_res_samples[1],
            down_block_res_samples[2],
            down_block_res_samples[3],
            down_block_res_samples[4],
            down_block_res_samples[5],
            down_block_res_samples[6],
            down_block_res_samples[7],
            down_block_res_samples[8],
            down_block_res_samples[9],
            down_block_res_samples[10],
            down_block_res_samples[11],
            mid_block_res_sample,
        )


class ControlNetUNet2DConditionModel(nn.Module):
    r"""
    UNet2DConditionModel is a conditional 2D UNet model that takes in a noisy sample, conditional state, and a timestep
    and returns sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)

    Parameters:
        sample_size (`int`, *optional*): The size of the input sample.
        in_channels (`int`, *optional*, defaults to 4): The number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4): The number of channels in the output.
        center_input_sample (`bool`, *optional*, defaults to `False`): Whether to center the input sample.
        flip_sin_to_cos (`bool`, *optional*, defaults to `False`):
            Whether to flip the sin to cos in the time embedding.
        freq_shift (`int`, *optional*, defaults to 0): The frequency shift to apply to the time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`):
            The tuple of downsample blocks to use.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",)`):
            The tuple of upsample blocks to use.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        downsample_padding (`int`, *optional*, defaults to 1): The padding to use for the downsampling convolution.
        mid_block_scale_factor (`float`, *optional*, defaults to 1.0): The scale factor to use for the mid block.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        norm_num_groups (`int`, *optional*, defaults to 32): The number of groups to use for the normalization.
        norm_eps (`float`, *optional*, defaults to 1e-5): The epsilon to use for the normalization.
        cross_attention_dim (`int`, *optional*, defaults to 1280): The dimension of the cross attention features.
        attention_head_dim (`int`, *optional*, defaults to 8): The dimension of the attention heads.
        use_linear_projection (`bool`, *optional*, defaults to False): Use linear projection instead of 1x1 convolution.
    """

    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types: Tuple[str] = (
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        use_linear_projection: bool = False,
    ):
        super().__init__()
        self.center_input_sample = center_input_sample
        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # input
        self.conv_in = nn.Conv2dBias(in_channels, block_out_channels[0], 3, 1, 1)
        # time
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                attn_num_head_channels=attention_head_dim[i],
                cross_attention_dim=cross_attention_dim,
                downsample_padding=downsample_padding,
                use_linear_projection=use_linear_projection,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2DCrossAttn(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift="default",
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attention_head_dim[-1],
            resnet_groups=norm_num_groups,
            use_linear_projection=use_linear_projection,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                attn_num_head_channels=reversed_attention_head_dim[i],
                cross_attention_dim=cross_attention_dim,
                use_linear_projection=use_linear_projection,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0],
            num_groups=norm_num_groups,
            eps=norm_eps,
            use_swish=True,
        )

        self.conv_out = nn.Conv2dBias(block_out_channels[0], out_channels, 3, 1, 1)

    def forward(
        self,
        sample,
        timesteps,
        encoder_hidden_states,
        down_block_residual_0,
        down_block_residual_1,
        down_block_residual_2,
        down_block_residual_3,
        down_block_residual_4,
        down_block_residual_5,
        down_block_residual_6,
        down_block_residual_7,
        down_block_residual_8,
        down_block_residual_9,
        down_block_residual_10,
        down_block_residual_11,
        mid_block_residual,
        return_dict: bool = True,
    ):
        """r
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, channel, height, width) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        down_block_additional_residuals = (
            down_block_residual_0,
            down_block_residual_1,
            down_block_residual_2,
            down_block_residual_3,
            down_block_residual_4,
            down_block_residual_5,
            down_block_residual_6,
            down_block_residual_7,
            down_block_residual_8,
            down_block_residual_9,
            down_block_residual_10,
            down_block_residual_11,
        )
        mid_block_additional_residual = mid_block_residual
        # 1. time
        t_emb = self.time_proj(timesteps)
        emb = self.time_embedding(t_emb)

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if (
                hasattr(downsample_block, "attentions")
                and downsample_block.attentions is not None
            ):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples
            # return sample

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample += down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        sample = self.mid_block(
            sample, emb, encoder_hidden_states=encoder_hidden_states
        )

        if mid_block_additional_residual is not None:
            sample += mid_block_additional_residual

        # 5. up
        upsample_size = None
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]
            if not is_final_block:
                upsample_size = ops.size()(down_block_res_samples[-1])

            if (
                hasattr(upsample_block, "attentions")
                and upsample_block.attentions is not None
            ):
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        # 6. post-process
        # make sure hidden states is in float32
        # when running in half-precision
        sample = self.conv_norm_out(sample)
        sample = self.conv_out(sample)
        return sample
