#%%
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

import click
import torch

from aitemplate.utils.import_path import import_parent
from diffusers import AutoencoderKL, StableDiffusionXLPipeline
from IPython.display import Image as IPythonImage, display
# if __name__ == "__main__":
    # import_parent(filepath=__file__, level=1)

from src.pipeline_stable_diffusion_xl_ait import StableDiffusionXLAITPipeline

#%%

#%%
hf_hub_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0"  # huggingface hub name or path to local model
apply_weights: bool = True  # apply weights to module, required for Windows
unet_module: str = 'tmp/SDXL_unet/SDXL_unet.so' # path to unet module
text_encoder_module: str = 'tmp/SDXL_text_encoder/SDXL_text_encoder.so' # path to text encoder module
text_encoder_2_module: str = 'tmp/SDXL_text_encoder_2/SDXL_text_encoder_2.so' # path to text encoder 2 module
time_embed_module: str = 'tmp/SDXL_addition_time_embed/SDXL_addition_time_embed.so' # path to time embed module
vae_module: str = 'tmp/SDXL_vae/SDXL_vae.so'  # path to vae module

diffusers_pipe = StableDiffusionXLPipeline.from_pretrained(
    hf_hub_or_path,
    use_safetensors=True,
    torch_dtype=torch.float16,
)
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    use_safetensors=True,
    torch_dtype=torch.float16,
)
pipe = StableDiffusionXLAITPipeline(
    vae,
    diffusers_pipe.text_encoder,
    diffusers_pipe.text_encoder_2,
    diffusers_pipe.tokenizer,
    diffusers_pipe.tokenizer_2,
    diffusers_pipe.unet,
    diffusers_pipe.scheduler,
    text_encoder_module,
    text_encoder_2_module,
    unet_module,
    vae_module,
    time_embed_module,
    apply_weights_to_modules=apply_weights,
)

#%%
%%time
width: int = 1024  # Width of generated image
height: int = 1024  # Height of generated image
batch: int = 1  # Batch size of generated image
prompt: str = "A vision of paradise, leonid afremov. colorful, 8k, masterpiece"  # prompt
negative_prompt: str = "blurry, overexposed, faded. 3d render, digital illustration. yellow, brown  "  # prompt
benchmark: bool = False  # run stable diffusion e2e benchmark
num_inference_steps = 15
guidance_scale = 15

prompt= ''' 
video game from 1770. 
'''

g = torch.Generator().manual_seed(0)

prompt = [prompt] * batch
images = pipe(
    prompt=prompt,
    prompt_2=prompt,
    height=height,
    width=width,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    _unload_modules=False,
    _callback=lambda x: display(faster_decode(x.pred_original_sample), clear=True),
    generator=g,
).images

display(images[0], clear=True)

#%%


from torch import nn
class VAEApprox(nn.Module):
    def __init__(self):
        super(VAEApprox, self).__init__()
        self.conv1 = nn.Conv2d(4, 8, (7, 7))
        self.conv2 = nn.Conv2d(8, 16, (5, 5))
        self.conv3 = nn.Conv2d(16, 32, (3, 3))
        self.conv4 = nn.Conv2d(32, 64, (3, 3))
        self.conv5 = nn.Conv2d(64, 32, (3, 3))
        self.conv6 = nn.Conv2d(32, 16, (3, 3))
        self.conv7 = nn.Conv2d(16, 8, (3, 3))
        self.conv8 = nn.Conv2d(8, 3, (3, 3))

    def forward(self, x):
        extra = 11
        x = nn.functional.interpolate(x, (x.shape[2] * 2, x.shape[3] * 2))
        x = nn.functional.pad(x, (extra, extra, extra, extra))
        x = x.to(torch.float16)

        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8, ]:
            x = layer(x)
            x = nn.functional.leaky_relu(x, 0.1)

        return x
loaded_model = VAEApprox().to(torch.float16).cuda().eval()
loaded_model.load_state_dict(torch.load('vaeapprox-sdxl.pt'))
# loaded_model = torch.compile(loaded_model)
#%%

def faster_decode(sample) -> IPythonImage:
    y = loaded_model(sample.float().cuda())[0]
    y = y * 0.5 + 0.5
    y = y * 255
    y = y.clamp(0, 255)
    y = y.to(torch.uint8)
    import torchvision.io as tvio
    data = tvio.encode_jpeg(y.cpu()).numpy().tobytes()
    decoded = IPythonImage(data, width=1024, height=1024)
    return decoded