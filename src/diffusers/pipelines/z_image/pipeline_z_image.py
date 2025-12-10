# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer, PreTrainedModel

from ...image_processor import VaeImageProcessor
from ...loaders import FromSingleFileMixin, ZImageLoraLoaderMixin
from ...models.autoencoders import AutoencoderKL
from ...models.transformers import ZImageTransformer2DModel
from ...pipelines.pipeline_utils import DiffusionPipeline
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from .pipeline_output import ZImagePipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import ZImagePipeline

        >>> pipe = ZImagePipeline.from_pretrained("Z-a-o/Z-Image-Turbo", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> prompt = "一幅创意海报"
        >>> image = pipe(prompt).images[0]
        >>> image.save("zimage.png")
        ```
"""


def calculate_shift(image_seq_len, base_seq_len: int = 256, max_seq_len: int = 4096, base_shift: float = 0.5, max_shift: float = 1.15):
    print("rahul- calculate_shift called")
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    print("rahul- calculate_shift result:", mu)
    return mu


def retrieve_timesteps(scheduler, num_inference_steps: Optional[int] = None, device: Optional[Union[str, torch.device]] = None,
                       timesteps: Optional[List[int]] = None, sigmas: Optional[List[float]] = None, **kwargs):
    print("rahul- retrieve_timesteps called")
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(f"Scheduler {scheduler.__class__} does not support custom timesteps.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(f"Scheduler {scheduler.__class__} does not support custom sigmas.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    print("rahul- retrieve_timesteps completed, steps:", len(timesteps))
    return timesteps, num_inference_steps


class ZImagePipeline(DiffusionPipeline, ZImageLoraLoaderMixin, FromSingleFileMixin):
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(self, scheduler: FlowMatchEulerDiscreteScheduler, vae: AutoencoderKL,
                 text_encoder: PreTrainedModel, tokenizer: AutoTokenizer, transformer: ZImageTransformer2DModel):
        print("rahul- __init__ called")
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            transformer=transformer,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        print("rahul- __init__ completed, vae_scale_factor:", self.vae_scale_factor)

    def encode_prompt(self, prompt: Union[str, List[str]], device: Optional[torch.device] = None,
                      do_classifier_free_guidance: bool = True,
                      negative_prompt: Optional[Union[str, List[str]]] = None,
                      prompt_embeds: Optional[List[torch.FloatTensor]] = None,
                      negative_prompt_embeds: Optional[torch.FloatTensor] = None,
                      max_sequence_length: int = 512):
        print("rahul- encode_prompt called")
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_embeds = self._encode_prompt(prompt=prompt, device=device, prompt_embeds=prompt_embeds, max_sequence_length=max_sequence_length)

        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = ["" for _ in prompt]
            else:
                negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_embeds = self._encode_prompt(prompt=negative_prompt, device=device, prompt_embeds=negative_prompt_embeds, max_sequence_length=max_sequence_length)
        else:
            negative_prompt_embeds = []
        print("rahul- encode_prompt completed")
        return prompt_embeds, negative_prompt_embeds

    def _encode_prompt(self, prompt: Union[str, List[str]], device: Optional[torch.device] = None,
                       prompt_embeds: Optional[List[torch.FloatTensor]] = None, max_sequence_length: int = 512) -> List[torch.FloatTensor]:
        print("rahul- _encode_prompt called")
        device = device or self._execution_device
        if prompt_embeds is not None:
            return prompt_embeds

        if isinstance(prompt, str):
            prompt = [prompt]

        for i, prompt_item in enumerate(prompt):
            messages = [{"role": "user", "content": prompt_item}]
            prompt_item = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
            prompt[i] = prompt_item

        text_inputs = self.tokenizer(prompt, padding="max_length", max_length=max_sequence_length, truncation=True, return_tensors="pt")
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()

        prompt_embeds = self.text_encoder(input_ids=text_input_ids, attention_mask=prompt_masks, output_hidden_states=True).hidden_states[-2]

        embeddings_list = [prompt_embeds[i][prompt_masks[i]] for i in range(len(prompt_embeds))]
        print("rahul- _encode_prompt completed")
        return embeddings_list

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        print("rahul- prepare_latents called")
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, num_channels_latents, height, width)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)
        print("rahul- prepare_latents completed, shape:", shape)
        return latents

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(self, *args, **kwargs):
        print("rahul- Pipeline __call__ invoked")
        result = super().__call__(*args, **kwargs)
        print("rahul- Pipeline __call__ completed")
        return result
