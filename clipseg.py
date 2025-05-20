# Copyright 2025 zhengxyz123
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import torch
from PIL import Image, ImageFilter
from torchvision.transforms import ToPILImage, ToTensor
from transformers import AutoProcessor, CLIPSegForImageSegmentation  # type: ignore


def tensor2image(tensor: torch.Tensor) -> Image.Image:
    if tensor.dim() == 4:
        return ToPILImage()(tensor[0].permute(2, 0, 1))
    else:
        return ToPILImage()(tensor)


def image2tensor(image: Image.Image) -> torch.Tensor:
    return ToTensor()(image)


class CLIPSegText:
    CATEGORY = "mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASK",)
    FUNCTION = "segment_image"

    def __init__(self) -> None:
        self.processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "text": ("STRING", {"multiline": True, "placeholder": "prompts"}),
            },
            "optional": {
                "blur_radius": (
                    "FLOAT",
                    {"min": 0, "max": 10, "step": 0.5, "default": 5.0},
                ),
                "threshold": (
                    "FLOAT",
                    {"min": 0, "max": 1, "step": 0.01, "default": 0.5},
                ),
            },
        }

    def segment_image(
        self, image: torch.Tensor, text: str, blur_radius: float, threshold: float
    ) -> Tuple[torch.Tensor]:
        original_img = tensor2image(image)
        tensor = torch.zeros(352, 352)
        prompts = [s.strip() for s in text.split(",")]

        inputs = self.processor(
            prompts, images=[original_img] * len(prompts), return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        preds = outputs.logits.unsqueeze(1)
        for i in range(len(prompts)):
            tensor += torch.sigmoid(preds[i][0])

        # do not ask why it's not `tensor > threshold`
        tensor = torch.where(tensor > threshold**2, 1.0, 0.0)
        tensor = image2tensor(
            tensor2image(tensor).filter(ImageFilter.GaussianBlur(blur_radius))
        )
        tensor = torch.where(tensor > 0, 1.0, 0.0)
        output_img = tensor2image(tensor)
        output_img = output_img.resize(
            (original_img.width, original_img.height),
            resample=Image.Resampling.BILINEAR,
        )

        return (image2tensor(output_img),)


class CLIPSegImage:
    CATEGORY = "mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASK",)
    FUNCTION = "segment_image"

    def __init__(self) -> None:
        self.processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",), "prompt": ("IMAGE",)},
            "optional": {
                "blur_radius": (
                    "FLOAT",
                    {"min": 0, "max": 10, "step": 0.5, "default": 5.0},
                ),
                "threshold": (
                    "FLOAT",
                    {"min": 0, "max": 1, "step": 0.01, "default": 0.5},
                ),
            },
        }

    def segment_image(
        self,
        image: torch.Tensor,
        prompt: torch.Tensor,
        blur_radius: float,
        threshold: float,
    ) -> Tuple[torch.Tensor]:
        original_img = tensor2image(image)
        input_image = self.processor(images=[original_img], return_tensors="pt")
        prompt_image = self.processor(
            images=[tensor2image(prompt)], return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(
                **input_image, conditional_pixel_values=prompt_image.pixel_values
            )
        preds = outputs.logits.unsqueeze(1)
        preds = torch.transpose(preds, 0, 1)
        tensor = torch.sigmoid(preds[0])

        # do not ask why it's not `tensor > threshold`
        tensor = torch.where(tensor > threshold**2, 1.0, 0.0)
        tensor = image2tensor(
            tensor2image(tensor).filter(ImageFilter.GaussianBlur(blur_radius))
        )
        tensor = torch.where(tensor > 0, 1.0, 0.0)
        output_img = tensor2image(tensor)
        output_img = output_img.resize(
            (original_img.width, original_img.height),
            resample=Image.Resampling.BILINEAR,
        )

        return (image2tensor(output_img),)


NODE_CLASS_MAPPINGS = {"CLIPSegText": CLIPSegText, "CLIPSegImage": CLIPSegImage}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPSegText": "CLIPSeg (Text)",
    "CLIPSegImage": "CLIPSeg (Image)",
}
