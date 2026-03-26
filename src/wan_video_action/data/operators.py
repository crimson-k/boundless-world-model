"""Data processing operators for robot action/state data.

This module provides operators for loading and normalizing robot
data from parquet files, supporting various action representations.
"""

import json
import math
import os
from typing import Dict, Any, Optional, List, Tuple

import imageio
import imageio.v3 as iio
import numpy as np
import pyarrow.parquet as pq
import torch
import torchvision
from PIL import Image

from diffsynth.core.data.operators import DataProcessingOperator, ToAbsolutePath, FrameSamplerByRateMixin


"""
Class: DataProcessingOperator
-----------------------------

Overloads the right-shift operator (`>>`) utilizing the `__rshift__` magic method.

This implementation facilitates intuitive pipeline composition, allowing multiple 
data processing operators to be chained together sequentially 
(e.g., `operator_A >> operator_B`).
"""
class ApplyOperatorToDict(DataProcessingOperator):
    """
    Applies a given operator to a specific key in a dictionary.

    Args:
        key: The dictionary key whose value will be processed.
        operator: The DataProcessingOperator to apply to the value.
        inplace: If True, modifies the input dictionary directly. 
                 If False, creates a shallow copy. Default is False.
    """
    def __init__(self, key: str, operator: DataProcessingOperator, inplace: bool = False):
        self.key = key
        self.operator = operator
        self.inplace = inplace

    def __call__(self, data: dict):
        if not isinstance(data, dict):
            raise TypeError(f"ApplyToKey expects a dictionary, got {type(data).__name__}.")
        
        if self.key not in data:
            raise KeyError(f"Key '{self.key}' not found in the input dictionary.")

        if self.inplace:
            data[self.key] = self.operator(data[self.key])
            return data
        else:
            updated_data = data.copy()
            updated_data[self.key] = self.operator(data[self.key])
            return updated_data


class ResolvePromptEmbPath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path

    def __call__(self, data: str):
        if os.path.isabs(data):
            return data
        return os.path.join(self.base_path, data)


class LoadVideoChunk(DataProcessingOperator, FrameSamplerByRateMixin):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x, frame_rate=24, fix_frame_rate=False):
        FrameSamplerByRateMixin.__init__(self, num_frames, time_division_factor, time_division_remainder, frame_rate, fix_frame_rate)
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor

    def __call__(self, data: str, start_frame=None, end_frame=None):
        reader = self.get_reader(data)
        raw_frame_rate = reader.get_meta_data()['fps']
        total_raw_frames = reader.count_frames()
        
        start = max(0, start_frame if start_frame is not None else 0)
        end = min(total_raw_frames, end_frame if end_frame is not None else total_raw_frames)
        clip_frames = max(0, end - start)

        # x / clip_frames = self.frame_rate / raw_frame_rate
        available_frames = int(clip_frames * self.frame_rate / raw_frame_rate) if self.fix_frame_rate else clip_frames
        num_frames = self.num_frames
        if available_frames < num_frames:
            num_frames = available_frames
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        
        frames = []
        for frame_id in range(num_frames):
            frame_id = self.map_single_frame_id(frame_id, raw_frame_rate, clip_frames)
            frame = reader.get_data(start + frame_id)
            frame = Image.fromarray(frame)
            frame = self.frame_processor(frame)
            frames.append(frame)
        reader.close()
        return frames
    

class LoadGIFChunk(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor

    def get_num_frames(self, clip_frames):
        num_frames = self.num_frames
        if clip_frames < num_frames:
            num_frames = clip_frames
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str, start_frame=None, end_frame=None):
        images = iio.imread(data, mode="RGB")
        total_raw_frames = len(images)
        
        start = max(0, start_frame if start_frame is not None else 0)
        end = min(total_raw_frames, end_frame if end_frame is not None else total_raw_frames)
        clip_frames = max(0, end - start)

        num_frames = self.get_num_frames(clip_frames)
        frames = []
        for img in images[start : start + num_frames]:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
        return frames


class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height=None, width=None, max_pixels=None, height_division_factor=1, width_division_factor=1, resize_mode: str = "fit"):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.resize_mode = resize_mode # "fit" / "crop"

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size

        if self.resize_mode == "crop":
            scale = max(target_width / width, target_height / height)
            image = torchvision.transforms.functional.resize(
                image,
                (round(height*scale), round(width*scale)),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            )
            image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
            return image

        elif self.resize_mode == "fit":
            image = torchvision.transforms.functional.resize(
                image, 
                [target_height, target_width], 
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            )
            return image
        
    def get_height_width(self, image):
        if self.resize_mode == "crop" and self.height is not None and self.width is not None:
            return self.height, self.width

        width, height = image.size
        max_area = self.height * self.width if (self.height is not None and self.width is not None) else self.max_pixels
        if max_area is not None and width * height > max_area:
            scale = (width * height / max_area) ** 0.5
            height, width = int(height / scale), int(width / scale)
        height = height // self.height_division_factor * self.height_division_factor
        width = width // self.width_division_factor * self.width_division_factor

        return height, width

    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image
    
    
class ToVideoTensor(DataProcessingOperator):
    """Convert loaded video frames to float tensor in (V, C, T, H, W), range [-1, 1].

    This operator converts a list of PIL Images or list of lists (for multi-view)
    into a normalized video tensor.
    """

    @staticmethod
    def _frame_to_tensor(frame: Image.Image) -> torch.Tensor:
        """Convert a single PIL Image to CHW tensor in range [-1, 1]."""
        if not isinstance(frame, Image.Image):
            raise TypeError(f"Expected PIL.Image, got {type(frame).__name__}")
        
        if frame.mode != "RGB":
            frame = frame.convert("RGB")
            
        array = np.asarray(frame, dtype=np.float32)
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()  # (C, H, W)
        tensor = tensor * (2.0 / 255.0) - 1.0
        return tensor

    def _frames_to_video_tensor(self, frames) -> torch.Tensor:
        """Convert a list of frames to (C, T, H, W) tensor."""
        if not isinstance(frames, (list, tuple)) or len(frames) == 0:
            raise ValueError("Expected non-empty frame list.")
        
        frame_tensors = [self._frame_to_tensor(frame) for frame in frames]
        video = torch.stack(frame_tensors, dim=1)  # (C, T, H, W)
        return video

    def __call__(self, data):
        """Convert data to video tensor.

        Args:
            data: One of:
                - torch.Tensor: Already a tensor, validate shape
                - PIL.Image: Single frame, treat as 1-frame video
                - list of PIL.Image: Single-view video
                - list of list of PIL.Image: Multi-view video

        Returns:
            torch.Tensor of shape (V, C, T, H, W) in range [-1, 1]
        """
        if isinstance(data, torch.Tensor):
            if data.ndim != 5:
                raise ValueError(f"Expected video tensor with shape (V,C,T,H,W), got {tuple(data.shape)}")
            
            return data.to(dtype=torch.float32)

        if isinstance(data, Image.Image):
            data = [data]

        if not isinstance(data, (list, tuple)) or len(data) == 0:
            raise TypeError("Expected loaded video frames as list/tuple.")

        # Check if multi-view (list of lists)
        if isinstance(data[0], (list, tuple)):
            views = [self._frames_to_video_tensor(view) for view in data]
            return torch.stack(views, dim=0)  # (V, C, T, H, W)

        # Single view
        video = self._frames_to_video_tensor(data).unsqueeze(0)  # (1, C, T, H, W)
        return video
    


