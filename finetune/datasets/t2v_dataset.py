from pathlib import Path
from typing import Any, Dict, List, Tuple
from typing_extensions import override

import torch
from accelerate.logging import get_logger
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import (
    load_prompts, load_videos,
    preprocess_video_with_resize,
    preprocess_video_with_buckets
)

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger(__name__)


class BaseT2VDataset(Dataset):
    """

    """
    def __init__(
        self,
        data_root: str,
        caption_column: str,
        video_column: str,
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        data_root = Path(data_root)
        self.prompts = load_prompts(data_root / caption_column)
        self.videos = load_videos(data_root / video_column)

        # Check if all video files exist
        if any(not path.is_file() for path in self.videos):
            raise ValueError(
                f"Some video files were not found. Please ensure that all video files exist in the dataset directory. Missing file: {next(path for path in self.videos if not path.is_file())}"
            )

        # Check if number of prompts matches number of videos
        if len(self.videos) != len(self.prompts):
            raise ValueError(
                f"Expected length of prompts and videos to be the same but found {len(self.prompts)=} and {len(self.videos)=}. Please ensure that the number of caption prompts and videos match in your dataset."
            )

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            # Here, index is actually a list of data objects that we need to return.
            # The BucketSampler should ideally return indices. But, in the sampler, we'd like
            # to have information about num_frames, height and width. Since this is not stored
            # as metadata, we need to read the video to get this information. You could read this
            # information without loading the full video in memory, but we do it anyway. In order
            # to not load the video twice (once to get the metadata, and once to return the loaded video
            # based on sampled indices), we cache it in the BucketSampler. When the sampler is
            # to yield, we yield the cache data instead of indices. So, this special check ensures
            # that data is not loaded a second time. PRs are welcome for improvements.
            return index

        prompt = self.prompts[index]

        # shape of frames: [F, C, H, W]
        frames = self.preprocess(self.videos[index])
        frames = self.video_transform(frames)

        return {
            "prompt": prompt,
            "video": frames,
            "video_metadata": {
                "num_frames": frames.shape[0],
                "height": frames.shape[2],
                "width": frames.shape[3],
            },
        }

    def preprocess(self, video_path: Path) -> torch.Tensor:
        """
        Loads and preprocesses a video.

        Args:
            video_path: Path to the video file to load.

        Returns:
            torch.Tensor: Video tensor of shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height
                - W is width
        """
        raise NotImplementedError("Subclass must implement this method")
    
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to a video.

        Args:
            frames (torch.Tensor): A 4D tensor representing a video
                with shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height 
                - W is width

        Returns:
            torch.Tensor: The transformed video tensor
        """
        raise NotImplementedError("Subclass must implement this method")


class T2VDatasetWithResize(BaseT2VDataset):
    """

    """
    def __init__(self, max_num_frames: int, height: int, width: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width

        self.__frame_transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)
            ]
        )
    
    @override
    def preprocess(self, video_path: Path) -> torch.Tensor:
        return preprocess_video_with_resize(
            video_path, self.max_num_frames, self.height, self.width,
        )
    
    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transform(f) for f in frames], dim=0)


class T2VDatasetWithBuckets(BaseT2VDataset):
    """

    """
    def __init__(self, video_resolution_buckets: List[Tuple[int, int, int]], *args, **kwargs) -> None:
        """
        Args:
            resolution_buckets: List of tuples representing the resolution buckets.
                Each tuple contains three integers: (max_num_frames, height, width).
        """
        super().__init__(*args, **kwargs)

        self.video_resolution_buckets = video_resolution_buckets

        self.__frame_transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)
            ]
        )
    
    @override
    def preprocess(self, video_path: Path) -> torch.Tensor:
        return preprocess_video_with_buckets(
            video_path, self.video_resolution_buckets
        )
    
    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transform(f) for f in frames], dim=0)
