from dataclasses import dataclass
from typing import Tuple, Sequence, Optional

from yaml import YAMLObject, SafeLoader


@dataclass
class Restructure(YAMLObject):
    yaml_tag = "!Restructure"
    yaml_loader = SafeLoader

    chunk_size: int
    rois: Sequence[Tuple[int, int, int, int]]


@dataclass
class SamplingConfig(YAMLObject):
    yaml_tag = "!Sampling"
    yaml_loader = SafeLoader

    patch_size: int
    n_samples: int
    oversampling_size: Optional[int] = 2
    never_nans: Optional[bool] = False
