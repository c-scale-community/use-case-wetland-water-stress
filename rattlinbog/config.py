from dataclasses import dataclass
from typing import Tuple, Sequence

from yaml import YAMLObject, SafeLoader


@dataclass
class Restructure(YAMLObject):
    yaml_tag = "!Restructure"
    yaml_loader = SafeLoader

    chunk_size: int
    rois: Sequence[Tuple[int, int, int, int]]
