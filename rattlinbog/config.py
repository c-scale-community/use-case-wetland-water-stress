from dataclasses import dataclass
from typing import Tuple, Sequence, Optional, Literal

from yaml import YAMLObject, SafeLoader


@dataclass
class Restructure(YAMLObject):
    yaml_tag = "!Restructure"
    yaml_loader = SafeLoader

    chunk_size: int
    rois: Sequence[Tuple[int, int, int, int]]
    parameter_type: Literal['hparam', 'mmeans']
    datetime_1_year: int = None
    datetime_2_year: int = None
    mask_extra_field: Optional[str] = None


@dataclass
class SamplingConfig(YAMLObject):
    yaml_tag = "!Sampling"
    yaml_loader = SafeLoader

    patch_size: int
    n_samples: int
    oversampling_size: Optional[int] = 2
    never_nans: Optional[bool] = False


@dataclass
class ParameterSelection(YAMLObject):
    yaml_tag = "!ParameterSelection"
    yaml_loader = SafeLoader

    root: str
    parameter_type: Literal['hparam', 'mmean']
    var_name: str
    datetime_1_year: int
    datetime_2_year: int
    extra_field: Optional[str] = None


@dataclass
class SamplesSelection(YAMLObject):
    yaml_tag = "!SamplesSelection"
    yaml_loader = SafeLoader

    root: str
    var_name: str
    sensor_field: str
    extra_field: str


@dataclass
class TrainCNN(YAMLObject):
    yaml_tag = "!TrainCNN"
    yaml_loader = SafeLoader

    parameter_selection: ParameterSelection
    samples_selection: SamplesSelection
