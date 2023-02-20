from dataclasses import dataclass
from typing import Dict


@dataclass
class RectGeo:
    north: float
    south: float
    east: float
    west: float

    def to_slice_dict(self) -> Dict[str, slice]:
        return {'y': slice(self.north, self.south),
                'x': slice(self.east, self.west)}
