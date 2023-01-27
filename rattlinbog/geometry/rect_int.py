from dataclasses import dataclass
from typing import Dict


@dataclass
class RectInt:
    min_x: int
    max_x: int
    min_y: int
    max_y: int

    def to_slice_dict(self) -> Dict[str, slice]:
        return {'y': slice(self.min_y, self.max_y),
                'x': slice(self.min_x, self.max_x)}
