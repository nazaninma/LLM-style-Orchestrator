from __future__ import annotations
from typing import Dict, Any


class BaseProblem:
    """
    Base class for all problem loaders.
    Each problem must implement build() and return standardized problem dict.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def build(self) -> Dict[str, Any]:
        raise NotImplementedError("Problem must implement build()")