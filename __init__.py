from importlib.metadata import version

__all__ = ["FusionModel", "load_pretrained"]
__version__ = version(__package__)

from .core import FusionModel              # re-export key classes
from .utils import load_pretrained