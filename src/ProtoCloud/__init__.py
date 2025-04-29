from .model import protoCloud
from . import model as model
from . import lrp as lrp
from . import data as data
from . import utils as utils
from . import vis as vis

from . import glo as _glo
import sys

sys.modules.setdefault("glo", _glo)


__all__ = ["glo", "protoCloud", "model", "scRNAData"]           # 让 `from Model import utils` 继续可用