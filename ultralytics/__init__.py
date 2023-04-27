# Ultralytics YOLO 🚀, GPL-3.0 license

__version__ = "8.0.26"

from ultralytics.yolo.engine.model import YOLO
from ultralytics.yolo.utils import ops
from ultralytics.yolo.utils.checks import check_yolo as checks

#ultralytics模块只能通过，from ultralytics访问以下四个接口
__all__ = ["__version__", "YOLO", "hub", "checks"]  # allow simpler import
