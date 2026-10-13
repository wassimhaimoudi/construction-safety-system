"""Application layer.

Use-cases that orchestrate the pure domain against abstract ports. This layer
must not import torch/ultralytics/cv2/fastapi; concrete adapters live under
``safety_monitor.infrastructure`` and are wired in at the edges.
"""

from .analyze_frame import FrameAnalysis, analyze_frame
from .detector import Detector, Frame

__all__ = ["Detector", "Frame", "FrameAnalysis", "analyze_frame"]
