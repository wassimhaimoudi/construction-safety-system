"""Safety Monitor — AI construction site safety service.

Pure domain logic lives under :mod:`safety_monitor.domain` and must never
import heavy runtime dependencies (torch, ultralytics, cv2, fastapi).
"""

__version__ = "0.1.0"
