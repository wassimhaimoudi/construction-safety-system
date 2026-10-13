"""Concrete adapters (torch/ultralytics/cv2/fastapi live under here).

This package is intentionally import-light: heavy runtimes are imported lazily
inside the functions that need them, so the pure layers and most tests never
pay for loading torch.
"""
