"""Hot-swappable camera adapter: Picamera2 or generic UVC (OpenCV).

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

The USB digital microscope enumerates as a standard UVC device, so OpenCV's
VideoCapture handles it; the Pi Camera path uses Picamera2. Selected via
config so a kit swap needs no code change.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Camera(Protocol):
    def capture(self) -> np.ndarray: ...
    def release(self) -> None: ...


class UVCCamera:
    def __init__(self, index: int = 0):
        import cv2

        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"UVC camera {index} not available")

    def capture(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Failed to read frame from UVC microscope")
        return frame

    def release(self) -> None:
        self.cap.release()


class PiCamera:
    def __init__(self):
        from picamera2 import Picamera2

        self.cam = Picamera2()
        self.cam.configure(self.cam.create_still_configuration())
        self.cam.start()

    def capture(self) -> np.ndarray:
        return self.cam.capture_array()

    def release(self) -> None:
        self.cam.stop()


def make_camera(kind: str = "uvc", index: int = 0) -> Camera:
    if kind == "uvc":
        return UVCCamera(index)
    if kind == "picamera":
        return PiCamera()
    raise ValueError(f"Unknown camera kind {kind!r}")
