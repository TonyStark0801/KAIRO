"""Shared camera thread — single cv2.VideoCapture, fans out frames."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)

FrameCallback = Callable[[np.ndarray], None]

_MAX_RETRIES = 3
_TARGET_FPS = 15


class CameraThread:
    def __init__(self, camera_index: int = 0) -> None:
        self._camera_index = camera_index
        self._subscribers: list[deque] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._healthy = False

    @property
    def healthy(self) -> bool:
        return self._healthy

    def add_subscriber(self) -> deque:
        d: deque = deque(maxlen=2)
        with self._lock:
            self._subscribers.append(d)
        return d

    def start(self) -> bool:
        cap = None
        for attempt in range(1, _MAX_RETRIES + 1):
            cap = cv2.VideoCapture(self._camera_index)
            if cap.isOpened():
                break
            logger.warning("Camera open attempt %d/%d failed", attempt, _MAX_RETRIES)
            cap.release()
            cap = None
            time.sleep(2 ** (attempt - 1))

        if cap is None:
            logger.error("Camera unavailable after %d retries", _MAX_RETRIES)
            return False

        self._healthy = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop, args=(cap,), daemon=True, name="camera"
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._healthy = False

    def _capture_loop(self, cap: cv2.VideoCapture) -> None:
        frame_interval = 1.0 / _TARGET_FPS
        try:
            while not self._stop_event.is_set():
                start = time.monotonic()
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Camera read failed, retrying")
                    time.sleep(0.1)
                    continue
                with self._lock:
                    for d in self._subscribers:
                        d.append(frame)
                elapsed = time.monotonic() - start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            cap.release()
            self._healthy = False
            logger.info("Camera thread stopped")
