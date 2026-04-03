"""Face verification via InsightFace — compares against enrolled embedding."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from runtime.event_bus import EventBus

_SIMILARITY_THRESHOLD = 0.5


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


class FaceVerifier:
    def __init__(
        self,
        frame_deque: deque,
        event_bus: EventBus,
        loop: asyncio.AbstractEventLoop,
        embedding_path: str = "~/.jarvis/face_embedding.npy",
        check_interval: float = 1.0,
    ) -> None:
        self._frame_deque = frame_deque
        self._bus = event_bus
        self._loop = loop
        self._embedding_path = Path(embedding_path).expanduser()
        self._check_interval = check_interval
        self._enrolled_embedding: np.ndarray | None = None
        self._app: object | None = None
        self._healthy = False

    @property
    def healthy(self) -> bool:
        return self._healthy

    def initialize(self) -> bool:
        if not self._embedding_path.exists():
            logger.error("No enrolled face at %s — run 'jarvis-enroll'", self._embedding_path)
            return False
        try:
            self._enrolled_embedding = np.load(str(self._embedding_path))
        except Exception:
            logger.exception("Failed to load face embedding")
            return False

        try:
            from insightface.app import FaceAnalysis
            self._app = FaceAnalysis(allowed_modules=["detection", "recognition"])
            self._app.prepare(ctx_id=-1, det_size=(640, 640))
        except Exception:
            logger.exception("Failed to initialize InsightFace")
            return False

        self._healthy = True
        return True

    def run(self, stop_event) -> None:
        if not self._healthy:
            return
        from runtime.event_bus import GestureEvent, GestureType
        import time as _time

        while not stop_event.is_set():
            if not self._frame_deque:
                _time.sleep(0.1)
                continue
            frame = self._frame_deque[-1]
            try:
                faces = self._app.get(frame)
                if not faces:
                    _time.sleep(self._check_interval)
                    continue
                embedding = faces[0].embedding
                similarity = _cosine_similarity(embedding, self._enrolled_embedding)
                if similarity >= _SIMILARITY_THRESHOLD:
                    event = GestureEvent(
                        type=GestureType.FACE_VERIFIED,
                        timestamp=time.time(),
                    )
                    self._loop.call_soon_threadsafe(
                        asyncio.ensure_future, self._bus.publish(event)
                    )
                    _time.sleep(3.0)
                else:
                    _time.sleep(self._check_interval)
            except Exception:
                logger.exception("Face verification error")
                _time.sleep(self._check_interval)


def enroll_cli() -> None:
    """One-time face enrollment — run as 'jarvis-enroll'."""
    import cv2

    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        print("InsightFace not installed. Run: pip install insightface")
        return

    dest = Path("~/.jarvis/face_embedding.npy").expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)

    app = FaceAnalysis(allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    embeddings: list[np.ndarray] = []
    captures_needed = 3
    print(f"Position your face in the frame. Press SPACE to capture ({captures_needed} captures needed). Press Q to quit.")

    while len(embeddings) < captures_needed:
        ret, frame = cap.read()
        if not ret:
            continue
        faces = app.get(frame)
        display = frame.copy()
        for face in faces:
            box = face.bbox.astype(int)
            cv2.rectangle(display, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.imshow("Jarvis Enrollment", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and faces:
            embeddings.append(faces[0].embedding)
            print(f"Captured {len(embeddings)}/{captures_needed}")

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) == captures_needed:
        avg = np.mean(embeddings, axis=0)
        np.save(str(dest), avg)
        print(f"Enrollment saved to {dest}")
    else:
        print("Enrollment cancelled — not enough captures")
