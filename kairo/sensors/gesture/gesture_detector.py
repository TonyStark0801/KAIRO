"""Gesture detection via MediaPipe — detects double clap and dual snap."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from runtime.event_bus import EventBus

_CLAP_DISTANCE_THRESHOLD = 0.15
_SNAP_WRIST_THRESHOLD = 0.08


class GestureDetector:
    def __init__(
        self,
        frame_deque: deque,
        event_bus: EventBus,
        loop: asyncio.AbstractEventLoop,
        check_interval: float = 0.1,
    ) -> None:
        self._frame_deque = frame_deque
        self._bus = event_bus
        self._loop = loop
        self._check_interval = check_interval
        self._hands = None
        self._healthy = False
        self._clap_timestamps: list[float] = []
        self._snap_count = 0

    @property
    def healthy(self) -> bool:
        return self._healthy

    def initialize(self) -> bool:
        try:
            import mediapipe as mp
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
            )
            self._healthy = True
            return True
        except Exception:
            logger.exception("Failed to initialize MediaPipe Hands")
            return False

    def run(self, stop_event) -> None:
        if not self._healthy:
            return
        import cv2
        from runtime.event_bus import GestureEvent, GestureType

        while not stop_event.is_set():
            if not self._frame_deque:
                time.sleep(0.05)
                continue
            frame = self._frame_deque[-1]
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self._hands.process(rgb)
                if not results.multi_hand_landmarks:
                    time.sleep(self._check_interval)
                    continue

                hands = results.multi_hand_landmarks
                now = time.time()

                if len(hands) == 2:
                    h1, h2 = hands[0].landmark, hands[1].landmark
                    palm_dist = abs(h1[9].x - h2[9].x) + abs(h1[9].y - h2[9].y)
                    if palm_dist < _CLAP_DISTANCE_THRESHOLD:
                        self._clap_timestamps = [
                            t for t in self._clap_timestamps if now - t < 2.0
                        ]
                        self._clap_timestamps.append(now)
                        logger.info("Clap detected (dist=%.3f, count=%d/2)", palm_dist, len(self._clap_timestamps))
                        if len(self._clap_timestamps) >= 2:
                            event = GestureEvent(
                                type=GestureType.DOUBLE_CLAP, timestamp=now
                            )
                            self._loop.call_soon_threadsafe(
                                asyncio.ensure_future, self._bus.publish(event)
                            )
                            self._clap_timestamps.clear()
                            time.sleep(0.5)
                            continue

                for hand_landmarks in hands:
                    lm = hand_landmarks.landmark
                    thumb_tip = lm[4]
                    middle_tip = lm[12]
                    wrist = lm[0]
                    dist = abs(thumb_tip.x - middle_tip.x) + abs(thumb_tip.y - middle_tip.y)
                    if dist < _SNAP_WRIST_THRESHOLD:
                        self._snap_count += 1
                        if self._snap_count >= 2:
                            event = GestureEvent(
                                type=GestureType.DUAL_SNAP, timestamp=now
                            )
                            self._loop.call_soon_threadsafe(
                                asyncio.ensure_future, self._bus.publish(event)
                            )
                            self._snap_count = 0
                            time.sleep(0.5)
                            continue

                time.sleep(self._check_interval)
            except Exception:
                logger.exception("Gesture detection error")
                time.sleep(self._check_interval)
