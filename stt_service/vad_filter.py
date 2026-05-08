"""Silero VAD pre-filter — drops non-speech audio before Whisper / verifier.

Cuts TV / room chatter / keyboard clatter that passes the energy gate but
isn't actually a voice speaking directly to the mic. ONNX, ~30MB, ~1ms per
utterance on CPU. Optional — passthrough if silero-vad isn't installed.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SileroVAD:
    def __init__(self, enabled: bool = True, min_speech_prob: float = 0.5) -> None:
        self._enabled = enabled
        self._min_prob = float(min_speech_prob)
        self._model: Any = None
        self._utils: Any = None
        self._lock = threading.Lock()
        self._noop_logged = False
        self._loaded = False

    def initialize(self) -> bool:
        if not self._enabled:
            self._log_noop_once("Silero VAD disabled — all audio accepted.")
            return False
        try:
            import torch
            # Official repo; cached under ~/.cache/torch/hub after first call.
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self._model = model
            self._utils = utils
            self._loaded = True
            logger.info("Silero VAD loaded (min_speech_prob=%.2f)", self._min_prob)
            return True
        except Exception:
            logger.exception("Silero VAD load failed — passthrough.")
            self._loaded = False
            return False

    def _log_noop_once(self, msg: str) -> None:
        if not self._noop_logged:
            self._noop_logged = True
            logger.warning(msg)

    def is_speech(self, audio_f32: np.ndarray, sample_rate: int = 16000) -> bool:
        """Return True if audio contains human speech. Fail-open on error."""
        if not self._loaded or self._model is None:
            return True
        if audio_f32.size == 0:
            return False
        try:
            import torch
            with self._lock:
                wav = torch.from_numpy(np.ascontiguousarray(audio_f32)).float()
                if wav.dim() > 1:
                    wav = wav.mean(dim=0)
                if sample_rate != 16000:
                    import torchaudio
                    wav = torchaudio.functional.resample(wav, sample_rate, 16000)
                # Official helper: returns list of {'start','end'} dicts.
                get_speech_ts = self._utils[0] if isinstance(self._utils, (list, tuple)) else self._utils.get("get_speech_timestamps")
                timestamps = get_speech_ts(
                    wav,
                    self._model,
                    sampling_rate=16000,
                    threshold=self._min_prob,
                )
            return bool(timestamps)
        except Exception:
            logger.exception("Silero VAD inference failed — accepting (fail-open).")
            return True
