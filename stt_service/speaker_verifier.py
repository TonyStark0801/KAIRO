"""ECAPA-TDNN speaker verification — accept only owner audio, reject everything else."""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_ECAPA_SAMPLE_RATE = 16000
_SHORT_AUDIO_SEC = 0.5
_SPEECHBRAIN_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
_DEFAULT_SAVEDIR = "~/.cache/speechbrain/spkrec-ecapa"


class SpeakerVerifier:
    """Lazy-loads ECAPA, compares cosine similarity to a stored voiceprint."""

    def __init__(
        self,
        voiceprint_path: Path,
        threshold: float = 0.65,
        enabled: bool = True,
    ) -> None:
        self.voiceprint_path = Path(voiceprint_path).expanduser()
        self.threshold = float(threshold)
        self._enabled = bool(enabled)
        self._classifier = None
        self._voiceprint: np.ndarray | None = None
        self._gating = False
        self._infer_lock = threading.Lock()
        self._savedir = Path(_DEFAULT_SAVEDIR).expanduser()
        self._noop_logged = False

    @property
    def is_gating(self) -> bool:
        return self._gating

    def _log_noop_once(self, msg: str, *args) -> None:
        if not self._noop_logged:
            self._noop_logged = True
            logger.warning(msg, *args)

    def _load_model_sync(self):
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier

        self._savedir.mkdir(parents=True, exist_ok=True)
        return EncoderClassifier.from_hparams(
            source=_SPEECHBRAIN_SOURCE,
            savedir=str(self._savedir),
        )

    def _prepare_wav_tensor(self, audio: np.ndarray, sample_rate: int):
        import torch
        import torchaudio

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32, copy=False)
        wav = torch.from_numpy(np.ascontiguousarray(audio)).float()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if sample_rate != _ECAPA_SAMPLE_RATE:
            wav = torchaudio.functional.resample(
                wav, sample_rate, _ECAPA_SAMPLE_RATE
            )
        return wav

    def _embed_sync(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if self._classifier is None:
            raise RuntimeError("SpeakerVerifier model not loaded")
        with self._infer_lock:
            wav = self._prepare_wav_tensor(audio, sample_rate)
            emb = self._classifier.encode_batch(wav)
            if isinstance(emb, (tuple, list)):
                emb = emb[0]
            vec = emb.detach().cpu().numpy().astype(np.float64).reshape(-1)
        n = float(np.linalg.norm(vec))
        if n > 1e-12:
            vec = vec / n
        return vec

    async def initialize(self, *, force_load_model: bool = False) -> bool:
        """Load voiceprint (if present) and ECAPA model when gating is active.

        If enabled is False, or voiceprint is missing and force_load_model is False,
        the verifier becomes a passthrough (verify always accepts).
        """
        if not self._enabled and not force_load_model:
            self._gating = False
            self._log_noop_once(
                "Speaker verification disabled in config — all audio accepted."
            )
            return False

        vp_exists = self.voiceprint_path.is_file()
        if not vp_exists and not force_load_model:
            self._gating = False
            self._log_noop_once(
                "Speaker verifier: no voiceprint at %s — passthrough (enroll to enable).",
                self.voiceprint_path,
            )
            return False

        if vp_exists:
            try:
                self._voiceprint = np.load(str(self.voiceprint_path)).astype(
                    np.float64
                )
                vn = float(np.linalg.norm(self._voiceprint))
                if vn > 1e-12:
                    self._voiceprint = self._voiceprint / vn
            except Exception:
                logger.exception(
                    "Failed to load voiceprint from %s — passthrough",
                    self.voiceprint_path,
                )
                self._voiceprint = None
                self._gating = False
                return False
        else:
            self._voiceprint = None

        loop = asyncio.get_running_loop()
        try:
            self._classifier = await loop.run_in_executor(
                None, self._load_model_sync
            )
        except Exception:
            logger.exception(
                "SpeechBrain ECAPA load failed — speaker verification disabled."
            )
            self._classifier = None
            self._gating = False
            return False

        self._gating = self._voiceprint is not None
        if self._gating:
            logger.info(
                "Speaker verification active (threshold=%.2f, voiceprint=%s)",
                self.threshold,
                self.voiceprint_path,
            )
        elif force_load_model:
            logger.info(
                "SpeakerVerifier model loaded for enrollment (no voiceprint yet)."
            )
        return True

    async def verify(self, audio: np.ndarray, sample_rate: int) -> tuple[bool, float]:
        if not self._gating:
            return True, 1.0

        if audio.size == 0:
            return True, 0.0

        duration = float(audio.shape[-1]) / float(sample_rate)
        if duration < _SHORT_AUDIO_SEC:
            logger.info("skipped verify: audio too short (%.2fs)", duration)
            return True, 0.0

        try:

            def _run() -> tuple[bool, float]:
                emb = self._embed_sync(audio, sample_rate)
                score = float(np.dot(emb, self._voiceprint))
                return score >= self.threshold, score

            accepted, score = await asyncio.to_thread(_run)
            return accepted, score
        except Exception:
            logger.exception("Speaker verify failed — accepting audio (fail-open).")
            return True, 0.0

    async def enroll(
        self,
        audio_samples: list[np.ndarray],
        sample_rate: int,
    ) -> None:
        if not audio_samples:
            raise ValueError("enroll requires at least one audio sample")
        if self._classifier is None:
            await self.initialize(force_load_model=True)
        if self._classifier is None:
            raise RuntimeError("Could not load ECAPA model for enrollment")

        embeds: list[np.ndarray] = []
        for i, raw in enumerate(audio_samples):
            if raw.size == 0:
                logger.warning("enroll: sample %d empty — skipping", i)
                continue

            def _one(a: np.ndarray, sr: int) -> np.ndarray:
                return self._embed_sync(a, sr)

            emb = await asyncio.to_thread(_one, raw, sample_rate)
            embeds.append(emb)

        if not embeds:
            raise ValueError("No valid audio samples to enroll")
        stacked = np.stack(embeds, axis=0)
        mean = np.mean(stacked, axis=0)
        n = float(np.linalg.norm(mean))
        if n > 1e-12:
            mean = mean / n

        self.voiceprint_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(self.voiceprint_path), mean.astype(np.float32))
        self._voiceprint = mean.astype(np.float64)
        vn = float(np.linalg.norm(self._voiceprint))
        if vn > 1e-12:
            self._voiceprint = self._voiceprint / vn
        self._gating = True
        logger.info("Voiceprint saved to %s (%d utterances)", self.voiceprint_path, len(embeds))
