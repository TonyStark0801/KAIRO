"""Whisper STT engine — multi-backend with priority order.

Priority (when engine="auto"):
  1. mlx-whisper     — Apple Neural Engine (macOS M-series; fastest, no download stall)
  2. faster-whisper  — CTranslate2 int8 CPU/CUDA (Windows, Linux, Intel Mac)
  3. pywhispercpp    — whisper.cpp bindings (last resort; uses cpp_fallback_model
                       directly — never tries to download a 3 GB large model)
"""

from __future__ import annotations

import logging

import numpy as np

from stt_service.base import STTEngine, TranscriptMeta

logger = logging.getLogger(__name__)

# mlx-whisper uses HuggingFace repo paths from mlx-community.
# Map friendly model names (same ones faster-whisper uses) to HF repos.
# Verified against https://huggingface.co/mlx-community — most use the -mlx suffix.
# large-v3-turbo is the exception: it exists without the suffix.
_MLX_MODEL_MAP: dict[str, str] = {
    "large-v3":        "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo":  "mlx-community/whisper-large-v3-turbo",   # no -mlx suffix
    "large-v2":        "mlx-community/whisper-large-v2-mlx",
    "medium":          "mlx-community/whisper-medium-mlx",
    "medium.en":       "mlx-community/whisper-medium.en-mlx",
    "small":           "mlx-community/whisper-small-mlx",
    "small.en":        "mlx-community/whisper-small.en-mlx",
    "base":            "mlx-community/whisper-base-mlx",
    "base.en":         "mlx-community/whisper-base.en-mlx",
    "tiny":            "mlx-community/whisper-tiny-mlx",
    "tiny.en":         "mlx-community/whisper-tiny.en-mlx",
}


class WhisperEngine(STTEngine):
    def __init__(
        self,
        model_name: str = "large-v3",
        cpp_fallback_model: str = "small.en",
        engine: str = "auto",
    ) -> None:
        self._model_name = model_name
        self._cpp_fallback_model = cpp_fallback_model
        self._engine_hint = engine.lower().strip()
        self._model = None
        self._backend: str = ""
        self._backend_failed: bool = False  # circuit breaker: True after unrecoverable error

    # ------------------------------------------------------------------
    # Initialisation — tries backends in order of preference
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        hint = self._engine_hint

        # Explicit engine selection
        if hint == "mlx-whisper":
            return self._try_mlx() or self._warn_and_fail("mlx-whisper")
        if hint == "faster-whisper":
            return self._try_faster_whisper() or self._warn_and_fail("faster-whisper")
        if hint == "pywhispercpp":
            return self._try_cpp()

        # "auto" — try in order: mlx → faster-whisper → cpp
        return self._try_mlx() or self._try_faster_whisper() or self._try_cpp()

    def _try_mlx(self) -> bool:
        """mlx-whisper — Apple Neural Engine, macOS M-series only.

        Eagerly downloads + loads the model so HuggingFace auth failures are
        caught ONCE at startup instead of exploding on every audio frame.
        Cached after first download (~150–500MB depending on model).
        """
        try:
            import mlx_whisper  # type: ignore[import-untyped]
            from huggingface_hub import snapshot_download  # type: ignore[import-untyped]

            repo = _MLX_MODEL_MAP.get(self._model_name, self._model_name)
            logger.info("Whisper: downloading/verifying mlx model '%s' …", repo)

            # token=False forces anonymous download — mlx-community repos are public.
            # Without this, huggingface_hub picks up any stale token from env vars
            # (HF_TOKEN, HUGGING_FACE_HUB_TOKEN) or old credential stores and sends
            # it to the API, getting a 401 even though no auth is needed.
            local_path = snapshot_download(repo_id=repo, token=False)

            self._model = local_path  # pass local path to transcribe(), not HF repo
            self._backend = "mlx-whisper"

            # Warmup: force model weights into memory NOW so first real audio frame
            # doesn't block with a cold load. mlx_whisper.ModelHolder caches by path.
            logger.info("Whisper: warming up mlx model (first load may take a few seconds) …")
            _dummy = np.zeros(1600, dtype=np.float32)  # 0.1s silence — minimal compute
            mlx_whisper.transcribe(_dummy, path_or_hf_repo=local_path, language="en", verbose=False)
            logger.info("Whisper backend: mlx-whisper ready (model=%s)", repo)
            return True
        except ImportError:
            logger.debug("mlx-whisper not installed — skipping")
        except Exception:
            logger.warning("mlx-whisper init failed (auth error? check ~/.cache/huggingface/token) — skipping", exc_info=True)
        return False

    def _try_faster_whisper(self) -> bool:
        """faster-whisper — CTranslate2 int8, works on CPU and CUDA."""
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]

            self._model = WhisperModel(
                self._model_name,
                device="cpu",
                compute_type="int8",
            )
            self._backend = "faster-whisper"
            logger.info("Whisper backend: faster-whisper (model=%s, int8 CPU)", self._model_name)
            return True
        except ImportError:
            logger.debug("faster-whisper not installed — skipping")
        except Exception:
            logger.warning("faster-whisper init failed — skipping", exc_info=True)
        return False

    def _try_cpp(self) -> bool:
        """pywhispercpp — whisper.cpp bindings.

        IMPORTANT: we use cpp_fallback_model (default: small.en) directly,
        NOT model_name. large-v3 via pywhispercpp would trigger a 3 GB
        download; the fallback model is intentionally small.
        """
        try:
            from pywhispercpp.model import Model  # type: ignore[import-untyped]

            self._model = Model(self._cpp_fallback_model)
            self._backend = "pywhispercpp"
            logger.info("Whisper backend: pywhispercpp (model=%s)", self._cpp_fallback_model)
            return True
        except ImportError:
            logger.debug("pywhispercpp not installed — skipping")
        except Exception:
            logger.exception("pywhispercpp init failed")
        return False

    @staticmethod
    def _warn_and_fail(backend: str) -> bool:
        logger.error(
            "Requested engine '%s' is unavailable. "
            "Install it with: pip install %s",
            backend,
            backend,
        )
        return False

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def transcribe(self, audio: np.ndarray) -> str:
        if self._model is None or self._backend_failed:
            return ""
        try:
            if self._backend == "mlx-whisper":
                return self._transcribe_mlx(audio)
            if self._backend == "faster-whisper":
                return self._transcribe_faster(audio)
            return self._transcribe_cpp(audio)
        except Exception:
            logger.exception("Whisper transcription error (backend=%s)", self._backend)
            return ""

    def _transcribe_mlx(self, audio: np.ndarray) -> str:
        return self._transcribe_mlx_meta(audio).text

    def _transcribe_mlx_meta(self, audio: np.ndarray) -> TranscriptMeta:
        import mlx_whisper  # type: ignore[import-untyped]

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._model,
            language="en",
            verbose=False,
        )
        if not isinstance(result, dict):
            return TranscriptMeta(text=str(result).strip().lower(), speech_prob=1.0)

        text = result.get("text", "").strip().strip(" .,!?\"'-").lower()
        segments = result.get("segments", [])
        if segments:
            avg_no_speech = sum(s.get("no_speech_prob", 0.0) for s in segments) / len(segments)
            speech_prob = 1.0 - avg_no_speech
        else:
            speech_prob = 1.0
        return TranscriptMeta(text=text, speech_prob=speech_prob)

    def _transcribe_faster(self, audio: np.ndarray) -> str:
        return self._transcribe_faster_meta(audio).text

    def _transcribe_faster_meta(self, audio: np.ndarray) -> TranscriptMeta:
        segments_iter, _ = self._model.transcribe(
            audio,
            language="en",
            beam_size=1,
            vad_filter=False,
        )
        segments = list(segments_iter)
        text = " ".join(seg.text for seg in segments).strip().strip(" .,!?\"'-").lower()
        if segments:
            avg_no_speech = sum(getattr(seg, "no_speech_prob", 0.0) for seg in segments) / len(segments)
            speech_prob = 1.0 - avg_no_speech
        else:
            speech_prob = 1.0
        return TranscriptMeta(text=text, speech_prob=speech_prob)

    def _transcribe_cpp(self, audio: np.ndarray) -> str:
        segments = self._model.transcribe(audio)
        text = " ".join(seg.text for seg in segments).strip()
        return text.strip(" .,!?\"'-").lower()

    # ------------------------------------------------------------------
    # Meta transcription — exposes speech probability
    # ------------------------------------------------------------------

    def transcribe_with_meta(self, audio: np.ndarray) -> TranscriptMeta:
        if self._model is None or self._backend_failed:
            return TranscriptMeta(text="", speech_prob=0.0)
        try:
            if self._backend == "mlx-whisper":
                return self._transcribe_mlx_meta(audio)
            if self._backend == "faster-whisper":
                return self._transcribe_faster_meta(audio)
            # pywhispercpp has no no_speech_prob — fall back to base
            return TranscriptMeta(text=self._transcribe_cpp(audio), speech_prob=1.0)
        except Exception as exc:
            # If this is an auth/network error, trip the circuit breaker so we don't
            # spam the log on every audio frame. A process restart resets this.
            err_str = str(exc).lower()
            if any(kw in err_str for kw in ("401", "403", "unauthorized", "repositorynotfound", "network")):
                self._backend_failed = True
                logger.error(
                    "Whisper backend '%s' failed with unrecoverable error — "
                    "disabling for this session. Fix: rm ~/.cache/huggingface/token && restart",
                    self._backend,
                )
            else:
                logger.exception("Whisper meta transcription error (backend=%s)", self._backend)
            return TranscriptMeta(text="", speech_prob=0.0)
