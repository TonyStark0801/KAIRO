"""Construct optional openWakeWord streaming detector from config."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.config.models import WakeConfig
    from sensors.wake.openwakeword_stream import OpenWakeWordStreamDetector

logger = logging.getLogger(__name__)


def try_create_openwakeword_stream(
    wake: WakeConfig,
) -> OpenWakeWordStreamDetector | None:
    if wake.engine != "openwakeword":
        return None
    try:
        from sensors.wake.openwakeword_stream import OpenWakeWordStreamDetector

        return OpenWakeWordStreamDetector(
            wakeword_models=wake.openwakeword_models or None,
            threshold=wake.openwakeword_threshold,
            inference_framework=wake.openwakeword_inference_framework,
        )
    except ImportError:
        logger.warning(
            "openWakeWord not installed — install with pip install 'kairo-runtime[wake]' "
            "or fall back to wake.engine=stt_keyword"
        )
        return None
    except Exception:
        logger.exception("openWakeWord init failed — use stt_keyword wake or fix models")
        return None
