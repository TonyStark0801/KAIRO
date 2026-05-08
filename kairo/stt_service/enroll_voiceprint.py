"""Record enrollment phrases and save an ECAPA voiceprint for speaker_id gating.

Run from the kairo package root:
  python -m stt_service.enroll_voiceprint
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


async def _async_main() -> int:
    from core.config.loader import load_config
    from memory_service.identity import IdentityMemory

    from stt_service.speaker_verifier import SpeakerVerifier

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    )

    try:
        import sounddevice as sd
    except ImportError:
        logger.error("sounddevice not installed — pip install sounddevice")
        return 1

    cfg = load_config()
    sid = cfg.speaker_id
    out_path = Path(sid.voiceprint_path).expanduser()

    identity = IdentityMemory()
    identity.load()
    owner = identity.owner_name
    assistant = identity.assistant_name

    # Keep samples consistent: same speaking pace, tone, and mic distance across
    # all phrases. ECAPA averages embeddings — varied prosody weakens the mean.
    phrases = [
        f"Hey {assistant}, can you hear me clearly right now",
        f"{assistant}, what's on my calendar for tomorrow morning",
        f"Hey {assistant}, play some background music while I keep working",
        f"{assistant}, decrease the volume by twenty percent please",
        f"Hey {assistant}, search the web for the latest news about artificial intelligence",
    ]

    verifier = SpeakerVerifier(
        voiceprint_path=out_path,
        threshold=sid.threshold,
        enabled=True,
    )
    if not await verifier.initialize(force_load_model=True):
        logger.error("Could not load ECAPA model (check speechbrain / network).")
        return 1

    sr = 16000
    duration = 5.0
    samples: list = []

    for i, phrase in enumerate(phrases, start=1):
        print()
        print(f'Phrase {i}/{len(phrases)} — {owner}, please read aloud:')
        print(f'  "{phrase}"')
        print(f"[recording {int(duration)}s...]")
        try:
            rec = sd.rec(
                int(duration * sr),
                samplerate=sr,
                channels=1,
                dtype="float32",
            )
            sd.wait()
        except Exception:
            logger.exception("Microphone recording failed")
            return 1
        audio = rec.reshape(-1)
        samples.append(audio)

    await verifier.enroll(samples, sr)
    print()
    print(f"Enrollment complete. Voiceprint saved to: {out_path}")
    return 0


def main() -> None:
    try:
        raise SystemExit(asyncio.run(_async_main()))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
