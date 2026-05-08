#!/usr/bin/env python3
"""Train a custom openWakeWord model for 'hey kairo'.

Generates synthetic audio via edge-tts (already installed), extracts
openWakeWord embeddings, trains a lightweight sklearn classifier, and
exports to ONNX for use with openWakeWord's streaming inference engine.

Usage:
    pip install torchinfo speechbrain datasets audiomentations onnx scikit-learn
    python scripts/train_wake_word.py

After training, config/kairo.yaml is auto-patched to:
    wake:
      engine: openwakeword
      openwakeword_models: [~/.kairo/models/hey_kairo.onnx]
      openwakeword_inference_framework: onnx
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import tempfile
import wave
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR   = Path.home() / ".kairo" / "models"
MODEL_NAME   = "hey_kairo"
N_PER_PHRASE = 60        # synthetic samples per phrase per voice (keep low for speed)
THRESHOLD    = 0.5

# Phrases — positive examples
POSITIVE_PHRASES = ["hey kairo", "kairo", "hi kairo", "ok kairo"]

# TTS voices — Indian English voices first so accent is covered
TTS_VOICES = [
    "en-IN-NeerjaNeural",
    "en-IN-PrabhatNeural",
    "en-US-GuyNeural",
    "en-US-AriaNeural",
    "en-GB-RyanNeural",
    "en-AU-WilliamNeural",
]

# Negative phrases — short common words that sound somewhat similar
NEGATIVE_PHRASES = [
    "okay", "hello", "hey", "hi there", "cairo", "hero", "mirror",
    "sierra", "care", "arrow", "tomorrow", "sorrow", "zero",
    "open", "close", "stop", "play", "go", "no", "yes",
]


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

REQUIRED = {
    "openwakeword": "openwakeword",
    "torch":        "torch",
    "torchaudio":   "torchaudio",
    "onnx":         "onnx",
    "sklearn":      "scikit-learn",
    "numpy":        "numpy",
    "edge_tts":     "edge-tts",
}

def check_deps() -> None:
    import importlib
    missing = []
    for mod, pkg in REQUIRED.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        logger.error(
            "Missing dependencies. Install with:\n"
            "  pip install %s", " ".join(missing)
        )
        sys.exit(1)
    logger.info("All dependencies present")


# ---------------------------------------------------------------------------
# Audio generation via edge-tts
# ---------------------------------------------------------------------------

async def _synthesize_one(phrase: str, voice: str, out_path: Path) -> bool:
    """Generate one WAV file. Returns True on success."""
    try:
        import edge_tts
        communicate = edge_tts.Communicate(phrase, voice)
        mp3_path = out_path.with_suffix(".mp3")
        await communicate.save(str(mp3_path))

        # Convert mp3 → 16kHz mono WAV using torchaudio
        import torchaudio
        wav, sr = torchaudio.load(str(mp3_path))
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        torchaudio.save(str(out_path), wav, 16000)
        mp3_path.unlink(missing_ok=True)
        return True
    except Exception as exc:
        logger.debug("TTS failed for '%s' / %s: %s", phrase, voice, exc)
        return False


async def generate_samples(phrases: list[str], out_dir: Path, n_per: int) -> list[Path]:
    """Generate n_per samples per phrase using all TTS voices, return list of WAV paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    paths = []
    count = 0
    for phrase in phrases:
        for voice in TTS_VOICES * (n_per // len(TTS_VOICES) + 1):
            if count >= n_per * len(phrases):
                break
            p = out_dir / f"sample_{count:04d}.wav"
            paths.append(p)
            tasks.append(_synthesize_one(phrase, voice, p))
            count += 1

    results = await asyncio.gather(*tasks, return_exceptions=True)
    ok = [p for p, r in zip(paths, results) if r is True]
    logger.info("Generated %d / %d audio samples in %s", len(ok), len(tasks), out_dir)
    return ok


# ---------------------------------------------------------------------------
# Feature extraction using openWakeWord's embedding model
# ---------------------------------------------------------------------------

def extract_embeddings(wav_paths: list[Path]) -> "np.ndarray":
    """Run openWakeWord's audio embedding model on each WAV file."""
    import numpy as np
    import torchaudio
    from openwakeword.model import Model

    # Load openWakeWord just to access its embedding model
    oww = Model(inference_framework="onnx")

    embeddings = []
    for p in wav_paths:
        try:
            wav, sr = torchaudio.load(str(p))
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            audio = (wav.squeeze().numpy() * 32767).astype("int16")

            # Feed 80ms chunks to the embedding model
            chunk = 1280  # OWW's standard chunk size
            feats = []
            for i in range(0, len(audio) - chunk, chunk):
                block = audio[i : i + chunk]
                oww.predict(block)
                # Pull the last embedding vector from the buffer
                for emb_key in oww.preprocessor.__dict__:
                    buf = getattr(oww.preprocessor, emb_key, None)
                    if hasattr(buf, "__len__") and len(buf) > 0:
                        feats.append(buf[-1])
                        break

            if feats:
                embeddings.append(np.mean(feats, axis=0))
        except Exception as exc:
            logger.debug("Embedding failed for %s: %s", p.name, exc)

    return np.array(embeddings) if embeddings else np.zeros((0, 96))


# ---------------------------------------------------------------------------
# Train + export
# ---------------------------------------------------------------------------

def train_classifier(
    pos_embeddings: "np.ndarray",
    neg_embeddings: "np.ndarray",
) -> "sklearn model":
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X = np.vstack([pos_embeddings, neg_embeddings])
    y = np.array([1] * len(pos_embeddings) + [0] * len(neg_embeddings))

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000)),
    ])
    pipe.fit(X, y)
    acc = pipe.score(X, y)
    logger.info("Classifier train accuracy: %.1f%%", acc * 100)
    return pipe


def export_onnx(model, embedding_dim: int, out_path: Path) -> None:
    """Export sklearn pipeline to ONNX via skl2onnx."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import numpy as np

        initial_type = [("float_input", FloatTensorType([None, embedding_dim]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        out_path.write_bytes(onnx_model.SerializeToString())
        logger.info("ONNX model saved: %s", out_path)
    except ImportError:
        # Fallback: pickle the sklearn model — openWakeWord can load sklearn models too
        import pickle
        pkl_path = out_path.with_suffix(".pkl")
        pkl_path.write_bytes(pickle.dumps(model))
        logger.info("sklearn model saved (skl2onnx not installed): %s", pkl_path)
        return pkl_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def patch_config(model_path: Path) -> None:
    config_path = Path(__file__).parent.parent / "config" / "kairo.yaml"
    if not config_path.exists():
        return
    text = config_path.read_text()
    framework = "onnx" if model_path.suffix == ".onnx" else "onnx"
    new_wake = (
        f"wake:\n"
        f"  engine: openwakeword\n"
        f"  openwakeword_models:\n"
        f"    - {model_path}\n"
        f"  openwakeword_threshold: {THRESHOLD}\n"
        f"  openwakeword_inference_framework: {framework}\n"
    )
    text = re.sub(r"^wake:.*?(?=^\w|\Z)", new_wake, text, flags=re.MULTILINE | re.DOTALL)
    config_path.write_text(text)
    logger.info("Patched config/kairo.yaml → engine=openwakeword")


async def async_main() -> None:
    check_deps()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="kairo_train_") as tmp:
        tmp = Path(tmp)

        logger.info("Generating POSITIVE samples (%d phrases × %d voices)…",
                    len(POSITIVE_PHRASES), len(TTS_VOICES))
        pos_dir = tmp / "positive"
        pos_wavs = await generate_samples(POSITIVE_PHRASES, pos_dir, N_PER_PHRASE)

        logger.info("Generating NEGATIVE samples (%d phrases × %d voices)…",
                    len(NEGATIVE_PHRASES), len(TTS_VOICES))
        neg_dir = tmp / "negative"
        neg_wavs = await generate_samples(NEGATIVE_PHRASES, neg_dir, N_PER_PHRASE)

        if not pos_wavs or not neg_wavs:
            logger.error("Not enough audio samples generated. Check edge-tts and network.")
            sys.exit(1)

        logger.info("Extracting embeddings for %d positive samples…", len(pos_wavs))
        pos_emb = extract_embeddings(pos_wavs)

        logger.info("Extracting embeddings for %d negative samples…", len(neg_wavs))
        neg_emb = extract_embeddings(neg_wavs)

        if pos_emb.shape[0] == 0 or neg_emb.shape[0] == 0:
            logger.error("Embedding extraction failed — check openWakeWord installation.")
            sys.exit(1)

        logger.info("Training classifier (pos=%d, neg=%d)…", len(pos_emb), len(neg_emb))
        clf = train_classifier(pos_emb, neg_emb)

        model_path = OUTPUT_DIR / f"{MODEL_NAME}.onnx"
        result_path = export_onnx(clf, pos_emb.shape[1], model_path) or model_path

    patch_config(result_path)

    logger.info("")
    logger.info("Done! Restart kairo — 'hey kairo' will now be detected via openWakeWord.")
    logger.info("Say it naturally from across the room. Threshold: %.2f", THRESHOLD)
    logger.info("Tune in kairo.yaml: lower threshold → more sensitive, higher → stricter")


def main() -> None:
    logger.info("=== Kairo Wake Word Training ===")
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
