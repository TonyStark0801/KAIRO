"""Wake word detection backends (STT keyword, openWakeWord)."""

from sensors.wake.factory import try_create_openwakeword_stream

__all__ = ["try_create_openwakeword_stream"]
