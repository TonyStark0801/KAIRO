"""Jarvis runtime daemon — entry point that wires everything together."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler
from pathlib import Path

from core.config.loader import load_config
from core.config.models import JarvisConfig
from core.intent.router import IntentRouter
from core.intent.slot_filler import SlotFiller
from core.pipeline.greeting_pipeline import GreetingPipeline
from core.pipeline.wake_pipeline import WakePipeline
from core.registry.executor import ToolExecutor
from core.registry.tool_registry import ToolRegistry
from core.session.state_machine import SessionStateMachine
from memory.behavioral.query import BehavioralQuery
from memory.behavioral.tracker import BehavioralTracker
from memory.session_cache.redis_client import SessionCache
from memory.vector.client import VectorMemoryClient
from memory.vector.embedder import Embedder
from runtime.event_bus import (
    EventBus,
    GestureEvent,
    IntentRoutedEvent,
    MemoryWriteEvent,
    SessionState,
    SessionStateChangedEvent,
    ToolCancelEvent,
    ToolExecutionEvent,
    VoiceTranscriptEvent,
)
from runtime.health import HealthStatus, HealthTracker

logger = logging.getLogger("jarvis")


def _setup_logging() -> None:
    log_dir = Path("~/.jarvis").expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        str(log_dir / "jarvis.log"), maxBytes=10 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


class JarvisDaemon:
    def __init__(self, config: JarvisConfig) -> None:
        self._config = config
        self._bus = EventBus()
        self._health = HealthTracker()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._executor_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sensor")
        self._stop_event = threading.Event()

        self._fsm: SessionStateMachine | None = None
        self._tool_registry: ToolRegistry | None = None
        self._tool_executor: ToolExecutor | None = None
        self._adapter = None
        self._router: IntentRouter | None = None
        self._slot_filler = SlotFiller()
        self._session_cache: SessionCache | None = None
        self._behavioral_tracker: BehavioralTracker | None = None
        self._behavioral_query: BehavioralQuery | None = None
        self._vector_client: VectorMemoryClient | None = None
        self._wake_pipeline: WakePipeline | None = None
        self._greeting_pipeline: GreetingPipeline | None = None
        self._fusion = None
        self._camera = None

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        await self._bus.start()

        await self._init_adapter()
        await self._init_memory()
        await self._init_tools()
        await self._init_intent()
        await self._init_sensors()
        await self._init_pipelines()
        await self._init_fsm()
        await self._wire_subscriptions()

        logger.info("Jarvis daemon started — health: %s", {
            k: v.status.name for k, v in self._health.get_status().items()
        })

    async def _init_adapter(self) -> None:
        from adapters.macos.adapter import MacOSAdapter
        self._adapter = MacOSAdapter()
        self._health.mark("adapter", HealthStatus.HEALTHY)

    async def _init_memory(self) -> None:
        self._session_cache = SessionCache(
            redis_enabled=self._config.redis.enabled,
            redis_url=self._config.redis.url,
        )
        await self._session_cache.initialize()
        self._health.mark("redis", HealthStatus.HEALTHY)

        self._behavioral_tracker = BehavioralTracker(self._config.memory, self._bus)
        await self._health.init_with_retry("sqlite", self._behavioral_tracker.initialize)
        self._behavioral_query = BehavioralQuery(self._behavioral_tracker)

        embedder = Embedder(self._config.ollama)
        await self._health.init_with_retry("ollama_embed", embedder.initialize)

        self._vector_client = VectorMemoryClient(self._config.memory, embedder, self._bus)
        await self._health.init_with_retry("chromadb", self._vector_client.initialize)

    async def _init_tools(self) -> None:
        self._tool_registry = ToolRegistry()
        self._tool_registry.discover()
        logger.info("Discovered %d tools", len(self._tool_registry.list_all()))

        self._tool_executor = ToolExecutor(
            self._bus, self._tool_registry, self._adapter,
            timeout=self._config.session.tool_timeout_seconds,
        )

    async def _init_intent(self) -> None:
        self._router = IntentRouter(
            host=self._config.ollama.host,
            port=self._config.ollama.port,
            model=self._config.ollama.model,
        )
        await self._health.init_with_retry("ollama", self._router.initialize)

    async def _init_sensors(self) -> None:
        try:
            from sensors.camera import CameraThread
            self._camera = CameraThread()
            started = self._camera.start()
            if started:
                self._health.mark("camera", HealthStatus.HEALTHY)
            else:
                self._health.mark("camera", HealthStatus.DOWN, "Camera unavailable")
        except Exception:
            logger.exception("Camera init failed")
            self._health.mark("camera", HealthStatus.DOWN, "Import or init error")

        if self._camera and self._camera.healthy:
            try:
                from sensors.gesture.face_verifier import FaceVerifier
                from sensors.gesture.gesture_detector import GestureDetector
                from sensors.gesture.fusion import GestureFusion

                face_deque = self._camera.add_subscriber()
                gesture_deque = self._camera.add_subscriber()

                face_verifier = FaceVerifier(
                    face_deque, self._bus, self._loop,
                    embedding_path=self._config.paths.face_embedding_path,
                )
                face_ok = face_verifier.initialize()
                if face_ok:
                    self._executor_pool.submit(face_verifier.run, self._stop_event)

                gesture_detector = GestureDetector(gesture_deque, self._bus, self._loop)
                gesture_ok = gesture_detector.initialize()
                if gesture_ok:
                    self._executor_pool.submit(gesture_detector.run, self._stop_event)

                self._fusion = GestureFusion(
                    self._bus, self._loop,
                    wake_window=self._config.session.wake_window_seconds,
                )
            except Exception:
                logger.exception("Gesture sensor init failed")

    async def _init_pipelines(self) -> None:
        self._wake_pipeline = WakePipeline(self._session_cache)

        process_manager = None
        try:
            from adapters.macos.process_manager import ProcessManager
            process_manager = ProcessManager(self._adapter)
        except Exception:
            pass

        self._greeting_pipeline = GreetingPipeline(
            self._adapter, self._behavioral_query, self._session_cache, process_manager,
        )

    async def _init_fsm(self) -> None:
        self._fsm = SessionStateMachine(self._bus)

    async def _wire_subscriptions(self) -> None:
        self._bus.subscribe(GestureEvent, self._fsm.handle_event)
        self._bus.subscribe(IntentRoutedEvent, self._fsm.handle_event)
        self._bus.subscribe(ToolExecutionEvent, self._fsm.handle_event)

        self._bus.subscribe(IntentRoutedEvent, self._tool_executor.on_intent_routed)
        self._bus.subscribe(ToolCancelEvent, self._tool_executor.on_cancel)

        if self._fusion:
            self._bus.subscribe(SessionStateChangedEvent, self._fusion.on_state_changed)
            self._bus.subscribe(GestureEvent, self._fusion.on_gesture)

        self._bus.subscribe(SessionStateChangedEvent, self._wake_pipeline.on_state_changed)
        self._bus.subscribe(SessionStateChangedEvent, self._greeting_pipeline.on_state_changed)

        if self._behavioral_tracker and self._behavioral_tracker.healthy:
            self._bus.subscribe(ToolExecutionEvent, self._behavioral_tracker.on_tool_execution)

        if self._vector_client and self._vector_client.healthy:
            self._bus.subscribe(MemoryWriteEvent, self._vector_client.on_memory_write)

        self._bus.subscribe(VoiceTranscriptEvent, self._on_voice_transcript)
        self._bus.subscribe(SessionStateChangedEvent, self._on_session_state_changed)

    async def _on_voice_transcript(self, event: VoiceTranscriptEvent) -> None:
        if self._router is None or not self._router.healthy:
            await self._adapter.send_notification("Jarvis", "Intent routing unavailable")
            return

        tool_metas = self._tool_registry.list_all()
        recent = await self._session_cache.get_recent_commands(event.session_id)
        recent_texts = [str(c) for c in recent]

        result = await self._router.route(event.text, tool_metas, recent_texts)

        if result.tool_name is None:
            await self._adapter.send_notification("Jarvis", "I didn't understand that")
            return

        tool = self._tool_registry.get(result.tool_name)
        if tool is None:
            await self._adapter.send_notification("Jarvis", f"Unknown tool: {result.tool_name}")
            return

        slot_result = await self._slot_filler.fill(
            result.params, tool.parameters_schema, recent,
            await self._behavioral_query.get_time_of_day_pattern() if self._behavioral_query else {},
        )

        if slot_result.unfilled:
            await self._adapter.send_notification("Jarvis", f"Missing info: {', '.join(slot_result.unfilled)}")
            return

        config_dict = self._config.model_dump()
        slot_result.params["_config"] = config_dict
        slot_result.params["_executor"] = self._tool_executor

        await self._bus.publish(IntentRoutedEvent(
            tool_name=result.tool_name, params=slot_result.params,
            confidence=result.confidence, session_id=event.session_id,
        ))
        await self._session_cache.append_command(
            event.session_id, {"tool": result.tool_name, "params": result.params, "text": event.text},
        )

    async def _on_session_state_changed(self, event: SessionStateChangedEvent) -> None:
        if event.new_state == SessionState.ACTIVE_SESSION:
            self._start_voice_pipeline()
            self._start_idle_timer()
        elif event.new_state in (SessionState.EXECUTING, SessionState.SLEEP, SessionState.IDLE_TIMEOUT):
            self._stop_voice_pipeline()
            self._cancel_idle_timer()

        if self._fsm:
            if self._wake_pipeline and self._wake_pipeline.current_context:
                self._fsm.session_id = self._wake_pipeline.current_context.session_id

    _voice_thread_stop: threading.Event | None = None

    def _start_voice_pipeline(self) -> None:
        self._stop_voice_pipeline()
        self._voice_thread_stop = threading.Event()
        self._executor_pool.submit(self._voice_capture_loop, self._voice_thread_stop)
        logger.info("Voice pipeline started")

    def _stop_voice_pipeline(self) -> None:
        if self._voice_thread_stop is not None:
            self._voice_thread_stop.set()
            self._voice_thread_stop = None
            logger.info("Voice pipeline stopped")

    def _voice_capture_loop(self, stop_event: threading.Event) -> None:
        try:
            import pyaudio
            from sensors.voice.vad import VoiceActivityDetector
            from sensors.voice.transcriber import Transcriber
            from sensors.voice.normalizer import normalize

            pa = pyaudio.PyAudio()
            stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=480)
            vad = VoiceActivityDetector()
            transcriber = Transcriber()
            if not transcriber.initialize():
                logger.error("Whisper init failed in voice pipeline")
                stream.close()
                pa.terminate()
                return

            while not stop_event.is_set():
                frame = stream.read(480, exception_on_overflow=False)
                audio_chunk = vad.process_frame(frame)
                if audio_chunk is not None:
                    import asyncio as _aio
                    _aio.run_coroutine_threadsafe(
                        self._process_audio_chunk(audio_chunk, transcriber, normalize), self._loop,
                    )
            stream.close()
            pa.terminate()
        except Exception:
            logger.exception("Voice capture loop error")

    async def _process_audio_chunk(self, audio: bytes, transcriber, normalize_fn) -> None:
        text = await transcriber.transcribe(audio)
        if not text:
            return
        text = normalize_fn(text)
        if not text:
            return
        session_id = ""
        if self._wake_pipeline and self._wake_pipeline.current_context:
            session_id = self._wake_pipeline.current_context.session_id
        await self._bus.publish(VoiceTranscriptEvent(text=text, confidence=1.0, session_id=session_id))

    _idle_task: asyncio.Task | None = None

    def _start_idle_timer(self) -> None:
        self._cancel_idle_timer()
        self._idle_task = asyncio.ensure_future(self._idle_timeout())

    def _cancel_idle_timer(self) -> None:
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
            self._idle_task = None

    async def _idle_timeout(self) -> None:
        try:
            await asyncio.sleep(self._config.session.idle_timeout_seconds)
            if self._fsm:
                await self._adapter.send_notification("Jarvis", "Session timed out — going to sleep")
                await self._fsm.trigger_idle_timeout()
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        logger.info("Shutting down Jarvis daemon")
        self._stop_event.set()
        if self._camera:
            self._camera.stop()
        self._executor_pool.shutdown(wait=False)
        await self._bus.stop()
        if self._behavioral_tracker:
            await self._behavioral_tracker.close()
        if self._session_cache:
            await self._session_cache.close()


async def _run() -> None:
    _setup_logging()
    config = load_config()
    daemon = JarvisDaemon(config)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.ensure_future(daemon.stop()))

    await daemon.start()

    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await daemon.stop()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
