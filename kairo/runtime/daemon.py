"""Kairo runtime daemon — slim wiring. No business logic here."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler
from pathlib import Path

from assistant_core.agent_loop import AgentLoop
from assistant_core.dialogue_planner import DialoguePlanner
from assistant_core.personality import Personality
from assistant_core.proactive import ProactiveEngine, ProactiveSuggestion
from assistant_core.reasoner import Reasoner, ReasonerAction, ReasonerResponse
from context_service.detector import ContextDetector
from core.config.loader import load_config
from core.config.models import KairoConfig
from core.pipeline.wake_pipeline import WakePipeline
from core.registry.executor import ToolExecutor
from core.registry.tool_registry import ToolRegistry
from core.session.state_machine import SessionStateMachine
from llm_router.groq_provider import GroqProvider
from llm_router.providers import create_local_chat_provider
from llm_router.protocol import LocalChatProvider
from memory_service.identity import IdentityMemory
from memory_service.preferences import PreferencesMemory
from memory_service.session_store import SessionStore
from memory.behavioral.query import BehavioralQuery
from memory.behavioral.tracker import BehavioralTracker
from memory.session_cache.redis_client import SessionCache
from memory.vector.client import VectorMemoryClient
from memory.vector.embedder import Embedder
from runtime.event_bus import (
    EventBus,
    GestureEvent,
    GestureType,
    IntentRoutedEvent,
    MemoryWriteEvent,
    SessionState,
    SessionStateChangedEvent,
    ToolCancelEvent,
    ToolExecutionEvent,
    VoiceTranscriptEvent,
)
from runtime.health import HealthStatus, HealthTracker
from sensors.voice.voice_verifier import VoiceVerifier
from sensors.wake.factory import try_create_openwakeword_stream
from stt_service.mic_listener import MicListener, MicMode
from stt_service.whisper_engine import WhisperEngine
from voice_service.piper_engine import PiperVoiceEngine

logger = logging.getLogger("kairo")

_MEDIA_TOOLS = {"youtube_pick", "youtube_playlist"}
_MEDIA_PLAY_ACTIONS = {"play"}
_MEDIA_PAUSE_ACTIONS = {"pause"}
_DUCK_LEVEL = 15


def _setup_logging() -> None:
    log_dir = Path("~/.kairo").expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        str(log_dir / "kairo.log"), maxBytes=10 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


class KairoDaemon:
    def __init__(self, config: KairoConfig) -> None:
        self._config = config
        self._bus = EventBus()
        self._health = HealthTracker()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._executor_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sensor")
        self._stop_event = threading.Event()

        # Services
        self._identity: IdentityMemory | None = None
        self._voice_verifier: VoiceVerifier | None = None
        self._preferences: PreferencesMemory | None = None
        self._session_store: SessionStore | None = None
        self._context_detector: ContextDetector | None = None
        self._voice: PiperVoiceEngine | None = None
        self._stt: WhisperEngine | None = None
        self._wake_stt: WhisperEngine | None = None
        self._mic: MicListener | None = None
        self._llm: LocalChatProvider | None = None
        self._planner: DialoguePlanner | None = None
        self._groq: GroqProvider | None = None
        self._personality: Personality | None = None
        self._reasoner: Reasoner | None = None
        self._agent_loop: AgentLoop | None = None
        self._proactive: ProactiveEngine | None = None
        self._todo_store = None

        # Existing components
        self._fsm: SessionStateMachine | None = None
        self._tool_registry: ToolRegistry | None = None
        self._tool_executor: ToolExecutor | None = None
        self._adapter = None
        self._session_cache: SessionCache | None = None
        self._behavioral_tracker: BehavioralTracker | None = None
        self._behavioral_query: BehavioralQuery | None = None
        self._vector_client: VectorMemoryClient | None = None
        self._fusion = None
        self._camera = None
        self._wake_pipeline: WakePipeline | None = None

        # Phase 2 — context / clipboard observers
        self._context_observer = None
        self._clipboard_monitor = None
        self._context_task: asyncio.Task | None = None

        # Runtime state
        self._current_session_id = ""
        self._speech_lock = asyncio.Lock()
        self._media_playing = False
        self._saved_volume: int | None = None

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        await self._bus.start()

        await self._init_identity()
        await self._init_adapter()
        await self._init_memory()
        await self._init_context()
        await self._init_voice()
        await self._init_stt()
        await self._init_tools()
        await self._init_llm()
        await self._init_groq()
        await self._init_brain()
        await self._init_proactive()
        await self._init_sensors()
        await self._init_fsm()
        await self._wire_subscriptions()

        if self._proactive:
            self._proactive_task = asyncio.ensure_future(self._proactive_loop())

        # Phase 2: start context observer (async) + clipboard monitor (thread)
        if self._context_observer:
            self._context_task = asyncio.ensure_future(self._context_observer.run())
        if self._clipboard_monitor:
            self._executor_pool.submit(self._clipboard_monitor.run, self._stop_event)

        logger.info("Kairo daemon started — health: %s", {
            k: v.status.name for k, v in self._health.get_status().items()
        })

    async def _init_identity(self) -> None:
        self._identity = IdentityMemory()
        self._identity.load()
        logger.info(
            "Identity: %s (owner: %s)",
            self._identity.assistant_name,
            self._identity.owner_name,
        )

    async def _init_adapter(self) -> None:
        from adapters.macos.adapter import MacOSAdapter
        self._adapter = MacOSAdapter()
        self._health.mark("adapter", HealthStatus.HEALTHY)

    async def _init_memory(self) -> None:
        self._preferences = PreferencesMemory()
        await self._health.init_with_retry("preferences", self._preferences.initialize)

        self._session_store = SessionStore()
        await self._health.init_with_retry("session_store", self._session_store.initialize)

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

    async def _init_context(self) -> None:
        self._context_detector = ContextDetector()

        obs_cfg = self._config.observer
        try:
            from sensors.observer.activity_classifier import ActivityClassifier
            from sensors.observer.context_observer import ContextObserver
            from sensors.observer.clipboard_monitor import ClipboardMonitor

            classifier = ActivityClassifier()

            if obs_cfg.capture_frontmost_window:
                self._context_observer = ContextObserver(
                    event_bus=self._bus,
                    detector=self._context_detector,
                    classifier=classifier,
                    poll_interval_sec=obs_cfg.poll_interval_sec,
                )
                logger.info(
                    "Context observer ready (poll=%.1fs, browser_tab=%s)",
                    obs_cfg.poll_interval_sec,
                    obs_cfg.capture_browser_tab,
                )

            if obs_cfg.capture_clipboard:
                self._clipboard_monitor = ClipboardMonitor(
                    event_bus=self._bus,
                    loop=self._loop,
                )
                logger.info("Clipboard monitor ready")

        except Exception:
            logger.exception("Observer services init failed — continuing without them")

        self._health.mark("context", HealthStatus.HEALTHY)

    async def _init_voice(self) -> None:
        model = self._identity.voice_model if self._identity else "en_US-amy-medium"
        self._voice = PiperVoiceEngine(model=model)
        await self._voice.initialize()

        # No mic muting during speech — built-in mic + headphones means no echo.
        # Mic stays in WAKE_WORD mode so user can interrupt.
        self._health.mark("voice", HealthStatus.HEALTHY)

    async def _init_stt(self) -> None:
        cfg = self._config.stt

        # Command engine — small.en: accurate, handles accented speech well
        self._stt = WhisperEngine(
            model_name=cfg.model,
            cpp_fallback_model=cfg.cpp_fallback_model,
            engine=cfg.engine,
        )
        if self._stt.initialize():
            self._health.mark("whisper", HealthStatus.HEALTHY)
            logger.info("Command STT ready (model=%s)", cfg.model)
        else:
            self._health.mark("whisper", HealthStatus.DOWN, "Whisper init failed")

        # Wake engine — tiny.en: ~80ms latency, just needs to spot "kairo"
        # Separate instance so model weights aren't shared with command path.
        self._wake_stt = WhisperEngine(
            model_name=cfg.wake_model,
            cpp_fallback_model=cfg.cpp_fallback_model,
            engine=cfg.engine,
        )
        if self._wake_stt.initialize():
            logger.info("Wake STT ready (model=%s)", cfg.wake_model)
        else:
            logger.warning("Wake STT init failed — using command engine for wake detection")
            self._wake_stt = self._stt  # fallback: share the command engine

    async def _init_tools(self) -> None:
        self._tool_registry = ToolRegistry()
        self._tool_registry.discover()
        logger.info("Discovered %d tools", len(self._tool_registry.list_all()))

        self._tool_executor = ToolExecutor(
            self._bus, self._tool_registry, self._adapter,
            timeout=self._config.session.tool_timeout_seconds,
        )

    async def _init_llm(self) -> None:
        self._llm = create_local_chat_provider(self._config.ollama)
        await self._health.init_with_retry("ollama", self._llm.initialize)

    async def _init_groq(self) -> None:
        cfg = self._config.groq
        self._groq = GroqProvider(
            api_key_env=cfg.api_key_env,
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        )
        ok = await self._groq.initialize()
        if ok:
            self._health.mark("groq", HealthStatus.HEALTHY)
            logger.info("Groq cloud LLM ready")
        else:
            self._health.mark("groq", HealthStatus.DOWN, "No API key or connection failed")
            logger.warning("Groq unavailable — Tier 3 will use local fallback")

    async def _init_brain(self) -> None:
        self._personality = Personality(
            self._identity, self._preferences,
            self._session_store, self._context_detector,
        )
        self._planner = DialoguePlanner()

        if self._groq and self._groq.healthy and self._tool_registry and self._adapter:
            self._agent_loop = AgentLoop(self._groq, self._tool_registry, self._adapter)
            logger.info("Agent loop ready (Groq + tool calling)")

        self._reasoner = Reasoner(
            self._personality, self._llm,
            self._preferences, self._session_store,
            groq=self._groq,
            agent_loop=self._agent_loop,
            behavioral_query=self._behavioral_query,
            vector_client=self._vector_client,
        )
        self._health.mark("brain", HealthStatus.HEALTHY)

    async def _init_proactive(self) -> None:
        if not self._config.proactive.enabled:
            logger.info("Proactive engine disabled by config")
            return

        from tools.tasks.todo_store import TodoStore
        self._todo_store = TodoStore()
        try:
            await self._todo_store.initialize()
        except Exception:
            logger.exception("TODO store init failed")
            self._todo_store = None

        self._proactive = ProactiveEngine(
            config=self._config.proactive,
            todo_store=self._todo_store,
            context_detector=self._context_detector,
            session_store=self._session_store,
        )
        logger.info("Proactive engine ready (interval=%ds)", self._config.proactive.check_interval)

    async def _init_sensors(self) -> None:
        verification_mode = self._identity.verification_mode if self._identity else "any"

        # Camera + face verifier — only when face verification is needed
        if verification_mode in ("face", "any", "both"):
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
                    face_deque = self._camera.add_subscriber()
                    face_verifier = FaceVerifier(
                        face_deque, self._bus, self._loop,
                        embedding_path=self._config.paths.face_embedding_path,
                    )
                    if face_verifier.initialize():
                        self._executor_pool.submit(face_verifier.run, self._stop_event)
                except Exception:
                    logger.exception("Face verifier init failed")
        else:
            logger.info("Camera skipped (verification_mode=%s)", verification_mode)

        # Voice verifier (speaker ID)
        if verification_mode in ("voice", "any", "both"):
            self._voice_verifier = VoiceVerifier()
            if self._voice_verifier.initialize():
                self._health.mark("voice_id", HealthStatus.HEALTHY)
                logger.info("Voice verification enabled")
            else:
                self._health.mark("voice_id", HealthStatus.DOWN, "No voice enrollment or resemblyzer missing")
                self._voice_verifier = None

        # Gesture fusion with verification mode
        from sensors.gesture.fusion import GestureFusion
        self._fusion = GestureFusion(
            self._bus, self._loop,
            wake_window=self._config.session.wake_window_seconds,
            verification_mode=verification_mode,
        )

        # Mic listener with optional voice verifier and openWakeWord streaming detector
        wake_words = self._identity.wake_words if self._identity else None
        oww = try_create_openwakeword_stream(self._config.wake)
        if oww:
            logger.info("Wake engine: openWakeWord streaming")
        elif self._config.wake.engine == "openwakeword":
            logger.warning("openWakeWord unavailable — falling back to STT keyword wake")
        self._mic = MicListener(
            stt_engine=self._stt,
            event_bus=self._bus,
            loop=self._loop,
            wake_words=wake_words,
            voice_verifier=self._voice_verifier,
            openwakeword_detector=oww,
            wake_stt_engine=self._wake_stt,
        )
        def _on_interrupt():
            logger.info("User interrupted speech")
            if self._voice:
                self._voice.stop_speaking()

        self._mic.set_interrupt_callback(_on_interrupt)
        self._executor_pool.submit(self._mic.run, self._stop_event)
        self._mic.set_mode(MicMode.WAKE_WORD)
        self._health.mark("mic", HealthStatus.HEALTHY)

    async def _init_fsm(self) -> None:
        self._wake_pipeline = WakePipeline(self._session_cache)
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

        if self._behavioral_tracker and self._behavioral_tracker.healthy:
            self._bus.subscribe(ToolExecutionEvent, self._behavioral_tracker.on_tool_execution)

        if self._vector_client and self._vector_client.healthy:
            self._bus.subscribe(MemoryWriteEvent, self._vector_client.on_memory_write)

        self._bus.subscribe(SessionStateChangedEvent, self._wake_pipeline.on_state_changed)

        self._bus.subscribe(VoiceTranscriptEvent, self._on_voice_transcript)
        self._bus.subscribe(SessionStateChangedEvent, self._on_session_state_changed)
        self._bus.subscribe(ToolExecutionEvent, self._on_tool_execution)
        self._bus.subscribe(GestureEvent, self._on_gesture_while_media)

        # Phase 2: context + clipboard observers
        from runtime.event_bus import ContextChangedEvent, ClipboardChangedEvent
        if self._context_observer:
            self._bus.subscribe(ContextChangedEvent, self._on_context_changed)
        if self._clipboard_monitor:
            self._bus.subscribe(ClipboardChangedEvent, self._on_clipboard_changed)

    # ------------------------------------------------------------------
    # Voice transcript → Reasoner
    # ------------------------------------------------------------------

    async def _on_voice_transcript(self, event: VoiceTranscriptEvent) -> None:
        self._start_idle_timer()

        # Skip empty or noise transcripts
        if not event.text or not event.text.strip():
            return

        if self._reasoner is None or self._planner is None:
            await self._speak("I'm not ready yet.")
            return

        plan = await self._planner.plan(event.text, event.session_id)
        effective = plan.transcript.strip()
        if not effective:
            return

        tool_metas = self._tool_registry.list_all()
        recent = await self._session_cache.get_recent_commands(event.session_id)
        recent_texts = [str(c) for c in recent]

        response = await self._reasoner.process(
            transcript=effective,
            session_id=event.session_id,
            tool_metas=tool_metas,
            recent_commands=recent_texts,
            use_llm=plan.use_llm,
            tone_hint=plan.tone_hint,
        )

        if not plan.allow_tool_execution and response.action in (
            ReasonerAction.EXECUTE,
            ReasonerAction.SPEAK_AND_EXECUTE,
        ):
            response = ReasonerResponse(
                action=ReasonerAction.SPEAK,
                tool_name=None,
                params={},
                message=response.message or "I'm not able to run that action right now.",
                confidence=response.confidence,
                raw=response.raw,
                tier=response.tier,
                interim_messages=list(response.interim_messages),
            )

        logger.info(
            "Reasoner (tier %d): action=%s tool=%s msg='%s'",
            response.tier,
            response.action.name,
            response.tool_name,
            response.message[:80] if response.message else "",
        )

        if response.action == ReasonerAction.EXECUTE:
            await self._dispatch_tool(response, event, persist_memory=plan.persist_memory)

        elif response.action == ReasonerAction.SPEAK_AND_EXECUTE:
            if response.message:
                await self._speak(response.message)
            await self._dispatch_tool(response, event, persist_memory=plan.persist_memory)

        elif response.action == ReasonerAction.SPEAK:
            for interim in getattr(response, "interim_messages", []):
                if interim:
                    await self._speak(interim)
            if response.message:
                await self._speak(response.message)
            if self._media_playing:
                await self._restore_system_volume()

    async def _dispatch_tool(
        self,
        response,
        event: VoiceTranscriptEvent,
        *,
        persist_memory: bool = True,
    ) -> None:
        tool = self._tool_registry.get(response.tool_name)
        if tool is None:
            await self._speak("I don't know how to do that.")
            return

        params = dict(response.params)
        config_dict = self._config.model_dump()
        params["_config"] = config_dict
        params["_executor"] = self._tool_executor

        await self._bus.publish(IntentRoutedEvent(
            tool_name=response.tool_name, params=params,
            confidence=response.confidence, session_id=event.session_id,
            persist_memory=persist_memory,
        ))
        await self._session_cache.append_command(
            event.session_id,
            {"tool": response.tool_name, "params": response.params, "text": event.text},
        )

    # ------------------------------------------------------------------
    # Speech — mutes mic during output
    # ------------------------------------------------------------------

    async def _speak(self, text: str) -> None:
        if not text or not self._voice:
            return
        async with self._speech_lock:
            await self._voice.speak(text)

    # ------------------------------------------------------------------
    # Tool execution result — media state + selective speech
    # ------------------------------------------------------------------

    async def _on_tool_execution(self, event: ToolExecutionEvent) -> None:
        feedback = event.result.message or ("All set." if event.success else "That didn't work.")

        if event.success:
            if event.tool_name in _MEDIA_TOOLS:
                self._set_media_state(True)
                logger.info("Media playing — mic switches to wake-word gate")
            elif event.tool_name == "youtube_control":
                action = event.result.data.get("action", "") if event.result.data else ""
                if action in _MEDIA_PAUSE_ACTIONS:
                    self._set_media_state(False)
                elif action in _MEDIA_PLAY_ACTIONS:
                    self._set_media_state(True)

        should_speak = event.result.data.get("speak_result", False) if event.result.data else False
        if not event.success:
            should_speak = True

        if should_speak:
            await self._speak(feedback)

        # Always force mic to WAKE_WORD after media tool — fixes race with _on_session_state_changed
        if self._media_playing and self._mic:
            if not self._speech_lock.locked():
                self._mic.set_mode(MicMode.WAKE_WORD, self._current_session_id)

        # Restore system volume after a command during media playback
        if self._media_playing:
            await self._restore_system_volume()

        if self._reasoner:
            self._reasoner.inject_tool_result(
                event.session_id, event.tool_name, feedback,
            )

    # ------------------------------------------------------------------
    # Session state management
    # ------------------------------------------------------------------

    async def _on_session_state_changed(self, event: SessionStateChangedEvent) -> None:
        session_id = ""
        if self._wake_pipeline and self._wake_pipeline.current_context:
            session_id = self._wake_pipeline.current_context.session_id
        session_id = session_id or event.session_id or self._current_session_id
        self._current_session_id = session_id

        if event.new_state == SessionState.ACTIVE_SESSION:
            if event.old_state in (SessionState.WAKE_PENDING, SessionState.SLEEP):
                await self._session_store.start_session(session_id)

                greeting = await self._personality.generate_greeting()
                if greeting:
                    await self._speak(greeting)

            # Only set mic mode if NOT coming from EXECUTING — let _on_tool_execution handle that
            if event.old_state != SessionState.EXECUTING:
                if self._mic and not self._speech_lock.locked():
                    target = MicMode.WAKE_WORD if self._media_playing else MicMode.COMMAND
                    self._mic.set_mode(target, session_id)
            elif not self._media_playing:
                if self._mic and not self._speech_lock.locked():
                    self._mic.set_mode(MicMode.COMMAND, session_id)

            self._start_idle_timer()

        elif event.new_state == SessionState.WAKE_PENDING:
            if self._mic:
                self._mic.set_mode(MicMode.WAKE_WORD, session_id)

        elif event.new_state in (SessionState.SLEEP, SessionState.IDLE_TIMEOUT):
            if self._mic:
                self._mic.set_mode(MicMode.WAKE_WORD, session_id)
            self._cancel_idle_timer()

            if self._reasoner and session_id:
                summary = await self._reasoner.generate_session_summary(session_id)
                tools = self._reasoner.get_session_tools(session_id)
                if summary:
                    await self._session_store.end_session(session_id, summary, tools)
                self._reasoner.clear_session(session_id)

            self._set_media_state(False)

        elif event.new_state == SessionState.EXECUTING:
            if self._mic and not self._speech_lock.locked():
                self._mic.set_mode(MicMode.IDLE)
            self._cancel_idle_timer()

        if self._fsm and session_id:
            self._fsm.session_id = session_id

    def _set_media_state(self, playing: bool) -> None:
        self._media_playing = playing
        if self._mic:
            self._mic.set_media_playing(playing)
        if self._reasoner:
            self._reasoner.set_media_playing(playing)

    # ------------------------------------------------------------------
    # Media gate — wake word during playback
    # ------------------------------------------------------------------

    async def _on_gesture_while_media(self, event: GestureEvent) -> None:
        if not self._media_playing:
            return
        if not self._fsm or self._fsm.state != SessionState.ACTIVE_SESSION:
            return
        if event.type not in (GestureType.WAKE_WORD_DETECTED, GestureType.ALL_SIGNALS_CONFIRMED):
            return

        logger.info("Wake word during media — ducking volume, listening for command")
        await self._duck_system_volume()
        if self._mic:
            self._mic.set_mode(MicMode.COMMAND, self._current_session_id)
        self._start_idle_timer()

    # ------------------------------------------------------------------
    # System volume ducking — works for any audio source
    # ------------------------------------------------------------------

    async def _duck_system_volume(self) -> None:
        try:
            proc = await asyncio.create_subprocess_exec(
                "osascript", "-e", "return output volume of (get volume settings)",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            current = int(stdout.decode().strip())
            self._saved_volume = current
            await asyncio.create_subprocess_exec(
                "osascript", "-e", f"set volume output volume {_DUCK_LEVEL}",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            logger.info("Volume ducked: %d → %d", current, _DUCK_LEVEL)
        except Exception:
            logger.debug("System volume duck failed")

    async def _restore_system_volume(self) -> None:
        if self._saved_volume is None:
            return
        try:
            target = self._saved_volume
            self._saved_volume = None
            proc = await asyncio.create_subprocess_exec(
                "osascript", "-e", f"set volume output volume {target}",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            logger.info("Volume restored to %d", target)
        except Exception:
            logger.debug("System volume restore failed")

    # ------------------------------------------------------------------
    # Phase 2 — context + clipboard handlers
    # ------------------------------------------------------------------

    async def _on_context_changed(self, event) -> None:
        """Invalidate cached Reasoner system prompts so context is reflected next turn."""
        logger.debug(
            "Context changed: app=%s activity=%s url=%s",
            event.app,
            event.activity_type.name,
            event.browser_url[:60] if event.browser_url else "",
        )
        # Clear the cached fast + deep system prompts so next Reasoner call
        # picks up the updated workspace context via Personality.
        if self._reasoner:
            self._reasoner._cached_fast_prompt = None
            self._reasoner._cached_prompts.clear()

    async def _on_clipboard_changed(self, event) -> None:
        logger.debug(
            "Clipboard: type=%s len=%d preview='%s'",
            event.content_type,
            len(event.content),
            event.content[:40].replace("\n", " "),
        )

    # ------------------------------------------------------------------
    # Proactive engine loop
    # ------------------------------------------------------------------

    _proactive_task: asyncio.Task | None = None

    async def _proactive_loop(self) -> None:
        interval = self._config.proactive.check_interval
        await asyncio.sleep(10)  # initial delay to let everything settle
        while True:
            try:
                suggestions = await self._proactive.check()
                for s in suggestions:
                    await self._handle_proactive(s)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Proactive check failed")
            await asyncio.sleep(interval)

    async def _handle_proactive(self, suggestion: ProactiveSuggestion) -> None:
        logger.info("Proactive trigger: %s", suggestion.trigger_type)
        if suggestion.message:
            await self._speak(suggestion.message)
        if suggestion.auto_act and suggestion.tool_name:
            tool = self._tool_registry.get(suggestion.tool_name) if self._tool_registry else None
            if tool:
                try:
                    params = dict(suggestion.tool_params)
                    params["_config"] = self._config.model_dump()
                    params["_executor"] = self._tool_executor
                    await tool.execute(params, self._adapter)
                except Exception:
                    logger.exception("Proactive tool execution failed: %s", suggestion.tool_name)

    # ------------------------------------------------------------------
    # Idle timer
    # ------------------------------------------------------------------

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
                if not self._media_playing:
                    await self._fsm.trigger_idle_timeout()
                else:
                    self._start_idle_timer()
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    _shutting_down = False

    async def stop(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        logger.info("Shutting down Kairo daemon")
        self._cancel_idle_timer()
        if self._proactive_task and not self._proactive_task.done():
            self._proactive_task.cancel()
        if self._context_task and not self._context_task.done():
            self._context_task.cancel()
        self._stop_event.set()
        if self._camera:
            self._camera.stop()
        self._executor_pool.shutdown(wait=False)
        await self._bus.stop()
        if self._behavioral_tracker:
            await self._behavioral_tracker.close()
        if self._session_cache:
            await self._session_cache.close()
        if self._preferences:
            await self._preferences.close()
        if self._session_store:
            await self._session_store.close()
        if self._todo_store:
            await self._todo_store.close()
        logger.info("Kairo daemon stopped")


async def _run() -> None:
    _setup_logging()
    config = load_config()
    daemon = KairoDaemon(config)
    shutdown_event = asyncio.Event()

    loop = asyncio.get_running_loop()

    def _signal_handler() -> None:
        if not shutdown_event.is_set():
            shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    await daemon.start()
    await shutdown_event.wait()
    await daemon.stop()


def main() -> None:
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
