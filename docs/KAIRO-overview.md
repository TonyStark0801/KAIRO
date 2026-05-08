You are reviewing an early-stage local AI assistant project named KAIRO.

KAIRO stands for:
Knowledge-Aware Interactive Runtime Operator

This project is evolving into a local-first developer companion agent that:

* listens via microphone (wake word: "Hey Kairo")
* performs speech-to-text locally
* reasons using a small local LLM (Qwen 3 4B or similar)
* speaks responses via local TTS
* observes system activity (active window, browser tab, file usage)
* stores structured long-term memory
* executes tool actions (open apps, run commands, read files)
* evolves contextually over time

Architecture target:

Mic → Wake Word → VAD → STT → Intent Detector → Dialogue Planner → LLM Brain → Tool Router → Memory Layer → TTS Output

System should be event-driven using Redis Streams between services.

Your task:

1. Analyze this repository structure
2. Identify which modules already exist
3. Identify missing architectural layers
4. Detect coupling problems
5. Suggest how to refactor toward a modular agent runtime
6. Suggest folder restructuring if needed
7. Suggest which parts belong in:
   observer/
   planner/
   brain/
   memory/
   connectors/
   stt/
   tts/
   bus/

Output format:

SECTION 1 — Current Architecture Summary
SECTION 2 — Missing Components
SECTION 3 — Coupling Issues
SECTION 4 — Refactor Suggestions
SECTION 5 — Recommended Next 5 Commits
SECTION 6 — Suggested Final Folder Structure
