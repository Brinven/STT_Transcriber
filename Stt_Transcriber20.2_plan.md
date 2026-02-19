# Stt Transcriber20

## PRD

## Overview
A desktop Python application that records and streams speech‑to‑text (STT) during live meetings on Zoom or Microsoft Teams, ingests existing WAV files, and exports transcripts. It offers a standard transcription mode and a medical mode that can forward the transcript to a local Medical‑trained LLM for SOAP formatted notes.

## Problem Statement
Transcriptionists and meeting participants need an offline, privacy‑first desktop tool that captures audio from common collaboration platforms, produces accurate text transcriptions, and supports medical note formatting. Existing solutions are web‑based, require internet connectivity, or lack dedicated medical workflows, leading to latency, data exposure risks, and fragmented user experience.

## Goals & Non‑Goals
**In Scope**
- Audio capture from Zoom/Teams or local WAV files  
- Offline STT processing with Python library (e.g., Vosk)  
- Two modes: general transcription and medical transcription that can forward text to a local LLM for SOAP formatting  
- Export transcript to plain text or markdown file  
- Simple, single‑user desktop GUI  

**Not in Scope**
- Online cloud APIs beyond offline local models  
- Multi‑user collaboration features  
- Real‑time web streaming of transcripts  
- Advanced audio editing or post‑processing tools

## Target Users
- **Transcriptionists** – Professionals who transcribe meetings and medical dictations, requiring high accuracy and privacy.  
- **Medical staff** – Doctors or nurses needing quick SOAP note generation from meeting content.  
- **General users** – Anyone who attends Zoom/Teams meetings and wants a local transcript.

## Functional Requirements
1. Launch desktop application (Python + PyQt5).  
2. Record audio from the system microphone during live Zoom or Teams sessions.  
3. Import existing WAV files for offline transcription.  
4. Start STT processing locally using an open‑source Python STT engine (e.g., Vosk).  
5. Display real‑time transcript in a scrollable text pane.  
6. Allow user to pause/resume recording and review the transcript before finalization.  
7. Offer two modes: *General* and *Medical*.  
   - **General**: output plain transcript.  
   - **Medical**: pass transcript to a local medical‑trained LLM (e.g., a lightweight model packaged with the app) that formats notes into SOAP structure, or present a manual “SOAPify” button.  
8. Export finalized transcript as `.txt` or `.md` file locally.  
9. Maintain all data on the user’s machine; no cloud uploads unless user explicitly enables them.

## Non‑Functional Requirements
- **Performance**: Transcription latency ≤ 2 s per minute of audio for general mode, ≤ 4 s for medical mode.  
- **Security & Privacy**: All processing is local; no data leaves the host machine.  
- **Reliability**: Application should run on Windows / macOS / Linux with Python ≥3.10 and PyQt5.  
- **Usability**: Single‑window UI, clear start/stop buttons, mode toggle, and a status bar for progress.  
- **Accessibility**: Font size adjustable; keyboard shortcuts for start/stop (Space) and export (Ctrl+E).  

## User Flows
1. **Launch & Mode Selection**  
   1️⃣ Open app → choose *General* or *Medical* from toggle.  
2. **Recording a Meeting**  
   1️⃣ Click “Start” → app captures audio from system mic.  
   2️⃣ Live STT shows text in main pane.  
   3️⃣ Press “Pause” to stop capture; “Resume” resumes.  
3. **Finalize & Export**  
   1️⃣ Once finished, click “Export”.  
   2️⃣ Choose file name/location → app writes `.txt` or `.md`.  
4. **Medical Mode Special Path**  
   1️⃣ After export, click “SOAPify” button.  
   2️⃣ App forwards transcript to local medical LLM, receives SOAP notes, displays in a new pane.  
   3️⃣ User can copy‑paste or export the SOAP output.

## Success Metrics
- **Accuracy**: ≥ 90 % WER for general mode; ≥ 85 % WER for medical mode (measured on a curated test set).  
- **Usability Score**: ≥ 80 % in SUS questionnaire after first‑month use.  
- **Runtime**: Application launches <5 s on target OSes; recording stays responsive under 2 s latency.  
- **Offline Coverage**: All core functions work with no internet connection for ≥ 90 % of daily usage scenarios.

## Open Questions
1. Which open‑source STT engine to embed (Vosk, Whisper) – decision required.  
2. Exact algorithm or model for the medical LLM used in offline mode – needs specification.  
3. Preferred file format for exported transcript (plain txt vs markdown).  

---

---

## User Stories & AC

### US-001: Record Live Meetings via Zoom  
**As a** transcriptionist, **I want** to capture audio from a live Zoom meeting so that I can transcribe it within the app.  

**Acceptance Criteria:**  
- [ ] AC‑1: The app can connect to an active Zoom window and capture its audio stream at 16 kHz.  
- [ ] AC‑2: Captured audio is written to an in‑memory buffer without noticeable delay (≤200 ms).  
- [ ] AC‑3: A start/stop button initiates and terminates recording, updating the UI status indicator accordingly.  

**Priority:** Must  

---  

### US-002: Record Live Meetings via Teams  
**As a** transcriptionist, **I want** to capture audio from a live Microsoft Teams meeting so that I can transcribe it within the app.  

**Acceptance Criteria:**  
- [ ] AC‑1: The app detects an active Teams window and captures its audio stream at 16 kHz.  
- [ ] AC‑2: Recording starts/stops with the same UI controls as Zoom.  
- [ ] AC‑3: No external API calls are required; all capture is local.  

**Priority:** Must  

---  

### US-003: Streaming STT for Live Meetings  
**As a** transcriptionist, **I want** real‑time speech‑to‑text output while recording so that I can monitor accuracy during meetings.  

**Acceptance Criteria:**  
- [ ] AC‑1: The app streams audio to the local STT engine and displays recognized text within one second of audio input.  
- [ ] AC‑2: A “stream” toggle enables or disables real‑time display without affecting file capture.  
- [ ] AC‑3: The UI shows a live character counter for the current transcript segment.  

**Priority:** Must  

---  

### US-004: Ingest External WAV Files  
**As a** user, **I want** to load existing WAV files into the app so that I can transcribe them offline.  

**Acceptance Criteria:**  
- [ ] AC‑1: A file dialog accepts .wav files up to 500 MB.  
- [ ] AC‑2: The selected file’s audio is read at 16 kHz and processed by the STT engine.  
- [ ] AC‑3: A progress bar reflects ingestion progress; upon completion a status “Ready” appears.  

**Priority:** Must  

---  

### US-005: General Transcription Mode  
**As a** transcriptionist, **I want** a standard mode that produces plain text transcripts so that I can use them directly or export to other tools.  

**Acceptance Criteria:**  
- [ ] AC‑1: The mode processes audio through the STT engine and outputs plain text with speaker timestamps.  
- [ ] AC‑2: No additional formatting is applied unless explicitly requested by the user.  
- [ ] AC‑3: Switching this mode clears any prior medical‑mode settings.  

**Priority:** Must  

---  

### US-006: Medical Transcription Mode  
**As a** medical transcriptionist, **I want** a mode that sends output to a medical LLM for SOAP formatting so that I can obtain structured clinical notes.  

**Acceptance Criteria:**  
- [ ] AC‑1: The mode forwards the raw transcript text to a local instance of the medical LLM via an API endpoint (local only).  
- [ ] AC‑2: The returned output follows SOAP structure (Subjective, Objective, Assessment, Plan).  
- [ ] AC‑3: A toggle button switches between general and medical modes without restarting the app.  

**Priority:** Must  

---  

### US-007: Export to Text File  
**As a** user, **I want** to export transcripts as plain text files so that I can share them easily.  

**Acceptance Criteria:**  
- [ ] AC‑1: A “Save As” dialog writes the current transcript (or selected segment) to a .txt file with UTF‑8 encoding.  
- [ ] AC‑2: The file name defaults to “Transcript_YYYYMMDD_HHMMSS.txt”.  
- [ ] AC‑3: No external network call is required for export.  

**Priority:** Should  

---  

### US-008: Export to CSV Format  
**As a** data analyst, **I want** to export transcripts as CSV files so that I can import them into spreadsheets or databases.  

**Acceptance Criteria:**  
- [ ] AC‑1: The exported CSV contains columns: Timestamp, Speaker, Text.  
- [ ] AC‑2: Each row represents one spoken segment with its start time.  
- [ ] AC‑3: Export is triggered by a single button click and writes to disk without external services.  

**Priority:** Could  

---  

### US-009: Data Privacy – Local Storage Only  
**As a** privacy‑conscious user, **I want** all audio, transcripts, and processing to remain on my local machine so that no data leaves the device.  

**Acceptance Criteria:**  
- [ ] AC‑1: The app stores all temporary files in a hidden folder under the user's home directory.  
- [ ] AC‑2: No network requests are made during normal operation (recording, streaming, exporting).  
- [ ] AC‑3: A “Clear Data” button removes all stored artifacts on confirmation.  

**Priority:** Must  

---  

### US-010: Offline Operation  
**As a** user in low‑bandwidth areas, **I want** the app to function fully offline so that I can transcribe without internet access.  

**Acceptance Criteria:**  
- [ ] AC‑1: All STT inference is performed by a local model (e.g., Whisper).  
- [ ] AC‑2: The application starts and records within 5 s of launch, with no external API calls.  
- [ ] AC‑3: A status indicator shows “Offline” when no network connectivity is detected.  

**Priority:** Must  

---  

### US-011: Minimalist User Interface  
**As a** user, **I want** a single main window with clear controls so that I can start recording, switch modes, and export without confusion.  

**Acceptance Criteria:**  
- [ ] AC‑1: The UI contains buttons for Record/Stop, Mode Switch, Medical Toggle, Export (TXT/CVS), and Clear Data.  
- [ ] AC‑2: Each button is labeled with an icon and tooltip that appears on hover.  
- [ ] AC‑3: The current mode indicator is always visible at the top of the window.  

**Priority:** Should  

---  

### US-012: Robust Error Handling  
**As a** user, **I want** clear error messages for failures so that I can correct problems quickly.  

**Acceptance Criteria:**  
- [ ] AC‑1: Any exception during recording or processing shows a dialog with the message and an “Retry” button.  
- [ ] AC‑2: A log file is written to the user’s local folder for debugging.  
- [ ] AC‑3: The app continues operation after a recoverable error without requiring restart.  

**Priority:** Should

---

## Architecture

## Architecture Overview  
The application is a **monolithic desktop program** that runs entirely offline on the user's machine. All processing—audio capture, speech‑to‑text conversion, medical formatting, and file I/O—is performed locally without any external servers or APIs.

---

## Component Diagram  

- **UI (Tkinter)** – Handles all user interactions; opens recording windows, selects modes, shows transcripts.  
- **Audio Recorder** – Captures microphone input or reads WAV files; streams raw PCM to the STT Engine.  
- **STT Engine (Whisper)** – Performs offline speech‑to‑text; returns plain text to the UI.  
- **Medical Formatter (Llama.cpp)** – In medical mode, feeds STT output into a local LLM and formats notes as SOAP.  
- **File Manager** – Persists transcripts or SOAP notes to disk; supports import/export of WAV and TXT files.  
- **Config Manager** – Reads/writes user settings (mode, directories) from a JSON file.  
- **Logger** – Records errors/operations to a local log file for debugging.

Communication is synchronous function calls within the same process; no networking or message queues are required.

---

## Technology Stack  

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python 3.12 | Mature, cross‑platform, supports all chosen libraries. |
| UI framework | Tkinter | Built‑in, lightweight, sufficient for simple desktop UI. |
| Speech‑to‑Text engine | Whisper (local) | Offline, high accuracy, pure python binding. |
| Medical LLM | Llama.cpp | Fully offline, small footprint, allows uncensored medical note formatting. |
| Packaging tool | PyInstaller | Generates single executable per OS; keeps deployment simple for a solo developer. |
| Logging | Python logging module | Standard library, no external deps. |
| Configuration format | JSON | Human‑readable, standard lib support. |

---

## Data Model  

**Session**  
- id (int): Unique session identifier.  
- start_time (datetime): Recording start.  
- end_time (datetime): Recording end.  
- mode (string): "general" or "medical".  

**Transcript**  
- session_id (int, FK to Session.id) : Associated session.  
- raw_text (text): Text from Whisper.  
- processed_text (text): Text after medical formatting (SOAP).  

**MedicalNote**  
- transcript_id (int, FK to Transcript.session_id) : Original transcript.  
- soap_section (string): "Subjective".  
- soap_section (string): "Objective".  
- soap_section (string): "Assessment".  
- soap_section (string): "Plan".  

---

## API Design  

No external API – all communication is internal via method calls within the same process.

---

## Directory Structure  

```
stt_app/
  src/
    __init__.py
    main.py                # Application entry point
    ui/
      __init__.py
      recorder_window.py   # Tkinter recording interface
    config_manager.py     # Handles JSON config file
    logger.py            # Logging wrapper
    audio_recorder.py    # Microphone/WAV ingestion
    stt_engine.py        # Wrapper for Whisper
    medical_formatter.py # Wrapper for Llama.cpp
    file_manager.py      # Read/write transcripts and notes
  data/
    config.json           # User settings (created on first run)
    logs/
      app.log            # Runtime log file
  tests/
    __init__.py
    test_audio_recorder.py
    test_stt_engine.py
  requirements.txt         # pip freeze of used packages
  README.md
```

---

## Infrastructure & Deployment  

- **Runtime** – Works on Windows/macOS/Linux desktop; Python interpreter is bundled with PyInstaller into a single executable.  
- **Installation** – End user runs the generated `stt_app.exe` (or `.app`, `.AppImage`). No admin rights required.  
- **Execution** – Launches Tkinter GUI; all processing stays on local machine.

---

## Security Considerations  

1. **Local Storage Encryption** – Configuration JSON is stored in plain text, but optional AES‑256 encryption of the file can be added later if needed.  
2. **File Permissions** – All generated files (transcripts, logs) are created with 0600 permissions on POSIX to limit external read/write.  
3. **No Network Exposure** – The app never contacts remote servers; all models run locally, preserving privacy of medical data and NSFW content.

---

## Key Technical Decisions  

- *Monolithic desktop architecture* – keeps deployment simple for a single user per instance.  
- *Tkinter UI* – avoids heavy dependencies while providing sufficient interaction.  
- *Whisper local engine* – delivers high‑accuracy STT without internet; satisfies offline constraint.  
- *Llama.cpp medical formatter* – enables uncensored, local LLM inference for SOAP formatting.  
- *PyInstaller packaging* – produces single binary executable, easing distribution to non‑programmer users.  

These choices align with the MVP priorities of core functionality, stability, and ease of use while respecting privacy and offline operation.

---

## Milestones

Assumptions  
- For the MVP no external API integrations will be used; all processing is offline as per user’s assumption that third‑party services are not required for MVP.

### Milestone 1: Project Setup & Basic UI
**Goal:** Establish a runnable Python desktop application skeleton with a main window and configuration system.

**Deliverables:**
- [ ] Create virtual environment and install dependencies (PyQt5, sounddevice, Whisper)
- [ ] Build base PyQt5 window with menu bar and status bar

**Key Tasks:**
1. Create `requirements.txt` and set up a Python 3.x virtual environment
2. Initialize a Git repository and commit initial structure
3. Scaffold main application file (`main.py`) that launches a PyQt5 `QMainWindow`
4. Add a basic configuration file (`config.json`) with default settings

**Exit Criteria:**  
- Running `python main.py` opens an undecorated window titled “STT Recorder” without errors  
- Configuration file exists and can be read into the application

**Dependencies:** None  

---

### Milestone 2: Offline Audio Capture & General Transcription
**Goal:** Enable users to record local audio and obtain accurate offline transcription with Whisper.

**Deliverables:**
- [ ] Recording module that captures microphone input and saves as WAV
- [ ] Whisper inference pipeline that accepts a WAV file and returns a plain text transcript

**Key Tasks:**
1. Implement audio capture using `sounddevice` or `pyaudio` to record into a temporary WAV file
2. Load the base Whisper model (`whisper-small`) at application start‑up
3. Add a “Record” button that triggers recording and, on completion, feeds the WAV into Whisper for transcription
4. Display the resulting transcript in a read‑only text widget

**Exit Criteria:**  
- User can record 30 s of audio and see a full transcript appear within 5 s after stopping recording  
- No crashes or hangs during capture or inference  

**Dependencies:** Milestone 1  

---

### Milestone 3: Live Meeting Integration & WAV Ingestion
**Goal:** Allow users to import existing WAV files and stream live audio from Zoom/Teams into the application for real‑time transcription.

**Deliverables:**
- [ ] File‑open dialog that accepts a WAV file and displays its transcript
- [ ] Virtual loopback capture that can be selected as audio input in Zoom/Teams and routed into Whisper

**Key Tasks:**
1. Add “Import WAV” menu action that opens a `QFileDialog` and runs the existing transcription pipeline on the chosen file
2. Create a virtual audio loopback device (e.g., using PyAudio’s `Stream` with `callback`) to capture system audio from Zoom/Teams
3. Feed captured frames into Whisper in small chunks, buffering as needed to avoid stutters
4. Show live transcription updates in the main text widget

**Exit Criteria:**  
- Importing a WAV shows its full transcript immediately  
- Starting the loopback stream with an open Zoom/Teams window yields live transcription within 10 s of lag  

**Dependencies:** Milestone 2  

---

### Milestone 4: Medical Mode & SOAP Formatting
**Goal:** Provide a medical transcription mode that formats raw output into a SOAP structure.

**Deliverables:**
- [ ] Toggle for “Medical Mode” in the UI
- [ ] Simple SOAP formatter that parses key phrases (subject, objective, assessment, plan) and returns formatted text

**Key Tasks:**
1. Add a checkbox or radio button to enable/disable medical mode
2. Implement a lightweight regex‑based parser that identifies typical medical terms and builds a SOAP skeleton
3. Hook the formatter into the transcription pipeline so that when medical mode is active, the displayed output is the formatted notes instead of raw transcript
4. Provide an option to export the SOAP notes to a text file

**Exit Criteria:**  
- In medical mode, feeding a sample transcript produces a correctly structured SOAP block (Subject:, Objective:, Assessment:, Plan:)  
- User can save the SOAP output and verify it in a plain‑text editor  

**Dependencies:** Milestone 3  

---

### Milestone 5: Polish & Release
**Goal:** Final polish, packaging, and release of a stable desktop installer.

**Deliverables:**
- [ ] Refactored codebase with clear module separation
- [ ] Comprehensive unit tests covering core functions
- [ ] Windows 64‑bit installer generated via PyInstaller

**Key Tasks:**
1. Clean up UI styling (fonts, spacing) for a polished look
2. Add error handling and logging to catch and report failures gracefully
3. Write a short user manual and include it in the installer
4. Build an executable with PyInstaller and test on fresh Windows 10 machine

**Exit Criteria:**  
- Installer runs without errors and installs all dependencies  
- All features (recording, import, live stream, medical mode) work end‑to‑end after installation  
- User manual is present and accurate  

**Dependencies:** Milestone 4

---

## Test Plan

**Testing Strategy**  
- **Approach** – All tests are automated with `pytest` to keep the suite lightweight for a solo developer.  
- **Tools** –  
  * Unit & integration tests: `pytest`, `pytest-cov`.  
  * End‑to‑end desktop automation: `pyautogui` (chosen because it works on Windows/macOS/Linux and requires no external drivers).  
  * Manual testing is kept to a few hand‑crafted scenarios.

**Test Levels**

| Level | Key Modules / Components | What to assert | Coverage Target |
|-------|--------------------------|---------------|----------------|
| **Unit** | `audio_capture.py`, `transcription_engine.py` (general & medical), `export_manager.py`, `config_manager.py` | - Correct sample rate and channels.<br>- Transcriber returns expected text for stubbed LLM.<br>- Export writes correct file format.| 80 % |
| **Integration** | `audio_capture.py ↔ transcription_engine.py`, `transcription_engine.py ↔ export_manager.py` | - Recorded audio is passed to engine without loss.<br>- Engine output can be consumed by export manager and written to disk. | 70 % |
| **End‑to‑end (desktop)** | Full GUI workflow – launch, record, stop, switch mode, export | - UI buttons enable/disable correctly.<br>- Switching between general & medical modes changes the LLM endpoint.<br>- Export button produces a valid transcript file. | Manual + 1 autotest |
| **Manual** | Privacy settings, NSFW content handling, offline operation | Verify that data never leaves local machine and that content is not censored. |

**Test Environment**

- Python 3.10+  
- `pytest`, `pytest-cov` for unit/integration tests  
- `pyautogui` for E2E scripts  
- FFmpeg (audio capture) installed locally  
- Sample WAV files placed under `tests/data/`  

**Test Data**

| Scenario | Seed Data | Edge Cases |
|----------|-----------|------------|
| Normal speech | `sample_normal.wav` | Empty audio file, 0 s |
| Medical terminology | `sample_medical.wav` | Corrupted wav, very long recording (>5 min) |
| NSFW content | `sample_nsfw.wav` | Mixed normal + medical content |

**Coverage Goals**

| Level | Target |
|-------|--------|
| Unit | ≥ 80% |
| Integration | ≥ 70% |

**Regression Strategy**

- Run the full test suite (`pytest --cov=src -m "not slow"`) locally before each commit.  
- On CI (e.g., GitHub Actions) trigger on every push to `main`.  
- If any test fails, halt deployment and notify via email or Slack.

**Performance Testing**

Not applicable for MVP.

**Accessibility Testing**

Not applicable – the app is a desktop GUI with no complex widgets.

---

## Release Checklist

# Release Checklist

## Pre‑Release
- [ ] All unit and integration tests pass.
- [ ] Code review completed (self‑review for solo devs).
- [ ] Audio capture modules correctly record `.wav` files at 48 kHz, 16‑bit depth.  
- [ ] Medical mode SOAP formatter outputs valid JSON adhering to the specified structure.

## Build & Package
- [ ] PyInstaller build succeeds with no errors or warnings.
- [ ] Executable includes all model weights and config files (no missing dependencies).
- [ ] Offline run test: application starts and completes a full transcription cycle without internet connectivity.
- [ ] Launch `.exe` on target Windows OS; verify UI appears and records audio.

## Documentation
- [ ] `README.md` updated with setup, build, and usage instructions for the desktop app.
- [ ] `CHANGELOG.md` updated with version history and key feature changes.
- [ ] API documentation (auto‑generated or hand‑written) added for internal modules used by the recorder and transcriber.
- [ ] Config guide explains LLM endpoint settings (`LLM_Studio`, `Ollama`) and privacy toggle usage.

## Deployment
- [ ] Copy the built `.exe` to the distribution folder on the target machine.
- [ ] Create a desktop shortcut that launches the application from the start menu.
- [ ] Post‑deployment verification: launch app, record a short clip, stream to transcription, and confirm output appears in the UI.

## Post‑Release
- [ ] Smoke test in release environment: run the full workflow once and check for crashes or hangs.
- [ ] Verify transcription accuracy on a sample meeting; error rate ≤ 5 % compared to reference transcript.
- [ ] Verify medical mode produces SOAP notes that match the required JSON schema and are exportable to a `.txt` file.

---

## Risk & Decision Log

## Risk Log

| ID   | Risk                                                                                           | Likelihood | Impact | Mitigation                                                                                                                                                                    | Owner    | Status |
|------|--------------------------------------------------------------------------------------------------|------------|--------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|--------|
| R-1  | Large Whisper model size causes slow startup and high disk usage.                                 | High        | Med    | Use the quantized “base” Whisper model and load only the required language weights at runtime.                                                                      | Developer | Open   |
| R-2  | Audio from Zoom/Teams may be in varying sample rates or channel counts, leading to transcription errors. | Med         | High   | Pre‑process all audio streams with ffmpeg to mono 16 kHz PCM before feeding into the recognizer; log and retry on failure.                                                 | Developer | Open   |
| R-3  | Medical LLM may produce inaccurate SOAP notes due to limited training data.                   | Med         | High   | Validate generated SOAPs against a small curated test set, use confidence thresholds to fallback to template‑based formatting when low.                                   | Developer | Open   |
| R-4  | CPU‑only inference on typical laptops can be slow and affect user experience.                  | Low         | Med    | Offer an optional GPU flag that loads the CUDA‑enabled Whisper backend; otherwise run a CPU‑quantized model and display progress bars.                                       | Developer | Open   |
| R-5  | Storing raw audio/transcripts on disk poses privacy risk if other apps access them.           | High        | Med    | Encrypt all temp files with Fernet key derived from user password; delete temporary files immediately after processing.                                                          | Developer | Open   |
| R-6  | Ingesting arbitrary WAV files may fail when format differs (e.g., 24‑bit, stereo).               | Med         | Low    | Wrap ffmpeg conversion in a subprocess and verify output format before transcription; retry with alternative codecs if needed.                                                | Developer | Open   |

## Decision Log

### D-001: UI Framework Selection
- **Date:** TBD
- **Status:** Accepted
- **Context:** Need a cross‑platform desktop GUI that supports custom widgets for audio controls.
- **Decision:** Use PyQt5 as the UI framework.
- **Alternatives Considered:** Tkinter, wxPython
- **Consequences:** Larger binary than Tkinter but provides richer media widgets; requires installing Qt libraries.

### D-002: Speech Recognition Engine
- **Date:** TBD
- **Status:** Accepted
- **Context:** Must transcribe audio offline with minimal external dependencies.
- **Decision:** Use the quantized Whisper “base” model from OpenAI’s open‑source library.
- **Alternatives Considered:** Vosk, DeepSpeech
- **Consequences:** Whisper offers best accuracy for generic English; requires ~500 MB of model files but can be trimmed.

### D-003: Medical Note Formatter
- **Date:** TBD
- **Status:** Accepted
- **Context:** Need to convert transcripts into SOAP format without external APIs.
- **Decision:** Integrate a local LLM via llama.cpp trained on medical notes and expose it through a simple prompt template.
- **Alternatives Considered:** OpenAI GPT‑3, Claude, third‑party medical LLMs
- **Consequences:** Keeps all processing offline; limited by local inference speed but satisfies privacy requirement.

### D-004: Application Architecture
- **Date:** TBD
- **Status:** Accepted
- **Context:** Single user per install, low concurrency.
- **Decision:** Build a monolithic single‑process application with separate modules for UI, audio ingestion, transcription, and note formatting.
- **Alternatives Considered:** Micro‑service or Electron wrapper
- **Consequences:** Simpler deployment; no inter‑process communication overhead.

---

## Backlog

| # | Item | Type | Priority | Effort | Milestone | Description |
|---|------|------|----------|--------|-----------|-------------|
| 1 | Audio Recording | Feature | Must | M | M1 | Capture microphone audio locally |
| 2 | Offline Whisper Transcription | Feature | Must | L | M1 | Run local Whisper model to produce transcript |
| 3 | File Export (TXT) | Feature | Must | S | M1 | Save transcript as plain text file |
| 4 | Zoom Live Capture | Feature | Must | L | M2 | Record audio from Zoom meeting window |
| 5 | Teams Live Capture | Feature | Must | L | M2 | Record audio from Microsoft Teams meeting window |
| 6 | Medical SOAP Formatter | Feature | Must | M | M3 | Send transcript to medical LLM for SOAP formatting (placeholder) |
| 7 | Wav Ingest & Transcribe | Feature | Should | M | M3 | Accept external wav files and transcribe |
| 8 | Export Formats (TXT/CSV) | Feature | Should | M | M3 | Provide txt and csv export options |
| 9 | PyQt5 GUI | Feature | Should | L | M4 | Build main window with controls using PyQt5 |
|10 | API Wrappers for LM Studios & Ollama | Chore | Should | S | M4 | Stubbed wrappers for future API calls |
|11 | Local Data Privacy | Feature | Must | S | M1 | Ensure all data stored locally, no external sync |
|12 | NSFW Content Support | Feature | Must | S | M1 | Allow uncensored transcription output |
|13 | Custom UI Themes | Feature | Could | S | M5 | Provide optional dark/light theme selection |
|14 | PDF Export | Feature | Could | L | M5 | Generate transcript in PDF format |

**Notes**

The backlog is split into five milestones.  
- **M1** gathers the core recording, offline Whisper transcription, basic export, and privacy guarantees – essential for a single‑user MVP.  
- **M2** tackles platform capture logic for Zoom and Teams, which are independent of the main transcribe pipeline.  
- **M3** bundles ingestion/export features and the medical SOAP formatter because they all rely on having a transcript already produced by M1.  
- **M4** focuses on UI polish (PyQt5) and any supporting tooling such as API wrappers; these are infrastructure that can be built once core logic works.  
- **M5** contains optional niceties like theming and PDF export, scheduled after the core experience is stable.  

This order keeps development linear: finish M1 first, then extend to external platforms (M2), add advanced processing (M3), solidify UI & privacy (M4), and finally deliver polish features (M5).
