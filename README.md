# AI Audio Detector

A forensic analysis system for detecting AI-generated music using signal processing, spectral analysis, and psychoacoustic metrics. Implements 17 detection checks across 7 analysis domains, producing a weighted probabilistic score (0-100%) with confidence level.

Built on research into the architectural origins of generative audio artifacts -- from convolutional upsampling checkerboard patterns to neural codec phase smearing -- this tool identifies the constellation of spectral, temporal, spatial, and structural anomalies that distinguish AI-synthesized audio from professional studio production.

---

## Table of Contents

- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Detection Domains](#detection-domains)
- [Analysis Depth Levels](#analysis-depth-levels)
- [API Reference](#api-reference)
- [Frontend](#frontend)
- [Scoring System](#scoring-system)
- [Installation](#installation)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Platform Support](#platform-support)
- [Limitations](#limitations)

---

## Quick Start

```bash
# Clone and enter project
cd ai-audio-detector

# macOS (Apple Silicon or Intel)
./scripts/setup-macos.sh

# Windows (auto-detects CUDA)
# powershell -ExecutionPolicy Bypass -File scripts/setup-windows.ps1

# Start the backend server
source .venv/bin/activate
uvicorn backend.app.main:app --reload

# Start the frontend (separate terminal)
cd frontend
npm install
npm run dev
# Open http://localhost:5174

# Or analyze via CLI
curl -X POST "http://localhost:8000/api/analyze?depth=standard" \
  -F "file=@your_track.wav"
```

---

## How It Works

The detector doesn't rely on a single "smoking gun." Instead, it performs **multi-modal triangulation** across seven independent analysis domains, each targeting a different category of artifact that AI generation systems produce. These artifacts arise from the mathematical properties of neural architectures themselves:

- **Autoregressive models** (MusicGen, AudioLM) generate audio token-by-token through neural codecs, introducing quantization noise and high-frequency phase smearing
- **Latent diffusion models** (AudioLDM, Riffusion) reconstruct magnitude spectrograms but approximate phase, causing transient smearing
- **CNN decoders** use transposed convolutions for upsampling, creating mathematically provable checkerboard spectral patterns

Each domain analyzer runs its checks independently, produces per-artifact probability scores, and feeds them into a weighted scoring engine that outputs an overall AI probability percentage.

---

## Detection Domains

### 1. Spectral Analysis (weight: 40%)

Examines frequency-domain characteristics for neural synthesis artifacts.

| Check | What It Detects | Method |
|-------|----------------|--------|
| **Checkerboard Artifacts** | Grid-like periodic patterns in the high-frequency spectrum caused by CNN transposed convolution upsampling | FFT of spectral slices in 4-20kHz band; measures peak-to-mean ratio of periodic components |
| **HF Harmonic-to-Noise Ratio** | Metallic/buzzy texture where AI forces harmonic structure into naturally noise-dominated high frequencies (>12kHz) | HPSS in 12-20kHz band; computes 10*log10(harmonic/noise energy). AI: >5dB, Studio: <0dB |
| **Spectral Rolloff** | Brick-wall frequency cutoffs from model bandwidth limits (e.g., 16kHz from 32kHz training data) | Finds effective bandwidth, measures rolloff slope (brick-wall vs. natural), checks for aliasing artifacts |
| **Spectral Flux Variance** | Static, smeared mixes where AI produces a "wall of sound" instead of clean instrument transitions | L2-norm spectral flux between consecutive STFT frames; measures coefficient of variation |

### 2. Spatial Analysis (weight: 5%)

Examines stereo imaging and phase coherence. Only active for stereo audio.

| Check | What It Detects | Method |
|-------|----------------|--------|
| **Phase Correlation** | Anti-correlated stereo channels, catastrophic mono cancellation, or near-mono stereo fields | Pearson correlation, mono summation loss test, per-band (low/mid/high) correlation analysis |
| **Bass Stereo Width** | Improperly wide bass frequencies (<200Hz) that should be centered in professional mixes | Mid-side decomposition of lowpass-filtered signal; computes side/mid RMS ratio |

### 3. Production Analysis (weight: 5%)

Evaluates mixing and mastering quality metrics.

| Check | What It Detects | Method |
|-------|----------------|--------|
| **Crest Factor** | Low peak-to-RMS ratio with soft tanh-shaped saturation instead of professional hard limiting | 20*log10(peak/RMS) plus amplitude histogram analysis for saturation curve shape |
| **Loudness Range (EBU R128)** | Flat dynamics where verse and chorus are equally loud with no emotional arc | Short-term loudness in 3s windows; LRA = 95th - 10th percentile. AI often <3 LU |
| **Reverb Tail Analysis** | Non-exponential reverb decay, abrupt gating, or morphing tails that violate room acoustics physics | Fits exponential decay to post-onset energy envelopes; measures R-squared goodness-of-fit |

### 4. Temporal Analysis (weight: 10%)

Analyzes micro-timing and transient characteristics.

| Check | What It Detects | Method |
|-------|----------------|--------|
| **Log-Attack Time** | Smeared transients from diffusion model phase errors -- drums that sound "thuddy" rather than "punchy" | Onset detection + envelope peak measurement; computes LAT = log10(attack_seconds). AI: >-1.5, Real: <-2.0 |
| **Rhythmic Jitter** | Uncorrelated random micro-timing (vs. human groove which is correlated) and tempo drift | Beat tracking, IBI coefficient of variation, and autocorrelation of timing deviations |

### 5. Structural Analysis (weight: 20%)

Evaluates musical form and compositional coherence. Deep depth only.

| Check | What It Detects | Method |
|-------|----------------|--------|
| **Self-Similarity Matrix** | Missing structural repetition ("hallucinated" choruses) or rigid looping (repetition trap) | Cosine similarity of chroma features; analyzes off-diagonal stripes at expected section intervals |
| **Structural Entropy** | Monotonous harmonic content and non-resolving cadences that violate tonal music grammar | Per-bar Shannon entropy of chroma distributions; checks cadence resolution at section boundaries |

### 6. Vocal Analysis (weight: 20%)

Detects vocal synthesis artifacts. Deep depth only. Automatically deactivates if no vocals are detected. Requires optional `parselmouth` dependency for formant analysis.

| Check | What It Detects | Method |
|-------|----------------|--------|
| **Formant Stability** | "Variable head size" effect where formants shift with pitch instead of staying anchored to vocal tract geometry | Pitch-formant correlation via Praat; real singers have near-zero correlation |
| **Breath Logic** | Physiologically impossible breathing: infinite phrases (>15s), mid-note breaths, double breaths | Energy envelope analysis with breath detection; counts physiological anomalies |
| **Phoneme Clarity** | Smeared consonants from neural codec HF artifacts -- "lispy" sibilants and missing plosive gaps | Sibilant frequency variance analysis (5-10kHz) and plosive gap detection |

### 7. Watermark Detection (currently disabled)

Checks for embedded AI provenance watermarks. Deep depth only. Requires optional `audioseal` + `torch` dependencies.

| Check | What It Detects | Method |
|-------|----------------|--------|
| **AudioSeal Watermark** | Meta AI's embedded audio watermark -- the only near-definitive proof of AI generation | AudioSeal detector model on 16kHz resampled audio; if detected, overrides overall score to 95%+ |

---

## Analysis Depth Levels

Control the trade-off between speed and thoroughness:

| Depth | Domains Active | Approx. Time | Best For |
|-------|---------------|--------------|----------|
| `quick` | Spectral, Spatial, Production (crest + loudness) | ~3-5 seconds | Batch screening, quick triage |
| `standard` | + Temporal, Production (reverb) | ~10-15 seconds | General-purpose analysis |
| `deep` | + Structural, Vocal, Watermark | ~30-45 seconds | Forensic investigation, disputed tracks |

Times are approximate for a 3-minute stereo track at 44.1kHz.

---

## API Reference

### `POST /api/analyze`

Upload an audio file for AI detection analysis.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `depth` | string | `standard` | Analysis depth: `quick`, `standard`, or `deep` |

**Request:** `multipart/form-data` with a `file` field.

**Supported formats:** WAV, MP3, FLAC, OGG, M4A, AAC, WMA

**Response:**

```json
{
  "scan_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "analyzed_at": "2025-06-15T12:00:00+00:00",
  "tool_version": "0.1.0",
  "filename": "track.wav",
  "duration_seconds": 180.5,
  "sample_rate": 44100,
  "channels": 2,
  "peak_db": -0.3,
  "rms_db": -14.2,
  "overall_score": 72.4,
  "confidence": "high",
  "confidence_value": 0.85,
  "depth_used": "deep",
  "overall_ai_likelihood": "likely",
  "domain_results": [
    {
      "domain": "spectral",
      "display_name": "Spectral Analysis",
      "score": 0.68,
      "active": true,
      "weight": 0.40,
      "artifacts": [
        {
          "name": "checkerboard_artifacts",
          "detected": true,
          "severity": "medium",
          "value": 3.5,
          "description": "Spectral periodicity ratio: 3.5x. Periodic grid pattern detected...",
          "probability": 0.6,
          "domain": "spectral",
          "weight": 5.0,
          "tier": 3
        }
      ]
    }
  ],
  "ai_artifacts": [],
  "high_freq_cutoff_hz": 15200.0,
  "stereo_correlation": 0.92
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `scan_id` | string | Unique UUID identifying this scan |
| `analyzed_at` | string | ISO 8601 timestamp of when the analysis was performed |
| `tool_version` | string | Software version that performed the analysis |
| `overall_score` | float | 0-100% probability the track is AI-generated |
| `confidence` | string | `low`, `medium`, or `high` -- based on how many domains contributed |
| `confidence_value` | float | 0.0-1.0 numeric confidence |
| `overall_ai_likelihood` | string | `unlikely` (<35%), `possible` (35-65%), `likely` (>65%) |
| `domain_results` | array | Per-domain breakdown with individual artifact details |
| `ai_artifacts` | array | Flat list of all artifact results across domains |
| artifact `tier` | int | 1=definitive, 2=strong, 3=moderate, 4=weak (see Artifact Tier System) |

### `GET /api/health`

Health check endpoint. Returns `{"status": "ok", "service": "ai-audio-detector"}`.

### Interactive API Docs

When the server is running, visit `http://localhost:8000/docs` for the Swagger UI.

---

## Frontend

A dark-themed single-page application built with React 18, TypeScript, Vite, and Tailwind CSS. Upload audio files, select analysis depth, and view results with per-domain breakdowns.

### Features

- **Drag-and-drop file upload** with format validation (WAV, MP3, FLAC, OGG, M4A, AAC, WMA)
- **Three analysis depths** with estimated time display (Quick ~5s, Standard ~15s, Deep ~45s)
- **Circular score gauge** (0-100%) color-coded green/yellow/red with likelihood and confidence badges
- **Collapsible domain cards** with per-artifact probability bars and severity indicators
- **Audio metadata display** showing duration, sample rate, channels, and peak/RMS levels
- **Domain preview grid** that highlights active domains based on selected depth
- **Responsive layout** with error handling for network failures and timeouts

### Tech Stack

| Package | Version | Purpose |
|---------|---------|---------|
| React | ^18.2.0 | UI framework |
| Vite | ^5.0.11 | Dev server and bundler |
| TypeScript | ^5.3.3 | Type safety |
| Tailwind CSS | ^3.4.1 | Utility-first styling |
| Axios | ^1.6.5 | HTTP client with 5-minute timeout for deep analysis |
| Lucide React | ^0.303.0 | Icons |

### Components

| Component | Description |
|-----------|-------------|
| `App.tsx` | Main app with state machine: idle → analyzing → results / error |
| `FileUploader.tsx` | Drag-and-drop upload zone with file info and format validation |
| `DepthSelector.tsx` | Quick / Standard / Deep button group with time estimates |
| `ScoreGauge.tsx` | SVG circular gauge with color gradient, likelihood badge, confidence badge |
| `DomainCard.tsx` | Collapsible card per domain with score bar, artifact rows, severity badges |
| `AudioInfo.tsx` | 4-column metadata grid (duration, sample rate, channels, peak/RMS) |
| `AnalysisResults.tsx` | Full results layout combining gauge, domain cards, and summary stats |

### Running the Frontend

```bash
cd frontend
npm install
npm run dev    # http://localhost:5174
```

The Vite dev server proxies all `/api` requests to the backend at `http://localhost:8000`. Both servers must be running simultaneously.

**Production build:**

```bash
npm run build  # outputs to frontend/dist/
```

---

## Scoring System

### Artifact Tier System

Every artifact check is classified into a reliability tier that determines its influence on the overall score. Tiers were calibrated against a 22-track empirical dataset (10 human, 12 AI):

| Tier | Label | Meaning | Examples |
|------|-------|---------|----------|
| 1 | Definitive | Near-certain AI indicator when detected | AudioSeal watermark, loudness range |
| 2 | Strong | Strongly suggests AI, rare in legitimate audio | Reverb tail analysis, HF harmonic-to-noise ratio, breath logic |
| 3 | Moderate | Useful supporting evidence | Checkerboard artifacts, spectral rolloff, structural entropy |
| 4 | Weak | Commonly triggered by legitimate audio | Crest factor, log-attack time, rhythmic jitter, bass stereo width |

Artifact weights and tiers are centralized in `backend/app/services/analyzers/weights.py` for transparency and easy tuning.

### Per-Domain Score

Each domain runs N checks. Each check produces a probability (0.0-1.0) and a weight (from `weights.py`). The domain score is a **blend** that prevents a single strong detection from being diluted by co-domain checks returning 0.0:

```
weighted_mean = sum(probability_i * weight_i) / sum(weight_i)
max_prob = max(probability_i for all checks)
domain_score = 0.7 * weighted_mean + 0.3 * max_prob
```

### Overall Score

The scoring engine applies two refinements beyond simple weighted averaging:

**1. Evidence gating** -- Domains scoring below 0.10 get their weight linearly scaled down, so near-zero domains don't consume weight budget from informative ones:

```
gate = min(domain_score / 0.10, 1.0)
effective_weight = base_weight * gate
```

**2. Concordance boost** -- When two or more domains independently score above 0.35, a bonus proportional to agreement strength is added:

```
if concordant_domains >= 2:
    bonus = (concordant_count / active_count) * mean_concordant_score * 0.25
    final_score = base_score + bonus
```

Inactive domains (skipped by depth, not applicable, or missing optional dependencies) are excluded and their weight redistributed among active domains.

### Domain Weights

| Domain | Weight | Rationale |
|--------|--------|-----------|
| Spectral | 0.40 | Primary low-level signal detection (4 checks) |
| Structural | 0.20 | Macro-level coherence and entropy |
| Vocal | 0.20 | Human voice discrimination (breath, formants, phonemes) |
| Temporal | 0.10 | Micro-timing and transient analysis |
| Spatial | 0.05 | Stereo field anomalies (stereo only) |
| Production | 0.05 | Dynamics and reverb analysis |
| Watermark | -- | Currently disabled; override logic remains active |

### Confidence Level

Based on the ratio of active domains to total possible (7):

| Active Domains | Confidence |
|----------------|------------|
| 5-7 (>=70%) | High |
| 3-4 (>=40%) | Medium |
| 1-2 (<40%) | Low |

### Watermark Override

If the AudioSeal watermark detector returns a score above 0.9, the overall score is floored at 95% regardless of other domain results. This is the only near-definitive positive proof.

---

## Installation

### Prerequisites

- **Python 3.11-3.14**
- **[uv](https://docs.astral.sh/uv/)** package manager (recommended)
- **Node.js >=18** and **npm** (for the frontend)

### Automated Setup

**macOS (Apple Silicon / Intel):**

```bash
cd ai-audio-detector
./scripts/setup-macos.sh
```

This installs all core dependencies, PyTorch with MPS acceleration (Apple Silicon), and parselmouth for vocal analysis.

**Windows (auto-detects CUDA):**

```powershell
cd ai-audio-detector
powershell -ExecutionPolicy Bypass -File scripts\setup-windows.ps1
```

Automatically detects NVIDIA GPUs and installs CUDA-enabled PyTorch if available, otherwise falls back to CPU.

### Manual Setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment
uv venv --python 3.12
source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1 on Windows

# Install core dependencies
uv pip install -e .

# Optional: vocal analysis (formant tracking via Praat)
uv pip install -e ".[vocal]"

# Optional: PyTorch for watermark detection
# macOS / CPU:
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
# Windows CUDA 12.1:
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Optional: AudioSeal watermark detection (requires torch)
uv pip install audioseal
```

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| librosa | >=0.11.0 | STFT, HPSS, onset detection, beat tracking, chroma features |
| scipy | >=1.14.0 | Signal filtering, spectrogram, curve fitting |
| numpy | >=1.26.4 | Array operations (numpy 2.x compatible) |
| pyloudnorm | >=0.2.0 | EBU R128 loudness measurement |
| soundfile | >=0.12.1 | Audio I/O |
| matplotlib | >=3.9.0 | Visualization (future spectrogram rendering) |
| FastAPI | latest | REST API framework |
| Pydantic | >=2.0 | Request/response validation |

### Optional Dependencies

| Package | Install | Purpose |
|---------|---------|---------|
| praat-parselmouth | `uv pip install praat-parselmouth` | Formant tracking for vocal analysis |
| audioseal | `uv pip install audioseal` | AudioSeal watermark detection |
| torch + torchaudio | See platform-specific instructions above | Required by audioseal |

All optional dependencies degrade gracefully -- if not installed, the corresponding domain returns `active: false` and its weight is redistributed.

---

## Configuration

Environment variables (prefix: `AI_DETECTOR_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_DETECTOR_HOST` | `0.0.0.0` | Server bind address |
| `AI_DETECTOR_PORT` | `8000` | Server port |
| `AI_DETECTOR_UPLOAD_DIR` | `backend/uploads` | Temporary upload directory |
| `AI_DETECTOR_MAX_FILE_SIZE_MB` | `500` | Maximum upload file size |
| `AI_DETECTOR_DEFAULT_ANALYSIS_DEPTH` | `standard` | Default depth when not specified |
| `AI_DETECTOR_ANALYSIS_TIMEOUT_SECONDS` | `120` | Analysis timeout |
| `AI_DETECTOR_ENABLE_VOCAL_ANALYSIS` | `true` | Enable/disable vocal domain |
| `AI_DETECTOR_ENABLE_WATERMARK_DETECTION` | `true` | Enable/disable watermark domain |

---

## Architecture

```
backend/app/
    main.py                          # FastAPI application entry point
    config.py                        # Settings via pydantic-settings
    models/
        schemas.py                   # Pydantic request/response models
    routers/
        analysis.py                  # POST /api/analyze, GET /api/health
    services/
        audio_analyzer.py            # Orchestrator: loads audio, runs analyzers, returns results
        analyzers/
            base.py                  # BaseAnalyzer ABC, ArtifactResult, DomainResult dataclasses
            weights.py               # Centralized artifact/domain weight config + 4-tier system
            scoring.py               # WeightedScoringEngine with evidence gating + concordance boost
            spectral.py              # SpectralAnalyzer  (4 checks, weight=0.40)
            spatial.py               # SpatialAnalyzer   (2 checks, weight=0.05)
            production.py            # ProductionAnalyzer(3 checks, weight=0.05)
            temporal.py              # TemporalAnalyzer  (2 checks, weight=0.10)
            structural.py            # StructuralAnalyzer(2 checks, weight=0.20)
            vocal.py                 # VocalAnalyzer     (3 checks, weight=0.20)
            watermark.py             # WatermarkAnalyzer (1 check,  currently disabled)

frontend/
    index.html                       # HTML entry point
    package.json                     # NPM dependencies (React, Vite, Tailwind, Axios)
    vite.config.ts                   # Vite dev server config with /api proxy to :8000
    tailwind.config.js               # Tailwind CSS content paths
    tsconfig.json                    # TypeScript configuration
    src/
        main.tsx                     # React 18 entry point (createRoot)
        App.tsx                      # Main app: idle → analyzing → results state machine
        index.css                    # Tailwind directives + dark theme + animations
        services/
            api.ts                   # Axios client + TypeScript interfaces for API schema
        components/
            FileUploader.tsx         # Drag-and-drop upload with format validation
            DepthSelector.tsx        # Quick / Standard / Deep toggle
            ScoreGauge.tsx           # SVG circular gauge (0-100%) with color gradient
            DomainCard.tsx           # Collapsible domain card with artifact breakdowns
            AudioInfo.tsx            # Audio metadata grid (duration, SR, channels, dB)
            AnalysisResults.tsx      # Full results layout with gauge + domain cards
```

### Data Flow

```
Browser (React SPA on :5174)
    |
    +--> FileUploader: drag-and-drop --> File object
    +--> DepthSelector: quick | standard | deep
    +--> Axios POST /api/analyze?depth=X (multipart/form-data)
    |
    v
FastAPI Backend (:8000)
    |
    v
AudioAnalyzer.analyze_file(path, depth)
    |
    +--> librosa.load() --> y_mono, y_stereo, sr
    |
    +--> For each analyzer (gated by depth):
    |       analyzer.analyze(y_mono, sr, y_stereo, depth)
    |           --> runs _check_*() methods
    |           --> returns DomainResult { score, artifacts[] }
    |
    +--> WeightedScoringEngine.calculate(domain_results[])
    |       --> evidence-gates low-scoring domains
    |       --> computes weighted overall score (0-100%)
    |       --> applies concordance boost if ≥2 domains agree
    |       --> determines confidence level
    |       --> applies watermark override if applicable
    |       --> returns ScoringResult
    |
    v
AnalysisResponse (JSON)
    |
    v
Browser renders: ScoreGauge + DomainCards + AudioInfo
```

### Adding a New Analyzer

1. Create `backend/app/services/analyzers/my_domain.py`
2. Subclass `BaseAnalyzer`, set `domain`, `display_name`, `base_weight`, `min_depth`
3. Implement `analyze()` method with `_check_*()` methods returning `ArtifactResult`
4. Add artifact weights and tiers to `weights.py`
5. Add domain weight to `DOMAIN_WEIGHTS` in `weights.py`
6. Register in `analyzers/__init__.py`
7. Add to `AudioAnalyzer._analyzers` list in `audio_analyzer.py`
8. The scoring engine automatically picks it up

---

## Platform Support

| Platform | Hardware Acceleration | Status |
|----------|-----------------------|--------|
| macOS (Apple Silicon) | MPS (Metal Performance Shaders) | Tested |
| macOS (Intel) | CPU | Supported |
| Windows | CUDA 12.1 (NVIDIA GPUs) | Supported |
| Windows | CPU | Supported |
| Linux | CUDA / CPU | Supported (manual setup) |

All core analysis (spectral, spatial, production, temporal, structural) runs on CPU and requires no GPU. GPU acceleration is only used by the optional AudioSeal watermark detector via PyTorch.

---

## Limitations

**This is a probabilistic tool, not a definitive classifier.** Important caveats:

- **No single artifact is proof.** Detection relies on the convergence of multiple independent indicators. A high score in one domain alone is not conclusive.
- **Professional electronic music may score higher** than expected because synthesized sounds share some characteristics with AI output (e.g., perfect harmonics, quantized timing).
- **AI models are improving rapidly.** As generation architectures adopt phase-aware loss functions and improved upsampling (e.g., removing checkerboard artifacts), low-level spectral signatures will become less reliable. Future detection will need to shift toward semantic and structural analysis.
- **Watermark detection is model-specific.** AudioSeal only detects Meta's watermark. Audio from Suno, Udio, or other platforms that don't embed AudioSeal watermarks will not trigger this check.
- **Mono audio disables spatial analysis.** The spatial domain (phase correlation, bass width) requires stereo input.
- **Short audio reduces confidence.** Structural analysis, loudness range, and beat tracking all require sufficient duration (>10 seconds minimum, >30 seconds recommended).
- **Stems and isolated instruments produce unreliable results.** The system assumes full-mix audio. Isolated stems (vocals, drums, bass, pads, etc.) will trigger significant false positives because many checks measure properties that only make sense in a complete mix -- e.g., structural repetition via chroma on a drum stem with no pitch content, loudness range on a sustained pad, HF harmonic-to-noise ratio on percussion. Analyze full mixes for reliable results.
- **The scoring thresholds have been empirically calibrated** against a 22-track dataset (10 human, 12 AI) but would benefit from validation against a larger labeled corpus. The current calibration reduced human false positives from ~24% to ~7% mean score while improving AI/human separation to ~23 percentage points.

---

## Research Basis

This system is grounded in the following technical observations from Music Information Retrieval (MIR) research and signal processing theory:

- **Checkerboard artifacts** from transposed convolutions are mathematically inherent to the architecture (Deezer, 2025) and independent of training data
- **Neural codec compression** (EnCodec, SoundStream) introduces codebook quantization noise and high-frequency phase smearing
- **Latent diffusion models** operating on Mel-spectrograms discard phase information, causing transient smearing measurable via Log-Attack Time
- **EBU R128 loudness range** provides a standardized, broadcast-grade metric for dynamic range that reliably differentiates mastered content from AI output
- **Self-similarity matrices** from chroma features reveal whether a track has genuine structural repetition (verse-chorus form) or "hallucinated" sections
- **Formant stability** across pitch is a physics-based constraint of the human vocal tract that AI voice synthesis frequently violates
- **AudioSeal** (Meta, 2024) provides the only current mechanism for definitive watermark-based provenance verification

---

## License

MIT
