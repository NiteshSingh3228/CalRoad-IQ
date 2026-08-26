# 🛣 CalRoad IQ

**AI-Powered Road Defect Detection & Road Quality Analysis System**

[![CI](https://github.com/<your-org>/calroad-iq/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-org>/calroad-iq/actions)

---

## 1. Project Purpose

CalRoad IQ is a Streamlit-based prototype that analyses road photos and:

- **Detects defects** (potholes, water logging, crack zones) using GPT-4o-mini via OpenRouter
- **Predicts GPS location** from visual image content (street signs, vegetation, lane markings) using GPT-4o-mini and GeoCLIP
- **Computes Road Quality Index (RQI)** — a project-defined heuristic score (0–100)
- **Recommends speeds** and generates analytics charts

---

## 2. Features

| Feature | Description |
|---|---|
| Image upload | JPG / PNG / WEBP up to 20 MB |
| EXIF GPS | Extracts exact coordinates from image metadata |
| AI Vision GPS | GPT-4o-mini reads visual clues (signs, vegetation, lanes) |
| GeoCLIP GPS | NeurIPS 2023 model predicts location from scene content |
| Manual GPS | User can enter/adjust coordinates |
| Defect detection | AI counts potholes, water logging, crack zones with bounding boxes |
| RQI score | Heuristic Road Quality Index 0–100 |
| Speed recommendation | RQI-adjusted avg and safe speeds |
| Folium map | Interactive map with GPS source indicators |
| Plotly analytics | 6 charts: RQI gauge, speed, traffic pattern, defects, severity, travel time |

---

## 3. Architecture

```
User uploads image
        ↓
Streamlit (Project.py)
        ↓
Image validation (size, type, Pillow decode)
        ↓
Image resize (max 1024px)
        ↓
Image hash (SHA-256 change detection)
        ↓
EXIF GPS extraction
        ↓ (if no EXIF)
AI Vision GPS (GPT-4o-mini via OpenRouter)
        ↓ (if AI GPS fails)
GeoCLIP location prediction (torch)
        ↓ (if all fail)
Manual / default coordinates
        ↓
AI defect detection (GPT-4o-mini, JSON response)
        ↓
Bounding-box annotation (OpenCV)
        ↓
RQI computation → Speed recommendation
        ↓
Folium map + Plotly charts
        ↓
Results rendered in Streamlit
```

---

## 4. Installation

**Minimum requirements** (no GPU):

```bash
pip install streamlit plotly folium streamlit-folium \
            pillow numpy opencv-python requests
```

**Full install** (with GeoCLIP):

```bash
pip install streamlit plotly folium streamlit-folium \
            pillow numpy opencv-python requests \
            torch torchvision geoclip
```

> ⚠️ GeoCLIP downloads ~1 GB model weights on first run. Subsequent runs use cached weights.
> The app runs without GeoCLIP — it falls back to AI Vision GPS.

---

## 5. Environment Setup

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env`:

```
OPENROUTER_API_KEY=your_openrouter_key_here
```

**Never commit `.env` to version control.**

---

## 6. API Key Setup

1. Go to [https://openrouter.ai/keys](https://openrouter.ai/keys)
2. Create a free account
3. Create an API key
4. Either:
   - Set `OPENROUTER_API_KEY` in your `.env` file (server-side default), or
   - Paste it into the UI text input (stored per-session only)

---

## 7. Running the App

```bash
streamlit run Project.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 8. Testing

```bash
# Run all unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=Project --cov-report=term-missing
```

Tests are deterministic and do **not** call OpenRouter or download GeoCLIP.

---

## 9. CI/CD

GitHub Actions workflow at `.github/workflows/ci.yml`:

- Triggers on push / PR to `main`
- Installs dependencies (excluding torch/geoclip for speed)
- Syntax check
- Lint (flake8)
- Unit tests (pytest)
- Security scan (bandit)
- Leaked-secret check

---

## 10. Known Limitations

| Limitation | Notes |
|---|---|
| GPT-4o-mini GPS accuracy | AI location estimation can be wrong — verify before use |
| GeoCLIP accuracy | Statistical prediction, not ground truth |
| RQI formula | Project-defined heuristic — not calibrated to engineering standards |
| Traffic chart | Illustrative traffic pattern — representative data for demonstration, not real FHWA measurements |
| `.pkl` files | `model.pkl`, `scaler.pkl`, `label_encoder.pkl`, `accident_severity_model.pkl` are present in the repository but **not connected** to the CalRoad IQ pipeline |
| `dataset/` | Excel/CSV files present but **not used** by the app |
| No authentication | No user login — anyone with network access can use the app |
| Streamlit reruns | Every UI interaction reruns the script — rate limiting is per-session |

---

## 11. AI / GeoCLIP Requirements

- **OpenRouter API key** — required for defect detection and AI GPS
- **GeoCLIP** — optional (`pip install torch torchvision geoclip`)
  - Downloads ~1 GB weights on first run
  - App works without it — falls back to AI Vision GPS

---

## 12. Dataset Usage

Files in `dataset/` (`road_traffic.xlsx`, `2021-ca-peak-hours.xlsx`, etc.) are
**not currently connected** to the application pipeline.

The traffic chart in the UI uses illustrative representative data.

---

## 13. RQI Methodology

The Road Quality Index is a **project-defined heuristic**:

```
RQI = 100 − (4 × potholes) − (5 × water_logging) − (3 × crack_zones)
```

- **RQI ≥ 70** — Good condition
- **RQI 40–69** — Moderate condition
- **RQI < 40** — Poor condition

Speed recommendations:

```
Avg Speed = 65 × (RQI / 100)    [minimum RQI clamped to 5]
Safe Speed = Avg Speed × 0.90
```

This is not an officially calibrated road quality standard.
