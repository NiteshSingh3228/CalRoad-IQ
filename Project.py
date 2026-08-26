"""
╔══════════════════════════════════════════════════════════════════╗
║  CalRoad IQ  —  Road Defect Analyser + AI Geo-Localisation       ║
║  OpenRouter AI (FREE) + GeoCLIP (NeurIPS 2023)                   ║
╠══════════════════════════════════════════════════════════════════╣
║  INSTALL (run once):                                             ║
║  pip install streamlit plotly folium streamlit-folium            ║
║              pillow numpy opencv-python requests                 ║
║              torch torchvision geoclip                           ║
║                                                                  ║
║  NOTE: GeoCLIP downloads ~1 GB model weights on first run.      ║
║        Subsequent runs use cached weights.                       ║
║                                                                  ║
║  RUN:                                                            ║
║  streamlit run Project.py                                        ║
║                                                                  ║
║  API KEY (free):                                                 ║
║  https://openrouter.ai/keys  →  Create Key  →  paste in app     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import cv2
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from PIL import Image, ImageDraw, ExifTags
import random, datetime, base64, json, io, os, hashlib, tempfile, html, contextlib, logging, time
import requests

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
_log = logging.getLogger("calroad")

st.set_page_config(page_title="CalRoad IQ", page_icon="🛣️", layout="wide")

# ══════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Space+Mono&family=DM+Sans:wght@400;600&display=swap');
html,body,[class*="css"]{background:#08090f!important;color:#b0c4de;font-family:'DM Sans',sans-serif;}
.block-container{padding:1rem 1.4rem!important;max-width:100%!important;}
.hdr{background:#0c1022;border:1px solid #1a2a4a;border-radius:10px;padding:14px 24px;margin-bottom:14px;display:flex;align-items:center;justify-content:space-between;}
.hdr-title{font-family:'Bebas Neue',sans-serif;font-size:34px;letter-spacing:6px;color:#eef4ff;}
.hdr-sub{font-family:'Space Mono',monospace;font-size:9px;color:#f5a020;letter-spacing:2px;margin-top:3px;}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:#20d47a;box-shadow:0 0 7px #20d47a;animation:blink 2s infinite;margin-right:5px;}
.dot-ai{display:inline-block;width:7px;height:7px;border-radius:50%;background:#7c3aed;box-shadow:0 0 7px #7c3aed;animation:blink 1.4s infinite;margin-right:5px;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.15}}
.sec{font-family:'Space Mono',monospace;font-size:9px;letter-spacing:3px;color:#f5a020;text-transform:uppercase;margin:12px 0 7px;display:flex;align-items:center;gap:8px;}
.sec::after{content:'';flex:1;height:1px;background:#1a2a4a;}
.card{background:#0c1022;border:1px solid #1a2a4a;border-radius:8px;padding:14px 16px;position:relative;overflow:hidden;}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:8px 8px 0 0;}
.cg::before{background:#20d47a;}.ca::before{background:#f5a020;}.cr::before{background:#f03a3a;}.cy::before{background:#f0d020;}
.cnt-card{background:#0c1022;border:1px solid #1a2a4a;border-radius:12px;padding:20px 16px;position:relative;overflow:hidden;text-align:center;}
.cnt-card::before{content:'';position:absolute;top:0;left:0;right:0;height:4px;border-radius:12px 12px 0 0;}
.cnt-red::before{background:linear-gradient(90deg,#f03a3a,#ff7070);}
.cnt-blue::before{background:linear-gradient(90deg,#3a8ef0,#70b8ff);}
.cnt-yellow::before{background:linear-gradient(90deg,#f0d020,#ffe870);}
.cnt-green::before{background:linear-gradient(90deg,#20d47a,#70ffb8);}
.cnt-num{font-family:'Bebas Neue',sans-serif;font-size:80px;line-height:1;margin:4px 0 2px;}
.cnt-lbl{font-family:'Space Mono',monospace;font-size:10px;letter-spacing:3px;text-transform:uppercase;margin-bottom:4px;}
.cnt-sub{font-family:'Space Mono',monospace;font-size:9px;color:#2a4060;margin-top:6px;}
.lbl{font-family:'Space Mono',monospace;font-size:8px;letter-spacing:2px;color:#2a4060;text-transform:uppercase;}
.val{font-family:'Bebas Neue',sans-serif;font-size:32px;line-height:1.1;}
.unt{font-family:'Space Mono',monospace;font-size:9px;color:#2a4060;margin-top:1px;}
.sub{font-family:'Space Mono',monospace;font-size:9px;margin-top:3px;}
.good{color:#20d47a;}.warn{color:#f0d020;}.bad{color:#f03a3a;}.dim{color:#2a4060;}
.dbrow{display:flex;gap:7px;flex-wrap:wrap;margin:7px 0;}
.db{font-family:'Space Mono',monospace;font-size:10px;padding:4px 11px;border-radius:20px;border:1px solid;}
.dp{background:rgba(240,58,58,.1);border-color:#f03a3a;color:#f03a3a;}
.dw{background:rgba(58,142,240,.1);border-color:#3a8ef0;color:#3a8ef0;}
.dc{background:rgba(240,208,32,.1);border-color:#f0d020;color:#f0d020;}
.dok{background:rgba(32,212,122,.1);border-color:#20d47a;color:#20d47a;}
.dai{background:rgba(124,58,237,.1);border-color:#7c3aed;color:#7c3aed;}
.dtot{background:rgba(255,255,255,.04);border-color:#2a4060;color:#607080;margin-left:auto;}
.fml{background:#0c1022;border:1px solid #1a2a4a;border-left:3px solid #f5a020;border-radius:0 8px 8px 0;padding:9px 13px;font-family:'Space Mono',monospace;font-size:11px;color:#eef4ff;line-height:2.2;margin:7px 0;}
.gbox{background:#0c1022;border:1px solid #1a2a4a;border-radius:8px;padding:11px 13px;}
.gr{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #10192e;}
.gr:last-child{border-bottom:none;}
.gk{font-family:'Space Mono',monospace;font-size:9px;color:#2a4060;}
.gv{font-family:'Space Mono',monospace;font-size:9px;color:#eef4ff;}
.loc-exif{background:rgba(32,212,122,.07);border:1px solid #20d47a;border-radius:8px;padding:10px 14px;font-family:'Space Mono',monospace;font-size:9px;color:#20d47a;margin-bottom:10px;}
.loc-geoclip{background:rgba(124,58,237,.07);border:1px solid #7c3aed;border-radius:8px;padding:10px 14px;font-family:'Space Mono',monospace;font-size:9px;color:#b090f0;margin-bottom:10px;}
.loc-aivision{background:rgba(240,148,32,.07);border:1px solid #f09420;border-radius:8px;padding:10px 14px;font-family:'Space Mono',monospace;font-size:9px;color:#f09420;margin-bottom:10px;}
.loc-manual{background:rgba(245,160,32,.07);border:1px solid #f5a020;border-radius:8px;padding:10px 14px;font-family:'Space Mono',monospace;font-size:9px;color:#f5a020;margin-bottom:10px;}
hr{border-color:#1a2a4a!important;margin:10px 0!important;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════
LA_ROADS = {
    "I-405 San Diego Fwy":   (34.05553,-118.44317),
    "I-110 Harbor Fwy":      (34.00651,-118.26527),
    "US-101 Hollywood Fwy":  (34.09848,-118.32891),
    "I-10 Santa Monica Fwy": (34.02321,-118.39812),
    "I-5 Golden State Fwy":  (34.05240,-118.22130),
    "CA-60 Pomona Fwy":      (34.02200,-117.95040),
    "CA-91 Artesia Fwy":     (33.88240,-117.90030),
    "I-105 Century Fwy":     (33.92030,-118.31540),
    "I-210 Foothill Fwy":    (34.19030,-117.82010),
    "I-605 San Gabriel Fwy": (34.02040,-118.03030),
}
LA_CENTER  = (34.0522, -118.2437)
BASE_SPEED = 65

LA_HOURLY  = [1100,750,480,380,580,1700,4000,6600,5800,4400,4100,4700,
              5000,5200,5600,6400,7000,7700,6100,4400,3700,3100,2700,1800]
NAT_HOURLY = [780,540,360,280,410,1100,2600,3900,3600,2900,2800,3100,
              3400,3500,3700,3900,4300,4700,3900,3000,2500,2100,1700,1100]

PT = dict(
    paper_bgcolor="#0c1022", plot_bgcolor="#08090f",
    font=dict(family="Space Mono",size=10,color="#607080"),
    xaxis=dict(gridcolor="#1a2a4a",linecolor="#1a2a4a",zerolinecolor="#1a2a4a"),
    yaxis=dict(gridcolor="#1a2a4a",linecolor="#1a2a4a",zerolinecolor="#1a2a4a"),
)

OPENROUTER_MODEL   = "openai/gpt-4o-mini"
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_REFERER = "https://calroad-iq.app"

# ══════════════════════════════════════════════════════════════════════
#  GeoCLIP — load once, cache forever
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_geoclip():
    """Load GeoCLIP model. Downloads ~1GB weights on first run."""
    try:
        from geoclip import GeoCLIP
        model = GeoCLIP()
        model.eval()
        return model, None
    except ImportError:
        return None, "geoclip not installed"
    except Exception as e:
        return None, str(e)

def predict_location_geoclip(pil_img, top_k=5):
    """
    Use GeoCLIP to predict GPS from image content.
    Returns (predictions_list, error_str)
    predictions_list = [{"lat":..,"lng":..,"prob":..}, ...]
    """
    model, err = load_geoclip()
    if model is None:
        return None, err
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            pil_img.save(f, format="JPEG", quality=92)
            tmp_path = f.name
        try:
            import torch
            with torch.no_grad():
                gps_preds, probs = model.predict(tmp_path, top_k=top_k)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)  # always delete, even on exception
        results = []
        for i in range(len(gps_preds)):
            lat, lng = float(gps_preds[i][0]), float(gps_preds[i][1])
            prob = float(probs[i])
            results.append({"lat": lat, "lng": lng, "prob": prob})
        # Normalize probabilities so they sum to 1.0
        total_prob = sum(r["prob"] for r in results) or 1.0
        for r in results:
            r["prob"] = round(r["prob"] / total_prob, 5)
        return results, None
    except Exception as e:
        _log.warning("GeoCLIP inference error: %s", e)
        return None, str(e)

# ══════════════════════════════════════════════════════════════════════
#  CORE HELPERS
# ══════════════════════════════════════════════════════════════════════
def compute_rqi(p, w, c):
    return max(0, min(100, 100 - 4*p - 5*w - 3*c))  # clamps [0, 100]

def compute_speeds(rqi):
    # Minimum effective RQI of 5 prevents 0 mph display
    n = max(rqi, 5) / 100.0
    return round(BASE_SPEED*n, 1), round(BASE_SPEED*n*0.9, 1)

def fmt_coord(lat, lng):
    return (f"{abs(lat):.5f}° {'N' if lat>=0 else 'S'}",
            f"{abs(lng):.5f}° {'E' if lng>=0 else 'W'}")

def img_hash(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=60)
    return hashlib.sha256(buf.getvalue()).hexdigest()

# ══════════════════════════════════════════════════════════════════════
#  FIX: EXIF GPS — handles all Pillow versions + tuple/IFDRational types
# ══════════════════════════════════════════════════════════════════════
def get_exif_gps(pil_img):
    """
    Extract GPS coords from image EXIF.
    Supports:
      - Modern Pillow (≥6.0): getexif() + get_ifd(34853)
      - Legacy Pillow: _getexif() with nested GPSInfo dict
      - GPS values as IFDRational objects, (num,den) tuples, or plain floats/ints
    Returns (lat, lng) as floats, or (None, None) if not found.
    """
    GPS_IFD_TAG = 34853  # standard EXIF tag for GPSInfo sub-IFD

    try:
        # ── Step 1: extract raw GPS dict ──────────────────────────────
        gps_raw = {}

        # Modern Pillow ≥ 6.0 path
        try:
            exif_obj = pil_img.getexif()
            if exif_obj:
                gps_raw = exif_obj.get_ifd(GPS_IFD_TAG)
        except (AttributeError, Exception):
            pass

        # Legacy fallback — _getexif() returns {tag_int: value}
        if not gps_raw:
            try:
                raw = pil_img._getexif()
                if raw:
                    # GPSInfo may be keyed by integer tag 34853 directly
                    gps_raw = raw.get(GPS_IFD_TAG, {})
                    # Some builds nest it under string key "GPSInfo"
                    if not gps_raw:
                        tags_str = {ExifTags.TAGS.get(k, k): v for k, v in raw.items()}
                        gps_raw  = tags_str.get("GPSInfo", {})
            except (AttributeError, Exception):
                pass

        if not gps_raw:
            return None, None

        # ── Step 2: map numeric GPS sub-tags → string names ───────────
        gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_raw.items()}

        if "GPSLatitude" not in gps or "GPSLongitude" not in gps:
            return None, None

        # ── Step 3: convert IFDRational / (num,den) tuple / number → float
        def _to_float(v):
            """Safely convert any GPS numeric form to Python float."""
            # (numerator, denominator) tuple  — old Pillow
            if isinstance(v, tuple) and len(v) == 2:
                num, den = v
                return float(num) / float(den) if den else 0.0
            # IFDRational or any other object with __float__
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        def dms_to_dd(dms):
            """Degrees-minutes-seconds → decimal degrees."""
            if len(dms) < 3:
                return 0.0
            return _to_float(dms[0]) + _to_float(dms[1]) / 60.0 + _to_float(dms[2]) / 3600.0

        lat = dms_to_dd(gps["GPSLatitude"])
        lng = dms_to_dd(gps["GPSLongitude"])

        # Apply hemisphere reference
        if str(gps.get("GPSLatitudeRef",  "N")).strip().upper() == "S":
            lat = -lat
        if str(gps.get("GPSLongitudeRef", "E")).strip().upper() == "W":
            lng = -lng

        # ── Step 4: sanity-check valid WGS-84 range ───────────────────
        if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lng <= 180.0):
            return None, None

        # ── Step 5: reject (0, 0) — almost never a real road photo ────
        if lat == 0.0 and lng == 0.0:
            return None, None

        return round(lat, 6), round(lng, 6)

    except Exception:
        return None, None

# ══════════════════════════════════════════════════════════════════════
#  AI VISION GPS — GPT-4o-mini estimates location from visual content
#  (runs when EXIF absent; no extra installs needed)
# ══════════════════════════════════════════════════════════════════════
GPS_SYSTEM_PROMPT = """You are an expert geo-locator AI.
Analyse the image for ANY location clues: road signs, lane markings, traffic lights,
vehicle plates, architecture, road paint colour/style, vegetation, sky, shadows,
building density, freeway structure, signage language, billboard text, utility poles,
bridge design, guardrail style, road surface type.
Return ONLY valid JSON — no markdown, no text outside JSON."""

GPS_USER_PROMPT = """Estimate the most likely GPS coordinates for where this photo was taken.
Use ALL available visual evidence in the image.

Evidence to look for:
- Road signs (text, colour, shape, font style — US green freeway signs, CA yellow warning signs)
- Lane markings (white/yellow, dashed/solid pattern — US style)
- Traffic signals and pole design
- Vehicle types and license plate shapes
- Architecture and building style
- Vegetation and trees (palm trees → Southern California / Florida)
- Mountains or hills in background
- Sky colour and sun angle
- Freeway / highway structure design
- Any visible text, logos, store names

For Los Angeles specifically watch for:
- Green rectangular freeway signs with white text (I-405, I-110, US-101, I-10 etc.)
- White lane markings on grey asphalt
- Palm trees along roadside
- Concrete sound walls on freeways
- HOV diamond lane markings
- Blue CA freeway marker shields

Return ONLY this JSON (no other text):
{
  "lat": 34.05,
  "lng": -118.24,
  "confidence": 0.72,
  "city": "Los Angeles",
  "state": "California",
  "country": "USA",
  "reasoning": "Green US freeway signs, palm trees, I-405 shield visible",
  "alt_predictions": [
    {"lat": 34.09, "lng": -118.33, "confidence": 0.15, "label": "Hollywood Fwy area"},
    {"lat": 34.02, "lng": -118.40, "confidence": 0.08, "label": "Santa Monica Fwy area"},
    {"lat": 33.92, "lng": -118.31, "confidence": 0.05, "label": "Century Fwy area"}
  ]
}

Rules:
- lat/lng must be valid WGS-84 decimal degrees
- confidence 0.0–1.0 (how certain you are of the exact spot)
- If image clearly shows LA freeway, pick the specific freeway corridor lat/lng
- alt_predictions: 3 plausible alternative spots
- If truly impossible to determine, use best continent/country estimate
- NEVER return 0,0 or null"""

def predict_location_ai(pil_img, api_key):
    """
    Use GPT-4o-mini to predict GPS from image visual content.
    Returns (result_dict, error_str)
    result_dict = {"lat":..,"lng":..,"confidence":..,"city":..,"reasoning":..,"alt_predictions":[...]}
    """
    try:
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": GPS_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{pil_to_b64(pil_img)}",
                        "detail": "high"
                    }},
                    {"type": "text", "text": GPS_USER_PROMPT},
                ]},
            ],
            "max_tokens": 800,
            "temperature": 0.1,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": OPENROUTER_REFERER,
            "X-Title": "CalRoad IQ",
        }
        resp = requests.post(OPENROUTER_URL, headers=headers,
                             data=json.dumps(payload), timeout=60)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()

        # Strip markdown fences if present
        if "```" in raw:
            for part in raw.split("```"):
                part = part.strip()
                if part.startswith("json"): part = part[4:].strip()
                if part.startswith("{"): raw = part; break
        s, e = raw.find("{"), raw.rfind("}")
        if s != -1 and e != -1:
            raw = raw[s:e+1]

        r = json.loads(raw)

        # Validate
        lat = float(r.get("lat", 0))
        lng = float(r.get("lng", 0))
        if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
            return None, "AI returned out-of-range coordinates"
        if lat == 0.0 and lng == 0.0:
            return None, "AI returned (0,0) — unusable"

        r["lat"] = round(lat, 5)
        r["lng"] = round(lng, 5)
        r.setdefault("confidence", 0.5)
        r.setdefault("city", "")
        r.setdefault("state", "")
        r.setdefault("country", "")
        r.setdefault("reasoning", "")
        r.setdefault("alt_predictions", [])
        return r, None

    except Exception as ex:
        return None, str(ex)


SYSTEM_PROMPT = """You are a road infrastructure inspector AI.
COUNT and LOCATE every road defect. Return ONLY valid JSON — no markdown, no text outside JSON."""

USER_PROMPT = """Analyse this road image. COUNT and LOCATE every visible defect.

POTHOLES — each separate hole:
  Bowl-shaped depression, dark shadowed interior, broken edges (dry/wet/muddy).
  NOT: manhole covers, patches, tyre skids, painted marks.

WATER LOGGING — each separate pool:
  Standing water ON road surface, shiny/reflective.
  NOT: wet-looking asphalt, shadows, paint gloss.

CRACKS — each contiguous crack zone:
  Alligator, longitudinal, transverse, or edge cracks.
  NOT: tar strips, lane markings.

BBOX (image fractions 0.0-1.0): x,y=top-left; w,h=size. Tight fit per defect.
SEVERITY: "mild"|"moderate"|"severe"  CONFIDENCE: "high"(>85%)|"medium"(70-85%)
Only report ≥70% confidence detections.

Return ONLY this JSON:
{
  "pothole_count":2,"water_count":1,"crack_count":1,
  "potholes":[{"x":0.10,"y":0.50,"w":0.12,"h":0.09,"severity":"severe","confidence":"high"}],
  "water":[{"x":0.30,"y":0.55,"w":0.20,"h":0.14,"severity":"moderate","confidence":"high"}],
  "cracks":[{"x":0.05,"y":0.30,"w":0.40,"h":0.06,"severity":"mild","confidence":"medium"}],
  "overall_condition":"Poor",
  "repair_urgency":"Immediate",
  "surface_type":"Asphalt",
  "visibility":"Clear"
}
Rules: counts=array lengths. overall_condition: Good(0)/Moderate(1-4)/Poor(5+).
repair_urgency: Immediate|Soon|Routine|None. Use [] for zero detections."""

def pil_to_b64(img):
    buf=io.BytesIO(); img.save(buf,format="JPEG",quality=92)
    return base64.standard_b64encode(buf.getvalue()).decode()

def run_detection(pil_img, api_key):
    try:
        payload={
            "model":OPENROUTER_MODEL,
            "messages":[
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":[
                    {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{pil_to_b64(pil_img)}","detail":"high"}},
                    {"type":"text","text":USER_PROMPT},
                ]},
            ],
            "max_tokens":2000,"temperature":0.05,
        }
        headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json",
                 "HTTP-Referer":OPENROUTER_REFERER,"X-Title":"CalRoad IQ"}
        resp=requests.post(OPENROUTER_URL,headers=headers,data=json.dumps(payload),timeout=90)
        resp.raise_for_status()
        raw=resp.json()["choices"][0]["message"]["content"].strip()
        if "```" in raw:
            for part in raw.split("```"):
                part=part.strip()
                if part.startswith("json"): part=part[4:].strip()
                if part.startswith("{"): raw=part; break
        s,e=raw.find("{"),raw.rfind("}")
        if s!=-1 and e!=-1: raw=raw[s:e+1]
        r=json.loads(raw)
        for k,v in [("potholes",[]),("water",[]),("cracks",[]),
                    ("pothole_count",0),("water_count",0),("crack_count",0),
                    ("overall_condition","Unknown"),("repair_urgency","Unknown"),
                    ("surface_type","Unknown"),("visibility","Unknown")]:
            r.setdefault(k,v)
        # ── AI output validation ─────────────────────────────────────
        # Ensure list fields are actually lists
        for k in ("potholes", "water", "cracks"):
            if not isinstance(r[k], list):
                _log.warning("AI returned non-list for '%s': %s", k, type(r[k]).__name__)
                r[k] = []
        # Validate bounding-box dicts (fraction values 0.0–1.0)
        def _valid_bbox(d):
            try:
                x,y,w,h = float(d["x"]),float(d["y"]),float(d["w"]),float(d["h"])
                return 0.0<=x<=1.0 and 0.0<=y<=1.0 and w>0 and h>0
            except (KeyError, TypeError, ValueError):
                return False
        for k in ("potholes", "water", "cracks"):
            before = len(r[k])
            r[k] = [d for d in r[k] if isinstance(d, dict) and _valid_bbox(d)]
            dropped = before - len(r[k])
            if dropped:
                _log.warning("Dropped %d invalid bbox(es) from '%s'", dropped, k)
        # Sanitise repair_urgency to known safe values
        VALID_URGENCY = {"Immediate", "Soon", "Routine", "None", "Unknown"}
        if r["repair_urgency"] not in VALID_URGENCY:
            r["repair_urgency"] = "Unknown"
        r["pothole_count"]=len(r["potholes"])
        r["water_count"]  =len(r["water"])
        r["crack_count"]  =len(r["cracks"])
        return r
    except requests.exceptions.HTTPError as e:
        try:
            body = e.response.json()
        except Exception:
            body = ""
        _log.warning("OpenRouter HTTP error %s: %s", e.response.status_code, body)
        st.error(f"⚠️ OpenRouter {e.response.status_code}: {body}")
        return _empty()
    except Exception as e:
        _log.warning("Detection error: %s", e)
        st.error(f"⚠️ Error: {e}")
        return _empty()

def _empty():
    return {"potholes":[],"water":[],"cracks":[],
            "pothole_count":0,"water_count":0,"crack_count":0,
            "overall_condition":"Unknown","repair_urgency":"Unknown",
            "surface_type":"Unknown","visibility":"Unknown"}

# ══════════════════════════════════════════════════════════════════════
#  ANNOTATE IMAGE
# ══════════════════════════════════════════════════════════════════════
COLS={"pothole":((240,58,58),(160,18,18),"POTHOLE"),
      "water":  ((50,150,245),(15,55,140),"WATER"),
      "crack":  ((240,200,30),(130,110,0),"CRACK")}

def annotate(pil_img, ai):
    img=np.array(pil_img.convert("RGB")); H,W=img.shape[:2]; ctr={"pothole":0,"water":0,"crack":0}
    def draw(dets,kind):
        bc,fc,lbl=COLS[kind]
        for d in dets:
            try:
                ctr[kind]+=1; idx=ctr[kind]
                x=max(0,int(d["x"]*W)); y=max(0,int(d["y"]*H))
                w=min(int(d["w"]*W),W-x); h=min(int(d["h"]*H),H-y)
                if w<8 or h<8: continue
                sev=d.get("severity",""); tag=f"#{idx} {lbl}"+(f" [{sev}]" if sev else "")
                thick=3 if sev=="severe" else 2
                cv2.rectangle(img,(x,y),(x+w,y+h),bc,thick)
                cl=min(16,w//3,h//3)
                for cx,cy in [(x,y),(x+w,y),(x,y+h),(x+w,y+h)]:
                    dx=1 if cx==x else -1; dy=1 if cy==y else -1
                    cv2.line(img,(cx,cy),(cx+dx*cl,cy),(255,255,255),2)
                    cv2.line(img,(cx,cy),(cx,cy+dy*cl),(255,255,255),2)
                tw=len(tag)*7+10
                cv2.rectangle(img,(x,max(0,y-26)),(x+tw,y),fc,-1)
                cv2.putText(img,tag,(x+4,max(5,y-7)),cv2.FONT_HERSHEY_SIMPLEX,0.36,(255,255,255),1,cv2.LINE_AA)
            except Exception:
                continue
    draw(ai.get("potholes",[]),"pothole"); draw(ai.get("water",[]),"water"); draw(ai.get("cracks",[]),"crack")
    ov=img.copy()
    for d in ai.get("potholes",[]):
        x,y=max(0,int(d["x"]*W)),max(0,int(d["y"]*H)); w,h=int(d["w"]*W),int(d["h"]*H)
        ov[y:y+h,x:x+w]=np.clip(ov[y:y+h,x:x+w].astype(int)+[45,-12,-12],0,255)
    for d in ai.get("water",[]):
        x,y=max(0,int(d["x"]*W)),max(0,int(d["y"]*H)); w,h=int(d["w"]*W),int(d["h"]*H)
        ov[y:y+h,x:x+w]=np.clip(ov[y:y+h,x:x+w].astype(int)+[-12,-12,45],0,255)
    for d in ai.get("cracks",[]):
        x,y=max(0,int(d["x"]*W)),max(0,int(d["y"]*H)); w,h=int(d["w"]*W),int(d["h"]*H)
        ov[y:y+h,x:x+w]=np.clip(ov[y:y+h,x:x+w].astype(int)+[10,10,-20],0,255)
    return Image.fromarray(cv2.addWeighted(ov,0.45,img,0.55,0))

def make_demo():
    img=Image.new("RGB",(640,360)); d=ImageDraw.Draw(img)
    for y in range(360):
        v=int(62-(y/360)*22); d.line([(0,y),(640,y)],fill=(v,int(v*.95),int(v*.85)))
    rng=random.Random(77)
    for _ in range(5):
        x,y=rng.randint(60,550),rng.randint(150,310); rx,ry=rng.randint(15,32),rng.randint(11,24)
        d.ellipse([x-rx,y-ry,x+rx,y+ry],fill=(9,6,3))
    for _ in range(6):
        x,y=rng.randint(50,540),rng.randint(140,300); pts=[(x,y)]
        for _ in range(7): pts.append((pts[-1][0]+rng.randint(8,20),pts[-1][1]+rng.randint(-6,6)))
        for i in range(len(pts)-1): d.line([pts[i],pts[i+1]],fill=(18,14,8),width=2)
    d.ellipse([370,210,560,290],fill=(22,48,82))
    return img

# ══════════════════════════════════════════════════════════════════════
#  GRAPHS
# ══════════════════════════════════════════════════════════════════════
def build_graphs(rqi, ph, wt, cr, now_h, avg_spd, safe_spd, sevs=None):
    rc="#20d47a" if rqi>=70 else "#f0d020" if rqi>=40 else "#f03a3a"
    rl="Good"   if rqi>=70 else "Moderate" if rqi>=40 else "Poor"
    BASE={k:v for k,v in PT.items() if k not in ("xaxis","yaxis")}
    AX,AY=PT["xaxis"],PT["yaxis"]
    if sevs is None:
        sevs={"potholes":[],"water":[],"cracks":[]}

    # ── Fig 1: RQI Gauge ─────────────────────────────────────────────
    fig1=go.Figure(go.Indicator(
        mode="gauge+number",value=rqi,
        number=dict(font=dict(size=40,color=rc,family="Bebas Neue"),suffix=" / 100"),
        gauge=dict(
            axis=dict(range=[0,100],tickfont=dict(size=9,color="#2a4060")),
            bar=dict(color=rc,thickness=0.28),bgcolor="#08090f",borderwidth=1,bordercolor="#1a2a4a",
            steps=[dict(range=[0,40],color="#120808"),dict(range=[40,70],color="#121008"),dict(range=[70,100],color="#081208")],
            threshold=dict(line=dict(color="#eef4ff",width=3),thickness=0.75,value=rqi),
        ),
    ))
    fig1.add_annotation(x=0.5,y=0.02,xref="paper",yref="paper",
        text=f"{rl} · P={ph} · W={wt} · C={cr}",
        showarrow=False,font=dict(size=9,color="#2a4060",family="Space Mono"))
    fig1.update_layout(**BASE,height=300,margin=dict(l=20,r=20,t=42,b=40),
        title=dict(text=f"RQI: {rqi}/100 — {rl}",font=dict(size=12,color=rc)))

    # ── Fig 2: Avg vs Safe Speed ─────────────────────────────────────
    reduction=round((BASE_SPEED-avg_spd)/BASE_SPEED*100,1)
    buf_delta=round(avg_spd-safe_spd,1)
    spd_col="#20d47a" if safe_spd>=50 else "#f0d020" if safe_spd>=35 else "#f03a3a"
    fig2=go.Figure()
    for lbl,val,col in [("Avg Speed\n(RQI adj.)",avg_spd,"#f5a020"),("Safe Speed\n(10% margin)",safe_spd,spd_col)]:
        fig2.add_trace(go.Bar(x=[lbl],y=[val],marker_color=col,marker_line_width=0,width=0.44,
            text=[f"{val} mph"],textposition="outside",textfont=dict(size=14,color="#eef4ff"),showlegend=False,
            hovertemplate=f"{lbl.replace(chr(10),' ')}: %{{y}} mph<extra></extra>"))
    fig2.add_annotation(x="Avg Speed\n(RQI adj.)",y=avg_spd+3,text=f"▼ {reduction}% vs {BASE_SPEED}mph",
        showarrow=False,font=dict(size=9,color="#f03a3a"))
    fig2.add_annotation(x="Safe Speed\n(10% margin)",y=safe_spd+3,text=f"−{buf_delta} mph buffer",
        showarrow=False,font=dict(size=9,color="#607080"))
    fig2.update_layout(**BASE,height=300,barmode="group",margin=dict(l=50,r=30,t=42,b=55),
        title=dict(text="Avg Speed vs Safe Speed (mph)",font=dict(size=12,color="#f5a020")),
        xaxis=dict(**AX),yaxis=dict(range=[0,BASE_SPEED*1.4],title="Speed (mph)",**AY))

    # ── Fig 3: Traffic vs National ───────────────────────────────────
    hours=list(range(24)); la_avg=round(sum(LA_HOURLY)/24); nat_avg=round(sum(NAT_HOURLY)/24)
    diff=round((la_avg-nat_avg)/nat_avg*100,1); sign="+" if diff>0 else ""
    fig3=go.Figure()
    fig3.add_trace(go.Scatter(x=hours,y=LA_HOURLY,fill=None,mode="none",showlegend=False))
    fig3.add_trace(go.Scatter(x=hours,y=NAT_HOURLY,fill="tonexty",fillcolor="rgba(245,160,32,0.07)",mode="none",showlegend=False))
    fig3.add_trace(go.Scatter(x=hours,y=NAT_HOURLY,mode="lines+markers",name="National Pattern (Illustrative)",
        line=dict(color="#3a8ef0",width=2,dash="dot"),marker=dict(size=4,color="#3a8ef0"),
        hovertemplate="Hour %{x}:00 · National: %{y:,}<extra></extra>"))
    fig3.add_trace(go.Scatter(x=hours,y=LA_HOURLY,mode="lines+markers",name="LA Pattern (Illustrative)",
        line=dict(color="#f5a020",width=2.5),marker=dict(size=5,color="#f5a020"),
        hovertemplate="Hour %{x}:00 · LA: %{y:,}<extra></extra>"))
    fig3.add_vline(x=now_h,line_color="#7c3aed",line_dash="dot",line_width=2,
        annotation_text=f"Now {now_h:02d}:00",annotation_font_color="#7c3aed",annotation_font_size=9)
    fig3.add_hline(y=la_avg,line_color="#f5a020",line_dash="dash",line_width=1,
        annotation_text=f"LA {la_avg:,}",annotation_font_color="#f5a020",annotation_font_size=8,annotation_position="right")
    fig3.add_hline(y=nat_avg,line_color="#3a8ef0",line_dash="dash",line_width=1,
        annotation_text=f"Nat {nat_avg:,}",annotation_font_color="#3a8ef0",annotation_font_size=8,annotation_position="right")
    fig3.add_annotation(x=17,y=max(LA_HOURLY)+260,text=f"LA is {sign}{diff}% vs national",
        showarrow=False,font=dict(size=9,color="#20d47a" if diff>0 else "#f03a3a"))
    fig3.update_layout(**BASE,height=300,margin=dict(l=50,r=100,t=42,b=50),
        title=dict(text="Illustrative Traffic Pattern (24 hr) — Representative Data",font=dict(size=12,color="#f5a020")),
        xaxis=dict(title="Hour",tickmode="linear",tick0=0,dtick=3,**AX),
        yaxis=dict(title="Vehicles / Hour",**AY),
        legend=dict(bgcolor="rgba(12,16,34,0.85)",bordercolor="#1a2a4a",borderwidth=1,
                    font=dict(size=9,color="#b0c4de"),x=0.01,y=0.99))

    # ── Fig 4: Defect Type Distribution (stacked by severity) ────────
    sev_order = ["mild","moderate","severe"]
    sev_colors = {"mild":"#f0d020","moderate":"#f5a020","severe":"#f03a3a"}
    defect_types = ["Potholes","Water Logging","Cracks"]
    sev_key_map  = ["potholes","water","cracks"]
    defect_totals= [ph, wt, cr]
    defect_bar_cols = ["#f03a3a","#3a8ef0","#f0d020"]

    fig4 = go.Figure()
    for sev in sev_order:
        counts = []
        for key in sev_key_map:
            counts.append(sevs[key].count(sev))
        fig4.add_trace(go.Bar(
            name=sev.capitalize(),
            x=defect_types,
            y=counts,
            marker_color=sev_colors[sev],
            marker_line_width=0,
            text=[str(c) if c>0 else "" for c in counts],
            textposition="inside",
            textfont=dict(size=11,color="#08090f"),
            hovertemplate="%{x} — " + sev + ": %{y}<extra></extra>",
        ))
    # Overlay total labels on top
    fig4.add_trace(go.Bar(
        name="Total",
        x=defect_types,
        y=defect_totals,
        marker_color="rgba(0,0,0,0)",
        marker_line_width=0,
        text=[f"Total: {v}" for v in defect_totals],
        textposition="outside",
        textfont=dict(size=10,color="#eef4ff"),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig4.update_layout(
        **BASE, height=300, barmode="stack",
        margin=dict(l=50,r=20,t=42,b=50),
        title=dict(text="Defect Type Distribution by Severity",font=dict(size=12,color="#f5a020")),
        xaxis=dict(title="Defect Type",**AX),
        yaxis=dict(title="Count",**AY),
        legend=dict(bgcolor="rgba(12,16,34,0.85)",bordercolor="#1a2a4a",borderwidth=1,
                    font=dict(size=9,color="#b0c4de"),x=0.01,y=0.99),
    )

    # ── Fig 5: Road Defect Severity vs Speed Reduction ───────────────
    # Show how each severity level across all defect types drives speed down
    sev_labels = ["Mild","Moderate","Severe"]
    # Worst-case speed reduction per severity bucket (representative values)
    # mild defects: small penalty, moderate: medium, severe: large
    # We compute from actual detected defect counts weighted by severity
    def sev_weighted_reduction(sev_list, weight):
        """Return speed reduction % contributed by defects at each severity."""
        m = sev_list.count("mild")
        mo= sev_list.count("moderate")
        sv= sev_list.count("severe")
        base = BASE_SPEED
        # Each unit of weight reduces speed: mild=0.5×, moderate=1×, severe=2×
        red_mild  = round(min(m  * weight * 0.5 / base * 100, 30), 1)
        red_mod   = round(min(mo * weight * 1.0 / base * 100, 40), 1)
        red_sev   = round(min(sv * weight * 2.0 / base * 100, 50), 1)
        return red_mild, red_mod, red_sev

    all_sev = sevs["potholes"] + sevs["water"] + sevs["cracks"]
    # Per-type breakdown for grouped bar
    types_labels = ["Potholes\n(w=4)","Water\n(w=5)","Cracks\n(w=3)"]
    type_weights  = [4, 5, 3]
    type_sevs     = [sevs["potholes"], sevs["water"], sevs["cracks"]]

    fig5 = go.Figure()
    sev_bar_cols = {"Mild":"#f0d020","Moderate":"#f5a020","Severe":"#f03a3a"}
    for i, (sev_lbl, sev_key) in enumerate(zip(["Mild","Moderate","Severe"],["mild","moderate","severe"])):
        y_vals = []
        for slist, wt_ in zip(type_sevs, type_weights):
            cnt  = slist.count(sev_key)
            red  = round(cnt * wt_ * (0.5 if sev_key=="mild" else 1.0 if sev_key=="moderate" else 2.0)
                         / BASE_SPEED * 100, 1)
            y_vals.append(min(red, 60))
        fig5.add_trace(go.Bar(
            name=sev_lbl,
            x=types_labels,
            y=y_vals,
            marker_color=sev_bar_cols[sev_lbl],
            marker_line_width=0,
            text=[f"{v}%" if v > 0 else "" for v in y_vals],
            textposition="outside",
            textfont=dict(size=10,color="#eef4ff"),
            hovertemplate="%{x}<br>" + sev_lbl + " severity: %{y}% speed reduction<extra></extra>",
        ))
    # Reference line — total overall reduction
    total_red = round((BASE_SPEED - avg_spd) / BASE_SPEED * 100, 1)
    fig5.add_hline(y=total_red, line_color="#7c3aed", line_dash="dot", line_width=2,
        annotation_text=f"Overall ▼{total_red}%",
        annotation_font_color="#7c3aed", annotation_font_size=9, annotation_position="right")
    fig5.update_layout(
        **BASE, height=300, barmode="group",
        margin=dict(l=50,r=80,t=42,b=60),
        title=dict(text="Road Defect Severity vs Speed Reduction (%)",font=dict(size=12,color="#f5a020")),
        xaxis=dict(title="Defect Type",**AX),
        yaxis=dict(title="Speed Reduction (%)",range=[0,70],**AY),
        legend=dict(bgcolor="rgba(12,16,34,0.85)",bordercolor="#1a2a4a",borderwidth=1,
                    font=dict(size=9,color="#b0c4de"),x=0.01,y=0.99),
    )

    # ── Fig 6: Road Condition vs Travel Time ─────────────────────────
    # Baseline travel time = 30 min at 65 mph over ~32.5 miles
    BASELINE_DIST_MI = 32.5  # miles (representative LA freeway segment)
    cond_labels  = ["Good\n(RQI 70-100)","Moderate\n(RQI 40-69)","Poor\n(RQI 0-39)","Current\nRoad"]
    cond_speeds  = [BASE_SPEED, BASE_SPEED*0.65, BASE_SPEED*0.35, avg_spd]
    travel_times = [round(BASELINE_DIST_MI / s * 60, 1) for s in cond_speeds]
    delays       = [round(t - travel_times[0], 1) for t in travel_times]
    bar_colors   = ["#20d47a","#f0d020","#f03a3a",
                    "#20d47a" if rqi>=70 else "#f0d020" if rqi>=40 else "#f03a3a"]

    fig6 = go.Figure()
    fig6.add_trace(go.Bar(
        name="Travel Time",
        x=cond_labels,
        y=travel_times,
        marker_color=bar_colors,
        marker_line_width=0,
        text=[f"{t} min" for t in travel_times],
        textposition="outside",
        textfont=dict(size=11,color="#eef4ff"),
        hovertemplate="%{x}<br>Travel time: %{y} min<extra></extra>",
        showlegend=False,
    ))
    # Delay annotation bars (stacked transparent layer)
    fig6.add_trace(go.Bar(
        name="Extra Delay vs Good",
        x=cond_labels,
        y=[0, delays[1], delays[2], delays[3]],
        marker_color=["rgba(0,0,0,0)","rgba(240,208,32,0.18)","rgba(240,58,58,0.18)",
                      "rgba(124,58,237,0.18)"],
        marker_line_width=0,
        base=travel_times,
        text=["","",f"+{delays[2]} min delay",
              f"+{delays[3]} min" if delays[3]>0 else "On time"],
        textposition="outside",
        textfont=dict(size=9,color="#607080"),
        showlegend=False,
        hoverinfo="skip",
    ))
    # Mark current condition with a vertical dashed line
    fig6.add_vline(x=3, line_color="#7c3aed", line_dash="dot", line_width=2)
    fig6.add_annotation(
        x=3, y=max(travel_times)*1.18,
        text=f"Current: {travel_times[3]} min",
        showarrow=False, font=dict(size=9,color="#7c3aed"))
    # Baseline reference
    fig6.add_hline(y=travel_times[0], line_color="#20d47a", line_dash="dash", line_width=1,
        annotation_text=f"Baseline {travel_times[0]} min",
        annotation_font_color="#20d47a", annotation_font_size=8, annotation_position="right")
    fig6.update_layout(
        **BASE, height=300, barmode="overlay",
        margin=dict(l=50,r=90,t=42,b=60),
        title=dict(text=f"Road Condition vs Travel Time ({BASELINE_DIST_MI:.0f} mi segment)",
                   font=dict(size=12,color="#f5a020")),
        xaxis=dict(title="Road Condition",**AX),
        yaxis=dict(title="Travel Time (min)",range=[0,max(travel_times)*1.35],**AY),
    )

    return fig1,fig2,fig3,fig4,fig5,fig6


# ══════════════════════════════════════════════════════════════════════
#  GeoCLIP — load once at module level (cache_resource handles repeat calls)
# ══════════════════════════════════════════════════════════════════════
_GC_MODEL, _GC_ERR = load_geoclip()
gc_model, gc_err   = _GC_MODEL, _GC_ERR

# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🛣 CalRoad IQ")
    st.markdown("---")
    st.markdown("#### 📦 Install")
    st.code(
        "pip install streamlit plotly folium\n"
        "  streamlit-folium pillow numpy\n"
        "  opencv-python requests\n"
        "  torch torchvision geoclip",
        language="bash")
    st.markdown("#### ▶ Run")
    st.code("streamlit run Project.py", language="bash")
    st.markdown("---")
    st.markdown("#### 🌍 Location Priority")
    st.markdown(
        "1. **EXIF GPS** — if photo has it embedded\n"
        "2. **GeoCLIP AI** — predicts from image content (NeurIPS 2023)\n"
        "3. **Manual** — enter coords yourself\n\n"
        "_GeoCLIP downloads ~1 GB on first run, then cached._")
    st.markdown("---")
    st.markdown("#### 📐 RQI Formula")
    st.code(
        "RQI = 100\n"
        "    − (4 × potholes)\n"
        "    − (5 × waterlogging)\n"
        "    − (3 × cracks)",
        language="text")
    st.markdown("---")
    st.markdown("#### 🤖 GeoCLIP Status")
    # Reuse module-level cached result — do NOT call load_geoclip() again here
    if _GC_MODEL is not None:
        st.success("✅ GeoCLIP loaded & ready")
    else:
        st.warning(f"⚠️ GeoCLIP unavailable\n\n`{_GC_ERR}`\n\nRun: `pip install torch geoclip`")


gc_status = "GeoCLIP AI Geo" if gc_model else "Manual Coords"

st.markdown(f"""
<div class="hdr">
  <div>
    <div class="hdr-title">🛣 CalRoad IQ</div>
    <div class="hdr-sub">AI defects · GeoCLIP location · RQI · speed · LA freeways</div>
  </div>
  <div style="font-family:'Space Mono',monospace;font-size:9px;color:#2a4060;text-align:right;line-height:2.2">
    <span class="dot-ai"></span>GPT-4o-mini · OpenRouter&nbsp;
    <span style="background:rgba(32,212,122,.15);border:1px solid #20d47a;color:#20d47a;padding:1px 6px;border-radius:10px;font-size:8px">FREE</span><br>
    <span class="dot-ai" style="background:#7c3aed;box-shadow:0 0 7px #7c3aed"></span>{gc_status}&nbsp;
    <span style="background:rgba(124,58,237,.15);border:1px solid #7c3aed;color:#b090f0;padding:1px 6px;border-radius:10px;font-size:8px">NeurIPS 2023</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── API Key ───────────────────────────────────────────────────────────
# Initialise API key in session state from env var (server-side default) or prior entry
if "api_key" not in st.session_state:
    st.session_state["api_key"] = os.environ.get("OPENROUTER_API_KEY", "")

api_key_input = st.text_input(
    "🔑 OpenRouter API Key",
    type="password",
    placeholder="sk-or-v1-...   (free key at openrouter.ai/keys)",
    value=st.session_state["api_key"],
)
if api_key_input:
    # Store ONLY in session_state — never in os.environ (shared across all users)
    st.session_state["api_key"] = api_key_input

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════
now_h = datetime.datetime.now().hour
for k, v in [
    ("result",       None),
    ("demo",         False),
    ("last_hash",    None),
    ("man_lat",      LA_CENTER[0]),
    ("man_lng",      LA_CENTER[1]),
    ("last_api_call", 0.0),   # rate-limiting timestamp
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════
#  UPLOAD
# ══════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec">📸 Upload Road Image</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop road photo — AI analyses defects + predicts location automatically",
    type=["jpg","jpeg","png","webp"],
    label_visibility="collapsed",
)

col_demo, col_clear, _ = st.columns([1,1,4])
with col_demo:
    if st.button("▶  Demo Image"):
        st.session_state["demo"] = True
        st.session_state["last_hash"] = None
with col_clear:
    if st.button("✕  Clear"):
        for k in ["result","demo","last_hash","man_lat","man_lng"]:
            st.session_state.pop(k, None)
        st.rerun()

use_demo = st.session_state["demo"]
pil_img  = None

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB hard limit

if uploaded:
    raw_bytes = uploaded.getvalue()
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        st.error(f"⚠️ Image too large ({len(raw_bytes)//1024//1024} MB). Please upload an image under 20 MB.")
        st.stop()
    pil_img  = Image.open(io.BytesIO(raw_bytes))
    use_demo = False
    st.session_state["demo"] = False
elif use_demo:
    pil_img = make_demo()

# ══════════════════════════════════════════════════════════════════════
#  AUTO-PROCESS on image change
# ══════════════════════════════════════════════════════════════════════
if pil_img is not None:
    w, h = pil_img.size
    if max(w,h) > 1024:
        s = 1024/max(w,h)
        pil_img = pil_img.resize((int(w*s),int(h*s)), Image.LANCZOS)

    current_hash = img_hash(pil_img)

    if current_hash != st.session_state["last_hash"]:
        akey = st.session_state.get("api_key", "")
        if not akey:
            st.error("⚠️ Enter your OpenRouter API key above.")
            st.stop()

        # ── Rate limiting: minimum 5 s between analyses ────────────
        elapsed = time.time() - st.session_state["last_api_call"]
        if elapsed < 5.0:
            st.warning(f"⏳ Please wait {round(5.0 - elapsed, 1)}s before analysing another image.")
            st.stop()

        # ── 1. EXIF GPS (exact — from image metadata) ──────────────
        exif_lat, exif_lng = get_exif_gps(pil_img)

        # ── 2. AI Vision GPS — GPT-4o-mini reads the image visually ─
        #    Runs whenever EXIF is missing. No extra packages needed.
        ai_gps      = None
        ai_gps_err  = None
        gc_preds    = None
        gc_error    = None

        if exif_lat is None:
            with st.spinner("🔍 AI Vision analysing image for location clues…"):
                ai_gps, ai_gps_err = predict_location_ai(pil_img, akey)

        # ── 3. GeoCLIP fallback (only if AI Vision also failed) ─────
        if exif_lat is None and ai_gps is None and gc_model is not None:
            with st.spinner("🌍 GeoCLIP predicting location from image content..."):
                gc_preds, gc_error = predict_location_geoclip(pil_img, top_k=5)

        # ── 4. Resolve final coordinates ────────────────────────────
        if exif_lat is not None:
            map_lat, map_lng = exif_lat, exif_lng
            gps_src   = "EXIF GPS"
            gps_mode  = "exif"
            gps_conf  = 1.0
            gps_city  = ""
            gps_reason= ""
            gc_preds  = None
            ai_gps    = None

        elif ai_gps is not None:
            map_lat   = ai_gps["lat"]
            map_lng   = ai_gps["lng"]
            gps_src   = "AI Vision GPS"
            gps_mode  = "aivision"
            gps_conf  = float(ai_gps.get("confidence", 0.5))
            gps_city  = ", ".join(filter(None, [
                ai_gps.get("city",""), ai_gps.get("state",""), ai_gps.get("country","")
            ]))
            gps_reason= ai_gps.get("reasoning", "")

        elif gc_preds and len(gc_preds) > 0:
            map_lat   = gc_preds[0]["lat"]
            map_lng   = gc_preds[0]["lng"]
            gps_src   = "GeoCLIP AI"
            gps_mode  = "geoclip"
            gps_conf  = gc_preds[0]["prob"]
            gps_city  = ""
            gps_reason= ""

        else:
            map_lat   = st.session_state["man_lat"]
            map_lng   = st.session_state["man_lng"]
            gps_src   = "Manual / Default"
            gps_mode  = "manual"
            gps_conf  = 0.0
            gps_city  = ""
            gps_reason= ""

        st.session_state["last_api_call"] = time.time()

        # ── 5. Defect detection ────────────────────────────────────
        with st.spinner("🤖 AI counting potholes, water & cracks..."):
            ai = run_detection(pil_img, akey)
        with st.spinner("🎨 Annotating..."):
            ann = annotate(pil_img, ai)

        np_  = ai["pothole_count"]
        nw   = ai["water_count"]
        nc   = ai["crack_count"]
        td   = np_ + nw + nc
        rqi  = compute_rqi(np_, nw, nc)
        rn   = rqi/100.0
        avg_spd, safe_spd = compute_speeds(rqi)
        rc   = "#20d47a" if rqi>=70 else "#f0d020" if rqi>=40 else "#f03a3a"
        rl   = "Good"    if rqi>=70 else "Moderate" if rqi>=40 else "Poor"
        uc   = {"Immediate":"#f03a3a","Soon":"#f0d020","Routine":"#20d47a",
                "None":"#20d47a"}.get(ai.get("repair_urgency",""),"#607080")

        st.session_state["result"] = dict(
            ann_img=ann, pil_img=pil_img,
            n_potholes=np_, n_water=nw, n_cracks=nc, total_def=td,
            rqi_score=rqi, rqi_norm=rn, rc=rc, rl=rl,
            avg_speed=avg_spd, safe_spd=safe_spd,
            map_lat=map_lat, map_lng=map_lng,
            gps_src=gps_src, gps_mode=gps_mode, gps_conf=gps_conf,
            gps_city=gps_city, gps_reason=gps_reason,
            gc_preds=gc_preds,
            ai_gps=ai_gps,
            ai_urgency   =ai.get("repair_urgency",""),
            ai_surface   =ai.get("surface_type","Unknown"),
            ai_visibility=ai.get("visibility","Unknown"),
            urgency_col  =uc,
            sevs={
                "potholes":[d.get("severity","—") for d in ai.get("potholes",[])],
                "water":   [d.get("severity","—") for d in ai.get("water",   [])],
                "cracks":  [d.get("severity","—") for d in ai.get("cracks",  [])],
            },
        )
        st.session_state["last_hash"] = current_hash

# ══════════════════════════════════════════════════════════════════════
#  RENDER
# ══════════════════════════════════════════════════════════════════════
res = st.session_state["result"]

if res:
    # ── Images ────────────────────────────────────────────────────
    ca, cb = st.columns(2)
    with ca:
        st.image(res["pil_img"], use_container_width=True, caption="Original")
    with cb:
        cap = (f"AI Detected — {res['total_def']} defect(s)"
               if res["total_def"]>0 else "AI — No Defects Detected")
        st.image(res["ann_img"], use_container_width=True, caption=cap)

    # ── Location status badge ──────────────────────────────────────
    lat_str, lng_str = fmt_coord(res["map_lat"], res["map_lng"])
    mode = res["gps_mode"]

    if mode == "exif":
        st.markdown(
            f'<div class="loc-exif">✅ &nbsp;<b>EXIF GPS</b> — coordinates extracted from image metadata<br>'
            f'<span style="color:#eef4ff;font-size:11px">{lat_str} &nbsp;·&nbsp; {lng_str}</span></div>',
            unsafe_allow_html=True)

    elif mode == "aivision":
        conf_pct = round(res["gps_conf"]*100, 1)
        city_txt = res.get("gps_city","")
        reason   = res.get("gps_reason","")
        # XSS fix: escape all AI-returned strings before HTML injection
        city_txt_safe = html.escape(city_txt)
        reason_safe   = html.escape(reason)
        st.markdown(
            f'<div class="loc-aivision">'
            f'🔍 &nbsp;<b>AI Vision GPS</b> — GPT-4o-mini read visual clues in the image<br>'
            f'<span style="color:#eef4ff;font-size:12px;font-weight:600">{lat_str} &nbsp;·&nbsp; {lng_str}</span>'
            f'{"&nbsp; <span style=color:#f09420>" + city_txt_safe + "</span>" if city_txt_safe else ""}<br>'
            f'<span style="color:#f5a020;font-size:9px">Confidence: {conf_pct}%</span>'
            f'{"<br><span style=color:#607080;font-size:9px>👁 Visual evidence: " + reason_safe + "</span>" if reason_safe else ""}'
            f'</div>',
            unsafe_allow_html=True)
        # Show alternate AI predictions
        ai_gps = res.get("ai_gps")
        if ai_gps and ai_gps.get("alt_predictions"):
            with st.expander("📍 AI Vision — alternate location predictions"):
                for i, p in enumerate(ai_gps["alt_predictions"]):
                    try:
                        alt_lat = round(float(p.get("lat", 0)), 5)
                        alt_lng = round(float(p.get("lng", 0)), 5)
                        alt_conf= round(float(p.get("confidence", 0))*100, 1)
                        alt_lbl_raw = str(p.get("label",""))
                        alt_lbl = html.escape(alt_lbl_raw)  # XSS: escape AI label
                        als, alns = fmt_coord(alt_lat, alt_lng)
                        bar = "█" * max(1, min(30, int(float(p.get("confidence",0))*30)))
                        st.markdown(
                            f"`#{i+2}` &nbsp; **{als} · {alns}** &nbsp;"
                            f"<span style='color:#f09420'>{alt_conf}% &nbsp;{bar}</span>"
                            f"{'&nbsp; <span style=color:#607080;font-size:9px>' + alt_lbl + '</span>' if alt_lbl else ''}",
                            unsafe_allow_html=True)
                    except Exception:
                        continue

    elif mode == "geoclip":
        conf_pct = round(res["gps_conf"]*100, 1)
        st.markdown(
            f'<div class="loc-geoclip">🌍 &nbsp;<b>GeoCLIP AI</b> — location predicted from image visual content (NeurIPS 2023)<br>'
            f'<span style="color:#eef4ff;font-size:11px">{lat_str} &nbsp;·&nbsp; {lng_str}</span>'
            f' &nbsp;<span style="color:#7c3aed">confidence: {conf_pct}%</span><br>'
            f'<span style="color:#2a4060;font-size:9px">'
            f'No EXIF GPS found — GeoCLIP estimated location from visual scene features.</span></div>',
            unsafe_allow_html=True)
        if res.get("gc_preds") and len(res["gc_preds"]) > 1:
            with st.expander("📍 GeoCLIP top-5 location predictions"):
                for i, p in enumerate(res["gc_preds"]):
                    ls, lns = fmt_coord(p["lat"], p["lng"])
                    bar = "█" * int(p["prob"]*30)
                    st.markdown(
                        f"`#{i+1}` &nbsp; **{ls} · {lns}** &nbsp; "
                        f"<span style='color:#7c3aed'>{round(p['prob']*100,2)}% &nbsp;{bar}</span>",
                        unsafe_allow_html=True)

    else:
        # ── MANUAL mode — FIX: persist coords back to session state ──
        st.markdown(
            f'<div class="loc-manual">📌 &nbsp;<b>Manual / Default</b> — no GPS data found. '
            f'Enter coordinates below.<br>'
            f'<span style="color:#eef4ff;font-size:11px">{lat_str} &nbsp;·&nbsp; {lng_str}</span></div>',
            unsafe_allow_html=True)
        lc1, lc2 = st.columns(2)
        with lc1:
            ml = st.number_input(
                "📌 Latitude",
                value=float(res["map_lat"]),
                format="%.6f",
                step=0.0001,
                key=f"lat_{st.session_state.get('last_hash') or 'default'}",
            )
        with lc2:
            mln = st.number_input(
                "📌 Longitude",
                value=float(res["map_lng"]),
                format="%.6f",
                step=0.0001,
                key=f"lng_{st.session_state.get('last_hash') or 'default'}",
            )
        # Clamp to valid WGS-84 ranges
        ml  = max(-90.0,  min(90.0,  ml))
        mln = max(-180.0, min(180.0, mln))
        res["map_lat"] = ml
        res["map_lng"] = mln
        res["gps_src"] = "Manual"
        st.session_state["man_lat"] = ml
        st.session_state["man_lng"] = mln
        lat_str, lng_str = fmt_coord(ml, mln)

    st.markdown("---")

    # ── Count Cards ────────────────────────────────────────────────
    st.markdown('<div class="sec">🔢 Detected Defect Count</div>', unsafe_allow_html=True)
    cc1,cc2,cc3,cc4 = st.columns(4)
    with cc1:
        sv=res["sevs"]["potholes"].count("severe"); mo=res["sevs"]["potholes"].count("moderate"); ml_=res["sevs"]["potholes"].count("mild")
        st.markdown(f'<div class="cnt-card cnt-red"><div class="cnt-lbl" style="color:#f03a3a">🔴 Potholes</div>'
            f'<div class="cnt-num" style="color:#f03a3a">{res["n_potholes"]}</div>'
            f'<div class="cnt-sub">{sv} severe · {mo} moderate · {ml_} mild</div></div>',unsafe_allow_html=True)
    with cc2:
        sv=res["sevs"]["water"].count("severe"); mo=res["sevs"]["water"].count("moderate"); ml_=res["sevs"]["water"].count("mild")
        st.markdown(f'<div class="cnt-card cnt-blue"><div class="cnt-lbl" style="color:#3a8ef0">💧 Water Logging</div>'
            f'<div class="cnt-num" style="color:#3a8ef0">{res["n_water"]}</div>'
            f'<div class="cnt-sub">{sv} severe · {mo} moderate · {ml_} mild</div></div>',unsafe_allow_html=True)
    with cc3:
        sv=res["sevs"]["cracks"].count("severe"); mo=res["sevs"]["cracks"].count("moderate"); ml_=res["sevs"]["cracks"].count("mild")
        st.markdown(f'<div class="cnt-card cnt-yellow"><div class="cnt-lbl" style="color:#f0d020">⚡ Crack Zones</div>'
            f'<div class="cnt-num" style="color:#f0d020">{res["n_cracks"]}</div>'
            f'<div class="cnt-sub">{sv} severe · {mo} moderate · {ml_} mild</div></div>',unsafe_allow_html=True)
    with cc4:
        tc_cls="cnt-red" if res["total_def"]>=5 else "cnt-yellow" if res["total_def"]>=2 else "cnt-green"
        tc_col="#f03a3a" if res["total_def"]>=5 else "#f0d020"    if res["total_def"]>=2 else "#20d47a"
        st.markdown(f'<div class="cnt-card {tc_cls}"><div class="cnt-lbl" style="color:{tc_col}">📊 Total Defects</div>'
            f'<div class="cnt-num" style="color:{tc_col}">{res["total_def"]}</div>'
            f'<div class="cnt-sub">RQI: {res["rqi_score"]}/100 · {res["rl"]}</div></div>',unsafe_allow_html=True)

    st.markdown("---")

    # ── Badges ─────────────────────────────────────────────────────
    bdg='<div class="dbrow"><div class="db dai">🤖 gpt-4o-mini · high-detail</div>'
    if res["n_potholes"]>0:
        s=", ".join(res["sevs"]["potholes"])
        bdg+=f'<div class="db dp">🔴 Potholes ×{res["n_potholes"]} <span style="opacity:.6;font-size:8px">({s})</span></div>'
    if res["n_water"]>0:
        s=", ".join(res["sevs"]["water"])
        bdg+=f'<div class="db dw">💧 Water ×{res["n_water"]} <span style="opacity:.6;font-size:8px">({s})</span></div>'
    if res["n_cracks"]>0:
        s=", ".join(res["sevs"]["cracks"])
        bdg+=f'<div class="db dc">⚡ Cracks ×{res["n_cracks"]} <span style="opacity:.6;font-size:8px">({s})</span></div>'
    if res["total_def"]==0:
        bdg+='<div class="db dok">✓ No Defects Detected</div>'
    bdg+=f'<div class="db dtot">Total: {res["total_def"]}</div></div>'
    st.markdown(bdg, unsafe_allow_html=True)

    # ── Stat Cards ─────────────────────────────────────────────────
    st.markdown('<div class="sec">📊 Road Analysis</div>', unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1:
        rqi_=res["rqi_score"]
        qcol="cr" if rqi_<40 else "cy" if rqi_<70 else "cg"
        qcls="bad" if rqi_<40 else "warn" if rqi_<70 else "good"
        st.markdown(f'<div class="card {qcol}"><div class="lbl">RQI Score</div>'
            f'<div class="val" style="color:{res["rc"]}">{res["rqi_score"]}</div>'
            f'<div class="unt">out of 100</div>'
            f'<div class="sub {qcls}">{res["rl"]} condition</div></div>',unsafe_allow_html=True)
    with c2:
        scls="bad" if res["avg_speed"]<35 else "warn" if res["avg_speed"]<52 else "good"
        st.markdown(f'<div class="card ca"><div class="lbl">Avg Speed</div>'
            f'<div class="val" style="color:#f5a020">{res["avg_speed"]}</div>'
            f'<div class="unt">mph · RQI adjusted</div>'
            f'<div class="sub {scls}">Safe: {res["safe_spd"]} mph</div></div>',unsafe_allow_html=True)
    with c3:
        ucol="cr" if res["ai_urgency"]=="Immediate" else "cy" if res["ai_urgency"]=="Soon" else "cg"
        # Escape AI-generated text before HTML insertion
        safe_urgency    = html.escape(str(res["ai_urgency"]))
        safe_surface    = html.escape(str(res["ai_surface"]))
        safe_visibility = html.escape(str(res["ai_visibility"]))
        st.markdown(f'<div class="card {ucol}"><div class="lbl">Repair Urgency</div>'
            f'<div class="val" style="color:{res["urgency_col"]};font-size:22px;padding-top:6px">{safe_urgency}</div>'
            f'<div class="unt">AI recommendation</div>'
            f'<div class="sub dim">Surface: {safe_surface} · Vis: {safe_visibility}</div></div>',
            unsafe_allow_html=True)

    p,wl,cr_=res["n_potholes"],res["n_water"],res["n_cracks"]
    st.markdown(
        f'<div class="fml">'
        f'RQI = 100 − (4×p) − (5×w) − (3×c) &nbsp;<span style="color:#7c3aed;font-size:9px">← AI counted</span><br>'
        f'&nbsp;&nbsp;&nbsp;&nbsp; = 100 − (4×{p}) − (5×{wl}) − (3×{cr_})'
        f' = <span style="color:{res["rc"]}">{res["rqi_score"]}/100</span> [{res["rl"]}]<br>'
        f'Avg = {BASE_SPEED}×{res["rqi_norm"]:.2f} = <span style="color:#f5a020">{res["avg_speed"]} mph</span>'
        f' · Safe = ×0.90 = <span style="color:#20d47a">{res["safe_spd"]} mph</span>'
        f'</div>',unsafe_allow_html=True)

    st.markdown("---")

    # ── MAP ────────────────────────────────────────────────────────
    mode_label = {"exif":"📡 EXIF GPS","aivision":"🔍 AI Vision GPS","geoclip":"🌍 GeoCLIP AI","manual":"📌 Manual"}.get(res["gps_mode"],"")
    st.markdown(
        f'<div class="sec">🗺 Location — {lat_str} · {lng_str} &nbsp;'
        f'<span style="color:#2a4060;font-size:8px">{mode_label}</span></div>',
        unsafe_allow_html=True)

    cg, cm = st.columns([1,3.5])
    rc2 = res["rc"]
    with cg:
        conf_txt = (f'{round(res["gps_conf"]*100,1)}%' if res["gps_mode"] in ("aivision","geoclip")
                    else "exact" if res["gps_mode"]=="exif" else "—")
        city_row = (f'<div class="gr"><span class="gk">Location</span>'
                    f'<span class="gv" style="color:#f09420;font-size:8px">{html.escape(str(res.get("gps_city","—")))}</span></div>'
                    if res.get("gps_city") else "")
        st.markdown(
            f'<div class="gbox">'
            f'<div style="font-family:Space Mono,monospace;font-size:8px;letter-spacing:2px;color:#f5a020;margin-bottom:7px">{mode_label}</div>'
            f'<div class="gr"><span class="gk">Latitude</span><span class="gv">{lat_str}</span></div>'
            f'<div class="gr"><span class="gk">Longitude</span><span class="gv">{lng_str}</span></div>'
            f'<div class="gr"><span class="gk">Confidence</span><span class="gv" style="color:#f09420">{conf_txt}</span></div>'
            f'{city_row}'
            f'<div class="gr"><span class="gk">Potholes</span><span class="gv" style="color:#f03a3a">{res["n_potholes"]}</span></div>'
            f'<div class="gr"><span class="gk">Water</span><span class="gv" style="color:#3a8ef0">{res["n_water"]}</span></div>'
            f'<div class="gr"><span class="gk">Cracks</span><span class="gv" style="color:#f0d020">{res["n_cracks"]}</span></div>'
            f'<div class="gr"><span class="gk">RQI</span><span class="gv" style="color:{rc2}">{res["rqi_score"]}/100</span></div>'
            f'<div class="gr"><span class="gk">Avg Speed</span><span class="gv" style="color:#f5a020">{res["avg_speed"]} mph</span></div>'
            f'<div class="gr"><span class="gk">Safe Speed</span><span class="gv" style="color:#20d47a">{res["safe_spd"]} mph</span></div>'
            f'<div class="gr"><span class="gk">Urgency</span><span class="gv" style="color:{res["urgency_col"]}">{res["ai_urgency"]}</span></div>'
            f'</div>',unsafe_allow_html=True)
    with cm:
        zoom = 15 if res["gps_mode"]=="exif" else 14 if res["gps_mode"]=="aivision" else 13 if res["gps_mode"]=="geoclip" else 11
        m = folium.Map(location=[res["map_lat"],res["map_lng"]], zoom_start=zoom, tiles="CartoDB dark_matter")

        # LA freeway markers
        for rname,(rlat,rlng) in LA_ROADS.items():
            folium.CircleMarker([rlat,rlng],radius=5,color="#f5a020",fill=True,
                fill_color="#1a2a4a",fill_opacity=0.8,weight=2,tooltip=rname).add_to(m)

        # AI Vision alternate predictions (dimmed orange dots)
        ai_gps_data = res.get("ai_gps")
        if ai_gps_data and res["gps_mode"]=="aivision":
            for i, p in enumerate(ai_gps_data.get("alt_predictions", [])):
                try:
                    alt_lat = float(p.get("lat", 0))
                    alt_lng = float(p.get("lng", 0))
                    alt_conf= float(p.get("confidence", 0))
                    alt_lbl = p.get("label", f"Alt #{i+2}")
                    if alt_lat == 0 and alt_lng == 0:
                        continue
                    folium.CircleMarker(
                        [alt_lat, alt_lng], radius=7,
                        color="#f09420", fill=True, fill_color="#f09420",
                        fill_opacity=0.25, weight=1,
                        tooltip=f"AI Alt #{i+2}: {alt_lat:.4f}, {alt_lng:.4f} ({round(alt_conf*100,1)}%) — {alt_lbl}",
                    ).add_to(m)
                except Exception:
                    continue

        # GeoCLIP top-5 alternate predictions (dimmed purple)
        if res.get("gc_preds") and res["gps_mode"]=="geoclip":
            for i, p in enumerate(res["gc_preds"][1:], start=2):
                folium.CircleMarker(
                    [p["lat"],p["lng"]], radius=6,
                    color="#7c3aed", fill=True, fill_color="#7c3aed",
                    fill_opacity=0.25, weight=1,
                    tooltip=f"GeoCLIP #{i}: {p['lat']:.4f}, {p['lng']:.4f} ({round(p['prob']*100,1)}%)",
                ).add_to(m)

        # Main detection pin
        pin_color = {"exif":"#20d47a","aivision":"#f09420","geoclip":"#7c3aed","manual":"#f5a020"}.get(res["gps_mode"],"#f5a020")
        city_disp = html.escape(res.get("gps_city", ""))  # XSS fix for Folium popup
        folium.Marker(
            [res["map_lat"],res["map_lng"]],
            icon=folium.DivIcon(
                html=(f'<div style="width:74px;height:74px;background:rgba(8,9,15,.95);'
                      f'border:3px solid {rc2};border-radius:50%;display:flex;flex-direction:column;'
                      f'align-items:center;justify-content:center;font-family:monospace;'
                      f'box-shadow:0 0 22px {rc2}88,0 0 0 2px {pin_color}44;">'
                      f'<span style="font-size:15px;font-weight:700;color:{rc2}">{res["rqi_score"]}</span>'
                      f'<span style="font-size:7px;color:#607080">RQI</span>'
                      f'<span style="font-size:9px">🔴{res["n_potholes"]} 💧{res["n_water"]}</span>'
                      f'</div>'),
                icon_size=(74,74),icon_anchor=(37,37)),
            tooltip=f"RQI {res['rqi_score']} · {res['n_potholes']}p · {res['n_water']}w · {res['n_cracks']}c",
            popup=folium.Popup(
                f'<div style="font-family:monospace;font-size:11px;background:#0c1022;color:#8ba5c8;'
                f'padding:12px;border:1px solid #1a2a4a;min-width:200px">'
                f'<b style="color:{rc2}">RQI: {res["rqi_score"]}/100 — {res["rl"]}</b><br><br>'
                f'🔴 Potholes: <b style="color:#f03a3a">{res["n_potholes"]}</b><br>'
                f'💧 Water: <b style="color:#3a8ef0">{res["n_water"]}</b><br>'
                f'⚡ Cracks: <b style="color:#f0d020">{res["n_cracks"]}</b><br><br>'
                f'<span style="color:{pin_color}">📍 {res["gps_src"]}</span><br>'
                f'{"<span style=color:#f09420;font-size:9px>" + city_disp + "</span><br>" if city_disp else ""}'
                f'<span style="color:#2a4060;font-size:9px">{lat_str} · {lng_str}</span>'
                f'</div>',max_width=260),
        ).add_to(m)
        folium.Circle([res["map_lat"],res["map_lng"]],radius=120,color=rc2,fill=True,
            fill_color=rc2,fill_opacity=0.07,weight=2).add_to(m)

        m.get_root().html.add_child(folium.Element(
            '<div style="position:fixed;bottom:14px;left:10px;z-index:9999;background:rgba(8,9,15,.95);'
            'border:1px solid #1a2a4a;padding:9px 13px;border-radius:7px;font-family:monospace;font-size:11px">'
            '<div style="color:#f5a020;font-size:7px;letter-spacing:2px;margin-bottom:4px">RQI</div>'
            '<div><span style="color:#20d47a">●</span> Good ≥70</div>'
            '<div><span style="color:#f0d020">●</span> Moderate 40–69</div>'
            '<div><span style="color:#f03a3a">●</span> Poor &lt;40</div>'
            '<div style="margin-top:5px;border-top:1px solid #1a2a4a;padding-top:5px;">'
            '<span style="color:#f5a020;font-size:8px">● LA Freeways</span></div>'
            '<div style="margin-top:3px;"><span style="color:#f09420;font-size:8px">◉ AI Vision alt. spots</span></div>'
            '<div style="margin-top:3px;"><span style="color:#7c3aed;font-size:8px">◉ GeoCLIP alt. predictions</span></div>'
            '</div>'))
        st_folium(m, width="100%", height=440, returned_objects=[])

    st.markdown("---")
    g_rqi=res["rqi_score"]; g_ph=res["n_potholes"]
    g_wt=res["n_water"];    g_cr=res["n_cracks"]
    g_avg=res["avg_speed"]; g_safe=res["safe_spd"]

else:
    st.markdown(
        '<div style="text-align:center;padding:60px 20px;background:#0c1022;'
        'border:2px dashed #1a2a4a;border-radius:12px;margin:20px 0">'
        '<div style="font-size:48px;margin-bottom:16px">🛣️</div>'
        '<div style="font-family:Space Mono,monospace;font-size:13px;color:#2a4060;line-height:2.6">'
        'Upload a road image to start<br>'
        '<span style="color:#f03a3a">🔴 AI counts potholes</span> &nbsp;'
        '<span style="color:#3a8ef0">💧 water logging</span> &nbsp;'
        '<span style="color:#f0d020">⚡ cracks</span><br>'
        '<span style="color:#7c3aed">🌍 GeoCLIP predicts location from image content</span><br>'
        '<span style="color:#f5a020">RQI · speed · map — all update automatically</span>'
        '</div></div>',unsafe_allow_html=True)
    st.markdown("---")
    g_rqi=100; g_ph=0; g_wt=0; g_cr=0
    g_avg=float(BASE_SPEED); g_safe=round(BASE_SPEED*0.9,1)

# ══════════════════════════════════════════════════════════════════════
#  GRAPHS
# ══════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec">📈 Analytics</div>', unsafe_allow_html=True)

g_sevs = res["sevs"] if res else {"potholes":[],"water":[],"cracks":[]}
fig1,fig2,fig3,fig4,fig5,fig6 = build_graphs(
    g_rqi,g_ph,g_wt,g_cr,now_h,g_avg,g_safe,sevs=g_sevs)

# ── Row 1 ─────────────────────────────────────────────────────────────
g1,g2,g3=st.columns(3)
with g1:
    st.markdown('<div style="font-family:Space Mono,monospace;font-size:9px;letter-spacing:2px;color:#f5a020;margin-bottom:5px">① ROAD QUALITY INDEX (RQI)</div>',unsafe_allow_html=True)
    st.plotly_chart(fig1,use_container_width=True,config={"displayModeBar":False})
with g2:
    st.markdown('<div style="font-family:Space Mono,monospace;font-size:9px;letter-spacing:2px;color:#f5a020;margin-bottom:5px">② AVG SPEED vs SAFE SPEED</div>',unsafe_allow_html=True)
    st.plotly_chart(fig2,use_container_width=True,config={"displayModeBar":False})
with g3:
    st.markdown('<div style="font-family:Space Mono,monospace;font-size:9px;letter-spacing:2px;color:#f5a020;margin-bottom:5px">③ OVERALL AVG TRAFFIC vs NATIONAL AVG</div>',unsafe_allow_html=True)
    st.plotly_chart(fig3,use_container_width=True,config={"displayModeBar":False})

# ── Row 2 ─────────────────────────────────────────────────────────────
g4,g5,g6=st.columns(3)
with g4:
    st.markdown('<div style="font-family:Space Mono,monospace;font-size:9px;letter-spacing:2px;color:#f5a020;margin-bottom:5px">④ DEFECT TYPE DISTRIBUTION</div>',unsafe_allow_html=True)
    st.plotly_chart(fig4,use_container_width=True,config={"displayModeBar":False})
with g5:
    st.markdown('<div style="font-family:Space Mono,monospace;font-size:9px;letter-spacing:2px;color:#f5a020;margin-bottom:5px">⑤ DEFECT SEVERITY vs SPEED REDUCTION</div>',unsafe_allow_html=True)
    st.plotly_chart(fig5,use_container_width=True,config={"displayModeBar":False})
with g6:
    st.markdown('<div style="font-family:Space Mono,monospace;font-size:9px;letter-spacing:2px;color:#f5a020;margin-bottom:5px">⑥ ROAD CONDITION vs TRAVEL TIME</div>',unsafe_allow_html=True)
    st.plotly_chart(fig6,use_container_width=True,config={"displayModeBar":False})

# ══════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════
_lat = res["map_lat"] if res else LA_CENTER[0]
_lng = res["map_lng"] if res else LA_CENTER[1]
fl, fw = fmt_coord(_lat, _lng)
st.markdown(
    f'<div style="text-align:center;padding:10px 0 3px;font-family:Space Mono,monospace;'
    f'font-size:8px;color:#2a4060;letter-spacing:2px;border-top:1px solid #1a2a4a;margin-top:6px">'
    f'CALROAD IQ · GPT-4o-mini · GEOCLIP NeurIPS 2023 · RQI=100−(4P)−(5W)−(3C) · {fl} {fw}'
    f'</div>',unsafe_allow_html=True)
