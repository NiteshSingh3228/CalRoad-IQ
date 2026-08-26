"""
test_core.py — CalRoad IQ unit tests
Run with:  pytest test_core.py -v
"""
import sys, os, io, html
sys.path.insert(0, os.path.dirname(__file__))

from Project import (
    compute_rqi, compute_speeds, fmt_coord,
    get_exif_gps, img_hash, _empty
)
from PIL import Image


# ── RQI ──────────────────────────────────────────────────────────────────

def test_rqi_perfect():
    assert compute_rqi(0, 0, 0) == 100

def test_rqi_clamp_to_zero():
    assert compute_rqi(100, 100, 100) == 0

def test_rqi_formula():
    # 100 - 4*3 - 5*1 - 3*2 = 77
    assert compute_rqi(3, 1, 2) == 77

def test_rqi_boundary_good():
    assert compute_rqi(0, 0, 10) == 70

def test_rqi_boundary_moderate():
    assert compute_rqi(0, 0, 20) == 40

def test_rqi_min_clamped():
    assert compute_rqi(999, 999, 999) >= 0

def test_rqi_max_clamped():
    assert compute_rqi(0, 0, 0) <= 100


# ── Speeds ───────────────────────────────────────────────────────────────

def test_speeds_full_rqi():
    avg, safe = compute_speeds(100)
    assert avg == 65.0
    assert safe == 58.5

def test_speeds_zero_rqi_no_zero_mph():
    """RQI=0 must never return 0 mph."""
    avg, safe = compute_speeds(0)
    assert avg > 0.0
    assert safe > 0.0

def test_speeds_safe_always_less():
    for rqi in [0, 25, 50, 75, 100]:
        avg, safe = compute_speeds(rqi)
        assert safe <= avg


# ── Coordinates ───────────────────────────────────────────────────────────

def test_fmt_coord_north_west():
    lat_s, lng_s = fmt_coord(34.052, -118.243)
    assert "N" in lat_s
    assert "W" in lng_s

def test_fmt_coord_south_east():
    lat_s, lng_s = fmt_coord(-33.87, 151.21)
    assert "S" in lat_s
    assert "E" in lng_s


# ── EXIF GPS ──────────────────────────────────────────────────────────────

def test_exif_no_exif_image():
    img = Image.new("RGB", (100, 100))
    lat, lng = get_exif_gps(img)
    assert lat is None
    assert lng is None


# ── Image hashing ─────────────────────────────────────────────────────────

def test_img_hash_deterministic():
    img = Image.new("RGB", (200, 200), color=(10, 20, 30))
    assert img_hash(img) == img_hash(img)

def test_img_hash_different_images():
    img1 = Image.new("RGB", (200, 200), color=(10, 20, 30))
    img2 = Image.new("RGB", (200, 200), color=(200, 100, 50))
    assert img_hash(img1) != img_hash(img2)


# ── _empty() ──────────────────────────────────────────────────────────────

def test_empty_zero_counts():
    e = _empty()
    assert e["pothole_count"] == 0
    assert e["water_count"] == 0
    assert e["crack_count"] == 0

def test_empty_all_keys():
    e = _empty()
    for key in ["potholes","water","cracks","pothole_count","water_count",
                "crack_count","overall_condition","repair_urgency",
                "surface_type","visibility"]:
        assert key in e, f"Missing: {key}"


# ── XSS safety ───────────────────────────────────────────────────────────

def test_html_escape_strips_script():
    escaped = html.escape("<script>alert(1)</script>")
    assert "<script>" not in escaped

def test_html_escape_safe_text_unchanged():
    assert html.escape("Los Angeles, CA") == "Los Angeles, CA"
