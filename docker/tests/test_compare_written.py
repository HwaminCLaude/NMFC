"""verify 판정 테스트 — compare_written(신선도+class별 내용), png_compare, compare_float_platform."""

import copy
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

DOCKER = Path(__file__).resolve().parents[1]
REPO = DOCKER.parent
FIX = Path(__file__).resolve().parent / "fixtures"
if str(DOCKER) not in sys.path:
    sys.path.insert(0, str(DOCKER))

import nmfc_compare as C  # noqa: E402


def fx(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def fails(fs):
    return [f for f in fs if f["status"] == "FAIL"]


def make_png(path: Path, seed=0, size=(20, 10), compress_level=6):
    from PIL import Image

    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size[1], size[0], 4), dtype=np.uint8)
    Image.fromarray(arr, "RGBA").save(path, compress_level=compress_level)
    return arr


# --------------------------------------------------------------------------- png_compare

def test_png_same_file_and_reencoded_are_pixel_equal(tmp_path):
    from PIL import Image

    a, b = tmp_path / "a.png", tmp_path / "b.png"
    make_png(a, compress_level=9)
    with Image.open(a) as im:
        im.save(b, compress_level=0)
    assert a.read_bytes() != b.read_bytes()           # 압축 바이트는 다르다
    assert C.png_compare(a, a)["pixel_equal"] is True
    r = C.png_compare(a, b)
    assert r["pixel_equal"] is True and r["size_equal"] is True
    assert r["changed_ratio"] == 0.0 and r["max_channel_diff"] == 0


def test_png_one_pixel_changed_is_not_equal(tmp_path):
    from PIL import Image

    a, b = tmp_path / "a.png", tmp_path / "b.png"
    arr = make_png(a)
    arr2 = arr.copy()
    arr2[3, 7, 1] = (int(arr2[3, 7, 1]) + 1) % 256
    Image.fromarray(arr2, "RGBA").save(b)
    r = C.png_compare(a, b)
    assert r["pixel_equal"] is False and r["size_equal"] is True
    assert r["changed_ratio"] == pytest.approx(1 / 200)
    assert r["max_channel_diff"] >= 1


def test_png_size_mismatch(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    make_png(a, size=(20, 10))
    make_png(b, size=(21, 10))
    r = C.png_compare(a, b)
    assert r["size_equal"] is False and r["pixel_equal"] is False


def test_png_rgb_vs_rgba_same_pixels_equal(tmp_path):
    from PIL import Image

    a, b = tmp_path / "a.png", tmp_path / "b.png"
    arr = np.random.default_rng(1).integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(a)
    Image.fromarray(np.dstack([arr, np.full((8, 8), 255, np.uint8)]), "RGBA").save(b)
    assert C.png_compare(a, b)["pixel_equal"] is True


def test_png_real_repo_figure_self_compare():
    p = REPO / "results" / "openset_noise_grid" / "curves_auc.png"
    if not p.exists():
        pytest.skip("그림 없음")
    assert C.png_compare(p, p)["pixel_equal"] is True


# --------------------------------------------------------------------------- float_platform

def test_theory_audit_t3_passes_float_platform():
    ref, t3 = fx("ref/theory_audit.json"), fx("theory_audit_t3.json")
    fs = C.compare_float_platform(ref, t3)
    assert not fails(fs)
    assert {f["path"] for f in fs} == {"invariance_maxerr/직교변환", "checks/8/detail"}
    assert all(f["status"] == "PASS" for f in fs)


def _mut(data, path, value):
    d = data
    for p in path[:-1]:
        d = d[p]
    if callable(value):
        d[path[-1]] = value(d[path[-1]])
    else:
        d[path[-1]] = value


@pytest.mark.parametrize("path,value", [
    (("R", "R"), lambda v: v + 1e-6),                                   # 숫자 1e-6 → FAIL
    (("config", "sigma2"), lambda v: v * (1 + 1e-8)),                    # rel 1e-9 초과
    (("checks", 0, "name"), "P1 상한 E⁺ ≤ d_min² (수정)"),               # 이름은 정확 일치
    (("checks", 0, "passed"), False),
    (("checks", 0, "detail"), "max(E⁺−d_min²)=-2.195e+01"),             # 문자열 속 숫자 변화
    (("checks", 0, "detail"), "min(E⁺−d_min²)=-2.194e+01"),             # 숫자를 뺀 텍스트 변화
    (("n_passed",), 15),
    (("n_total",), 17),
    (("config", "exp"), "c100v_mhrb"),
    (("config", "k1"), "3"),                                            # 형 변화
    (("sigma_limits", "sigma0_convergence", 0, "maxerr"), None),
])
def test_float_platform_mutations_fail(path, value):
    ref = fx("ref/theory_audit.json")
    new = copy.deepcopy(fx("theory_audit_t3.json"))
    _mut(new, path, value)
    assert fails(C.compare_float_platform(ref, new)), path


def test_float_platform_structure_changes_fail():
    ref = fx("ref/theory_audit.json")
    for mutate in (lambda d: d.pop("margin"), lambda d: d.__setitem__("new_key", 1),
                   lambda d: d["checks"].pop(), lambda d: d["sigma_limits"]["sigma0_convergence"].append({})):
        new = copy.deepcopy(ref)
        mutate(new)
        assert fails(C.compare_float_platform(ref, new))


def test_float_platform_tiny_noise_passes():
    ref = fx("ref/theory_audit.json")
    new = copy.deepcopy(ref)
    new["prop1_bounds"]["max_upper_violation"] += 1e-12           # rel 1e-9·21.9 안
    new["prototype_reduction_maxerr"] = 3.1e-15                  # abs 1e-12 안
    new["checks"][3]["detail"] = "max|Δ|=3.100e-15"
    fs = C.compare_float_platform(ref, new)
    assert fs and not fails(fs)


# --------------------------------------------------------------------------- compare_written

def _snap(root: Path, rels):
    out = {}
    for rel in rels:
        p = root / rel
        out[rel] = ({"exists": True, "mtime_ns": p.stat().st_mtime_ns, "sha256": C._norm_sha256(p)}
                    if p.exists() else {"exists": False, "mtime_ns": None, "sha256": None})
    return out


def _touch_later(p: Path, before_ns: int):
    t = before_ns + 2_000_000_000
    os.utime(p, ns=(t, t))


@pytest.fixture
def written(tmp_path):
    base, cur = tmp_path / "base", tmp_path / "cur"
    for root in (base, cur):
        (root / "results").mkdir(parents=True)
        (root / "figs").mkdir()
    shutil.copyfile(FIX / "ref" / "theory_audit.json", base / "results" / "theory_audit.json")
    shutil.copyfile(FIX / "ref" / "theory_audit.json", cur / "results" / "theory_audit.json")
    make_png(base / "figs" / "en.png", seed=3)
    make_png(cur / "figs" / "en.png", seed=3)
    make_png(base / "figs" / "ko.png", seed=4)
    make_png(cur / "figs" / "ko.png", seed=4)
    (base / "results" / "note.md").write_bytes(b"line1\nline2\n")
    (cur / "results" / "note.md").write_bytes(b"line1\nline2\n")
    writes = [{"path": "results/theory_audit.json", "class": "float_platform"},
              {"path": "figs/en.png", "class": "pixel_equal"},
              {"path": "figs/ko.png", "class": "font_dependent"},
              {"path": "results/note.md", "class": "exact"}]
    before = _snap(cur, [w["path"] for w in writes])
    return base, cur, writes, before


def _rewrite_all(cur, before, ta_bytes=None, en=None, ko=None, note=None):
    from PIL import Image

    (cur / "results" / "theory_audit.json").write_bytes(
        ta_bytes if ta_bytes is not None else (FIX / "theory_audit_t3.json").read_bytes())
    with Image.open(cur / "figs" / "en.png") as im:
        im.load()
        img = im.copy()
    if en is not None:
        img.putpixel((0, 0), en)
    img.save(cur / "figs" / "en.png", compress_level=1)          # 바이트는 바뀌어도 픽셀은 동일
    with Image.open(cur / "figs" / "ko.png") as im:
        im.load()
        img2 = im.copy()
    if ko is not None:
        img2.putpixel((1, 1), ko)
    img2.save(cur / "figs" / "ko.png", compress_level=1)
    (cur / "results" / "note.md").write_bytes(note if note is not None else b"line1\r\nline2\r\n")
    for rel, s in before.items():
        _touch_later(cur / rel, s["mtime_ns"])


def test_compare_written_all_pass(written):
    base, cur, writes, before = written
    _rewrite_all(cur, before)
    fs = C.compare_written(base, cur, writes, before)
    assert [f["status"] for f in fs] == ["PASS", "PASS", "REPORT", "PASS"], fs
    assert all(f["data"]["fresh"] for f in fs)
    assert fs[2]["data"]["pixel_equal"] is True and fs[3]["detail"] == "LF 정규화 sha256 일치"


def test_compare_written_not_rewritten_fails(written):
    base, cur, writes, before = written
    fs = C.compare_written(base, cur, writes, before)              # 아무것도 다시 쓰지 않음
    assert [f["status"] for f in fs] == ["FAIL"] * 4
    assert all("mtime_ns 불변" in f["detail"] for f in fs)


def test_compare_written_content_mismatch_fails(written):
    base, cur, writes, before = written
    bad_ta = json.loads((FIX / "theory_audit_t3.json").read_text(encoding="utf-8"))
    bad_ta["R"]["R"] += 1e-6
    _rewrite_all(cur, before, ta_bytes=json.dumps(bad_ta).encode("utf-8"), en=(9, 9, 9, 255),
                 ko=(9, 9, 9, 255), note=b"line1\nline2 changed\n")
    fs = C.compare_written(base, cur, writes, before)
    assert [f["status"] for f in fs] == ["FAIL", "FAIL", "REPORT", "FAIL"], fs
    assert fs[2]["data"]["pixel_equal"] is False and fs[2]["data"]["changed_ratio"] > 0


def test_compare_written_missing_snapshot_or_file_fails(written):
    base, cur, writes, before = written
    _rewrite_all(cur, before)
    partial = {k: v for k, v in before.items() if k != "results/note.md"}
    fs = C.compare_written(base, cur, writes, partial)
    assert fs[3]["status"] == "FAIL" and "사전 스냅샷" in fs[3]["detail"]
    (cur / "figs" / "en.png").unlink()
    fs = C.compare_written(base, cur, writes, before)
    assert fs[1]["status"] == "FAIL" and "파일이 없음" in fs[1]["detail"]
    (base / "figs" / "ko.png").unlink()
    fs = C.compare_written(base, cur, writes, before)
    assert fs[2]["status"] == "REPORT" and "기준본 없음" in fs[2]["detail"]


def test_compare_written_new_file_counts_as_fresh(written):
    base, cur, writes, before = written
    before = dict(before, **{"results/note.md": {"exists": False, "mtime_ns": None, "sha256": None}})
    fs = C.compare_written(base, cur, [writes[3]], before)
    assert fs[0]["status"] == "PASS" and fs[0]["data"]["fresh"] is True


def test_compare_written_unknown_class_fails(written):
    base, cur, writes, before = written
    _rewrite_all(cur, before)
    fs = C.compare_written(base, cur, [{"path": "results/note.md", "class": "pixel_close"}], before)
    assert fs[0]["status"] == "FAIL"
