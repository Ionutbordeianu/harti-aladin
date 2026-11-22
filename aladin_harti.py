#!/usr/bin/env python3
"""
Hărți meteo ALADIN CHMI 2,3 km pentru România (pagina index.html din acest folder).

Sursa: modelul ALADIN al Institutului Hidrometeorologic Ceh (ČHMÚ / CHMI), domeniul
Europa Centrală la 2,3 km, republicat de Open-Meteo pe serverul public Amazon
(openmeteo.s3.amazonaws.com, format .om). Fișierele .om sunt împărțite pe bucăți de
32 x 32 puncte, așa că scriptul descarcă DOAR bucățile care acoperă România
(~5 MB pe oră de prognoză, nu 29 MB), cu cereri HTTP Range în paralel.

Ce face:
  1. citește latest.json și folosește doar rulările de 00 și 12 UTC (CHMI rulează și la 06/18)
  2. pentru 28 de momente (+1..+12h orar, +15..+48h la 3 ore, +54..+72h la 6 ore)
     citește parametrii de la sol și 9 niveluri de presiune, doar zona României
  3. citește precipitațiile/ninsoarea pentru fiecare oră (ca să le adune pe intervale)
  4. reproiectează din Lambert pe o grilă geografică 0,03° x 0,021° (~2,3 km)
     peste 19–30°E, 42,5–49,5°N; calculează izoterma de 0°C și grosimea 500–1000 hPa
  5. scrie data/ (ca la HartiALARO4) și, dacă există github.txt, publică pe GitHub Pages

Date: ČHMÚ – Czech Hydrometeorological Institute, ALADIN model, CC BY 4.0;
      via Open-Meteo (open-meteo.com, CC BY 4.0).
Jurnal: aladin_harti.log
"""
import base64
import bz2
import gzip
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import omfiles

try:
    import requests
except ImportError:          # merge și fără, doar mai încet
    requests = None

S3 = "https://openmeteo.s3.amazonaws.com/data_spatial/chmi_aladin_central_europe_2km/"
CHMI_DIR = "https://opendata.chmi.cz/meteorology/weather/nwp_aladin/Lambert_2.3km/00/"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FORMAT = "aladin1"
WORKERS = 16

# grila nativă ALADIN (Lambert conformal, sferă R=6371229)
NXc, NYc, DXc = 1053, 837, 2325.0
LA1, LO1, LOV, LATIN = 38.599, 1.334, 17.0, 46.244
# grila de ieșire
LON0, LAT0, DLON, DLAT = 19.0, 42.5, 0.03, 0.021
CW, CH = 367, 334                      # 19.0–29.98°E, 42.5–49.49°N
STEPS = list(range(1, 13)) + list(range(15, 49, 3)) + [54, 60, 66, 72]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 200]          # salvate (la ~9 km)
FRZ_LEVELS = [1000, 950, 925, 850, 800, 700, 600, 500, 400]      # pentru izoterma 0°C
UA_STEP = 4
MG_STEP, MG_TILE = 3, 16

SURF_SRC = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m",
            "wind_gusts_10m", "pressure_msl", "cloud_cover", "cloud_cover_low", "cloud_cover_mid",
            "cloud_cover_high", "cape", "visibility"]
ACC_SRC = ["precipitation", "snowfall_water_equivalent"]
UA_SRC = {"t": "temperature", "z": "geopotential_height", "u": "wind_u_component",
          "v": "wind_v_component", "rh": "relative_humidity"}

# pas de cuantizare, ales după precizia afișată (None = automat)
STEP = {"t2m": 0.1, "rh2m": 1, "u10": 0.25, "v10": 0.25, "gust": 0.5, "mslp": 0.1,
        "prectot": 0.02, "lssnow": 0.02, "tcc": 5, "lcc": 5, "mcc": 5, "hcc": 5,
        "cape": 10, "vis": 100, "frz": 50, "thick": 0.5}
UA_STEP_BY = {"t": 0.2, "z": 5, "u": 0.5, "v": 0.5, "rh": 2}
MG_VARS = ["t2m", "rh2m", "u10", "v10", "gust", "mslp", "prectot", "lssnow", "tcc", "lcc", "mcc", "hcc", "frz"]

log = logging.getLogger("aladin")


# ---------------- proiecția Lambert ----------------
_R = 6371229.0
_lat1 = np.radians(LATIN)
_n = np.sin(_lat1)
_F = np.cos(_lat1) * np.tan(np.pi / 4 + _lat1 / 2) ** _n / _n
_rho = lambda lat: _R * _F / np.tan(np.pi / 4 + lat / 2) ** _n
_rho0 = _rho(_lat1)


def _fwd(lon, lat):
    lon = np.radians(lon); lat = np.radians(lat)
    r = _rho(lat); th = _n * (lon - np.radians(LOV))
    return r * np.sin(th), _rho0 - r * np.cos(th)


_X0, _Y0 = _fwd(LO1, LA1)
_lon, _lat = np.meshgrid(LON0 + np.arange(CW) * DLON, LAT0 + np.arange(CH) * DLAT)
_x, _y = _fwd(_lon, _lat)
FI = (_x - _X0) / DXc
FJ = (_y - _Y0) / DXc
I0, I1 = int(np.floor(FI.min())) - 1, int(np.ceil(FI.max())) + 2
J0, J1 = int(np.floor(FJ.min())) - 1, int(np.ceil(FJ.max())) + 2
I1, J1 = min(I1, NXc), min(J1, NYc)
_ii = FI - I0; _jj = FJ - J0
_ia = np.clip(np.floor(_ii).astype(int), 0, I1 - I0 - 2)
_ja = np.clip(np.floor(_jj).astype(int), 0, J1 - J0 - 2)
_wi = (_ii - _ia).astype(np.float32); _wj = (_jj - _ja).astype(np.float32)


def regrid(s):
    """subdomeniul nativ [J0:J1, I0:I1] -> grila geografică (biliniar)"""
    s = np.nan_to_num(np.asarray(s, dtype=np.float32))
    a = s[_ja, _ia]; b = s[_ja, _ia + 1]; c = s[_ja + 1, _ia]; d = s[_ja + 1, _ia + 1]
    return ((a * (1 - _wi) + b * _wi) * (1 - _wj) + (c * (1 - _wi) + d * _wi) * _wj).astype(np.float32)


# ---------------- acces HTTP pe bucăți ----------------
_tls = threading.local()
STATS = {"req": 0, "bytes": 0}
_stats_lock = threading.Lock()


def http(url, rng=None, tries=6, want_total=False):
    """rng = (start, end_inclusiv) sau ("tail", n) pentru ultimii n octeți"""
    for k in range(tries):
        try:
            headers = {"User-Agent": "HartiAladin/1.0"}
            total = None
            if rng and rng[0] == "tail":
                headers["Range"] = "bytes=-%d" % rng[1]
            elif rng:
                headers["Range"] = "bytes=%d-%d" % rng
            if requests is not None:
                s = getattr(_tls, "s", None)
                if s is None:
                    s = _tls.s = requests.Session()
                r = s.get(url, headers=headers, timeout=120)
                if r.status_code not in (200, 206):
                    raise IOError("HTTP %d pentru %s" % (r.status_code, url))
                data = r.content
                total = r.headers.get("Content-Range", "/").split("/")[-1]
            else:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                    total = (r.headers.get("Content-Range") or "/").split("/")[-1]
            with _stats_lock:
                STATS["req"] += 1; STATS["bytes"] += len(data)
            if want_total:
                return data, int(total)
            return data
        except Exception as e:
            if k == tries - 1:
                raise
            if k >= 1:
                log.warning("eroare %s (%s), reîncerc", url.rsplit("/", 1)[-1], e)
            _tls.s = None                      # conexiune nouă la următoarea încercare
            time.sleep(2 * (k + 1) + (hash(url) % 7) / 7)


class RangeFS:
    """sistem de fișiere minimal pentru omfiles: citește doar octeții ceruți
    (+ o rezervă de 4 KB, pentru că bucățile vecine sunt lipite în fișier)"""
    READAHEAD = 4096

    def __init__(self, size, tail=None):
        self._size = size
        self.cache = {}
        self.tail = tail            # (start, octeți) de la sfârșitul fișierului

    def size(self, path):
        return self._size

    def cat_file(self, path, start=None, end=None, **kw):
        start = start or 0
        end = self._size if end is None else end
        for blk in (self.cache.get(path), self.tail):
            if blk and blk[0] <= start and end <= blk[0] + len(blk[1]):
                return blk[1][start - blk[0]:end - blk[0]]
        e2 = min(self._size, max(end, start + self.READAHEAD))
        data = http(path, (start, e2 - 1))
        self.cache[path] = (start, data)
        return data[:end - start]


_tails = {}
_tails_lock = threading.Lock()
TAIL = 65536


def file_tail(url):
    """mărimea fișierului și ultimii 64 KB (acolo e cuprinsul: variabile și tabele de poziții)"""
    with _tails_lock:
        if url in _tails:
            return _tails[url]
    data, total = http(url, ("tail", TAIL), want_total=True)
    res = (total, (total - len(data), data))
    with _tails_lock:
        _tails[url] = res
        if len(_tails) > 200:
            _tails.pop(next(iter(_tails)))
    return res


def read_vars(url, names):
    """citește variabilele cerute dintr-un fișier .om, doar zona României"""
    size, tail = file_tail(url)
    fs = RangeFS(size, tail)
    rd = omfiles.OmFileReader.from_fsspec(fs, url)
    out = {}
    for nm in names:
        v = rd.get_child_by_name(nm)
        out[nm] = regrid(v[J0:J1, I0:I1])
    return out


RUN_HOURS = (0, 12)          # folosim doar rulările de 00 și 12 UTC


def latest_run():
    """ultima rulare de 00 sau 12 UTC (dacă cea mai nouă e de 06/18, o luăm pe cea dinainte)"""
    j = json.loads(http(S3 + "latest.json"))
    ref = datetime.strptime(j["reference_time"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    if ref.hour not in RUN_HOURS:
        ref = ref - timedelta(hours=6)
        j = json.loads(http(S3 + ref.strftime("%Y/%m/%d/%H%MZ/") + "meta.json"))
    valid = [datetime.strptime(v, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc) for v in j["valid_times"]]
    return ref, valid, j.get("completed", True)


def om_url(ref, t):
    return S3 + ref.strftime("%Y/%m/%d/%H%MZ/") + t.strftime("%Y-%m-%dT%H%M") + ".om"


# ---------------- relieful modelului (o singură dată, de la CHMI) ----------------
def orography():
    html = http(CHMI_DIR).decode("utf-8", "replace")
    names = sorted(set(re.findall(r'href="(ALADLAMB4opendata_\d{10}_SURFGEOPOTENTIEL\.grb\.bz2)"', html)))
    if not names:
        raise RuntimeError("nu găsesc SURFGEOPOTENTIEL la CHMI")
    buf = bz2.decompress(http(CHMI_DIR + names[-1]))
    return regrid(decode_grib1_lambert(buf) / 9.80665)


def decode_grib1_lambert(d):
    """un singur mesaj GRIB1, grilă Lambert NXc x NYc, împachetare simplă"""
    u24 = lambda i: (d[i] << 16) | (d[i + 1] << 8) | d[i + 2]
    if d[:4] != b"GRIB" or d[7] != 1:
        raise ValueError("nu e GRIB1")
    p = 8; pl = u24(p); P = d[p:p + pl]
    D = s16(P[26], P[27]); p += pl
    if P[7] & 0x80:
        g = p
        nx, ny = (d[g + 6] << 8) | d[g + 7], (d[g + 8] << 8) | d[g + 9]
        if d[g + 5] != 3 or (nx, ny) != (NXc, NYc):
            raise ValueError("grilă neașteptată")
        p += u24(p)
    B = d[p:p + u24(p)]
    E = s16(B[4], B[5]); R = ibm_float(B[6:10]); nb = B[10]
    x = unpack_bits(bytes(B[11:]), nb, NXc * NYc).reshape(NYc, NXc)
    v = (R + x * 2.0 ** E) * 10.0 ** (-D)
    return v[J0:J1, I0:I1]


# ---------------- calcule ----------------
def freezing_level(t2m, zs, T, Z):
    """altitudinea (m) a izotermei de 0°C, de la sol în sus (T în °C, Z în m, niveluri de jos în sus)"""
    prev_h = zs + 2.0
    prev_t = t2m.astype(np.float64)
    res = np.full(t2m.shape, np.nan)
    res[prev_t <= 0] = 0.0
    for k in range(T.shape[0]):
        h = Z[k].astype(np.float64); t = T[k].astype(np.float64)
        above = h > prev_h + 1
        t = np.where(above, t, prev_t); h = np.where(above, h, prev_h)
        cross = np.isnan(res) & (prev_t > 0) & (t <= 0) & above
        w = np.where(cross, prev_t / np.maximum(prev_t - t, 1e-6), 0)
        res = np.where(cross, prev_h + w * (h - prev_h), res)
        prev_t, prev_h = t, h
    res[np.isnan(res)] = prev_h[np.isnan(res)]
    return res.astype(np.float32)


def write_hour(tmp, h, f):
    out = {"h": h, "f": {}}
    for name in STEP:
        k0, st, d = enc_field(f[name], STEP[name])
        out["f"][name] = {"k0": k0, "st": st, "p1": f["p1"], "p2": h, "d": d}
    write_js(os.path.join(tmp, "h%02d.js" % h), "ALARO_DATA", h, out)
    ua = {"h": h, "f": {}}
    for short in UA_SRC:
        for L in LEVELS:
            v = f["%s%d" % (short, L)][::UA_STEP, ::UA_STEP]
            k0, st, d = enc_field(v, UA_STEP_BY[short])
            ua["f"]["%s%d" % (short, L)] = {"k0": k0, "st": st, "d": d}
    write_js(os.path.join(tmp, "h%02d_p.js" % h), "ALARO_UA", h, ua)


def write_orog(tmp, hgt):
    k0, st, d = enc_field(hgt, 1)
    rec = {"w": CW, "h": CH, "k0": k0, "st": st, "d": d}
    with open(os.path.join(tmp, "orog.js"), "w", encoding="utf-8") as fh:
        fh.write("window.ALARO_OROG=" + json.dumps(rec, separators=(",", ":")) + ";\n")


def load_orog_cached():
    """relieful e static: îl păstrăm între rulări în orog.npy"""
    p = os.path.join(HERE, "orog_aladin.npy")
    if os.path.exists(p):
        a = np.load(p)
        if a.shape == (CH, CW):
            return a
    a = orography()
    np.save(p, a)
    log.info("relief descărcat de la CHMI")
    return a

# ---- comune (din alaro_harti.py) ----
def ibm_float(b):
    s = -1.0 if b[0] & 0x80 else 1.0
    e = (b[0] & 0x7F) - 64
    m = ((b[1] << 16) | (b[2] << 8) | b[3]) / 16777216.0
    return s * m * 16.0 ** e


def s16(hi, lo):
    v = ((hi & 0x7F) << 8) | lo
    return -v if hi & 0x80 else v


def unpack_bits(data, nb, n):
    """valori întregi fără semn de câte nb biți, n bucăți"""
    if nb == 0:
        return np.zeros(n, dtype=np.int64)
    if nb == 8:
        return np.frombuffer(data, dtype=np.uint8, count=n).astype(np.int64)
    if nb == 16:
        return np.frombuffer(data, dtype=">u2", count=n).astype(np.int64)
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8, count=(n * nb + 7) // 8))[: n * nb]
    bits = bits.reshape(n, nb).astype(np.int64)
    return bits @ (1 << np.arange(nb - 1, -1, -1, dtype=np.int64))


def med_residual(x):
    """predicție din vecini (stânga, sus, stânga-sus) ca în LOCO-I; rămâne doar diferența"""
    a = np.zeros_like(x); b = np.zeros_like(x); c = np.zeros_like(x)
    a[:, 1:] = x[:, :-1]; b[1:, :] = x[:-1, :]; c[1:, 1:] = x[:-1, :-1]
    mx = np.maximum(a, b); mn = np.minimum(a, b)
    p = np.where(c >= mx, mn, np.where(c <= mn, mx, a + b - c))
    p[0, 1:] = x[0, :-1]; p[1:, 0] = x[:-1, 0]; p[0, 0] = 0
    return x - p


def pack_residual(r):
    """zigzag pe 16 biți, apoi octeții mici și cei mari separat (se arhivează mai bine)"""
    z = (r << 1) ^ (r >> 63)
    if z.max() >= 65536:
        raise ValueError("valori în afara intervalului")
    z = z.astype(np.uint16).ravel()
    return np.concatenate([(z & 255).astype(np.uint8), (z >> 8).astype(np.uint8)]).tobytes()


def enc_field(v, st):
    if st is None:
        rng = float(np.nanmax(v) - np.nanmin(v)) or 1.0
        st = float("%.2g" % (rng / 200))
    k = np.round(v.astype(np.float64) / st).astype(np.int64)
    k0 = int(k.min())
    return k0, st, base64.b64encode(gzip.compress(pack_residual(med_residual(k - k0)), 9)).decode()


def write_js(path, var, key, obj):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("window.%s=window.%s||{};window.%s[%s]=" % (var, var, var, json.dumps(key))
                 + json.dumps(obj, separators=(",", ":")) + ";\n")


def write_meteograms(tmp, mg, nh):
    os.makedirs(os.path.join(tmp, "mg"))
    ny, nx = mg[MG_VARS[0]].shape[1:]
    for ty in range(0, ny, MG_TILE):
        for tx in range(0, nx, MG_TILE):
            th, tw = min(MG_TILE, ny - ty), min(MG_TILE, nx - tx)
            obj = {"tw": tw, "th": th, "v": {}}
            for name in MG_VARS:
                a = mg[name][:, ty:ty + th, tx:tx + tw].astype(np.float64)
                st = STEP[name] or 0.01
                k = np.round(a / st).astype(np.int64)
                k0 = int(k.min())
                r = np.stack([med_residual(k[n] - k0) for n in range(k.shape[0])])
                obj["v"][name] = {"k0": k0, "st": st, "d": base64.b64encode(
                    gzip.compress(pack_residual(r), 9)).decode()}
            key = "%d_%d" % (tx // MG_TILE, ty // MG_TILE)
            write_js(os.path.join(tmp, "mg", "m_%s.js" % key), "ALARO_MG", key, obj)



# ---------------- istoric: ultimele rulări, fiecare în data/<rulare>/ ----------------
KEEP_RUNS = 4
_RUN_RE = re.compile(r"^\d{10}$")


def _read_meta_file(path):
    with open(path, encoding="utf-8") as fh:
        return json.loads(fh.read().split("=", 1)[1].rstrip().rstrip(";"))


def hist_runs():
    """rulările păstrate, cea mai nouă prima"""
    if not os.path.isdir(DATA):
        return []
    return sorted([d for d in os.listdir(DATA) if _RUN_RE.match(d) and os.path.exists(os.path.join(DATA, d, "meta.js"))],
                  reverse=True)


def hist_index():
    runs = []
    for r in hist_runs():
        try:
            m = _read_meta_file(os.path.join(DATA, r, "meta.js"))
            runs.append({"run": r, "updated": m.get("updated"), "h0": m["hours"][0], "h1": m["hours"][-1]})
        except Exception:
            log.warning("rularea %s din istoric e incompletă, o ignor", r)
    with open(os.path.join(DATA, "runs.js"), "w", encoding="utf-8") as fh:
        fh.write("window.ALARO_RUNS=" + json.dumps(runs) + ";\n")


def hist_migrate():
    """mută datele din vechiul format (direct în data/) în data/<rulare>/"""
    flat = os.path.join(DATA, "meta.js")
    if not os.path.exists(flat):
        return
    try:
        run = str(_read_meta_file(flat).get("run"))
    except Exception:
        run = ""
    if not _RUN_RE.match(run):
        return
    dst = os.path.join(DATA, run)
    if os.path.isdir(dst):
        shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    for name in os.listdir(DATA):
        if name == run or _RUN_RE.match(name) or name == "runs.js":
            continue
        os.rename(os.path.join(DATA, name), os.path.join(dst, name))
    log.info("datele existente au fost mutate în istoric (data/%s)", run)
    hist_index()


def hist_install(tmp, run):
    """pune rularea nouă în data/<rulare>/, păstrează doar ultimele KEEP_RUNS"""
    os.makedirs(DATA, exist_ok=True)
    hist_migrate()
    dst = os.path.join(DATA, run)
    old_dir = dst + "_vechi"
    shutil.rmtree(old_dir, ignore_errors=True)
    if os.path.isdir(dst):
        os.rename(dst, old_dir)
    os.rename(tmp, dst)
    shutil.rmtree(old_dir, ignore_errors=True)
    for r in hist_runs()[KEEP_RUNS:]:
        shutil.rmtree(os.path.join(DATA, r), ignore_errors=True)
        log.info("rularea %s a ieșit din istoric", r)
    for name in os.listdir(DATA):          # resturi de la întreruperi
        if name.endswith("_vechi"):
            shutil.rmtree(os.path.join(DATA, name), ignore_errors=True)
    hist_index()


def current_meta():
    try:
        hist_migrate()
    except Exception:
        log.exception("mutarea în istoric a eșuat")
    runs = hist_runs()
    try:
        return _read_meta_file(os.path.join(DATA, runs[0], "meta.js")) if runs else None
    except Exception:
        return None


def _git_sha(data):
    import hashlib
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def gh_existing(conf):
    """fișierele deja publicate (cale -> sha), ca să trimitem doar ce s-a schimbat"""
    try:
        ref = gh(conf, "GET", "/git/refs/heads/gh-pages", ok404=True)
        if ref.get("_status") or "object" not in ref:
            return {}
        com = gh(conf, "GET", "/git/commits/" + ref["object"]["sha"])
        t = gh(conf, "GET", "/git/trees/%s?recursive=1" % com["tree"]["sha"])
        if t.get("truncated"):
            return {}
        return {e["path"]: e["sha"] for e in t.get("tree", []) if e.get("type") == "blob"}
    except Exception as e:
        log.warning("nu pot citi ce e deja publicat (%s), trimit tot", e)
        return {}


def gh_blob_sha(conf, existing, local):
    with open(local, "rb") as fh:
        data = fh.read()
    sha = _git_sha(data)
    if existing.get("__any__", {}).get(sha):
        return sha, False
    b = gh(conf, "POST", "/git/blobs", {"content": base64.b64encode(data).decode(), "encoding": "base64"})
    return b["sha"], True


# ---------------- publicare pe GitHub Pages (opțional) ----------------
GH_API = os.environ.get("ALARO_GH_API", "https://api.github.com")
GH_CONF = os.path.join(HERE, "github.txt")
GH_MARK = os.path.join(HERE, "github_publicat.txt")


def gh_config():
    if not os.path.exists(GH_CONF):
        return None
    conf = {}
    with open(GH_CONF, encoding="utf-8-sig") as fh:
        for line in fh:
            if "=" in line:
                k, v = line.split("=", 1)
                conf[k.strip().lower()] = v.strip()
    if conf.get("repo") and conf.get("token"):
        return conf
    log.warning("github.txt incomplet (trebuie repo= și token=)")
    return None


def gh(conf, method, path, data=None, ok404=False):
    for i in range(5):
        try:
            return _gh1(conf, method, path, data, ok404)
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            if i == 4:
                raise
            log.warning("GitHub: conexiune întreruptă (%s), reîncerc...", e)
            time.sleep(5 * (i + 1))


def _gh1(conf, method, path, data=None, ok404=False):
    url = "%s/repos/%s%s" % (GH_API, conf["repo"], path)
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, method=method, headers={
        "Authorization": "Bearer " + conf["token"], "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "HartiAladin/1.0",
        "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            txt = r.read().decode() or "{}"
            return json.loads(txt)
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "replace")[:300]
        if ok404 and e.code in (403, 404, 409, 422):
            return {"_status": e.code, "_msg": msg}
        raise RuntimeError("GitHub %s %s -> %s %s" % (method, path, e.code, msg))


def publish(force=False):
    conf = gh_config()
    if not conf:
        return
    meta = current_meta()
    if not meta:
        return
    import hashlib
    pages = [n for n in ("index.html", "comparare.html") if os.path.exists(os.path.join(HERE, n))]
    ph = hashlib.md5(b"".join(open(os.path.join(HERE, n), "rb").read() for n in pages)).hexdigest()[:10]
    stamp = "%s %s %s" % (meta.get("run"), meta.get("updated"), ph)
    if not force and os.path.exists(GH_MARK):
        with open(GH_MARK, encoding="utf-8") as fh:
            if fh.read().strip() == stamp:
                return
    log.info("public pe GitHub (%s)...", conf["repo"])
    files = [(n, os.path.join(HERE, n)) for n in pages]
    for root, _dirs, names in os.walk(DATA):
        for name in sorted(names):
            if name.endswith(".js"):
                full = os.path.join(root, name)
                files.append(("data/" + os.path.relpath(full, DATA).replace(os.sep, "/"), full))
    # un depozit gol nu acceptă încă obiecte git: îl inițializăm cu un README
    first = gh(conf, "POST", "/git/blobs", {"content": "", "encoding": "utf-8"}, ok404=True)
    if first.get("_status") == 409:
        gh(conf, "PUT", "/contents/README.md", {"message": "init", "content": base64.b64encode(
            "Hărți meteo ALARO-4 (date ANM, CC BY 4.0)\n".encode()).decode()})
    tree = [{"path": ".nojekyll", "mode": "100644", "type": "blob",
             "sha": gh(conf, "POST", "/git/blobs", {"content": "", "encoding": "utf-8"})["sha"]}]
    existing = gh_existing(conf)
    existing["__any__"] = {s: True for p, s in existing.items() if p != "__any__"}
    sent = 0
    for path, local in files:
        sha, up = gh_blob_sha(conf, existing, local)
        sent += up
        tree.append({"path": path, "mode": "100644", "type": "blob", "sha": sha})
    log.info("fișiere: %d în total, %d trimise acum", len(files), sent)
    tsha = gh(conf, "POST", "/git/trees", {"tree": tree})["sha"]
    csha = gh(conf, "POST", "/git/commits", {"message": "date " + stamp, "tree": tsha, "parents": []})["sha"]
    r = gh(conf, "PATCH", "/git/refs/heads/gh-pages", {"sha": csha, "force": True}, ok404=True)
    if r.get("_status"):
        gh(conf, "POST", "/git/refs", {"ref": "refs/heads/gh-pages", "sha": csha})
    pages = gh(conf, "GET", "/pages", ok404=True)
    if pages.get("_status") in (403, 404):
        r = gh(conf, "POST", "/pages", {"source": {"branch": "gh-pages", "path": "/"}}, ok404=True)
        if r.get("_status"):
            log.warning("activează GitHub Pages manual: Settings > Pages > Branch: gh-pages / (root)")
    owner, name = conf["repo"].split("/", 1)
    with open(GH_MARK, "w", encoding="utf-8") as fh:
        fh.write(stamp)
    log.info("publicat: https://%s.github.io/%s/", owner.lower(), name)




# ---------------- actualizare ----------------
def step_names():
    names = list(SURF_SRC)
    for short, src in UA_SRC.items():
        for L in LEVELS:
            names.append("%s_%dhPa" % (src, L))
    for L in FRZ_LEVELS:
        for src in ("temperature", "geopotential_height"):
            nm = "%s_%dhPa" % (src, L)
            if nm not in names:
                names.append(nm)
    return names


def chunks(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def build_fields(raw, acc, p1, h, zs):
    f = {"p1": p1}
    f["t2m"] = raw["temperature_2m"]
    f["rh2m"] = np.clip(raw["relative_humidity_2m"], 0, 100)
    ws = raw["wind_speed_10m"]; wd = np.radians(raw["wind_direction_10m"])
    f["u10"] = (-ws * np.sin(wd)).astype(np.float32); f["v10"] = (-ws * np.cos(wd)).astype(np.float32)
    f["gust"] = raw["wind_gusts_10m"]
    f["mslp"] = raw["pressure_msl"]
    f["tcc"], f["lcc"], f["mcc"], f["hcc"] = (raw[k] for k in ("cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"))
    f["cape"] = np.clip(raw["cape"], 0, None); f["vis"] = np.clip(raw["visibility"], 0, None)
    f["prectot"] = np.clip(sum(acc[k]["precipitation"] for k in range(p1 + 1, h + 1)), 0, None)
    f["lssnow"] = np.clip(sum(acc[k]["snowfall_water_equivalent"] for k in range(p1 + 1, h + 1)), 0, None)
    T = np.stack([raw["temperature_%dhPa" % L] for L in FRZ_LEVELS])
    Z = np.stack([raw["geopotential_height_%dhPa" % L] for L in FRZ_LEVELS])
    f["frz"] = freezing_level(f["t2m"], zs, T, Z)
    f["thick"] = ((raw["geopotential_height_500hPa"] - raw["geopotential_height_1000hPa"]) / 10).astype(np.float32)
    for short, src in UA_SRC.items():
        for L in LEVELS:
            f["%s%d" % (short, L)] = raw["%s_%dhPa" % (src, L)]
    return f


def update(force):
    ref, valid, completed = latest_run()
    run = ref.strftime("%Y%m%d%H")
    avail = {int(round((t - ref).total_seconds() / 3600)): t for t in valid}
    steps = [s for s in STEPS if s in avail and all(k in avail for k in range(1, s + 1))]
    if not steps:
        raise RuntimeError("rularea %s nu are încă ore disponibile" % run)
    old = current_meta()
    if not force and old and old.get("format") == FORMAT and old.get("run") == run and old.get("hours") == steps:
        log.info("deja la zi: rularea %s, %d momente", run, len(steps))
        return
    log.info("rularea %s (%s), momente: %s", run, "completă" if completed else "în curs", steps)
    zs = load_orog_cached()
    tmp = DATA + "_nou"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        # precipitații și ninsoare pentru fiecare oră (se adună pe intervale)
        hrs = list(range(1, steps[-1] + 1))
        accl = list(ex.map(lambda k: read_vars(om_url(ref, avail[k]), ACC_SRC), hrs))
        acc = dict(zip(hrs, accl))
        log.info("precipitații orare citite (%d ore, %.1f MB)", len(hrs), STATS["bytes"] / 1e6)
        names = step_names()
        mg = {}
        pint = []
        p1 = 0
        def submit(h):
            url = om_url(ref, avail[h])
            return [ex.submit(read_vars, url, c) for c in chunks(names, 4)]
        pending = {steps[0]: submit(steps[0])}
        for n, h in enumerate(steps):
            if n + 1 < len(steps):
                pending[steps[n + 1]] = submit(steps[n + 1])     # următorul moment se descarcă în paralel
            raw = {}
            for fut in pending.pop(h):
                raw.update(fut.result())
            f = build_fields(raw, acc, p1, h, zs)
            write_hour(tmp, h, f)
            pint.append([p1, h])
            for name in MG_VARS:
                sub = f[name][::MG_STEP, ::MG_STEP]
                if name not in mg:
                    mg[name] = np.zeros((len(steps),) + sub.shape, dtype=np.float32)
                mg[name][n] = sub
            log.info("+%02dh gata (%d cereri, %.1f MB până acum)", h, STATS["req"], STATS["bytes"] / 1e6)
            p1 = h
            del raw, f
    write_orog(tmp, zs)
    write_meteograms(tmp, mg, len(steps))
    meta = {"format": FORMAT, "model": "aladin", "run": run, "hours": steps, "pint": pint, "levels": LEVELS,
            "grid": {"w": CW, "h": CH, "lon0": LON0, "lat0": LAT0, "dlon": DLON, "dlat": DLAT},
            "ua_step": UA_STEP, "mg": {"step": MG_STEP, "tile": MG_TILE},
            "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    with open(os.path.join(tmp, "meta.js"), "w", encoding="utf-8") as fh:
        fh.write("window.ALARO_META=" + json.dumps(meta) + ";\n")
    hist_install(tmp, run)
    log.info("gata: %d momente în %.0f s, %d cereri, %.0f MB descărcați", len(steps), time.time() - t0, STATS["req"], STATS["bytes"] / 1e6)


def main():
    force = "--force" in sys.argv
    try:
        update(force)
    finally:
        try:
            publish("--publica" in sys.argv)
        except Exception:
            log.exception("publicarea pe GitHub a eșuat (se reîncearcă la următoarea rulare)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(os.path.join(HERE, "aladin_harti.log"), encoding="utf-8")]
                        + ([logging.StreamHandler(sys.stdout)] if sys.stdout else []))
    try:
        main()
    except Exception:
        log.exception("actualizarea a eșuat")
        sys.exit(1)
