#!/usr/bin/env python3
"""Render one 30-second vertical rain video (1080x1920, 30 fps) with rain sound.

Every video also has a thunderstorm (lightning flashes + rolling thunder), water
dripping from a roof edge (visible drops + drip sounds) and birds tweeting.

Usage:
    python3 make_rain.py            # renders the next day (counts files in videos/)
    python3 make_rain.py 7          # renders day 7
    python3 make_rain.py 7 --frames out_dir   # only writes 3 preview frames

Prints one JSON line: {"day", "file", "title", "description"}.
"""
import json
import math
import os
import random
import subprocess
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H, FPS, SECONDS = 1080, 1920, 30, 30
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "videos")

# preset: sky gradient (top, bottom), silhouette kind, bokeh colours, rain tint
PRESETS = {
    "city": dict(top=(8, 12, 30), bottom=(30, 34, 62), sil="skyline",
                 bokeh=[(255, 190, 90), (255, 120, 80), (120, 180, 255)], tint=(200, 215, 255)),
    "neon": dict(top=(14, 6, 34), bottom=(44, 14, 60), sil="skyline",
                 bokeh=[(255, 60, 170), (60, 220, 255), (170, 90, 255)], tint=(225, 205, 255)),
    "cozy": dict(top=(26, 18, 22), bottom=(70, 44, 30), sil="window",
                 bokeh=[(255, 170, 70), (255, 210, 130)], tint=(255, 235, 215)),
    "forest": dict(top=(22, 40, 38), bottom=(10, 28, 20), sil="pines",
                   bokeh=[], tint=(210, 235, 225)),
    "monsoon": dict(top=(70, 80, 88), bottom=(40, 62, 50), sil="hills",
                    bokeh=[], tint=(230, 238, 240)),
    "storm": dict(top=(12, 16, 28), bottom=(18, 34, 50), sil="waves",
                  bokeh=[], tint=(205, 220, 245)),
    "dawn": dict(top=(60, 70, 100), bottom=(150, 120, 110), sil="hills",
                 bokeh=[(255, 220, 170)], tint=(240, 235, 235)),
}

# (title, preset, rain intensity 1-3, heavy thunder) - every day has thunder; True = closer and more of it
DAYS = [
    ("Rain Over City Lights at Night 🌧️ 30 Seconds of Calm", "city", 2, False),
    ("Cozy Window Rain 🕯️ Relax in 30 Seconds", "cozy", 1, False),
    ("Heavy Monsoon Rain Over Green Hills 🌿", "monsoon", 3, False),
    ("Rain in a Misty Pine Forest 🌲", "forest", 2, False),
    ("Thunderstorm Over the Sea ⚡", "storm", 3, True),
    ("Neon City Rain 🌃 Night Vibes", "neon", 2, False),
    ("Soft Rain at Dawn 🌅 Peaceful Morning", "dawn", 1, False),
    ("Gentle Night Rain for Sleep 😴", "city", 1, False),
    ("Warm Lights & Rainy Window ☕", "cozy", 2, False),
    ("Forest Downpour 🌧️ Nature Sounds", "forest", 3, False),
    ("Monsoon Drizzle on the Hills 🍃", "monsoon", 1, False),
    ("Night Storm at Sea 🌊 Thunder & Rain", "storm", 2, True),
    ("Purple Neon Rain 💜 City at Midnight", "neon", 3, False),
    ("Rainy Sunrise Over the Valley 🌄", "dawn", 2, False),
    ("City Thunderstorm at Night ⚡🌧️", "city", 3, True),
    ("Fireside Glow & Rain 🔥 Ultimate Cozy", "cozy", 1, False),
    ("Light Rain Between the Pines 🌲 Calm", "forest", 1, False),
    ("Barish in the Hills 🌧️ Monsoon Mood", "monsoon", 2, False),
    ("Heavy Rain on Dark Waves 🌊", "storm", 3, False),
    ("Neon Drizzle 🌃 Lo-fi Rain Mood", "neon", 1, False),
    ("Morning Rain Over the Hills 🌦️", "dawn", 3, False),
    ("Midnight City Rain 🌙 Deep Relaxation", "city", 2, False),
    ("Rainy Evening by the Window 🪟", "cozy", 3, False),
    ("Thunder in the Forest ⚡🌲", "forest", 2, True),
    ("Monsoon Thunderstorm ⛈️ Heavy Barish", "monsoon", 3, True),
    ("Calm Rain on the Ocean 🌊 Sleep Sounds", "storm", 1, False),
    ("Pink & Blue Neon Rain 🌧️ Night Walk", "neon", 2, False),
    ("Quiet Rain at First Light 🌤️", "dawn", 1, False),
    ("Rain & City Bokeh ✨ Stress Relief", "city", 3, False),
    ("Last Rain of the Night 🌧️ Fall Asleep Fast", "cozy", 2, True),
]

DESCRIPTION = (
    "30 seconds of relaxing rain with distant thunder, water dripping from the roof and birdsong "
    "to calm your mind. Loop it for sleep, study or stress relief.\n\n"
    "#rain #rainsounds #thunderstorm #birdsong #relaxing #sleep #barish #shorts"
)


def gradient(top, bottom):
    t = np.linspace(0, 1, H)[:, None, None]
    col = np.array(top)[None, None, :] * (1 - t) + np.array(bottom)[None, None, :] * t
    return np.repeat(col, W, axis=1)


def build_background(p, rng):
    img = Image.fromarray(gradient(p["top"], p["bottom"]).astype(np.uint8))
    # bokeh lights (blurred glowing discs)
    if p["bokeh"]:
        glow = Image.new("RGB", (W, H), (0, 0, 0))
        d = ImageDraw.Draw(glow)
        for _ in range(60):
            x, y = rng.randint(0, W), rng.randint(int(H * 0.35), int(H * 0.95))
            r = rng.randint(18, 70)
            c = rng.choice(p["bokeh"])
            k = rng.uniform(0.35, 0.9)
            d.ellipse([x - r, y - r, x + r, y + r], fill=tuple(int(v * k) for v in c))
        glow = glow.filter(ImageFilter.GaussianBlur(14))
        img = Image.fromarray(np.minimum(np.asarray(img, np.uint16) + np.asarray(glow, np.uint16), 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    sil = p["sil"]
    dark = tuple(int(v * 0.25) for v in p["top"])
    if sil == "skyline":
        x = 0
        while x < W:
            bw, bh = rng.randint(70, 190), rng.randint(300, 900)
            d.rectangle([x, H - bh, x + bw, H], fill=dark)
            for wy in range(H - bh + 20, H - 30, 46):
                for wx in range(x + 12, x + bw - 20, 34):
                    if rng.random() < 0.28:
                        c = rng.choice(p["bokeh"])
                        d.rectangle([wx, wy, wx + 14, wy + 22], fill=tuple(int(v * 0.8) for v in c))
            x += bw + rng.randint(0, 14)
    elif sil == "pines":
        for layer, shade in enumerate((0.9, 0.55, 0.25)):
            col = tuple(int(v * shade) for v in p["bottom"])
            base = H - 250 + layer * 120
            for _ in range(16):
                x = rng.randint(-60, W + 60)
                h = rng.randint(500, 1000) - layer * 120
                w = h // 4
                d.polygon([(x, base - h), (x - w, base + 300), (x + w, base + 300)], fill=col)
            d.rectangle([0, min(base + 200, H - 1), W, H], fill=col)
    elif sil in ("hills", "waves"):
        amp = 90 if sil == "hills" else 28
        for layer, shade in enumerate((0.85, 0.6, 0.38, 0.2)):
            col = tuple(int(v * shade) for v in p["bottom"])
            base = H * (0.55 + 0.11 * layer)
            ph, f = rng.uniform(0, 6.28), rng.uniform(0.003, 0.008) * (3 if sil == "waves" else 1)
            pts = [(x, base + amp * math.sin(x * f + ph) + amp * 0.4 * math.sin(x * f * 2.7 + ph * 2))
                   for x in range(0, W + 20, 20)]
            d.polygon(pts + [(W, H), (0, H)], fill=col)
    elif sil == "window":
        frame = tuple(int(v * 0.35) for v in p["top"])
        for x in (0, W // 2 - 14, W - 28):
            d.rectangle([x, 0, x + 28, H], fill=frame)
        for y in (0, H // 2 - 14, H - 28):
            d.rectangle([0, y, W, y + 28], fill=frame)
    img = img.filter(ImageFilter.GaussianBlur(1.2))
    arr = np.asarray(img, np.float32)
    # vignette
    yy, xx = np.mgrid[0:H, 0:W]
    v = 1 - 0.45 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 2
    return (arr * v[..., None]).clip(0, 255).astype(np.uint16)


def rain_layer(rng, count, length, width, alpha, blur, slant):
    """A tile that wraps seamlessly in x and y, holding rain streaks."""
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    for _ in range(count):
        x, y = rng.randint(0, W), rng.randint(0, H)
        ln = int(length * rng.uniform(0.6, 1.3))
        a = int(alpha * rng.uniform(0.5, 1.0))
        dx = int(ln * slant)
        for ox in (-W, 0, W):
            for oy in (-H, 0, H):
                d.line([x + ox, y + oy, x + ox + dx, y + oy + ln], fill=a, width=width)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(img, np.uint16)


SR = 44100
EAVE_H = 64          # height of the roof edge drawn along the top of the frame
DROP_G = 5.0         # px/frame^2, so a drop crosses the frame in ~0.9 s


def drip_schedule(rng, total):
    """Roof drips: a few fixed spots along the eave, each releasing drops at its own uneven pace.

    Returns [(release_frame, x, pitch)].
    """
    drops = []
    for _ in range(rng.randint(5, 7)):
        x = rng.randint(60, W - 60)
        period = rng.uniform(1.1, 2.8) * FPS
        pitch = rng.uniform(0.8, 1.35)
        t = rng.uniform(0, period)
        while t < total:
            drops.append((int(t), x, pitch))
            t += period * rng.uniform(0.7, 1.4)
    return sorted(drops)


def drop_sprite():
    """A soft teardrop, brighter at the bottom."""
    h, w = 48, 18
    yy, xx = np.mgrid[0:h, 0:w]
    cx, r = (w - 1) / 2, 1.5 + 6.5 * (yy / (h - 1)) ** 1.5
    a = np.clip(1 - np.abs(xx - cx) / r, 0, 1) * np.clip((h - yy) / 5, 0, 1)
    return (a * 255).astype(np.uint16)


def add_eave(bg, p):
    """Dark roof edge along the top with a faint wet highlight on its lower lip."""
    dark = np.array(p["top"], np.float32) * 0.18
    bg[:EAVE_H] = dark.astype(np.uint16)
    bg[EAVE_H:EAVE_H + 3] = np.minimum(bg[EAVE_H:EAVE_H + 3] + 55, 255)
    return bg


def extras_audio(rng, drops, path):
    """Synthesise the drip and bird sounds into a mono WAV at `path`."""
    n = SR * SECONDS
    out = np.zeros(n + SR, np.float32)

    def put(t0, sig):
        i = int(t0 * SR)
        if 0 <= i < n:
            out[i:i + len(sig)] += sig[:len(out) - i]

    # drips: a short rising "plink" when each drop lands
    fall = math.sqrt(2 * (H - EAVE_H) / DROP_G) / FPS
    for f, x, pitch in drops:
        d = rng.uniform(0.09, 0.14)
        t = np.arange(int(d * SR)) / SR
        f0 = 620 * pitch * rng.uniform(0.95, 1.05)
        freq = f0 * (1 + 1.6 * (t / d) ** 0.6)
        sig = np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-t * 38) * np.minimum(t / 0.003, 1)
        put(f / FPS + fall, (sig * rng.uniform(0.30, 0.46)).astype(np.float32))

    # birds: a few phrases of quick chirps, two "birds" with their own pitch range
    t_phrase = rng.uniform(1.0, 2.5)
    while t_phrase < SECONDS - 3:
        base = rng.choice([rng.uniform(2600, 3300), rng.uniform(3800, 4800)])
        gain = rng.uniform(0.15, 0.24)
        tn = t_phrase
        for _ in range(rng.randint(3, 7)):
            d = rng.uniform(0.06, 0.16)
            t = np.arange(int(d * SR)) / SR
            u = t / d
            sweep = rng.choice([-1, 1]) * rng.uniform(0.15, 0.45)
            freq = base * rng.uniform(0.9, 1.15) * (1 + sweep * (u - 0.5)) \
                * (1 + 0.04 * np.sin(2 * np.pi * rng.uniform(25, 45) * t))
            ph = 2 * np.pi * np.cumsum(freq) / SR
            sig = (np.sin(ph) + 0.25 * np.sin(2 * ph)) * np.sin(np.pi * u) ** 2
            put(tn, (sig * gain).astype(np.float32))
            tn += d + rng.uniform(0.03, 0.12)
        t_phrase = tn + rng.uniform(1.5, 4.5)

    pcm = (np.clip(out[:n], -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def render(day, frames_dir=None):
    title, preset, intensity, thunder = DAYS[(day - 1) % len(DAYS)]
    p = PRESETS[preset]
    rng = random.Random(1000 + day)
    bg = add_eave(build_background(p, rng), p)
    slant = rng.uniform(0.04, 0.16)
    n = {1: 500, 2: 900, 3: 1500}[intensity]
    # (tile, speed px/frame) for far, mid and near rain; speeds divide evenly so the clip loops
    layers = [
        (rain_layer(rng, n, 40, 1, 70, 1.0, slant), 32),
        (rain_layer(rng, n // 2, 80, 2, 110, 0.6, slant), 48),
        (rain_layer(rng, n // 5, 150, 3, 150, 0, slant), 64),
    ]
    tint = np.array(p["tint"], np.float32)[None, None, :] / 255.0
    total = FPS * SECONDS
    flashes = sorted(rng.sample(range(60, total - 150), 3 if thunder else 2))
    fx = random.Random(5000 + day)   # separate stream so the rain itself is unchanged
    drops = drip_schedule(fx, total)
    sprite = drop_sprite()
    sh, sw = sprite.shape

    def frame(i):
        rain = np.zeros((H, W), np.uint16)
        for tile, speed in layers:
            dy = (i * speed) % H
            dx = int(dy * slant) % W
            rain += np.roll(np.roll(tile, dy, axis=0), dx, axis=1)
        rain[:EAVE_H] = 0   # the roof keeps the rain off itself
        out = bg + (rain[..., None] * tint).astype(np.uint16)
        for f, x, _ in drops:
            k = i - f
            if k < -12 or k > 40:
                continue
            if k < 0:   # bead swelling on the lip of the roof before it lets go
                y, sc = EAVE_H - sh + 10, (k + 12) / 12
            else:
                y, sc = EAVE_H - sh + 10 + int(0.5 * DROP_G * k * k), 1.0
            if y >= H:
                continue
            y0, y1 = max(y, EAVE_H), min(y + sh, H)
            if y1 <= y0:
                continue
            x0 = x - sw // 2
            out[y0:y1, x0:x0 + sw] += (sprite[y0 - y:y1 - y, :, None] * sc).astype(np.uint16)
        for f in flashes:
            k = i - f
            if 0 <= k < 14:
                out = out + int(120 * (1 - k / 14) * (1 if k % 5 < 3 else 0.4))
        return np.minimum(out, 255).astype(np.uint8)

    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)
        for i in ([0, total // 2] + ([flashes[0] + 1] if flashes else [total - 1])):
            Image.fromarray(frame(i)).save(os.path.join(frames_dir, f"day{day:02d}_f{i:04d}.jpg"), quality=88)
        return None

    os.makedirs(OUT_DIR, exist_ok=True)
    final = os.path.join(OUT_DIR, f"day{day:03d}.mp4")
    tmp = os.path.join(OUT_DIR, f"day{day:03d}.tmp.mp4")
    vol = {1: 0.5, 2: 0.7, 3: 0.9}[intensity]
    wav = os.path.join(OUT_DIR, f"day{day:03d}.tmp.wav")
    extras_audio(fx, drops, wav)
    tvol = 5 if thunder else 3.2
    audio = (f"anoisesrc=color=pink:duration={SECONDS}:seed={day}:amplitude=0.9,"
             f"highpass=f=500,lowpass=f=9000,volume={vol}[r];"
             f"anoisesrc=color=brown:duration={SECONDS}:seed={day + 99}:amplitude=0.9,"
             f"lowpass=f=350,volume=0.35[b]")
    audio += ";[1:a]aresample=44100[x]"
    mix = "[r][b][x]"
    nmix = 3
    for j, f in enumerate(flashes):
        t0 = f / FPS + 0.4
        audio += (f";anoisesrc=color=brown:duration=4:seed={day + j}:amplitude=1,lowpass=f=140,"
                  f"afade=t=in:d=0.15,afade=t=out:st=0.6:d=3.4,volume={tvol},adelay={int(t0 * 1000)}:all=1[t{j}]")
        mix += f"[t{j}]"
        nmix += 1
    audio += (f";{mix}amix=inputs={nmix}:normalize=0:duration=first,"
              f"afade=t=in:d=1,afade=t=out:st={SECONDS - 1.5}:d=1.5,alimiter=limit=0.9[a]")
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", wav,
           "-filter_complex", audio, "-map", "0:v", "-map", "[a]",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-maxrate", "6M", "-bufsize", "12M",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
           "-t", str(SECONDS), "-movflags", "+faststart", tmp]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(total):
        proc.stdin.write(frame(i).tobytes())
    proc.stdin.close()
    rc = proc.wait()
    os.remove(wav)
    if rc != 0:
        raise SystemExit("ffmpeg failed")
    os.replace(tmp, final)
    return dict(day=day, file=os.path.relpath(final, ROOT), title=title, description=DESCRIPTION)


def next_day():
    if not os.path.isdir(OUT_DIR):
        return 1
    done = [f for f in os.listdir(OUT_DIR) if f.startswith("day") and f.endswith(".mp4") and ".tmp" not in f]
    return len(done) + 1


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    day = int(args[0]) if args and args[0].isdigit() else next_day()
    if "--frames" in sys.argv:
        render(day, frames_dir=sys.argv[sys.argv.index("--frames") + 1])
    else:
        print(json.dumps(render(day), ensure_ascii=False))
