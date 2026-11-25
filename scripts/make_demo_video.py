"""Turn frames from scripts/record_demo.mjs into demo/demo.mp4 and demo/demo.gif.

The screencast only sends a frame when the page changes, so each frame is shown
until the next one's timestamp (ffmpeg concat list with per-frame durations).
Uses the ffmpeg binary from imageio-ffmpeg:

    pip install imageio-ffmpeg
    python scripts/make_demo_video.py [framesDir]
"""
import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent.parent
FRAMES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "demo" / "frames"
OUT_DIR = ROOT / "demo"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def frame_path(frame):
    """Absolute path with forward slashes, quoted for the concat list."""
    return (FRAMES / frame["name"]).resolve().as_posix().replace("'", "'\''")


def concat_list(meta):
    """Write an ffmpeg concat list (file + duration per frame) and return its path."""
    frames = meta["frames"]
    # Skip a stall at the very start (the screencast can pause while it warms up).
    for i in range(min(30, len(frames) - 1)):
        if frames[i + 1]["t"] - frames[i]["t"] > 1.0:
            frames = frames[i + 1:]
            break
    lines = []
    for cur, nxt in zip(frames, frames[1:] + [None]):
        dur = (nxt["t"] - cur["t"]) if nxt else 1.0  # hold the last frame for a second
        lines += [f"file '{frame_path(cur)}'", f"duration {max(dur, 0.001):.4f}"]
    lines.append(f"file '{frame_path(frames[-1])}'")  # concat demuxer quirk: repeat the last file
    path = FRAMES / "list.txt"
    path.write_text("\n".join(lines) + "\n")
    return path, frames[-1]["t"] - frames[0]["t"] + 1.0


def ffmpeg(*args):
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def main():
    meta = json.loads((FRAMES / "frames.json").read_text())
    lst, seconds = concat_list(meta)
    OUT_DIR.mkdir(exist_ok=True)
    src = ["-f", "concat", "-safe", "0", "-i", str(lst)]

    mp4 = OUT_DIR / "demo.mp4"
    ffmpeg(*src, "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-preset", "slow", "-crf", "24",
           "-movflags", "+faststart", str(mp4))
    print(f"{mp4}  {seconds:.1f}s  {mp4.stat().st_size / 1e6:.1f} MB")

    gif = OUT_DIR / "demo.gif"
    scale = "fps=10,scale=720:-1:flags=lanczos"
    ffmpeg(*src, "-vf", f"{scale},split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];"
           "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle", str(gif))
    print(f"{gif}  {gif.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
