#!/usr/bin/env python3

"""
Download the weights for the YOLO model from the official YOLO website and save them to the specified directory.

written on plain python so that it can be used in any environment without the need for additional dependencies.
"""


import argparse
import os
import hashlib
import re
import sys
import urllib.request

from pathlib import Path


HERE = Path(__file__).resolve().parent
CFG = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg"
REL = "https://github.com/AlexeyAB/darknet/releases/download/yolov4"
DATA = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data"


TINY = [
    (f"{CFG}/yolov4-tiny.cfg", "yolov4-tiny.cfg"),
    (f"{REL}/yolov4-tiny.weights", "yolov4-tiny.weights"),
    (f"{DATA}/coco.names", "coco.names")
]

FULL = [
    (f"{CFG}/yolov4.cfg", "yolov4.cfg"),
    (f"{REL}/yolov4.weights", "yolov4.weights"),
]


def _progress(blocks, block_size, total):
    if total > 0 and sys.stdout.isatty():
        done = min(100, blocks * block_size * 100 // total)
        sys.stdout.write(f"\r {done:3d}%")
        sys.stdout.flush()
        
        
        

def fetch(url, name):
    """Download one file unless a non-empty copy is already sitting here."""
    dest = HERE / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip  {name} (already present)")
        return
    print(f"get   {name}", flush=True)
    urllib.request.urlretrieve(url, dest, _progress)
    if sys.stdout.isatty():
        print()


def short_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def human(size):
    mb = size / (1024 * 1024)
    return f"{mb:.1f}MB" if mb >= 1 else f"{size / 1024:.1f}KB"


def main():
    ap = argparse.ArgumentParser(description="fetch pretrained YOLOv4-Tiny weights")
    ap.add_argument("--full", action="store_true", help="also fetch full YOLOv4 (~250 MB)")
    args = ap.parse_args()

    for url, name in TINY + (FULL if args.full else []):
        fetch(url, name)

    print(f"\nfiles in {HERE}:")
    for name in ("yolov4-tiny.cfg", "yolov4-tiny.weights", "coco.names",
                 "yolov4.cfg", "yolov4.weights"):
        path = HERE / name
        if path.exists():
            print(f"  {name:<22} {human(path.stat().st_size):>8}  sha256:{short_sha(path)}")


if __name__ == "__main__":
    main()