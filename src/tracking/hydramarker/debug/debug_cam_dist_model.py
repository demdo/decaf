"""Print the active RealSense stream distortion model once and exit."""

from __future__ import annotations

import argparse
from typing import Any

import pyrealsense2 as rs


def enum_name(enum_owner: Any, value: Any) -> str:
    for name in dir(enum_owner):
        if name.startswith("_"):
            continue
        try:
            if getattr(enum_owner, name) == value:
                return name
        except Exception:
            continue
    return str(value)


def call_or_unknown(obj: Any, name: str) -> Any:
    try:
        return getattr(obj, name)()
    except Exception:
        return "unknown"


def print_intrinsics(profile: Any) -> None:
    stream_profile = profile.get_stream(rs.stream.color)
    video_profile = stream_profile.as_video_stream_profile()
    intr = video_profile.get_intrinsics()
    stream_type = call_or_unknown(stream_profile, "stream_type")
    stream_format = call_or_unknown(stream_profile, "format")
    stream_fps = call_or_unknown(stream_profile, "fps")

    print("RealSense active color stream")
    print("-----------------------------")
    print(f"stream:      {enum_name(rs.stream, stream_type)}")
    print(f"format:      {enum_name(rs.format, stream_format)}")
    print(f"fps:         {stream_fps}")
    print(f"resolution:  {intr.width} x {intr.height}")
    print()
    print("Intrinsics")
    print(f"fx:          {intr.fx}")
    print(f"fy:          {intr.fy}")
    print(f"ppx:         {intr.ppx}")
    print(f"ppy:         {intr.ppy}")
    print()
    print("Distortion")
    print(f"model:       {enum_name(rs.distortion, intr.model)}")
    print(f"coeffs:      {[float(x) for x in intr.coeffs]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print RealSense color-stream intrinsics/distortion once.",
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color,
        int(args.width),
        int(args.height),
        rs.format.bgr8,
        int(args.fps),
    )

    profile = None
    try:
        profile = pipeline.start(config)
        # Let the device settle and make sure the active profile is negotiated.
        for _ in range(5):
            pipeline.wait_for_frames()
        print_intrinsics(profile)
    finally:
        if profile is not None:
            pipeline.stop()


if __name__ == "__main__":
    main()
