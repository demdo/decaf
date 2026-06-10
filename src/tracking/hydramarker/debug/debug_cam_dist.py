import pyrealsense2 as rs


def main():
    pipeline = rs.pipeline()
    config = rs.config()

    # Use the same stream settings as in your tracking setup
    config.enable_stream(
        rs.stream.color,
        1920,
        1080,
        rs.format.bgr8,
        30
    )

    try:
        profile = pipeline.start(config)

        color_stream = profile.get_stream(rs.stream.color)
        video_profile = color_stream.as_video_stream_profile()
        intr = video_profile.get_intrinsics()

        print("RealSense color stream intrinsics")
        print("--------------------------------")
        print(f"Resolution: {intr.width} x {intr.height}")
        print(f"fx: {intr.fx}")
        print(f"fy: {intr.fy}")
        print(f"ppx: {intr.ppx}")
        print(f"ppy: {intr.ppy}")
        print()
        print(f"Distortion model: {intr.model}")
        print(f"Distortion coeffs: {intr.coeffs}")
        print()

        if intr.model == rs.distortion.none:
            print("Detected model: rs.distortion.none")
        elif intr.model == rs.distortion.modified_brown_conrady:
            print("Detected model: rs.distortion.modified_brown_conrady")
        elif intr.model == rs.distortion.inverse_brown_conrady:
            print("Detected model: rs.distortion.inverse_brown_conrady")
        elif intr.model == rs.distortion.brown_conrady:
            print("Detected model: rs.distortion.brown_conrady")
        elif intr.model == rs.distortion.kannala_brandt4:
            print("Detected model: rs.distortion.kannala_brandt4")
        elif intr.model == rs.distortion.ftheta:
            print("Detected model: rs.distortion.ftheta")
        else:
            print("Detected model: unknown / unsupported RealSense distortion model")

    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()