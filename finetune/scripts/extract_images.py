import argparse
import os
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir", type=str, required=True, help="Root directory containing videos.txt and video subdirectory"
    )
    return parser.parse_args()


args = parse_args()

# Create data/images directory if it doesn't exist
data_dir = Path(args.datadir)
image_dir = data_dir / "images"
image_dir.mkdir(exist_ok=True)

# Read videos.txt
videos_file = data_dir / "videos.txt"
with open(videos_file, "r") as f:
    video_paths = [line.strip() for line in f.readlines() if line.strip()]

# Process each video file and collect image paths
image_paths = []
for video_rel_path in video_paths:
    video_path = data_dir / video_rel_path

    # Open video
    cap = cv2.VideoCapture(str(video_path))

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read video: {video_path}")
        continue

    # Save frame as PNG with same name as video
    image_name = f"images/{video_path.stem}.png"
    image_path = data_dir / image_name
    cv2.imwrite(str(image_path), frame)

    # Release video capture
    cap.release()

    print(f"Extracted first frame from {video_path} to {image_path}")
    image_paths.append(image_name)

# Write images.txt
images_file = data_dir / "images.txt"
with open(images_file, "w") as f:
    for path in image_paths:
        f.write(f"{path}\n")
