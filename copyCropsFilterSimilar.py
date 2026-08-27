# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:15:02 2026

@author: Kim Bjerge (ChatGPT)

Yes. Below is a script that:

Recursively finds all *.jpg files.
Processes each source subdirectory independently.
Uses imagehash.phash() to compare images.
Uses a similarity threshold of 15.
Keeps only different images by default.
Allows you to keep 1–4 images from each group of similar images.
Preserves the original subdirectory structure in the output directory.
Copies the selected images rather than moving them.
Prints progress and a summary.

python copyCropsFilterSimilar.py "D:/crops" "D:/crops_filtered" --threshold 15 --keep 2

"""

from pathlib import Path
import shutil
import imagehash
from PIL import Image
import argparse


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DEFAULT_THRESHOLD = 15
DEFAULT_SIMILAR_IMAGES = 2


# ---------------------------------------------------------
# Calculate perceptual hash
# ---------------------------------------------------------

def calculate_hash(image_file):
    try:
        with Image.open(image_file) as img:
            return imagehash.phash(img)

    except Exception as e:
        print(f"ERROR reading {image_file}: {e}")
        return None


# ---------------------------------------------------------
# Select images
# ---------------------------------------------------------

def select_images(image_files, threshold, keep_similar):
    """
    Select images so that only different images are kept.

    threshold:
        Maximum phash distance for images to be considered similar.

    keep_similar:
        Number of similar images to keep from each similarity group.
    """

    selected = []
    selected_hashes = []

    for image_file in image_files:

        img_hash = calculate_hash(image_file)

        if img_hash is None:
            continue

        # Find how many already selected images are similar
        similar_count = 0

        for selected_hash in selected_hashes:

            distance = img_hash - selected_hash

            if distance <= threshold:
                similar_count += 1

        # Keep image if it is sufficiently different
        # or if we have not yet reached the requested number
        # of similar images.
        if similar_count < keep_similar:

            selected.append(image_file)
            selected_hashes.append(img_hash)

    return selected


# ---------------------------------------------------------
# Main processing
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Copy different JPG images using perceptual image hashing."
    )

    parser.add_argument(
        "input_dir",
        help="Input directory containing subdirectories with JPG images"
    )

    parser.add_argument(
        "output_dir",
        help="Output directory"
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD,
        help="pHash similarity threshold (default: 15)"
    )

    parser.add_argument(
        "--keep",
        type=int,
        choices=[1, 2, 3, 4],
        default=DEFAULT_SIMILAR_IMAGES,
        help="Number of similar images to keep (1-4, default: 1)"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all JPG files recursively
    jpg_files = list(input_dir.rglob("*.jpg"))

    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Threshold       : {args.threshold}")
    print(f"Keep similar    : {args.keep}")
    print(f"Total JPG files : {len(jpg_files)}")
    print()

    # Group files by their parent directory
    directories = {}

    for image_file in jpg_files:
        directories.setdefault(image_file.parent, []).append(image_file)

    total_selected = 0
    total_skipped = 0

    # Process each directory independently
    for directory, files in sorted(directories.items()):

        print("----------------------------------------")
        print(f"Directory: {directory}")
        print(f"Images   : {len(files)}")

        # Sort to make the selection reproducible
        files = sorted(files)

        selected = select_images(
            files,
            threshold=args.threshold,
            keep_similar=args.keep
        )

        # Create corresponding output directory
        relative_dir = directory.relative_to(input_dir)
        destination_dir = output_dir / relative_dir

        destination_dir.mkdir(parents=True, exist_ok=True)

        # Copy selected images
        for image_file in selected:

            destination_file = destination_dir / image_file.name

            shutil.copy2(image_file, destination_file)

        skipped = len(files) - len(selected)

        total_selected += len(selected)
        total_skipped += skipped

        print(f"Selected : {len(selected)}")
        print(f"Skipped  : {skipped}")

    # Final summary
    print()
    print("========================================")
    print("SUMMARY")
    print("========================================")
    print(f"Total input images : {len(jpg_files)}")
    print(f"Images copied      : {total_selected}")
    print(f"Images skipped     : {total_skipped}")
    print("========================================")


if __name__ == "__main__":
    main()