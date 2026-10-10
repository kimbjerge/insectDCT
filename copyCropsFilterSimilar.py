"""
Created on 2026-10-10.

@author: Otar Sakhiashvili (Claude)

Below is a script that finds all images in a directory and its subdirectories.
To run the script, the path to the target folder must be specified.
The script creates a new folder next to the scanned folder, with the same name
plus the "_filtered" suffix. The images that are kept are copied into it with the
same subfolder structure, leaving the original folder untouched.
The default similarity threshold is 15, but the user can change it when prompted
in the terminal. In most cases 18 works well, so it is recommended.
The script performs a dry run first, showing how many images would be removed at
each threshold from 0 to 20, and then asks the user to choose a threshold for the
cleanup and how many copies of each similar group should be kept.

"""



import os
import shutil

import numpy as np
import imagehash
from PIL import Image


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

# Which image is kept out of each group of similar images.
KEEPER_RULES = {
    "resolution":    lambda m: (-(m['width'] * m['height']), -m['size'], m['path']),  # most pixels
    "largest":       lambda m: (-m['size'], m['path']),                               # biggest file
    "smallest":      lambda m: (m['size'], m['path']),                                # smallest file
}

# Number of set bits for every byte value (used for fast Hamming distance)
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def get_image_info(filepath, hash_size):

    try:
        stat = os.stat(filepath)
        with Image.open(filepath) as img:
            width, height = img.size
            img_hash = imagehash.dhash(img, hash_size=hash_size)
        return {
            'path': filepath,
            'bits': np.packbits(img_hash.hash.flatten()),  # compact form for fast comparison
            'width': width,
            'height': height,
            'size': stat.st_size,
        }
    except Exception:
        return None


def group_similar_images(images, similarity_threshold, keeper_rule, scope="global"):
    """
    scope="global"     -> every image is compared with every other image, across all folders.
    scope="per_folder" -> images are only compared with images in the SAME folder
                          (their immediate parent directory), so a duplicate in another
                          folder can never cause an image to be removed.
    """
    if scope == "global":
        return _group_pool(images, similarity_threshold, keeper_rule)
    if scope == "per_folder":
        by_folder = {}
        for img in images:
            by_folder.setdefault(os.path.dirname(img['path']), []).append(img)
        groups = []
        for folder in sorted(by_folder):
            groups.extend(_group_pool(by_folder[folder], similarity_threshold, keeper_rule))
        return groups
    raise ValueError(f"Unknown scope '{scope}' (use 'global' or 'per_folder')")


def _group_pool(images, similarity_threshold, keeper_rule):
    """
    Keeper-first grouping of one pool of images.
    Images are sorted so the best copy (per keeper_rule) comes first. Each image
    not yet in a group becomes the group's keeper and claims every remaining image
    within `similarity_threshold` bits of IT. Results no longer depend on the
    order in which os.walk happens to find files, and the best image is always
    the group's representative.
    Returns only groups with more than one image, each ordered best -> worst as
    a list of (image, distance_from_keeper).
    """
    images = sorted(images, key=KEEPER_RULES[keeper_rule])
    packed = np.stack([img['bits'] for img in images])  # shape (n, hash_bytes)
    assigned = np.zeros(len(images), dtype=bool)
    groups = []

    for i in range(len(images)):
        if assigned[i]:
            continue
        assigned[i] = True
        remaining = np.where(~assigned)[0]
        group = [(images[i], 0)]

        if remaining.size:
            distances = POPCOUNT[np.bitwise_xor(packed[remaining], packed[i])].sum(axis=1)
            close = distances <= similarity_threshold
            for j, d in zip(remaining[close], distances[close]):
                group.append((images[j], int(d)))
            assigned[remaining[close]] = True

        groups.append(group)  # already ordered best -> worst

    return [g for g in groups if len(g) > 1]


def run_threshold_sweep(images, keeper_rule, hash_size, thresholds, scope="global"):

    total = len(images)
    total_bits = hash_size * hash_size
    scale = total_bits / 64.0  # the threshold list is defined for an 8x8 (64-bit) hash
    bit_thresholds = sorted({int(round(t * scale)) for t in thresholds})

    print(f"\n=== THRESHOLD SWEEP (dry run, nothing is deleted, scope: {scope}) ===")
    print(f"{'threshold':>9} {'match %':>8} {'groups':>8} {'removed':>9} {'removed %':>10} {'remaining':>10}")

    for t in bit_thresholds:
        groups = group_similar_images(images, t, keeper_rule, scope)
        removed = sum(len(g) - 1 for g in groups)
        match_pct = 100.0 * (1 - t / total_bits)
        print(f"{t:>9} {match_pct:>7.1f}% {len(groups):>8} {removed:>9} "
              f"{100.0 * removed / total:>9.1f}% {total - removed:>10}")
    print()


def make_output_dir(root_dir, suffix="_filtered"):
    root = os.path.abspath(root_dir.rstrip("\\/"))
    parent = os.path.dirname(root)
    return os.path.join(parent, os.path.basename(root) + suffix)


def copy_kept_images(all_images, paths_to_drop, root_dir, output_dir):
    root = os.path.abspath(root_dir)
    copied = 0
    failed = 0
    for img in all_images:
        if img['path'] in paths_to_drop:
            continue
        rel = os.path.relpath(img['path'], root)
        destination = os.path.join(output_dir, rel)
        try:
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copy2(img['path'], destination)
            copied += 1
        except Exception as e:
            failed += 1
            print(f"Failed to copy {img['path']}: {e}")
    return copied, failed


def analyze_image_folder(root_dir, similarity_threshold=5, keeper_rule="resolution",
                         hash_size=8, scope="global", sweep_thresholds=(0, 2, 5, 8, 10, 15, 16, 17, 18, 19, 20),
                         output_suffix="_filtered"):
    images = []
    total_images = 0

    print(f"Scanning directory: {os.path.abspath(root_dir)}\n" + "-" * 50)

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS:
                total_images += 1
                info = get_image_info(os.path.join(dirpath, filename), hash_size)
                if info is not None:
                    images.append(info)

    print(f"Scan complete. Total images processed: {total_images} "
          f"({total_images - len(images)} unreadable)\n")

    if not images:
        return

    # --- optional threshold sweep -------------------------------------------
    if sweep_thresholds:
        choice = input("Run a threshold sweep first (dry run, deletes nothing)? (yes/no): ").strip().lower()
        if choice in ['yes', 'y']:
            run_threshold_sweep(images, keeper_rule, hash_size, sweep_thresholds, scope)
            raw = input(f"Threshold to use for the real cleanup "
                        f"[Enter = {similarity_threshold}, or type 'q' to quit]: ").strip().lower()
            if raw == 'q':
                print("Quit. No files were modified.")
                return
            if raw:
                try:
                    similarity_threshold = int(raw)
                except ValueError:
                    print(f"Not a number, using {similarity_threshold}.")

    duplicate_groups = group_similar_images(images, similarity_threshold, keeper_rule, scope)

    max_bits = hash_size * hash_size
    print(f"=== VISUAL DUPLICATES (Threshold: {similarity_threshold} of {max_bits} bits, "
          f"keeper rule: {keeper_rule}, scope: {scope}) ===")

    if not duplicate_groups:
        print("No visual duplicates found.\n")
        duplicate_groups = []

    total_dup_files = sum(len(group) for group in duplicate_groups)
    print(f"Found {len(duplicate_groups)} group(s) of similar images spanning {total_dup_files} total files.\n")

    output_dir = make_output_dir(root_dir, output_suffix)

    print("=== OUTPUT ===")
    print(f"Nothing in the scanned folder is deleted. The images that are kept will be")
    print(f"copied, with the original folder structure, to:\n  {output_dir}\n")

    copy_choice = input("Continue? (yes/no): ").strip().lower()
    if copy_choice not in ['yes', 'y']:
        print("Cancelled. Nothing was copied.")
        return

    while True:
        try:
            keep_count = int(input("How many copies of each similar group should be kept? (Usually 1): ").strip())
            if keep_count < 1:
                print("Please enter 1 or more.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a whole number.")

    # Images that will NOT be copied: everything past the first `keep_count` in each group.
    paths_to_drop = {item['path'] for group in duplicate_groups
                     for item, _ in group[keep_count:]}

    to_copy = len(images) - len(paths_to_drop)
    print(f"\nKeeping the best {keep_count} per group (by '{keeper_rule}').")
    print(f"Images to copy   : {to_copy}")
    print(f"Images left out  : {len(paths_to_drop)}\n")

    if os.path.exists(output_dir) and os.listdir(output_dir):
        overwrite = input(f"'{output_dir}' already exists and is not empty. Add to it anyway? (yes/no): ").strip().lower()
        if overwrite not in ['yes', 'y']:
            print("Cancelled. Nothing was copied.")
            return

    copied, failed = copy_kept_images(images, paths_to_drop, root_dir, output_dir)

    print()
    print("========================================")
    print("SUMMARY")
    print("========================================")
    print(f"Total images scanned : {len(images)}")
    print(f"Images copied        : {copied}")
    print(f"Images left out      : {len(paths_to_drop)}")
    if failed:
        print(f"Copy failures        : {failed}")
    print(f"Output directory     : {output_dir}")
    print(f"Source folder        : unchanged")
    print("========================================")


# --- HOW TO RUN ---
# insert the path to your target folder.
if __name__ == "__main__":
    target_directory = r"D:\folder\path\to\your\images"

    # Default is 15, but you can change it when prompted in the terminal.
    SIMILARITY_THRESHOLD = 15

    # Which image survives in each group: resolution, largest, smallest
    KEEPER_RULE = "resolution"

    HASH_SIZE = 8

    # "global"     : compare images across ALL folders (finds duplicates spread over folders)
    # "per_folder" : only compare images inside the same folder (safest when folders are

    SCOPE = "per_folder"


    OUTPUT_SUFFIX = "_filtered"

    SWEEP_THRESHOLDS = (0, 2, 5, 8, 10, 15, 16, 17, 18, 19, 20)

    if os.path.exists(target_directory):
        analyze_image_folder(target_directory,
                             similarity_threshold=SIMILARITY_THRESHOLD,
                             keeper_rule=KEEPER_RULE,
                             hash_size=HASH_SIZE,
                             scope=SCOPE,
                             sweep_thresholds=SWEEP_THRESHOLDS,
                             output_suffix=OUTPUT_SUFFIX)
    else:
        print(f"Error: Directory '{target_directory}' does not exist. Please update the path.")
