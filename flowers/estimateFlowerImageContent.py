# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 12:23:58 2026

@author: Kim Bjerge (ChatGPT)
"""

import os
import csv
import argparse
import datetime 
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from PIL.ExifTags import TAGS

from ultralytics.models.sam import SAM3SemanticPredictor


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}


#%% Return datetime based on file exifdata
def getFrameTime(image_filename): 

    # open the image
    image = Image.open(image_filename)
     
    # extracting the exif metadata
    exifdata = image.getexif()
    
    dateTimeStr = datetime.datetime.now().strftime('%Y%m%d%H%M%S') # Current time YYYYMMDDHHMMSS
    
    # looping through all the tags present in exifdata
    for tagid in exifdata:
        # getting the tag name instead of tag id
        tagname = TAGS.get(tagid, tagid)
        if tagname == "DateTime": # Check tag date time
            # passing the tagid to get its respective value
            value = exifdata.get(tagid)
            # printing the final result
            #print(f"{tagname:25}: {value}")
            
            # reformat time stamp to "YYYYMMDDHHMMSS"
            timestamp = value.replace(':', '')
            dateTimeStr = timestamp.replace(' ', '')
       
    # close the image
    image.close()
    
    image_time = datetime.datetime.strptime(dateTimeStr, "%Y%m%d%H%M%S")

    return image_time, dateTimeStr
        

def find_images(input_dir):
    """Find all images recursively."""
    input_dir = Path(input_dir)

    return sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def create_mask_visualization(
    image,
    masks,
    output_path,
    alpha=0.45,
    draw_contours=True,
    draw_numbers=True
):
    """
    Create and save an image showing SAM 3 flower masks
    overlaid on the original image.

    Parameters
    ----------
    image : numpy.ndarray
        Original BGR image.

    masks : numpy.ndarray
        Boolean masks with shape [N, H, W].

    output_path : str or Path
        Output image path.

    alpha : float
        Mask transparency.

    draw_contours : bool
        Draw contours around each flower.

    draw_numbers : bool
        Draw flower number at the centroid of each mask.
    """

    # ---------------------------------------------------------
    # Make sure output directory exists
    # ---------------------------------------------------------
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Copy original image
    # ---------------------------------------------------------
    visualization = image.copy()

    # ---------------------------------------------------------
    # No masks
    # ---------------------------------------------------------
    if masks is None or len(masks) == 0:

        cv2.imwrite(
            str(output_path),
            visualization
        )

        return

    # ---------------------------------------------------------
    # Generate visually distinct colours
    #
    # HSV gives reasonably separated colours.
    # ---------------------------------------------------------
    number_of_masks = len(masks)

    colors = []

    for i in range(number_of_masks):

        hue = int(
            179 * i / max(number_of_masks, 1)
        )

        hsv_color = np.uint8(
            [[[hue, 220, 255]]]
        )

        bgr_color = cv2.cvtColor(
            hsv_color,
            cv2.COLOR_HSV2BGR
        )[0, 0]

        colors.append(
            tuple(int(x) for x in bgr_color)
        )

    # ---------------------------------------------------------
    # Draw each mask
    # ---------------------------------------------------------
    for i, mask in enumerate(masks):

        color = colors[i]

        # Boolean mask -> uint8
        mask_uint8 = (
            mask.astype(np.uint8) * 255
        )

        # -----------------------------------------------------
        # Create solid colour image
        # -----------------------------------------------------
        colour_layer = np.zeros_like(
            image,
            dtype=np.uint8
        )

        colour_layer[:, :] = color

        # -----------------------------------------------------
        # Alpha blend only inside mask
        # -----------------------------------------------------
        mask_pixels = mask.astype(bool)

        visualization[mask_pixels] = cv2.addWeighted(
            visualization[mask_pixels],
            1.0 - alpha,
            colour_layer[mask_pixels],
            alpha,
            0
        )

        # -----------------------------------------------------
        # Find contours
        # -----------------------------------------------------
        if draw_contours:

            contours, _ = cv2.findContours(
                mask_uint8,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            cv2.drawContours(
                visualization,
                contours,
                -1,
                color,
                thickness=2
            )

        # -----------------------------------------------------
        # Flower number
        # -----------------------------------------------------
        if draw_numbers:

            moments = cv2.moments(
                mask_uint8
            )

            if moments["m00"] != 0:

                cx = int(
                    moments["m10"] /
                    moments["m00"]
                )

                cy = int(
                    moments["m01"] /
                    moments["m00"]
                )

                label = str(i + 1)

                # Text size
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = max(
                    0.5,
                    min(image.shape[:2]) / 1200
                )

                thickness = max(
                    1,
                    int(font_scale * 2)
                )

                text_size, baseline = cv2.getTextSize(
                    label,
                    font,
                    font_scale,
                    thickness
                )

                text_width, text_height = text_size

                # -------------------------------------------------
                # Background rectangle for readability
                # -------------------------------------------------
                x1 = cx - text_width // 2 - 4
                y1 = cy - text_height - 6

                x2 = cx + text_width // 2 + 4
                y2 = cy + baseline + 4

                cv2.rectangle(
                    visualization,
                    (x1, y1),
                    (x2, y2),
                    color,
                    thickness=-1
                )

                # -------------------------------------------------
                # Draw number
                # -------------------------------------------------
                cv2.putText(
                    visualization,
                    label,
                    (
                        cx - text_width // 2,
                        cy
                    ),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA
                )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    success = cv2.imwrite(
        str(output_path),
        visualization
    )

    if not success:
        raise RuntimeError(
            f"Could not save mask visualization: "
            f"{output_path}"
        )


def analyse_image(
    image_path,
    predictor,
    prompt="flower",
    mask_output_path=None
):
    """
    Analyse one image with SAM 3.

    Returns one dictionary containing image-level statistics.
    """
    
    frame_time, dateTimeStr = getFrameTime(str(image_path))
    timestamp_year_str = frame_time.strftime("%Y")
    timestamp_date_str = frame_time.strftime("%Y%m%d")
    timestamp_time_str = frame_time.strftime("%H%M%S")
    
    # ---------------------------------------------------------
    # Read image
    # ---------------------------------------------------------
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    height, width = image.shape[:2]

    image_pixel_area = width * height

    # ---------------------------------------------------------
    # Give image to SAM 3
    # ---------------------------------------------------------
    predictor.set_image(str(image_path))

    # ---------------------------------------------------------
    # Text prompt
    # ---------------------------------------------------------
    results = predictor(
        text=[prompt]
    )

    # SAM 3 returns a list of Results objects
    result = results[0]

    # ---------------------------------------------------------
    # Extract masks
    # ---------------------------------------------------------
    if result.masks is None:

        # Save original image if requested
        if mask_output_path is not None:

            create_mask_visualization(
                image=image,
                masks=None,
                output_path=mask_output_path
            )

        return {
            "year": timestamp_year_str,
            "date": timestamp_date_str,
            "time": timestamp_time_str,

            "image": str(image_path),
            #"filename": image_path.name,
            "width": width,
            "height": height,
            "image_pixel_area": image_pixel_area,

            "number_of_flowers": 0,

            "flower_pixel_area": 0,
            "flower_percentage": 0.0,

            "mean_flower_area": 0.0,
            #"median_flower_area": 0.0,
            "smallest_flower_area": 0,
            "largest_flower_area": 0,

            "mean_confidence": 0.0,
            #"min_confidence": 0.0,
            #"max_confidence": 0.0,

            "mask_image": (
                str(mask_output_path)
                if mask_output_path is not None
                else ""
            ),
        }

    # ---------------------------------------------------------
    # Masks
    #
    # result.masks.data has shape:
    #
    #     [N, H, W]
    #
    # where N is the number of detected instances.
    # ---------------------------------------------------------
    masks = result.masks.data

    if torch.is_tensor(masks):

        masks = (
            masks
            .detach()
            .cpu()
            .numpy()
        )

    masks = masks.astype(bool)

    number_of_flowers = len(masks)

    # ---------------------------------------------------------
    # Save mask visualization
    # ---------------------------------------------------------
    if mask_output_path is not None:

        create_mask_visualization(
            image=image,
            masks=masks,
            output_path=mask_output_path
        )

    # ---------------------------------------------------------
    # Individual flower areas
    # ---------------------------------------------------------
    individual_areas = masks.reshape(
        number_of_flowers,
        -1
    ).sum(axis=1)

    # ---------------------------------------------------------
    # Combine all flower masks
    #
    # np.any() creates the union of all masks.
    #
    # This prevents overlapping masks from being counted twice.
    # ---------------------------------------------------------
    combined_mask = np.any(
        masks,
        axis=0
    )

    flower_pixel_area = int(
        combined_mask.sum()
    )

    flower_percentage = (
        100.0 *
        flower_pixel_area /
        image_pixel_area
    )

    # ---------------------------------------------------------
    # Confidence scores
    # ---------------------------------------------------------
    if (
        result.boxes is not None
        and result.boxes.conf is not None
    ):

        confidence = (
            result.boxes.conf
            .detach()
            .cpu()
            .numpy()
        )

        mean_confidence = float(
            np.mean(confidence)
        )

        min_confidence = float(
            np.min(confidence)
        )

        max_confidence = float(
            np.max(confidence)
        )

    else:

        mean_confidence = np.nan
        min_confidence = np.nan
        max_confidence = np.nan

    # ---------------------------------------------------------
    # Return results
    # ---------------------------------------------------------
    return {

        "year": timestamp_year_str,
        "date": timestamp_date_str,
        "time": timestamp_time_str,
            
        "image": str(image_path),
        #"filename": image_path.name,

        "width": width,
        "height": height,
        "image_pixel_area": image_pixel_area,

        "number_of_flowers": number_of_flowers,

        "flower_pixel_area": flower_pixel_area,
        "flower_percentage": flower_percentage,

        "mean_flower_area": float(
            np.mean(individual_areas)
        ),

        #"median_flower_area": float(
        #    np.median(individual_areas)
        #),

        "smallest_flower_area": int(
            np.min(individual_areas)
        ),

        "largest_flower_area": int(
            np.max(individual_areas)
        ),

        "mean_confidence": mean_confidence,
        #"min_confidence": min_confidence,
        #"max_confidence": max_confidence,

        "mask_image": (
            str(mask_output_path)
            if mask_output_path is not None
            else ""
        ),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        #required=True,
        default="./flowerImages",        
        help="Directory containing images"
    )

    parser.add_argument(
        "--output",
        default="flower_analysis.csv",
        help="Output CSV file"
    )

    parser.add_argument(
        "--mask-output",
        default="./flowerMasks",
        help=(
            "Directory where SAM 3 mask "
            "visualizations are saved"
        )
    )

    parser.add_argument(
        "--skip",
        type=int,
        default=1, # 60
        help="Skip number of images" # When sampled each 1 minute then analyse an image each hour (skip=60)
    )

    parser.add_argument(
        "--model",
        default="sam3.pt",
        help="SAM 3 model"
    )

    parser.add_argument(
        "--prompt",
        default="flower",
        help="SAM 3 text prompt"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="SAM 3 confidence threshold"
    )

    parser.add_argument(
        "--quantize",
        type=int,
        default=16,
        choices=[16, 8],
        help="SAM 3 quantization"
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    # ---------------------------------------------------------
    # SAM 3 predictor
    # ---------------------------------------------------------
    overrides = {
        "conf": args.conf,
        "task": "segment",
        "mode": "predict",
        "model": args.model,
        "quantize": args.quantize,
        "device": device,
        "verbose": False,
        "save": False,
        "save_txt": False,
        "save_crop": False
    }

    predictor = SAM3SemanticPredictor(
        overrides=overrides
    )

    # ---------------------------------------------------------
    # Find images
    # ---------------------------------------------------------
    image_paths = find_images(
        args.input
    )

    print(
        f"Found {len(image_paths)} images."
    )

    results = []

    # ---------------------------------------------------------
    # Process images
    # ---------------------------------------------------------
    for i, image_path in enumerate(
        image_paths
    ):
        count = i + 1
        if (count % args.skip == 0): # Analyse every skip images (Skip == 1) every

            print(
                f"[{i + 1}/{len(image_paths)}] "
                f"{image_path}"
            )
    
            # -----------------------------------------------------
            # Create corresponding mask-output path
            #
            # Preserve the input directory structure.
            # -----------------------------------------------------
            relative_path = image_path.relative_to(
                Path(args.input)
            )
    
            mask_output_path = (
                Path(args.mask_output)
                #/ relative_path.parent
                / f"{relative_path.stem}_masks.png"
            )
    
            try:
    
                result = analyse_image(
                    image_path,
                    predictor,
                    prompt=args.prompt,
                    mask_output_path=mask_output_path
                )
    
                results.append(result)
    
            except Exception as e:
    
                print(
                    f"ERROR: {image_path}: {e}"
                )
    
                results.append({
                    
                    "year": None,
                    "date": None,
                    "time": None,
                    
                    "image": str(image_path),
                    #"filename": image_path.name,
    
                    "width": None,
                    "height": None,
                    "image_pixel_area": None,
    
                    "number_of_flowers": None,
    
                    "flower_pixel_area": None,
                    "flower_percentage": None,
    
                    "mean_flower_area": None,
                    #"median_flower_area": None,
                    "smallest_flower_area": None,
                    "largest_flower_area": None,
    
                    "mean_confidence": None,
                    #"min_confidence": None,
                    #"max_confidence": None,
    
                    "mask_image": str(
                        mask_output_path
                    ),
                })

    # ---------------------------------------------------------
    # Write CSV
    # ---------------------------------------------------------
    outputFile = "No output"
    if results:

        fieldnames = results[0].keys()

        if '.csv' in args.output:
            outputFile = args.output
        else:
            inputSplit = args.input.split('/')
            fileName = inputSplit[-3] + '-' + inputSplit[-2] + '.csv'
            outputFile = args.output + '-' + fileName
            
        with open(
            outputFile,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(results)

    print()
    print("Finished.")
    print(
        f"Output CSV: {outputFile}"
    )
    print(
        f"Mask images: {args.mask_output}"
    )


if __name__ == "__main__":
    main()