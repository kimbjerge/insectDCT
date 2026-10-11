#!/usr/bin/env python3
"""
Plot daily mean flower coverage and flower counts from one or more CSV files.

Usage examples:
    python plot_flower_by_date.py /path/to/csv_directory --site "Abisko" --year 2022    
    python plot_flower_by_date.py /path/to/csv_directory --pattern "*ABIS_01*.csv" --site "Abisko" --year 2022

All CSV files in the directory are combined. Rows are grouped by the `date`
column, and the daily means of `flower_percentage` and `number_of_flowers`
are plotted. The site name is used as the plot title.
Only the months from May to September is plotted.


Author: Kim Bjerge + ChatGPT
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_csv_files(directory: Path, pattern: str, recursive: bool) -> pd.DataFrame:
    """Read and combine CSV files from a directory."""
    paths = sorted(directory.rglob(pattern) if recursive else directory.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No CSV files matching {pattern!r} found in {directory}"
        )

    frames = []
    required = {"date", "flower_percentage", "number_of_flowers"}

    for path in paths:
        try:
            frame = pd.read_csv(path)
            
            # Convert dates and mark invalid formats as NaT
            parsed_dates = pd.to_datetime(
                frame["date"].astype(str),
                format="%Y%m%d",
                errors="coerce",
            )
            
            # Drop rows with invalid dates
            frame = frame.loc[parsed_dates.notna()].copy()
            
            # Keep date as a datetime column if needed
            frame["date"] = parsed_dates.loc[frame.index]

        except Exception as exc:
            print(f"Warning: skipping {path}: {exc}")
            continue

        missing = required - set(frame.columns)
        if missing:
            print(f"Warning: skipping {path}; missing columns: {', '.join(sorted(missing))}")
            continue

        frame["source_file"] = str(path)
        frames.append(frame)

    if not frames:
        raise ValueError(
            "No usable CSV files found. Each file must contain date, "
            "flower_percentage, and number_of_flowers columns."
        )

    data = pd.concat(frames, ignore_index=True)

    # Parse dates robustly whether stored as 20220621 or 2022-06-21.
    date_text = data["date"].astype(str).str.strip()

    parsed = pd.to_datetime(date_text, format="%Y%m%d", errors="coerce")
    missing_dates = parsed.isna()
    if missing_dates.any():
        parsed.loc[missing_dates] = pd.to_datetime(
            date_text.loc[missing_dates], errors="coerce"
        )
    data["date"] = parsed

    for column in ("flower_percentage", "number_of_flowers"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    invalid = data["date"].isna()
    if invalid.any():
        print(f"Warning: dropping {int(invalid.sum())} rows with unparseable dates.")
        data = data.loc[~invalid].copy()

    return data

# Narsarsuaq_23-A3-107_WSCT",
# Narsarsuaq_23-A5-103_WSCT

def main():
    parser = argparse.ArgumentParser(
        description="Plot daily mean flower coverage and number of flowers."
    )
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        #default=Path("./flowersTundra/Abisko"),
        #default=Path("./flowersTundra/Narsarsuaq"),
        default=Path("./flowersTundra/Toolik"),
        help="Directory containing the flower-analysis CSV files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="File glob pattern (default: *.csv).",
    )
    parser.add_argument(
        "--site",
        #default="Abisko",
        #default="Narsarsuaq",
        default="Toolik",
        help="Recording site name shown as the plot title (default: directory name).",
    )
    parser.add_argument(
        "--year",
        type=int,
        #default=2021,
        #default=2022,
        #default=2023,
        default=2024,
        #default=2025,
        help="The year to be displayed.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also search for CSV files in subdirectories.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively (default behavior if --output is omitted).",
    )

    args = parser.parse_args()

    if not args.directory.is_dir():
        parser.error(f"Not a directory: {args.directory}")

    site = args.site or args.directory.name
    data = load_csv_files(args.directory, args.pattern, args.recursive)


    if args.year is not None:
        data = data[data['year'] == args.year]
        #data = data[data['date'].dt.year == args.year]

    # Keep only dates from May (05) through September (09)
    data = data[data["date"].dt.month.between(5, 9)]

    daily = (
        data.groupby("date", as_index=False)
        .agg(
            flower_percentage=("flower_percentage", "mean"),
            number_of_flowers=("number_of_flowers", "mean"),
            image_count=("date", "size"),
        )
        .sort_values("date")
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Flower coverage and number of flowers — {site} {args.year}", fontsize=15)

    axes[0].plot(
        daily["date"],
        daily["flower_percentage"],
        "ro",
        linewidth=1.8,
    )
    axes[0].set_ylabel("Mean flower coverage (%)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        daily["date"],
        daily["number_of_flowers"],
        "o",
        linewidth=1.8,
    )
    axes[1].set_ylabel("Mean number of flowers per image")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    output = Path(args.site + '-' + str(args.year) + "-flowers.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {output}")
    plt.show()

    print("\nDaily summary:")
    print(daily.to_string(index=False, formatters={
        "date": lambda value: value.strftime("%Y-%m-%d"),
        "flower_percentage": lambda value: f"{value:.3f}",
        "number_of_flowers": lambda value: f"{value:.2f}",
    }))


if __name__ == "__main__":
    main()
