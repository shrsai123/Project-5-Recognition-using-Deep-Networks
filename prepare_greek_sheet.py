# Shreyas Raman
# Project 5: crop a handwritten Greek-letter sheet into ImageFolder layout

# import statements
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# useful functions with a comment for each function
def parse_args(argv):
    """Parses command line options for preparing custom Greek-letter crops."""
    parser = argparse.ArgumentParser(description="Crop a handwritten alpha/beta/gamma sheet into class folders.")
    parser.add_argument("image_path", type=Path, help="photo containing rows of alpha, beta, and gamma symbols")
    parser.add_argument("--output-dir", type=Path, default=Path("custom_greek_images"), help="output ImageFolder path")
    parser.add_argument("--labels", default="alpha,beta,gamma", help="comma-separated row labels from top to bottom")
    parser.add_argument("--examples-per-row", type=int, default=5, help="expected number of symbols in each row")
    parser.add_argument("--crop-size", type=int, default=128, help="saved square crop size")
    parser.add_argument("--padding", type=float, default=0.35, help="padding as a fraction of the symbol size")
    return parser.parse_args(argv[1:])


def otsu_threshold(grayscale):
    """Computes an Otsu threshold for a grayscale uint8 image."""
    histogram = np.bincount(grayscale.reshape(-1), minlength=256).astype(np.float64)
    total = grayscale.size
    sum_total = np.dot(np.arange(256), histogram)
    weight_background = 0.0
    sum_background = 0.0
    best_threshold = 0
    best_variance = -1.0

    for threshold in range(256):
        weight_background += histogram[threshold]
        if weight_background == 0:
            continue

        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += threshold * histogram[threshold]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold

    return best_threshold


def make_foreground_mask(image):
    """Creates a dark-ink foreground mask from a grayscale image."""
    grayscale = np.array(image.convert("L"), dtype=np.uint8)
    threshold = otsu_threshold(grayscale)
    mask = grayscale < min(110, max(70, threshold))
    mask = remove_border_components(mask)
    return grayscale, mask


def remove_border_components(mask):
    """Removes dark regions connected to the image border, such as table shadows."""
    cleaned = mask.copy()
    height, width = cleaned.shape
    stack = []

    for x in range(width):
        if cleaned[0, x]:
            stack.append((0, x))
        if cleaned[height - 1, x]:
            stack.append((height - 1, x))

    for y in range(height):
        if cleaned[y, 0]:
            stack.append((y, 0))
        if cleaned[y, width - 1]:
            stack.append((y, width - 1))

    while stack:
        y, x = stack.pop()
        if not cleaned[y, x]:
            continue
        cleaned[y, x] = False

        for next_y in range(max(0, y - 1), min(height, y + 2)):
            for next_x in range(max(0, x - 1), min(width, x + 2)):
                if cleaned[next_y, next_x]:
                    stack.append((next_y, next_x))

    return cleaned


def find_segments(values, min_gap, min_width):
    """Finds contiguous foreground segments in a row or column projection."""
    indices = np.flatnonzero(values)
    if len(indices) == 0:
        return []

    segments = []
    start = int(indices[0])
    previous = int(indices[0])

    for index in indices[1:]:
        index = int(index)
        if index - previous > min_gap:
            if previous - start + 1 >= min_width:
                segments.append((start, previous))
            start = index
        previous = index

    if previous - start + 1 >= min_width:
        segments.append((start, previous))

    return segments


def find_row_segments(mask, expected_rows):
    """Finds the top-to-bottom symbol rows from the foreground mask."""
    row_projection = mask.sum(axis=1) > max(8, mask.shape[1] * 0.003)
    segments = find_segments(row_projection, min_gap=45, min_width=25)

    if len(segments) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows but found {len(segments)} rows: {segments}")

    return segments


def find_symbol_boxes(mask, row_segment, examples_per_row):
    """Finds left-to-right symbol boxes inside one row."""
    y0, y1 = row_segment
    row_mask = mask[y0 : y1 + 1, :]
    column_projection = row_mask.sum(axis=0) > max(5, row_mask.shape[0] * 0.04)
    x_segments = find_segments(column_projection, min_gap=70, min_width=20)

    if len(x_segments) != examples_per_row:
        raise ValueError(f"Expected {examples_per_row} symbols in row {row_segment} but found {len(x_segments)}: {x_segments}")

    boxes = []
    for x0, x1 in x_segments:
        symbol_mask = row_mask[:, x0 : x1 + 1]
        ys, xs = np.nonzero(symbol_mask)
        if len(xs) == 0:
            continue
        tight_x0 = x0 + int(xs.min())
        tight_x1 = x0 + int(xs.max())
        tight_y0 = y0 + int(ys.min())
        tight_y1 = y0 + int(ys.max())
        boxes.append((tight_x0, tight_y0, tight_x1, tight_y1))

    return boxes


def find_component_boxes(mask, min_area=500):
    """Finds connected foreground components and returns their bounding boxes."""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    boxes = []

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue

            stack = [(y, x)]
            seen[y, x] = True
            xs = []
            ys = []

            while stack:
                current_y, current_x = stack.pop()
                xs.append(current_x)
                ys.append(current_y)

                for next_y in range(max(0, current_y - 1), min(height, current_y + 2)):
                    for next_x in range(max(0, current_x - 1), min(width, current_x + 2)):
                        if mask[next_y, next_x] and not seen[next_y, next_x]:
                            seen[next_y, next_x] = True
                            stack.append((next_y, next_x))

            area = len(xs)
            if area < min_area:
                continue

            boxes.append((min(xs), min(ys), max(xs), max(ys), area))

    return boxes


def group_component_boxes_by_row(component_boxes, expected_rows):
    """Groups component boxes into top-to-bottom rows."""
    if not component_boxes:
        return []

    sorted_boxes = sorted(component_boxes, key=lambda box: (box[1] + box[3]) / 2)
    median_height = float(np.median([box[3] - box[1] + 1 for box in sorted_boxes]))
    row_threshold = max(90.0, median_height * 0.9)
    rows = []

    for box in sorted_boxes:
        center_y = (box[1] + box[3]) / 2
        if not rows:
            rows.append([box])
            continue

        row_centers = [(other[1] + other[3]) / 2 for other in rows[-1]]
        if abs(center_y - float(np.mean(row_centers))) <= row_threshold:
            rows[-1].append(box)
        else:
            rows.append([box])

    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows but found {len(rows)} rows.")

    return rows


def merge_row_components(row_boxes, examples_per_row):
    """Merges nearby components in one row and returns one box per symbol."""
    ordered = sorted(row_boxes, key=lambda box: box[0])
    merged = []

    for box in ordered:
        x0, y0, x1, y1, area = box
        if not merged:
            merged.append([x0, y0, x1, y1, area])
            continue

        previous = merged[-1]
        gap = x0 - previous[2]
        previous_width = previous[2] - previous[0] + 1
        current_width = x1 - x0 + 1

        if gap < max(35, min(previous_width, current_width) * 0.65):
            previous[0] = min(previous[0], x0)
            previous[1] = min(previous[1], y0)
            previous[2] = max(previous[2], x1)
            previous[3] = max(previous[3], y1)
            previous[4] += area
        else:
            merged.append([x0, y0, x1, y1, area])

    if len(merged) != examples_per_row:
        raise ValueError(f"Expected {examples_per_row} symbols in row but found {len(merged)}: {merged}")

    return [(box[0], box[1], box[2], box[3]) for box in merged]


def crop_symbol(source_image, box, crop_size, padding):
    """Crops one symbol, pads it to a square, and resizes it."""
    x0, y0, x1, y1 = box
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    pad = int(max(width, height) * padding)

    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(source_image.width - 1, x1 + pad)
    y1 = min(source_image.height - 1, y1 + pad)

    crop = source_image.crop((x0, y0, x1 + 1, y1 + 1)).convert("RGB")
    side = max(crop.size)
    square = Image.new("RGB", (side, side), "white")
    left = (side - crop.width) // 2
    top = (side - crop.height) // 2
    square.paste(crop, (left, top))
    return square.resize((crop_size, crop_size), Image.Resampling.LANCZOS)


def save_debug_image(source_image, boxes, output_path):
    """Saves the original sheet with detected crop boxes drawn on it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    debug = source_image.convert("RGB").copy()
    draw = ImageDraw.Draw(debug)

    for index, (x0, y0, x1, y1) in enumerate(boxes):
        draw.rectangle((x0, y0, x1, y1), outline="red", width=4)
        draw.text((x0, max(0, y0 - 22)), str(index), fill="red")

    debug.save(output_path)


def prepare_sheet(image_path, output_dir, labels, examples_per_row, crop_size, padding):
    """Crops a Greek-letter sheet and writes class-folder images."""
    source_image = Image.open(image_path).convert("RGB")
    _grayscale, mask = make_foreground_mask(source_image)
    component_boxes = find_component_boxes(mask)
    rows = group_component_boxes_by_row(component_boxes, len(labels))

    all_boxes = []
    saved_paths = []
    for row_index, label in enumerate(labels):
        class_dir = output_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)
        boxes = merge_row_components(rows[row_index], examples_per_row)
        all_boxes.extend(boxes)

        for example_index, box in enumerate(boxes, start=1):
            crop = crop_symbol(source_image, box, crop_size, padding)
            output_path = class_dir / f"{label}_{example_index:02d}.jpg"
            crop.save(output_path, quality=95)
            saved_paths.append(output_path)

    save_debug_image(source_image, all_boxes, output_dir / "sheet_detected_boxes.jpg")
    return saved_paths


# main function
def main(argv):
    """Crops the custom Greek sheet into alpha, beta, and gamma image folders."""
    args = parse_args(argv)
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    saved_paths = prepare_sheet(args.image_path, args.output_dir, labels, args.examples_per_row, args.crop_size, args.padding)

    print(f"Saved {len(saved_paths)} crops to {args.output_dir}")
    for path in saved_paths:
        print(path)
    print(f"Saved debug boxes to {args.output_dir / 'sheet_detected_boxes.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
