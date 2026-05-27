#!/usr/bin/env python3
import argparse
import math
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image


def box_blur_bool(mask, radius):
    """Fast square blur for boolean masks, returning neighborhood hit counts."""
    if radius <= 0:
        return mask.astype(np.int32)

    padded = np.pad(mask.astype(np.int32), radius, mode="constant")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    h, w = mask.shape
    size = radius * 2 + 1

    return (
        integral[size : size + h, size : size + w]
        - integral[:h, size : size + w]
        - integral[size : size + h, :w]
        + integral[:h, :w]
    )


def detect_dark_background(rgb, dark_threshold, tile_size, tile_ratio):
    arr = rgb.astype(np.float32)
    luma = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    dark = luma < dark_threshold

    h, w = dark.shape
    bg_tiles = np.zeros_like(dark, dtype=bool)

    for y in range(0, h, tile_size):
        y2 = min(y + tile_size, h)
        for x in range(0, w, tile_size):
            x2 = min(x + tile_size, w)
            tile = dark[y:y2, x:x2]
            if tile.mean() >= tile_ratio:
                bg_tiles[y:y2, x:x2] = True

    # Include nearby anti-aliased edges and text pixels that sit inside dark panels.
    nearby_dark_panel = box_blur_bool(bg_tiles, tile_size // 2) > 0
    local_dark_density = box_blur_bool(dark, tile_size // 2) / float(tile_size * tile_size)
    block_like_dark = local_dark_density > max(tile_ratio, 0.35)
    return nearby_dark_panel | block_like_dark, dark, luma


def remove_dark_regions_keep_text(original_rgb, rgb, threshold=135):
    arr = original_rgb.astype(np.float32)
    luma = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    dark = luma < threshold
    h, w = dark.shape
    seen = np.zeros_like(dark, dtype=bool)
    output = rgb.copy()

    min_large_area = max(1200, int(w * h * 0.0002))
    min_long_width = int(w * 0.28)
    min_long_height = int(h * 0.035)
    maxc = original_rgb.max(axis=2).astype(np.int16)
    minc = original_rgb.min(axis=2).astype(np.int16)
    low_saturation = (maxc - minc) < 40
    original_light_text = low_saturation & (luma > 165)

    for sy in range(h):
        for sx in range(w):
            if seen[sy, sx] or not dark[sy, sx]:
                continue

            queue = deque([(sy, sx)])
            seen[sy, sx] = True
            pixels = []
            min_y = max_y = sy
            min_x = max_x = sx

            while queue:
                y, x = queue.popleft()
                pixels.append((y, x))
                if y < min_y:
                    min_y = y
                elif y > max_y:
                    max_y = y
                if x < min_x:
                    min_x = x
                elif x > max_x:
                    max_x = x

                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and dark[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))

            comp_h = max_y - min_y + 1
            comp_w = max_x - min_x + 1
            area = len(pixels)
            fill_ratio = area / float(comp_w * comp_h)
            is_horizontal_rule = comp_w >= min_long_width and comp_h <= 80 and fill_ratio > 0.35
            is_vertical_rule = comp_h >= min_long_height and comp_w <= 80 and fill_ratio > 0.35
            is_dark_block = area >= min_large_area and fill_ratio > 0.08
            is_frame_or_rule = is_horizontal_rule or is_vertical_rule or is_dark_block

            if is_frame_or_rule:
                ys, xs = zip(*pixels)
                output[ys, xs] = [255, 255, 255]

                region_mask = np.zeros((h, w), dtype=bool)
                region_mask[ys, xs] = True
                near_region = box_blur_bool(region_mask, 12) > 0
                y1 = max(0, min_y - 12)
                y2 = min(h, max_y + 13)
                x1 = max(0, min_x - 12)
                x2 = min(w, max_x + 13)
                text_mask = original_light_text[y1:y2, x1:x2] & near_region[y1:y2, x1:x2]
                output_region = output[y1:y2, x1:x2]
                output_region[text_mask] = [0, 0, 0]

    return output


def convert_image(
    source,
    output_image,
    output_pdf,
    dark_threshold=115,
    white_threshold=190,
    tile_size=24,
    tile_ratio=0.24,
):
    image = Image.open(source).convert("RGB")
    rgb = np.array(image)

    dark_panel, dark_pixels, luma = detect_dark_background(
        rgb, dark_threshold, tile_size, tile_ratio
    )

    arr = rgb.copy()

    # White or near-white text sitting inside detected dark panels becomes black.
    maxc = rgb.max(axis=2).astype(np.int16)
    minc = rgb.min(axis=2).astype(np.int16)
    low_saturation = (maxc - minc) < 35
    white_text_on_dark = dark_panel & low_saturation & (luma > white_threshold)

    # Only the dark pixels inside dark panels become paper white. This avoids
    # changing ordinary black text on existing white areas.
    arr[dark_panel & dark_pixels] = [255, 255, 255]
    arr[white_text_on_dark] = [0, 0, 0]
    arr = remove_dark_regions_keep_text(rgb, arr)

    processed = Image.fromarray(arr, "RGB")
    processed.save(output_image, quality=95)

    width, height = processed.size
    page_height = round(width * 297 / 210)
    page_count = math.ceil(height / page_height)
    pages = []

    for i in range(page_count):
        top = i * page_height
        bottom = min(top + page_height, height)
        page = Image.new("RGB", (width, page_height), "white")
        page.paste(processed.crop((0, top, width, bottom)), (0, 0))
        pages.append(page)

    pages[0].save(output_pdf, save_all=True, append_images=pages[1:], resolution=150)
    return image.size, page_count


def main():
    parser = argparse.ArgumentParser(
        description="Convert black screenshot backgrounds to white, blacken white text on those backgrounds, then split into a multi-page PDF."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--dark-threshold", type=int, default=55)
    parser.add_argument("--white-threshold", type=int, default=190)
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--tile-ratio", type=float, default=0.45)
    args = parser.parse_args()

    stem = args.source.with_suffix("")
    output_image = args.image or stem.with_name(f"{stem.name}_print_friendly.jpg")
    output_pdf = args.pdf or stem.with_name(f"{stem.name}_print_friendly.pdf")

    size, page_count = convert_image(
        args.source,
        output_image,
        output_pdf,
        args.dark_threshold,
        args.white_threshold,
        args.tile_size,
        args.tile_ratio,
    )

    print(f"source: {args.source}")
    print(f"size: {size[0]}x{size[1]}")
    print(f"image: {output_image}")
    print(f"pdf: {output_pdf}")
    print(f"pages: {page_count}")


if __name__ == "__main__":
    main()
