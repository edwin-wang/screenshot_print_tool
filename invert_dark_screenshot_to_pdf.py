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


def invert_continuous_dark_regions(rgb, threshold=70):
    arr = rgb.copy()
    work = rgb.astype(np.float32)
    luma = 0.2126 * work[:, :, 0] + 0.7152 * work[:, :, 1] + 0.0722 * work[:, :, 2]
    dark = luma < threshold
    h, w = dark.shape

    mask = np.zeros_like(dark, dtype=bool)
    min_segment_width = int(w * 0.18)
    min_edge_width = int(w * 0.04)
    max_gap = int(w * 0.075)

    for y in range(h):
        xs = np.flatnonzero(dark[y])
        if xs.size == 0:
            continue

        start = prev = int(xs[0])
        dark_count = 1
        segments = []

        for x in xs[1:]:
            x = int(x)
            gap = x - prev - 1
            if gap <= max_gap:
                dark_count += 1
                prev = x
                continue

            segments.append((start, prev, dark_count))
            start = prev = x
            dark_count = 1

        segments.append((start, prev, dark_count))

        for x1, x2, count in segments:
            width = x2 - x1 + 1
            touches_edge = x1 <= 12 or x2 >= w - 13
            is_wide_block = width >= min_segment_width
            is_edge_block = touches_edge and width >= min_edge_width
            if (is_wide_block or is_edge_block) and count / width >= 0.42:
                pad = 8
                mask[y, max(0, x1 - pad) : min(w, x2 + pad + 1)] = True

    active_rows = mask.any(axis=1)
    y = 0
    while y < h:
        if not active_rows[y]:
            y += 1
            continue

        start = y
        while y < h and active_rows[y]:
            y += 1
        end = y

        if end - start < 18:
            mask[start:end, :] = False

    arr[mask] = 255 - arr[mask]
    arr = remove_edge_dark_bars(arr, threshold)
    arr = remove_horizontal_boundary_rules(arr)

    return arr


def remove_horizontal_boundary_rules(rgb):
    arr = rgb.copy()
    work = rgb.astype(np.float32)
    luma = 0.2126 * work[:, :, 0] + 0.7152 * work[:, :, 1] + 0.0722 * work[:, :, 2]
    maxc = rgb.max(axis=2).astype(np.int16)
    minc = rgb.min(axis=2).astype(np.int16)
    neutral_line = (luma < 225) & ((maxc - minc) < 22)
    h, w = neutral_line.shape
    rows_to_clear = np.zeros(h, dtype=bool)

    for y in range(h):
        xs = np.flatnonzero(neutral_line[y])
        if xs.size == 0:
            continue

        longest = 0
        start = prev = int(xs[0])
        for x in xs[1:]:
            x = int(x)
            if x == prev + 1:
                prev = x
                continue
            longest = max(longest, prev - start + 1)
            start = prev = x
        longest = max(longest, prev - start + 1)

        if longest >= int(w * 0.62) or neutral_line[y].mean() >= 0.48:
            rows_to_clear[max(0, y - 2) : min(h, y + 3)] = True

    arr[rows_to_clear, :] = [255, 255, 255]
    return arr


def crop_vertical_whitespace(rgb, margin=80):
    work = rgb.astype(np.float32)
    luma = 0.2126 * work[:, :, 0] + 0.7152 * work[:, :, 1] + 0.0722 * work[:, :, 2]
    maxc = rgb.max(axis=2).astype(np.int16)
    minc = rgb.min(axis=2).astype(np.int16)
    content = (luma < 245) | ((maxc - minc) > 24)
    rows = np.flatnonzero(content.mean(axis=1) > 0.003)
    if rows.size == 0:
        return rgb

    top = max(0, int(rows[0]) - margin)
    bottom = min(rgb.shape[0], int(rows[-1]) + margin + 1)
    return rgb[top:bottom, :]


def remove_edge_dark_bars(rgb, threshold=70):
    arr = rgb.copy()
    work = rgb.astype(np.float32)
    luma = 0.2126 * work[:, :, 0] + 0.7152 * work[:, :, 1] + 0.0722 * work[:, :, 2]
    dark = luma < threshold
    h, w = dark.shape

    max_edge_distance = int(w * 0.18)
    min_bar_width = max(4, int(w * 0.003))
    max_bar_width = int(w * 0.16)
    min_bar_height = max(90, int(h * 0.018))

    for side in ("left", "right"):
        x_range = range(0, max_edge_distance) if side == "left" else range(w - 1, w - max_edge_distance - 1, -1)
        for x in x_range:
            if dark[:, x].mean() < 0.06:
                continue

            y = 0
            while y < h:
                while y < h and not dark[y, x]:
                    y += 1
                start = y
                while y < h and dark[y, x]:
                    y += 1
                end = y

                if end - start < min_bar_height:
                    continue

                y1 = max(0, start - 4)
                y2 = min(h, end + 4)
                if side == "left":
                    left = x
                    while left > 0 and dark[start:end, left - 1].mean() > 0.18:
                        left -= 1
                    right = x
                    while right + 1 < w and dark[start:end, right + 1].mean() > 0.18:
                        right += 1
                else:
                    left = x
                    while left > 0 and dark[start:end, left - 1].mean() > 0.18:
                        left -= 1
                    right = x
                    while right + 1 < w and dark[start:end, right + 1].mean() > 0.18:
                        right += 1

                bar_width = right - left + 1
                touches_side = left <= max_edge_distance or right >= w - max_edge_distance
                if min_bar_width <= bar_width <= max_bar_width and touches_side:
                    pad = 8
                    arr[y1:y2, max(0, left - pad) : min(w, right + pad + 1)] = [255, 255, 255]

    return remove_tall_dark_rules(arr, threshold)


def remove_tall_dark_rules(rgb, threshold=70):
    arr = rgb.copy()
    work = rgb.astype(np.float32)
    luma = 0.2126 * work[:, :, 0] + 0.7152 * work[:, :, 1] + 0.0722 * work[:, :, 2]
    dark = luma < threshold
    h, w = dark.shape
    seen = np.zeros_like(dark, dtype=bool)

    min_height = max(110, int(h * 0.012))
    max_width = max(18, int(w * 0.025))

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
            center_x = (min_x + max_x) / 2
            near_page_edge = center_x < w * 0.16 or center_x > w * 0.62
            is_tall_rule = (
                near_page_edge
                and comp_h >= min_height
                and comp_w <= max_width
                and comp_h / max(comp_w, 1) >= 8
                and fill_ratio >= 0.35
            )

            if is_tall_rule:
                y1 = max(0, min_y - 4)
                y2 = min(h, max_y + 5)
                x1 = max(0, min_x - 6)
                x2 = min(w, max_x + 7)
                arr[y1:y2, x1:x2] = [255, 255, 255]

    return remove_edge_dark_artifacts(arr, threshold)


def remove_edge_dark_artifacts(rgb, threshold=85):
    arr = rgb.copy()
    work = rgb.astype(np.float32)
    luma = 0.2126 * work[:, :, 0] + 0.7152 * work[:, :, 1] + 0.0722 * work[:, :, 2]
    maxc = rgb.max(axis=2).astype(np.int16)
    minc = rgb.min(axis=2).astype(np.int16)
    dark = (luma < threshold) & ((maxc - minc) < 48)
    h, w = dark.shape
    seen = np.zeros_like(dark, dtype=bool)

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
            near_right_edge = max_x >= int(w * 0.78)
            small_enough = area < int(w * h * 0.004)
            rule_like = comp_w / max(comp_h, 1) >= 5 or comp_h / max(comp_w, 1) >= 5
            compact_edge_mark = comp_w <= int(w * 0.15) and comp_h <= int(h * 0.04)

            if near_right_edge and small_enough and fill_ratio > 0.28 and (rule_like or compact_edge_mark):
                y1 = max(0, min_y - 6)
                y2 = min(h, max_y + 7)
                x1 = max(0, min_x - 8)
                x2 = min(w, max_x + 9)
                arr[y1:y2, x1:x2] = [255, 255, 255]

    return arr


def convert_image(
    source,
    output_image,
    output_pdf,
    dark_threshold=115,
    white_threshold=190,
    tile_size=24,
    tile_ratio=0.24,
    invert_regions=False,
    trim_vertical_whitespace=False,
):
    image = Image.open(source).convert("RGB")
    rgb = np.array(image)

    if invert_regions:
        arr = invert_continuous_dark_regions(rgb, dark_threshold)
        if trim_vertical_whitespace:
            arr = crop_vertical_whitespace(arr)
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
    parser.add_argument(
        "--invert-dark-regions",
        action="store_true",
        help="Invert each large continuous dark region instead of only whitening dark pixels.",
    )
    parser.add_argument(
        "--trim-vertical-whitespace",
        action="store_true",
        help="Remove large top/bottom whitespace after processing while keeping a print margin.",
    )
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
        args.invert_dark_regions,
        args.trim_vertical_whitespace,
    )

    print(f"source: {args.source}")
    print(f"size: {size[0]}x{size[1]}")
    print(f"image: {output_image}")
    print(f"pdf: {output_pdf}")
    print(f"pages: {page_count}")


if __name__ == "__main__":
    main()
