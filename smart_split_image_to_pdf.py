#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def row_content_score(image):
    rgb = np.array(image.convert("RGB")).astype(np.int16)
    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    luma = (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    )

    # Count black text, colored handwriting, and remaining marks as content.
    # Very light near-white paper is treated as empty.
    non_white = (luma < 246) | ((maxc - minc) > 24)
    return non_white.mean(axis=1)


def rolling_sum(values, radius):
    kernel = np.ones(radius * 2 + 1, dtype=np.float32)
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def choose_cut(scores, top, target_height, total_height, search_window, quiet_band):
    target = min(top + target_height, total_height)
    if target >= total_height:
        return total_height

    min_page_height = int(target_height * 0.62)
    lo = max(top + min_page_height, target - search_window)
    hi = min(total_height - 1, target)
    if lo >= hi:
        return target

    band_scores = rolling_sum(scores, quiet_band)
    candidates = np.arange(lo, hi + 1)

    # Prefer blank/quiet rows, with a light penalty for drifting too far from A4.
    local = band_scores[candidates]
    distance_penalty = np.abs(candidates - target) / max(search_window, 1) * 0.01
    best = candidates[np.argmin(local + distance_penalty)]
    return int(best)


def draw_page_number(page, page_index, page_count, margin=34):
    draw = ImageDraw.Draw(page)
    label = f"{page_index}/{page_count}"
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 24)
    except OSError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = page.width - margin - text_w
    y = page.height - margin - text_h
    pad_x = 10
    pad_y = 5
    draw.rounded_rectangle(
        (x - pad_x, y - pad_y, x + text_w + pad_x, y + text_h + pad_y),
        radius=6,
        fill=(255, 255, 255),
        outline=(210, 210, 210),
        width=1,
    )
    draw.text((x, y), label, fill=(90, 90, 90), font=font)


def split_image_to_pdf(
    source,
    output_pdf,
    search_window=420,
    quiet_band=28,
    resolution=150,
    add_page_numbers=True,
):
    image = Image.open(source).convert("RGB")
    width, height = image.size
    page_height = round(width * 297 / 210)
    scores = row_content_score(image)

    cuts = [0]
    top = 0
    while top < height:
        cut = choose_cut(scores, top, page_height, height, search_window, quiet_band)
        if cut <= top:
            cut = min(top + page_height, height)
        cuts.append(cut)
        top = cut

    pages = []
    for top, bottom in zip(cuts, cuts[1:]):
        page = Image.new("RGB", (width, page_height), "white")
        crop = image.crop((0, top, width, bottom))
        page.paste(crop, (0, 0))
        pages.append(page)

    if add_page_numbers:
        for index, page in enumerate(pages, start=1):
            draw_page_number(page, index, len(pages))

    pages[0].save(output_pdf, save_all=True, append_images=pages[1:], resolution=resolution)
    return image.size, len(pages), cuts


def main():
    parser = argparse.ArgumentParser(
        description="Split a long processed image into PDF pages, choosing page breaks near quiet rows instead of cutting through text."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--search-window", type=int, default=420)
    parser.add_argument("--quiet-band", type=int, default=28)
    parser.add_argument("--no-page-numbers", action="store_true")
    args = parser.parse_args()

    output_pdf = args.pdf or args.source.with_suffix("").with_name(
        f"{args.source.stem}_smart_split.pdf"
    )
    size, page_count, cuts = split_image_to_pdf(
        args.source,
        output_pdf,
        args.search_window,
        args.quiet_band,
        add_page_numbers=not args.no_page_numbers,
    )

    print(f"source: {args.source}")
    print(f"size: {size[0]}x{size[1]}")
    print(f"pdf: {output_pdf}")
    print(f"pages: {page_count}")
    print("cuts:", ", ".join(str(cut) for cut in cuts))


if __name__ == "__main__":
    main()
