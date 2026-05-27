# Ink-Saving Long Screenshot Printing Tool

This directory is for converting long screenshots into print-friendly PDFs:

1. Convert black/dark backgrounds to white.
2. Convert white text on dark backgrounds to black.
3. Intelligently split long images into multi-page PDFs, avoiding cuts through text or questions as much as possible.
4. Optionally add small page numbers to the bottom-right corner of each PDF page.

## Current Example Files

- `IMG_1749_print_friendly_no_frames.jpg`: processed long image.
- `IMG_1749_print_friendly_no_frames_smart_split.pdf`: intelligently split PDF without page numbers.
- `IMG_1749_print_friendly_no_frames_smart_split_numbered.pdf`: intelligently split PDF with bottom-right page numbers.

## Recommended Usage

If you already have a processed long image and only want to split it intelligently and add page numbers:

```bash
python3 smart_split_image_to_pdf.py IMG_1749_print_friendly_no_frames.jpg
```

The output file will be named automatically:

```text
IMG_1749_print_friendly_no_frames_smart_split.pdf
```

To specify the output PDF name:

```bash
python3 smart_split_image_to_pdf.py input_image.jpg --pdf output_file.pdf
```

To disable page numbers:

```bash
python3 smart_split_image_to_pdf.py input_image.jpg --pdf output_file.pdf --no-page-numbers
```

## Start From a Raw Long Screenshot

If you have a new raw screenshot, first convert the dark background to white:

```bash
python3 invert_dark_screenshot_to_pdf.py raw_screenshot.jpg
```

Then use the generated `_print_friendly.jpg` image for intelligent page splitting:

```bash
python3 smart_split_image_to_pdf.py raw_screenshot_print_friendly.jpg
```

## Page Splitting Adjustments

If the split still cuts through text, increase the search range:

```bash
python3 smart_split_image_to_pdf.py input_image.jpg --search-window 620
```

If split points are too easily affected by small handwriting marks, increase the quiet band width:

```bash
python3 smart_split_image_to_pdf.py input_image.jpg --quiet-band 40
```
