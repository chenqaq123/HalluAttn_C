#!/usr/bin/env python3
"""Build a paper-facing mechanism figure for semantic-neighbor POPE failures.

The figure visualizes saved examples where vanilla and attention-only methods
answer yes to an absent target, while semantic-neighbor evidence is stronger
than target evidence and TDEV answers no. It is intentionally a mechanism figure,
not a detector visualization: the image is paired with the target/neighbor
scores and predictions that instantiate `looking is not verifying`.
"""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SPLITS = ("random", "popular", "adversarial")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build POPE mechanism contact-sheet figure")
    p.add_argument("--examples_csv", default="mitigation/results/pope_mechanism_alignment_full/pope_mechanism_alignment_examples.csv")
    p.add_argument("--coco_dir", default="/home/chenguanxu/common_dataset/coco-2014-dataset/val2014")
    p.add_argument("--output_dir", default="mitigation/results/pope_mechanism_alignment_full/figure")
    p.add_argument("--examples_per_split", type=int, default=2)
    p.add_argument("--thumb_width", type=int, default=360)
    p.add_argument("--text_width", type=int, default=500)
    p.add_argument("--row_height", type=int, default=270)
    return p.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def image_path(coco_dir: Path, image_stem: str) -> Path:
    stem = image_stem if image_stem.endswith(".jpg") else f"{image_stem}.jpg"
    path = coco_dir / stem
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def fit_image(path: Path, width: int, height: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    x = (width - image.width) // 2
    y = (height - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def wrap_text(text: str, max_chars: int) -> list[str]:
    lines: list[str] = []
    for part in str(text).split("\n"):
        wrapped = textwrap.wrap(part, width=max_chars, break_long_words=False, replace_whitespace=False)
        lines.extend(wrapped or [""])
    return lines


def draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: str, max_chars: int, line_gap: int = 4) -> int:
    x, y = xy
    for line in wrap_text(text, max_chars):
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line or "A", font=font)
        y += bbox[3] - bbox[1] + line_gap
    return y


def select_rows(rows: list[dict[str, str]], examples_per_split: int) -> list[dict[str, str]]:
    by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
    selected = []
    for split in SPLITS:
        candidates = sorted(
            by_split.get(split, []),
            key=lambda r: (
                int(r.get("attention_yes_count", 0)),
                float(r["best_neighbor_score"]),
                -float(r["target_score"]),
            ),
            reverse=True,
        )
        selected.extend(candidates[:examples_per_split])
    return selected


def row_summary(row: dict[str, str]) -> dict[str, str | float]:
    return {
        "split": row["split"],
        "question_id": row["question_id"],
        "image": row["image"],
        "question": row["question"],
        "target": row["target"],
        "best_neighbor": row["best_neighbor"],
        "target_score": float(row["target_score"]),
        "best_neighbor_score": float(row["best_neighbor_score"]),
        "tdev_margin": float(row["tdev_margin"]),
        "vanilla": row["vanilla"],
        "pai": row["pai"],
        "clearsight": row["clearsight"],
        "visattnsink": row["visattnsink"],
        "tdev": row["tdev"],
        "related_present": row["related_present"],
    }


def build_figure(rows: list[dict[str, str]], coco_dir: Path, output_png: Path, args: argparse.Namespace) -> None:
    margin = 28
    gap = 22
    header_h = 78
    row_gap = 18
    row_w = args.thumb_width + gap + args.text_width
    width = row_w + 2 * margin
    height = header_h + len(rows) * args.row_height + (len(rows) - 1) * row_gap + 2 * margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(26, bold=True)
    subtitle_font = load_font(15)
    head_font = load_font(17, bold=True)
    body_font = load_font(15)
    small_font = load_font(13)

    y = margin
    draw.text((margin, y), "Looking Is Not Verifying: Semantic-Neighbor Failures", font=title_font, fill="#111111")
    y += 34
    draw.text((margin, y), "Absent target, related object present; attention-style methods say yes, TDEV says no.", font=subtitle_font, fill="#444444")
    y = margin + header_h

    for idx, row in enumerate(rows, start=1):
        x_img = margin
        img = fit_image(image_path(coco_dir, row["image"]), args.thumb_width, args.row_height)
        canvas.paste(img, (x_img, y))
        draw.rectangle((x_img, y, x_img + args.thumb_width - 1, y + args.row_height - 1), outline="#cccccc", width=1)

        x_text = x_img + args.thumb_width + gap
        ty = y + 4
        label = f"{idx}. {row['split']} qid={row['question_id']} | target absent: {row['target']}"
        ty = draw_wrapped(draw, (x_text, ty), label, head_font, "#111111", 54, line_gap=5)
        ty += 2
        ty = draw_wrapped(draw, (x_text, ty), row["question"], body_font, "#222222", 58, line_gap=4)
        ty += 6
        score_line = (
            f"Evidence: target {row['target_score']} vs best neighbor "
            f"{row['best_neighbor']} {row['best_neighbor_score']} "
            f"(margin {row['tdev_margin']})"
        )
        ty = draw_wrapped(draw, (x_text, ty), score_line, body_font, "#111111", 58, line_gap=4)
        ty += 4
        related = f"Related present: {row['related_present']}"
        ty = draw_wrapped(draw, (x_text, ty), related, small_font, "#444444", 70, line_gap=3)
        ty += 5
        preds = f"Predictions: vanilla={row['vanilla']}, PAI={row['pai']}, ClearSight={row['clearsight']}, VisAttnSink={row['visattnsink']}, TDEV={row['tdev']}"
        ty = draw_wrapped(draw, (x_text, ty), preds, body_font, "#222222", 60, line_gap=4)
        ty += 6
        draw.text((x_text, ty), "Mechanism: related evidence is real, but it supports the wrong object claim.", font=small_font, fill="#7a2e1f")

        y += args.row_height + row_gap

    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_png)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = select_rows(read_rows(Path(args.examples_csv)), args.examples_per_split)
    if not rows:
        raise RuntimeError("No mechanism rows selected")
    png_path = output_dir / "pope_mechanism_alignment_contact_sheet.png"
    json_path = output_dir / "pope_mechanism_alignment_contact_sheet.json"
    build_figure(rows, Path(args.coco_dir), png_path, args)
    payload = {
        "examples_csv": args.examples_csv,
        "coco_dir": args.coco_dir,
        "examples_per_split": args.examples_per_split,
        "figure_png": str(png_path),
        "examples": [row_summary(row) for row in rows],
        "caption": "Semantic-neighbor failure examples: vanilla and attention-only methods answer yes to absent targets when related evidence is present; TDEV rejects the target because neighbor evidence is stronger.",
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {png_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
