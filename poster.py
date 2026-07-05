"""
Share poster generation: PNG composition.
Used by Flask /generate-poster route only (not inference).
"""

import io
import json
import os
import re

from PIL import Image, ImageDraw, ImageFilter, ImageFont

import train

POSTER_WIDTH = 1080
POSTER_HEIGHT = 1350
MARGIN = 48
FOOTER_H = 56
POSTER_COLORS = {
    "bg_cream": "#f7f5ef",
    "gradient_top": "#f7f5ef",
    "gradient_bottom": "#f1f8f4",
    "card_white": "#ffffff",
    "primary": "#1b4332",
    "accent_green": "#40916c",
    "title": "#1b4332",
    "subtitle": "#40916c",
    "body": "#2d3436",
    "muted": "#636e72",
    "badge_bg": "#d8f3dc",
    "badge_text": "#1b4332",
    "warning_bg": "#fde8d8",
    "warning_text": "#8b4513",
    "warning_icon": "#e67e22",
    "footer_bg": "#1b4332",
    "footer_text": "#ffffff",
    "dotted": "#b2bec3",
    "section_label": "#1b4332",
    "leaf_decor": "#95d5b2",
}

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\ariali.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def _load_font(size, bold=False, italic=False):
    candidates = list(_FONT_CANDIDATES)
    if italic:
        italic_c = [c for c in candidates if "i.ttf" in c.lower() or "Oblique" in c]
        candidates = italic_c + [c for c in candidates if c not in italic_c]
    elif bold:
        candidates = [c for c in candidates if "Bold" in c or "b.ttf" in c.lower()] + [
            c for c in candidates if "Bold" not in c and "b.ttf" not in c.lower()
        ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_gradient_background(width, height, top_hex, bottom_hex):
    # ponytail: O(height) Python loop; upgrade path: numpy meshgrid or solid fill
    top = _hex_to_rgb(top_hex)
    bottom = _hex_to_rgb(bottom_hex)
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        ratio = y / max(height - 1, 1)
        color = tuple(int(top[i] + (bottom[i] - top[i]) * ratio) for i in range(3))
        draw.line([(0, y), (width, y)], fill=color)
    return img


def _rounded_rect(draw, bbox, radius, fill, outline=None, width=1):
    x0, y0, x1, y1 = bbox
    r = min(radius, (x1 - x0) // 2, (y1 - y0) // 2)
    draw.rounded_rectangle(bbox, radius=r, fill=fill, outline=outline, width=width)


def _rounded_image(img, radius):
    img = img.convert("RGBA")
    mask = Image.new("L", img.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle([(0, 0), img.size], radius=radius, fill=255)
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.paste(img, (0, 0), mask)
    return out


def _dotted_line(draw, x1, y1, x2, y2, fill, dash=6, gap=4):
    if abs(x2 - x1) >= abs(y2 - y1):
        x, end = min(x1, x2), max(x1, x2)
        while x < end:
            seg_end = min(x + dash, end)
            draw.line([(x, y1), (seg_end, y1)], fill=fill, width=1)
            x += dash + gap
    else:
        y, end = min(y1, y2), max(y1, y2)
        while y < end:
            seg_end = min(y + dash, end)
            draw.line([(x1, y), (x1, seg_end)], fill=fill, width=1)
            y += dash + gap


def _dotted_leader(draw, x, y, label, value, font, width, fill, value_fill):
    label_w = draw.textlength(label, font=font)
    value_w = draw.textlength(value, font=font)
    draw.text((x, y), label, fill=fill, font=font)
    draw.text((x + width - value_w, y), value, fill=value_fill, font=font)
    line_x0 = x + label_w + 8
    line_x1 = x + width - value_w - 8
    if line_x1 > line_x0:
        _dotted_line(draw, line_x0, y + 11, line_x1, y + 11, _hex_to_rgb(POSTER_COLORS["dotted"]))


def _draw_pill_badge(draw, x, y, text, font, bg_hex, text_hex, pad_x=14, pad_y=6):
    tw = draw.textlength(text, font=font)
    th = 18
    bbox = (x, y, x + tw + pad_x * 2, y + th + pad_y * 2)
    _rounded_rect(draw, bbox, radius=20, fill=_hex_to_rgb(bg_hex))
    draw.text((x + pad_x, y + pad_y), text, fill=_hex_to_rgb(text_hex), font=font)
    return bbox[3] - bbox[1]


def _draw_logo(draw, cx, cy, size, primary, accent):
    r = size // 2
    draw.ellipse(
        [(cx - r, cy - r), (cx + r, cy + r)],
        fill=_hex_to_rgb(primary),
    )
    leaf = [
        (cx - 4, cy + 8),
        (cx - 10, cy - 6),
        (cx, cy - 14),
        (cx + 10, cy - 6),
        (cx + 4, cy + 8),
    ]
    draw.polygon(leaf, fill=_hex_to_rgb(accent))
    draw.ellipse(
        [(cx + 6, cy + 2), (cx + 18, cy + 14)],
        outline=(255, 255, 255),
        width=2,
    )
    draw.line([(cx + 18, cy + 14), (cx + 24, cy + 20)], fill=(255, 255, 255), width=2)


def _draw_leaf_decor(canvas, w):
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    color = _hex_to_rgb(POSTER_COLORS["leaf_decor"]) + (50,)
    for ox, oy, scale in [(820, 20, 1.0), (920, 50, 0.7), (880, 90, 0.5)]:
        pts = [
            (ox, oy + int(30 * scale)),
            (ox - int(18 * scale), oy),
            (ox, oy - int(22 * scale)),
            (ox + int(18 * scale), oy),
        ]
        draw.polygon(pts, fill=color)
    canvas.paste(overlay, (0, 0), overlay)


def _draw_icon(draw, kind, x, y, size, color):
    c = _hex_to_rgb(color)
    s = size
    if kind == "warning":
        draw.polygon(
            [(x + s // 2, y), (x + s, y + s), (x, y + s)],
            fill=_hex_to_rgb(POSTER_COLORS["warning_icon"]),
        )
        draw.text((x + s // 2 - 3, y + s // 2 - 4), "!", fill=(255, 255, 255), font=_load_font(12, bold=True))
    elif kind == "calendar":
        _rounded_rect(draw, (x, y + 4, x + s, y + s), 3, c)
        draw.line([(x, y + 10), (x + s, y + 10)], fill=(255, 255, 255), width=1)
    elif kind == "leaf":
        draw.polygon(
            [(x + s // 2, y), (x, y + s // 2), (x + s // 2, y + s), (x + s, y + s // 2)],
            fill=c,
        )
    elif kind == "utensils":
        draw.line([(x + 4, y), (x + 4, y + s)], fill=c, width=2)
        draw.line([(x + s - 4, y), (x + s - 4, y + s)], fill=c, width=2)
        draw.arc([(x + 2, y + s // 2), (x + 8, y + s)], 0, 180, fill=c, width=2)
    elif kind == "pin":
        draw.ellipse([(x + 2, y), (x + s - 2, y + s - 6)], fill=c)
        draw.polygon([(x + s // 2 - 3, y + s - 8), (x + s // 2 + 3, y + s - 8), (x + s // 2, y + s)], fill=c)
    elif kind == "tag":
        draw.polygon([(x, y + 4), (x + s - 8, y + 4), (x + s, y + s // 2), (x + s - 8, y + s - 4), (x, y + s - 4)], fill=c)
    elif kind == "shield":
        draw.polygon(
            [(x + s // 2, y), (x + s, y + 4), (x + s - 2, y + s - 2), (x + s // 2, y + s), (x + 2, y + s - 2), (x, y + 4)],
            fill=c,
        )
    elif kind == "chart":
        draw.pieslice([(x, y), (x + s, y + s)], 200, 330, fill=c)
        draw.pieslice([(x, y), (x + s, y + s)], 330, 30, fill=_hex_to_rgb(POSTER_COLORS["accent_green"]))
    elif kind == "info":
        draw.ellipse([(x, y), (x + s, y + s)], outline=c, width=2)
        draw.text((x + s // 2 - 3, y + 2), "i", fill=c, font=_load_font(12, bold=True))
    elif kind == "book":
        draw.rectangle([(x + 2, y + 2), (x + s - 2, y + s - 2)], outline=c, width=2)
        draw.line([(x + s // 2, y + 2), (x + s // 2, y + s - 2)], fill=c, width=1)
    elif kind == "flame":
        draw.polygon([(x + s // 2, y), (x + s, y + s), (x, y + s)], fill=c)
    elif kind == "wheat":
        draw.line([(x + s // 2, y + s), (x + s // 2, y)], fill=c, width=2)
        draw.line([(x + 2, y + s // 2), (x + s - 2, y + 4)], fill=c, width=1)
    elif kind == "vitamin":
        draw.ellipse([(x, y), (x + s, y + s)], outline=c, width=2)
        draw.text((x + 3, y + 1), "+", fill=c, font=_load_font(11, bold=True))
    elif kind == "fiber":
        draw.arc([(x, y + 2), (x + s, y + s)], 180, 0, fill=c, width=2)


_NUTRITION_ICONS = ["flame", "wheat", "vitamin", "fiber"]


def _parse_nutrition_item(s):
    s = (s or "").strip()
    s = re.sub(r"^Approx\.\s*", "", s, flags=re.IGNORECASE)
    if ":" in s:
        label, value = s.split(":", 1)
        return label.strip(), value.strip().rstrip(".")
    return s, ""


def _fit_hero(img, max_w, max_h):
    ratio = min(max_w / img.width, max_h / img.height)
    new_w = max(1, int(img.width * ratio))
    new_h = max(1, int(img.height * ratio))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _add_drop_shadow(img, offset=(8, 12), blur=16, shadow_color=(0, 0, 0, 60)):
    pad = blur * 2
    canvas_w = img.width + pad * 2
    canvas_h = img.height + pad * 2
    shadow = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    shadow_layer = Image.new("RGBA", img.size, shadow_color)
    shadow.paste(shadow_layer, (pad + offset[0], pad + offset[1]))
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur))
    shadow.paste(img, (pad, pad), img if img.mode == "RGBA" else None)
    return shadow


def _card_with_shadow(w, h, radius=24, shadow_offset=(0, 6), blur=20):
    pad = blur * 2 + 8
    layer = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    card_bbox = (pad, pad, pad + w, pad + h)
    shadow_bbox = (pad + shadow_offset[0], pad + shadow_offset[1], pad + w + shadow_offset[0], pad + h + shadow_offset[1])
    _rounded_rect(draw, shadow_bbox, radius, fill=(0, 0, 0, 50))
    shadow_only = layer.filter(ImageFilter.GaussianBlur(blur))
    layer = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    layer.paste(shadow_only, (0, 0), shadow_only)
    draw = ImageDraw.Draw(layer)
    _rounded_rect(draw, card_bbox, radius, fill=_hex_to_rgb(POSTER_COLORS["card_white"]))
    return layer, pad


def _wrap_text(text, font, max_width, draw):
    words = text.split()
    if not words:
        return []
    lines = []
    current = words[0]
    for word in words[1:]:
        test = current + " " + word
        if draw.textlength(test, font=font) <= max_width:
            current = test
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _truncate(text, max_len=120):
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _draw_section_header(draw, x, y, title, icon_kind, colors):
    _draw_icon(draw, icon_kind, x, y, 18, colors["section_label"])
    font = _load_font(16, bold=True)
    draw.text((x + 26, y), title, fill=_hex_to_rgb(colors["section_label"]), font=font)
    return y + 28


def _draw_trait_column(draw, x, y, col_w, icon_kind, heading, body, colors):
    _draw_icon(draw, icon_kind, x + col_w // 2 - 9, y, 18, colors["accent_green"])
    font_head = _load_font(13, bold=True)
    font_body = _load_font(14)
    hw = draw.textlength(heading, font=font_head)
    draw.text((x + (col_w - hw) / 2, y + 24), heading, fill=_hex_to_rgb(colors["primary"]), font=font_head)
    for i, line in enumerate(_wrap_text(body, font_body, col_w - 16, draw)):
        lw = draw.textlength(line, font=font_body)
        draw.text(
            (x + (col_w - lw) / 2, y + 48 + i * 20),
            line,
            fill=_hex_to_rgb(colors["muted"]),
            font=font_body,
        )


def _draw_info_block(draw, x, y, icon_kind, text, max_w, colors):
    _draw_icon(draw, icon_kind, x, y + 2, 18, colors["accent_green"])
    font = _load_font(15)
    lines = _wrap_text(text, font, max_w - 30, draw)
    for i, line in enumerate(lines):
        draw.text((x + 28, y + i * 22), line, fill=_hex_to_rgb(colors["body"]), font=font)
    return y + max(len(lines), 1) * 22 + 12


def build_poster(image_bytes, metadata):
    """
    Build a shareable PNG poster.

    metadata: dict with prediction, confidence, fruit_info (optional).
    Returns PNG bytes.
    """
    colors = POSTER_COLORS
    w = POSTER_WIDTH
    h = POSTER_HEIGHT
    footer_top = h - FOOTER_H

    prediction = metadata.get("prediction", "Unknown")
    confidence = float(metadata.get("confidence", 0))
    fruit_info = metadata.get("fruit_info") or {}

    canvas = _draw_gradient_background(w, h, colors["gradient_top"], colors["gradient_bottom"])
    _draw_leaf_decor(canvas, w)
    draw = ImageDraw.Draw(canvas)

    # --- Header ---
    logo_cx, logo_cy = MARGIN + 28, 56
    _draw_logo(draw, logo_cx, logo_cy, 56, colors["primary"], colors["accent_green"])
    font_title_main = _load_font(36, bold=True)
    font_title_sub = _load_font(18)
    draw.text((MARGIN + 72, 36), "LOCAL FRUIT", fill=_hex_to_rgb(colors["primary"]), font=font_title_main)
    sub = "IDENTIFICATION SYSTEM"
    draw.text((MARGIN + 72, 78), sub, fill=_hex_to_rgb(colors["accent_green"]), font=font_title_sub)

    y = 120

    # --- Disclaimer ---
    if confidence < train.CONFIDENCE_MEDIUM:
        banner_h = 44
        banner_x0, banner_x1 = MARGIN, w - MARGIN
        _rounded_rect(
            draw,
            (banner_x0, y, banner_x1, y + banner_h),
            radius=10,
            fill=_hex_to_rgb(colors["warning_bg"]),
        )
        _draw_icon(draw, "warning", banner_x0 + 16, y + 10, 22, colors["warning_icon"])
        font_disc = _load_font(18)
        disc_text = "Low confidence — please verify before sharing."
        draw.text(
            (banner_x0 + 48, y + 11),
            disc_text,
            fill=_hex_to_rgb(colors["warning_text"]),
            font=font_disc,
        )
        y += banner_h + 16
    else:
        y += 8

    # --- Main identification card ---
    card_w = w - MARGIN * 2
    card_h = 520
    card_layer, card_pad = _card_with_shadow(card_w, card_h)
    card_x = MARGIN - card_pad
    card_y = y - card_pad
    canvas.paste(card_layer, (card_x, card_y), card_layer)
    card_draw = ImageDraw.Draw(canvas)

    inner_x = MARGIN + 24
    inner_y = y + 24
    hero_max_w = int(card_w * 0.40)
    hero_max_h = card_h - 160

    hero_src = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    hero = _fit_hero(hero_src, hero_max_w, hero_max_h)
    hero = _rounded_image(hero, 16)
    hero = _add_drop_shadow(hero, offset=(4, 6), blur=12, shadow_color=(0, 0, 0, 45))
    hero_x = inner_x + (hero_max_w - hero.width) // 2 + 8
    hero_y = inner_y + (hero_max_h - hero.height) // 2
    canvas.paste(hero, (hero_x, hero_y), hero)

    right_x = inner_x + hero_max_w + 32
    right_w = card_w - hero_max_w - 72
    ry = inner_y

    title = prediction.upper() if len(prediction) <= 24 else prediction
    font_fruit = _load_font(44, bold=True)
    for i, line in enumerate(_wrap_text(title, font_fruit, right_w, card_draw)):
        card_draw.text((right_x, ry + i * 48), line, fill=_hex_to_rgb(colors["title"]), font=font_fruit)
    ry += 52 * max(1, len(_wrap_text(title, font_fruit, right_w, card_draw)))

    english = fruit_info.get("english_name") or train.local_to_english_name(prediction)
    if english:
        font_sub = _load_font(22, italic=True)
        card_draw.text(
            (right_x, ry),
            f"({english})",
            fill=_hex_to_rgb(colors["subtitle"]),
            font=font_sub,
        )
        ry += 32

    badge_text = f"{confidence * 100:.2f}% CONFIDENCE"
    font_badge = _load_font(14, bold=True)
    badge_h = _draw_pill_badge(
        card_draw, right_x, ry, badge_text, font_badge, colors["badge_bg"], colors["badge_text"]
    )
    ry += badge_h + 14

    description = _truncate(fruit_info.get("description", ""), 220)
    if not description:
        description = f"{prediction} is a local fruit included in this dataset."
    font_desc = _load_font(16)
    for line in _wrap_text(description, font_desc, right_w, card_draw):
        card_draw.text((right_x, ry), line, fill=_hex_to_rgb(colors["body"]), font=font_desc)
        ry += 24

    trait_y = y + card_h - 120
    _dotted_line(card_draw, inner_x, trait_y, MARGIN + card_w - 24, trait_y, _hex_to_rgb(colors["dotted"]))
    trait_y += 16
    col_w = (card_w - 48) // 3
    season_text = _truncate(fruit_info.get("season") or "Limited availability in Malaysia", 55)
    uses_text = _truncate(fruit_info.get("uses") or "Eaten fresh or used in local dishes.", 55)
    traits = [
        ("calendar", "SEASONAL", season_text),
        ("leaf", "NATURAL", "Wild-harvested local fruit"),
        ("utensils", "CULINARY USE", uses_text),
    ]
    for i, (icon, heading, body) in enumerate(traits):
        col_x = inner_x + i * col_w
        _draw_trait_column(card_draw, col_x, trait_y, col_w, icon, heading, body, colors)
        if i < 2:
            div_x = col_x + col_w
            _dotted_line(card_draw, div_x, trait_y, div_x, trait_y + 90, _hex_to_rgb(colors["dotted"]))

    y += card_h + 24

    # --- Info grid ---
    grid_h = footer_top - y - 16
    col_gap = 24
    col_w = (card_w - col_gap) // 2
    left_x = MARGIN
    right_x_grid = MARGIN + col_w + col_gap

    gy = _draw_section_header(draw, left_x, y, "NUTRITIONAL INFORMATION", "chart", colors)
    nutrition = fruit_info.get("nutrition") or []
    font_row = _load_font(16)
    icon_y = gy
    for i, item in enumerate(nutrition[:4]):
        label, value = _parse_nutrition_item(item)
        icon_kind = _NUTRITION_ICONS[i] if i < len(_NUTRITION_ICONS) else "leaf"
        _draw_icon(draw, icon_kind, left_x, icon_y + 2, 16, colors["accent_green"])
        _dotted_leader(
            draw,
            left_x + 24,
            icon_y,
            label,
            value,
            font_row,
            col_w - 24,
            _hex_to_rgb(colors["body"]),
            _hex_to_rgb(colors["primary"]),
        )
        icon_y += 32

    font_foot = _load_font(13, italic=True)
    draw.text(
        (left_x, footer_top - 72),
        "*Values are approximate and may vary.",
        fill=_hex_to_rgb(colors["muted"]),
        font=font_foot,
    )

    gy2 = _draw_section_header(draw, right_x_grid, y, "ADDITIONAL INFORMATION", "info", colors)
    season_full = fruit_info.get("season") or "Availability varies by region in Malaysia."
    price_full = fruit_info.get("price") or "Price varies by market and season."
    uses_full = fruit_info.get("uses") or "Commonly consumed fresh or used in local dishes."
    block_y = gy2
    block_y = _draw_info_block(draw, right_x_grid, block_y, "pin", season_full, col_w, colors)
    block_y = _draw_info_block(draw, right_x_grid, block_y, "tag", price_full, col_w, colors)
    _draw_info_block(draw, right_x_grid, block_y, "shield", uses_full, col_w, colors)

    # --- Footer ---
    draw.rectangle([(0, footer_top), (w, h)], fill=_hex_to_rgb(colors["footer_bg"]))
    footer_text = "Identified with MobileNetV3 | Malaysian Local Fruits"
    font_footer = _load_font(17)
    ftw = draw.textlength(footer_text, font=font_footer)
    icon_w = 22
    total_w = icon_w + 8 + ftw
    fx = (w - total_w) / 2
    fy = footer_top + (FOOTER_H - 18) // 2
    _draw_icon(draw, "book", int(fx), int(fy - 2), 18, colors["footer_text"])
    draw.text(
        (fx + icon_w + 8, fy),
        footer_text,
        fill=_hex_to_rgb(colors["footer_text"]),
        font=font_footer,
    )

    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


if __name__ == "__main__":
    info_path = train.FRUIT_INFO_PATH
    with open(info_path, encoding="utf-8") as f:
        fruit_db = json.load(f)
    sample_info = fruit_db.get("Bambangan", {})
    placeholder = io.BytesIO()
    Image.new("RGB", (400, 500), (180, 180, 180)).save(placeholder, format="PNG")
    png = build_poster(
        placeholder.getvalue(),
        {
            "prediction": "Bambangan",
            "confidence": 0.242,
            "fruit_info": sample_info,
        },
    )
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "expected PNG output"
    assert len(png) > 10_000, f"poster too small: {len(png)} bytes"
    out_dir = train.OUTPUTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sample-poster.png")
    with open(out_path, "wb") as f:
        f.write(png)
    print(f"OK: wrote {out_path} ({len(png)} bytes)")
