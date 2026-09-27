
import os

from PIL import Image, ImageDraw, ImageFont

DPI = 300
A4_PX = (int(8.27 * DPI), int(11.69 * DPI))  # 2481 x 3507
MARGIN = int(0.9 * DPI)

HEADING_H = int(0.32 * DPI)
FIELD_H = int(0.24 * DPI)
GAP_H = int(0.30 * DPI)


def _font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


class Page:

    def __init__(self, title: str, subtitle: str = ""):
        self.title = title
        self.subtitle = subtitle
        self.blocks: list[str] = [title]
        self._ops: list[tuple[str, str]] = []
        if subtitle:
            self.blocks.append(subtitle)

    def heading(self, text: str) -> "Page":
        self._ops.append(("heading", text))
        self.blocks.append(text)
        return self

    def field(self, text: str) -> "Page":
        self._ops.append(("field", text))
        self.blocks.append(text)
        return self

    def item(self, text: str) -> "Page":
        return self.field(f"- {text}")

    def line(self, text: str) -> "Page":
        self._ops.append(("line", text))
        self.blocks.append(text)
        return self

    def gap(self) -> "Page":
        self._ops.append(("gap", ""))
        return self

    def groups(self) -> list[list[str]]:
        header = [self.title] + ([self.subtitle] if self.subtitle else [])
        groups: list[list[str]] = []
        current = list(header)
        for kind, text in self._ops:
            if kind == "gap":
                continue
            if kind == "heading" and len(current) > len(header):
                groups.append(current)
                current = list(header)
            current.append(text)
        groups.append(current)
        return groups

    def _header_height(self, first: bool) -> int:
        y = MARGIN + (int(0.42 * DPI) if first else GAP_H)
        if first and self.subtitle:
            y += GAP_H
        return y + int(0.40 * DPI)

    def paginate(self) -> list[list[tuple[str, str]]]:
        limit = A4_PX[1] - MARGIN
        pages: list[list[tuple[str, str]]] = []
        current: list[tuple[str, str]] = []
        y = self._header_height(first=True)

        for i, (kind, text) in enumerate(self._ops):
            height = {"gap": GAP_H, "heading": HEADING_H}.get(kind, FIELD_H)
            needed = height
            if kind == "heading":
                for next_kind, _ in self._ops[i + 1:i + 3]:
                    if next_kind != "gap":
                        needed += FIELD_H

            if current and y + needed > limit:
                pages.append(current)
                current = []
                y = self._header_height(first=False)
                if kind == "gap":
                    continue
            if not current and kind == "gap":
                continue
            if y + height > limit:
                raise ValueError(
                    f"page overflow rendering {self.title!r}: the block "
                    f"{text!r} does not fit on an empty page"
                )
            current.append((kind, text))
            y += height

        if current:
            pages.append(current)
        return pages or [[]]

    def _draw(self, ops: list[tuple[str, str]], first: bool) -> Image.Image:
        img = Image.new("RGB", A4_PX, "white")
        d = ImageDraw.Draw(img)

        title_f = _font(int(0.30 * DPI), bold=True)
        sub_f = _font(int(0.15 * DPI))
        head_f = _font(int(0.20 * DPI), bold=True)
        body_f = _font(int(0.145 * DPI))

        y = MARGIN
        if first:
            d.text((MARGIN, y), self.title, fill="black", font=title_f)
            y += int(0.42 * DPI)
            if self.subtitle:
                d.text((MARGIN, y), self.subtitle, fill="black", font=sub_f)
                y += GAP_H
        else:
            d.text((MARGIN, y), f"{self.title} (continued)", fill="black",
                   font=sub_f)
            y += GAP_H
        d.line([(MARGIN, y), (A4_PX[0] - MARGIN, y)], fill="black", width=4)
        y += int(0.40 * DPI)

        for kind, text in ops:
            if kind == "gap":
                y += GAP_H
            elif kind == "heading":
                d.text((MARGIN, y), text, fill="black", font=head_f)
                y += HEADING_H
            elif kind == "field":
                d.text((MARGIN + int(0.25 * DPI), y), text, fill="black",
                       font=body_f)
                y += FIELD_H
            else:
                d.text((MARGIN, y), text, fill="black", font=body_f)
                y += FIELD_H
        return img

    def to_images(self) -> list[Image.Image]:
        return [
            self._draw(ops, first=(i == 0))
            for i, ops in enumerate(self.paginate())
        ]

    def to_image(self) -> Image.Image:
        return self.to_images()[0]

    def save(self, path: str) -> str:
        images = self.to_images()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        images[0].save(path, "PDF", resolution=DPI, save_all=True,
                       append_images=images[1:])
        return path
