"""Selectable-text view: one SVG per page, the page image plus invisible
<text> elements on the detected line boxes (textLength stretches each
string to its box width)."""

import base64
import html
import io

_STYLE = ("text{fill:transparent;cursor:text;white-space:pre;"
          "font-family:sans-serif}"
          "text::selection{fill:transparent;"
          "background-color:rgba(26,115,232,.35)}")


def _data_uri(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def page_svg(image, results):
    w, h = image.size
    texts = []
    for r in results:
        x, y, bw, bh = r.bbox
        if not r.text.strip() or bw < 1 or bh < 2:
            continue
        # y is the baseline in SVG
        texts.append(
            f'<text x="{x:.0f}" y="{y + bh * 0.8:.0f}" '
            f'font-size="{bh * 0.8:.0f}" textLength="{bw:.0f}" '
            f'lengthAdjust="spacingAndGlyphs">{html.escape(r.text)}</text>')
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
            f'style="width:100%;height:auto;display:block">'
            f'<style>{_STYLE}</style>'
            f'<image href="{_data_uri(image)}" width="{w}" height="{h}"/>'
            + "".join(texts) + "</svg>")


def render(pages_results):
    """[(page_image, [LineResult, ...]), ...]"""
    return "".join(f'<div style="margin-bottom:12px">{page_svg(img, res)}'
                   "</div>" for img, res in pages_results)
