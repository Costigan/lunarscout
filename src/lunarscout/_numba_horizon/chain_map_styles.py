"""Chain-length intervals, their colours and the QGIS styles that use them (private).

Chains of 2 seeds are one interval; chains of 3 or more are cut at the quartiles of their lengths into up to four more,
so five intervals at most. Each gets a distinct, saturated hue and a brightness that rises with the length, so that the
lines stand out against black or a dark hillshade and the longer chains stand out most: violet, rose, orange, yellow and
green. Havens are white and route pixels that pass next to a permanently shadowed region (PSR) are aqua. The same
intervals colour the images and the GeoPackage line layers (rule-based, shortest first so longer chains are on top);
the GeoPackages carry their styles in the ``layer_styles`` table, which QGIS applies as the default when it adds a layer.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from xml.sax.saxutils import quoteattr

import numpy as np
import numpy.typing as npt

#: Distinct hues (violet, rose, orange, yellow, green) at full brightness and strong saturation, in order of rising
#: relative luminance.
PALETTE = ("#6a5cff", "#ff4f6e", "#ff7a1a", "#ffcc00", "#7dff3a")
LINE_WIDTHS_MM = (0.3, 0.45, 0.65, 0.9, 1.2)
HAVEN_COLOUR = "#ffffff"
PSR_COLOUR = "#00ffff"
#: The start of a chain in the monthly pixel file: bright magenta, in the widest gap of the hues above (violet to rose).
ROOT_COLOUR = "#ff00ff"


@dataclass(frozen=True)
class Interval:
    """The chains with ``first <= length <= last`` (``last`` None: no upper limit)."""

    first: int
    last: int | None
    label: str
    colour: str
    width_mm: float

    def filter(self, attribute: str) -> str:
        if self.last is None:
            return f'"{attribute}" >= {self.first}'
        if self.first == self.last:
            return f'"{attribute}" = {self.first}'
        return f'"{attribute}" >= {self.first} AND "{attribute}" <= {self.last}'


def _label(first: int, last: int | None, maximum: int) -> str:
    top = maximum if last is None else last
    return f"{first} seeds" if first == top else f"{first}-{top} seeds"


def build_intervals(lengths: npt.ArrayLike) -> list[Interval]:
    """The intervals of the chain lengths ``lengths`` (one per chain, each 2 or more): length 2 alone, and the chains of
    3 or more cut at the quartiles of their own lengths (the lengths at which 25, 50 and 75% of them are reached;
    repeated cut points merge, so there may be fewer than five intervals)."""
    lengths = np.asarray(lengths)
    longer = lengths[lengths >= 3]
    bounds: list[int] = []
    if len(longer):
        quartiles = np.percentile(longer, [25, 50, 75], method="inverted_cdf")
        bounds = sorted({int(q) for q in quartiles if int(q) < int(longer.max())})
    firsts = [2] + ([3] + [b + 1 for b in bounds] if len(longer) else [])
    lasts: list[int | None] = [2] + ([*bounds, None] if len(longer) else [])
    maximum = int(longer.max()) if len(longer) else 2
    picks = np.round(np.linspace(0, len(PALETTE) - 1, len(firsts))).astype(int)
    return [Interval(f, l, _label(f, l, maximum), PALETTE[p], LINE_WIDTHS_MM[p])
            for f, l, p in zip(firsts, lasts, picks)]


def class_of(length: npt.ArrayLike, intervals: list[Interval]) -> npt.NDArray[np.uint8]:
    """The interval class (1 to ``len(intervals)``) of chain lengths, 0 below the first interval; lengths above the
    last interval's limit belong to it."""
    length = np.asarray(length)
    classes = np.searchsorted([i.first for i in intervals], length, side="right")
    return classes.astype(np.uint8)


def _rgb(colour: str) -> tuple[int, int, int]:
    return int(colour[1:3], 16), int(colour[3:5], 16), int(colour[5:7], 16)


def lookup_table(intervals: list[Interval]) -> npt.NDArray[np.uint8]:
    """RGBA of the image classes: 0 transparent, 1 to ``n`` the intervals, ``n + 1`` aqua (a PSR pass), ``n + 2`` white
    (a haven)."""
    colours = [i.colour for i in intervals] + [PSR_COLOUR, HAVEN_COLOUR]
    table = np.zeros((len(colours) + 1, 4), dtype=np.uint8)
    for k, colour in enumerate(colours, start=1):
        table[k] = (*_rgb(colour), 255)
    return table


_NO_DATA_DEFINED = ('<data_defined_properties><Option type="Map"><Option name="name" value="" type="QString"/>'
                    '<Option name="properties"/><Option name="type" value="collection" type="QString"/></Option>'
                    '</data_defined_properties>')


def _rgba(colour: str, alpha: int = 255) -> str:
    return "{},{},{},{}".format(*_rgb(colour), alpha)


def _options(values: dict) -> str:
    body = "".join(f'<Option name="{name}" value={quoteattr(str(value))} type="QString"/>' for name, value in values.items())
    return f'<Option type="Map">{body}</Option>'


def _line_symbol(name: str, colour: str, width: float) -> str:
    options = {"capstyle": "round", "joinstyle": "round", "line_color": _rgba(colour), "line_style": "solid",
               "line_width": f"{width:g}", "line_width_unit": "MM"}
    return (f'<symbol name="{name}" type="line" alpha="1" force_rhr="0" clip_to_extent="1" is_animated="0" frame_rate="10">'
            f'{_NO_DATA_DEFINED}<layer id="{{{uuid.uuid4()}}}" enabled="1" class="SimpleLine" locked="0" pass="0">'
            f'{_options(options)}{_NO_DATA_DEFINED}</layer></symbol>')


def line_style_qml(intervals: list[Interval], attribute: str = "chain_length") -> str:
    """Rule-based by the intervals of ``attribute`` (the chain length in seeds): a colour and a width per interval,
    the longer chains drawn last so they are on top."""
    rules = "".join(
        f'<rule key="{{{uuid.uuid4()}}}" filter={quoteattr(i.filter(attribute))} label={quoteattr(i.label)} symbol="{k}"/>'
        for k, i in enumerate(intervals))
    symbols = "".join(_line_symbol(str(k), i.colour, i.width_mm) for k, i in enumerate(intervals))
    return ("<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
            '<qgis version="3.34.4-Prizren" styleCategories="Symbology">'
            '<renderer-v2 type="RuleRenderer" symbollevels="0" referencescale="-1" forceraster="0" enableorderby="0">'
            f'<rules key="{{{uuid.uuid4()}}}">{rules}</rules><symbols>{symbols}</symbols></renderer-v2>'
            '<layerOpacity>1</layerOpacity></qgis>\n')


def point_style_qml() -> str:
    """Small white dots (the havens' colour): the seeds that are alone, behind everything else."""
    options = {"angle": "0", "cap_style": "square", "color": _rgba(HAVEN_COLOUR), "horizontal_anchor_point": "1",
               "joinstyle": "bevel", "name": "circle", "offset": "0,0", "outline_color": "35,35,35,255",
               "outline_style": "no", "outline_width": "0", "outline_width_unit": "MM", "scale_method": "diameter",
               "size": "0.9", "size_unit": "MM", "vertical_anchor_point": "1"}
    symbol = ('<symbol name="0" type="marker" alpha="1" force_rhr="0" clip_to_extent="1" is_animated="0" frame_rate="10">'
              f'{_NO_DATA_DEFINED}<layer id="{{{uuid.uuid4()}}}" enabled="1" class="SimpleMarker" locked="0" pass="0">'
              f'{_options(options)}{_NO_DATA_DEFINED}</layer></symbol>')
    return ("<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
            '<qgis version="3.34.4-Prizren" styleCategories="Symbology">'
            '<renderer-v2 symbollevels="0" referencescale="-1" type="singleSymbol" forceraster="0" enableorderby="0">'
            f'<symbols>{symbol}</symbols><rotation/><sizescale/></renderer-v2><layerOpacity>1</layerOpacity></qgis>\n')
