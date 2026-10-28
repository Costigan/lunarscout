"""QGIS default styles stored inside the chain-map GeoPackages (private).

A GeoPackage can carry layer styles in a ``layer_styles`` table (QGIS's own convention, also written by
"Save as default" in the layer properties); QGIS applies the one marked as the default when it adds the layer.
The line layers are drawn by chain length, one colour and one width per number of seeds (2 to 10, then "more"),
brighter and wider for longer chains, longest on top. The isolated seeds are small translucent grey dots.
"""

from __future__ import annotations

import uuid
from xml.sax.saxutils import quoteattr

# Eight steps of a dark-to-bright sequential ramp (inferno-like): brighter means a longer chain, and every step is
# visible on a black background and on a grey DEM.
CHAIN_COLOURS = ("#420a68", "#6a176e", "#932667", "#bc3754", "#dd513a", "#f37819", "#fca50a", "#f6d746", "#fcffa4")
CHAIN_LENGTHS = tuple(range(2, 2 + len(CHAIN_COLOURS)))        # 2 .. 10 seeds
_FIRST_WIDTH_MM, _WIDTH_STEP_MM = 0.26, 0.14
_NO_DATA_DEFINED = ('<data_defined_properties><Option type="Map"><Option name="name" value="" type="QString"/>'
                    '<Option name="properties"/><Option name="type" value="collection" type="QString"/></Option>'
                    '</data_defined_properties>')


def _rgba(colour: str, alpha: int = 255) -> str:
    return f"{int(colour[1:3], 16)},{int(colour[3:5], 16)},{int(colour[5:7], 16)},{alpha}"


def _options(values: dict) -> str:
    body = "".join(f'<Option name="{name}" value={quoteattr(str(value))} type="QString"/>' for name, value in values.items())
    return f'<Option type="Map">{body}</Option>'


def _line_symbol(name: str, colour: str, width: float) -> str:
    options = {"capstyle": "round", "joinstyle": "round", "line_color": _rgba(colour), "line_style": "solid",
               "line_width": f"{width:g}", "line_width_unit": "MM"}
    return (f'<symbol name="{name}" type="line" alpha="1" force_rhr="0" clip_to_extent="1" is_animated="0" frame_rate="10">'
            f'{_NO_DATA_DEFINED}<layer id="{{{uuid.uuid4()}}}" enabled="1" class="SimpleLine" locked="0" pass="0">'
            f'{_options(options)}{_NO_DATA_DEFINED}</layer></symbol>')


def line_style_qml(attribute: str = "chain_length") -> str:
    """Categorized by ``attribute`` (the chain length in seeds), drawn shortest first so longer chains are on top."""
    colours = list(CHAIN_COLOURS)
    widths = [_FIRST_WIDTH_MM + _WIDTH_STEP_MM * k for k in range(len(colours))]
    categories = [f'<category uuid="{{{uuid.uuid4()}}}" value="{n}" type="long" render="true" symbol="{k}" label="{n} seeds"/>'
                  for k, n in enumerate(CHAIN_LENGTHS)]
    categories.append(f'<category uuid="{{{uuid.uuid4()}}}" value="" type="string" render="true" '
                      f'symbol="{len(colours)}" label="more"/>')
    symbols = [_line_symbol(str(k), colour, width) for k, (colour, width) in enumerate(zip(colours, widths))]
    symbols.append(_line_symbol(str(len(colours)), colours[-1], widths[-1]))
    return (
        "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
        '<qgis version="3.34.4-Prizren" styleCategories="Symbology">'
        f'<renderer-v2 attr="{attribute}" symbollevels="0" referencescale="-1" type="categorizedSymbol" forceraster="0" '
        'enableorderby="1">'
        f'<categories>{"".join(categories)}</categories><symbols>{"".join(symbols)}</symbols><rotation/><sizescale/>'
        f'<orderby><orderByClause nullsFirst="0" asc="1">{attribute}</orderByClause></orderby></renderer-v2>'
        '<layerOpacity>1</layerOpacity></qgis>\n')


def point_style_qml() -> str:
    """Small translucent grey dots: seeds that are alone, behind everything else."""
    options = {"angle": "0", "cap_style": "square", "color": "140,140,140,150", "horizontal_anchor_point": "1",
               "joinstyle": "bevel", "name": "circle", "offset": "0,0", "outline_color": "35,35,35,255",
               "outline_style": "no", "outline_width": "0", "outline_width_unit": "MM", "scale_method": "diameter",
               "size": "0.9", "size_unit": "MM", "vertical_anchor_point": "1"}
    symbol = (f'<symbol name="0" type="marker" alpha="1" force_rhr="0" clip_to_extent="1" is_animated="0" frame_rate="10">'
              f'{_NO_DATA_DEFINED}<layer id="{{{uuid.uuid4()}}}" enabled="1" class="SimpleMarker" locked="0" pass="0">'
              f'{_options(options)}{_NO_DATA_DEFINED}</layer></symbol>')
    return ("<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
            '<qgis version="3.34.4-Prizren" styleCategories="Symbology">'
            '<renderer-v2 symbollevels="0" referencescale="-1" type="singleSymbol" forceraster="0" enableorderby="0">'
            f'<symbols>{symbol}</symbols><rotation/><sizescale/></renderer-v2><layerOpacity>1</layerOpacity></qgis>\n')
