# Normalized crop coordinates measured from the supplied 1836x2376 Reading pages.
# x coordinates intentionally include the complete table width so OCR retains
# the left field labels and right-side values.

REFERENCE_WIDTH = 1836
REFERENCE_HEIGHT = 2376
X0 = 80
X1 = 1770

PAGE_A_BOXES = [
    ("general", 295, 453),
    ("power_speed", 453, 690),
    ("electronic_control", 690, 828),
    ("cylinder_pressure", 828, 1262),
    ("turbocharger", 1262, 1545),
    ("scavenge_air", 1545, 1933),
    ("exhaust_gas", 1933, 2277),
]

PAGE_B_BOXES = [
    ("fuel_oil", 295, 498),
    ("cylinder_lubrication", 498, 618),
    ("cylinder_condition", 618, 851),
    ("crankcase", 851, 1114),
    ("liner_wall", 1114, 1318),
    ("tc_sac_media", 1318, 1469),
    ("engine_media_plant", 1469, 1823),
]


def _scale(v, actual, reference):
    return int(round(v * actual / reference))


def boxes_for_page(slot, width, height):
    specs = PAGE_A_BOXES if slot == 0 else PAGE_B_BOXES
    x0 = _scale(X0, width, REFERENCE_WIDTH)
    x1 = _scale(X1, width, REFERENCE_WIDTH)
    out = []
    for region, y0, y1 in specs:
        out.append((region, x0, _scale(y0, height, REFERENCE_HEIGHT), x1, _scale(y1, height, REFERENCE_HEIGHT)))
    return out
