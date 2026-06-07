import numpy as np

from mlx_cv.ops import coords_to_pixels, token_to_coord

# LocateAnything coordinate-token scheme (§16): coord_start = 151677, range [0, 1000]
COORD_START = 151677


def test_token_to_coord():
    assert token_to_coord(151741, COORD_START) == 64       # the doc's <64> example
    assert token_to_coord(COORD_START, COORD_START) == 0    # <0>
    assert token_to_coord(152677, COORD_START) == 1000      # <1000> = coord_end
    assert token_to_coord(99, COORD_START) == 0             # clamp below


def test_coords_to_pixels():
    px = coords_to_pixels([[500, 250]], (200, 400))         # H=200, W=400
    assert np.allclose(px, [[200, 50]])                     # 500/1000*400, 250/1000*200
