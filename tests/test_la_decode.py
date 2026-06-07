from mlx_cv.models.locateanything import (
    LocateAnythingConfig,
    TokenScheme,
    parse_grounding_text,
    parse_grounding_tokens,
)

S = TokenScheme()


def _coords(vals):
    return [S.coord_start + v for v in vals]


def test_scheme_from_config():
    assert TokenScheme.from_config(LocateAnythingConfig()) == S


def test_parse_single_box_with_label():
    toks = [S.ref_start, 100, 101, S.ref_end,
            S.box_start, *_coords([64, 152, 273, 244]), S.box_end]
    items = parse_grounding_tokens(toks, S)
    assert len(items) == 1
    assert items[0].kind == "box"
    assert items[0].coords == [64, 152, 273, 244]
    assert items[0].label == [100, 101]


def test_parse_point():
    toks = [S.box_start, *_coords([10, 20]), S.box_end]
    items = parse_grounding_tokens(toks, S)
    assert items[0].kind == "point" and items[0].coords == [10, 20]


def test_parse_empty_box_skipped():
    assert parse_grounding_tokens([S.box_start, S.none_id, S.box_end], S) == []


def test_multi_instance_shares_label():
    toks = [S.ref_start, 200, S.ref_end,
            S.box_start, *_coords([1, 2, 3, 4]), S.box_end,
            S.box_start, *_coords([5, 6, 7, 8]), S.box_end]
    items = parse_grounding_tokens(toks, S)
    assert len(items) == 2
    assert all(it.label == [200] for it in items)
    assert items[1].coords == [5, 6, 7, 8]


def test_parse_text_form():
    text = "<ref>remote</ref><box><64><152><273><244></box><ref>cat</ref><box><1><2><3><4></box>"
    items = parse_grounding_text(text)
    assert [it.label for it in items] == ["remote", "cat"]
    assert items[0].coords == [64, 152, 273, 244] and items[0].kind == "box"
