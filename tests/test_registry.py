import pytest

from mlx_cv.core.registry import BACKBONES, Registry, register_backbone
import mlx_cv.heads.dense as _dense  # noqa: F401  (import self-registers)
from mlx_cv import HEADS


def test_register_get_list():
    r = Registry("t")

    @r.register("a")
    class A:
        pass

    assert "a" in r
    assert r.get("a") is A
    assert r.list() == ["a"]
    assert len(r) == 1


def test_register_direct_call():
    r = Registry("t")
    obj = object()
    assert r.register("x", obj) is obj
    assert r.get("x") is obj


def test_duplicate_raises():
    r = Registry("t")
    r.register("a", object())
    with pytest.raises(KeyError):
        r.register("a", object())


def test_unknown_raises():
    r = Registry("t")
    with pytest.raises(KeyError):
        r.get("nope")


def test_backbone_two_kinds():
    @register_backbone("dummy-vit", kind="vision")
    class V:
        pass

    @register_backbone("dummy-llm", kind="llm")
    class L:
        pass

    assert "dummy-vit" in BACKBONES.list(kind="vision")
    assert "dummy-llm" in BACKBONES.list(kind="llm")
    assert "dummy-vit" not in BACKBONES.list(kind="llm")


def test_dpt_head_registered():
    assert "dpt" in HEADS
