"""Golden-fixture parity contract + bisect harness (§11).

Trust is a first-class architectural element: every model ships fixed-input
reference outputs (and selected intermediate *taps*), and CI asserts the MLX
output matches within tolerance. :func:`bisect` localizes a divergence to the
first tap that drifts, turning "the boxes are wrong" into "the projector is wrong".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["ParityCase", "allclose_tree", "assert_parity", "bisect", "save_case", "load_case"]


@dataclass
class ParityCase:
    """A reproducible parity fixture for one model on one fixed input."""

    name: str
    inputs: Any
    expected: Any
    taps: dict[str, Any] | None = None     # ordered forward -> deeper


def _close(a, b, atol: float, rtol: float) -> bool:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and bool(np.allclose(a, b, atol=atol, rtol=rtol))


def allclose_tree(a, b, *, atol: float = 1e-4, rtol: float = 1e-4) -> bool:
    """Recursively compare nested dict / list / array trees within tolerance."""
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            return False
        return all(allclose_tree(a[k], b[k], atol=atol, rtol=rtol) for k in a)
    if isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            return False
        return all(allclose_tree(x, y, atol=atol, rtol=rtol) for x, y in zip(a, b))
    return _close(a, b, atol, rtol)


def assert_parity(got, expected, *, atol: float = 1e-4, rtol: float = 1e-4, name: str = "") -> None:
    """Raise ``AssertionError`` if ``got`` diverges from ``expected``."""
    if not allclose_tree(got, expected, atol=atol, rtol=rtol):
        raise AssertionError(f"parity failed for {name!r}")


def bisect(ref_taps: dict[str, Any], got_taps: dict[str, Any], *,
           atol: float = 1e-4, rtol: float = 1e-4) -> str | None:
    """Return the first (forward-ordered) tap whose value diverges, else ``None``."""
    for key in ref_taps:  # dict preserves insertion (forward) order
        if key not in got_taps or not allclose_tree(
            ref_taps[key], got_taps[key], atol=atol, rtol=rtol
        ):
            return key
    return None


# -- fixture (de)serialization -------------------------------------------------
# Schema: one ``.npz`` per case (numpy-native, no extra deps). ``inputs`` and
# ``expected`` are ``{name: array}`` dicts; ``taps`` is an *ordered* dict whose
# forward order is preserved out-of-band in ``__tap_order__`` (npz does not
# guarantee key order), so ``bisect`` stays meaningful after a round-trip.
_PREFIX = {"in.": "inputs", "exp.": "expected", "tap.": "taps"}


def _as_array_dict(obj: Any, what: str) -> dict[str, np.ndarray]:
    if not isinstance(obj, dict):
        raise TypeError(f"fixture {what} must be a dict[str, array], got {type(obj).__name__}")
    return {str(k): np.asarray(v) for k, v in obj.items()}


def save_case(case: ParityCase, path) -> None:
    """Serialize a ``ParityCase`` to a single ``.npz`` (tap order preserved)."""
    flat: dict[str, np.ndarray] = {"__name__": np.asarray(case.name)}
    for k, v in _as_array_dict(case.inputs, "inputs").items():
        flat[f"in.{k}"] = v
    for k, v in _as_array_dict(case.expected, "expected").items():
        flat[f"exp.{k}"] = v
    taps = case.taps or {}
    flat["__tap_order__"] = np.asarray(list(taps.keys()), dtype="U")
    for k, v in _as_array_dict(taps, "taps").items():
        flat[f"tap.{k}"] = v
    np.savez(path, **flat)


def load_case(path) -> ParityCase:
    """Load a ``ParityCase`` saved by :func:`save_case` (tap order restored)."""
    z = np.load(path, allow_pickle=False)
    buckets: dict[str, dict[str, np.ndarray]] = {"inputs": {}, "expected": {}, "taps": {}}
    for key in z.files:
        for pre, bucket in _PREFIX.items():
            if key.startswith(pre):
                buckets[bucket][key[len(pre):]] = z[key]
                break
    order = [str(s) for s in z["__tap_order__"]] if "__tap_order__" in z.files else []
    taps = {k: buckets["taps"][k] for k in order} if order else None
    return ParityCase(name=str(z["__name__"]), inputs=buckets["inputs"],
                      expected=buckets["expected"], taps=taps)
