import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_locateanything_staged_bf16_real_image_regression():
    required = os.environ.get("MLX_CV_REQUIRE_LOCATEANYTHING_RELEASE_GATE") == "1"
    package = os.environ.get("MLX_CV_LOCATEANYTHING_RELEASE_PACKAGE")
    if not required:
        pytest.skip("set MLX_CV_REQUIRE_LOCATEANYTHING_RELEASE_GATE=1 for the real package gate")
    if not package:
        pytest.fail("MLX_CV_LOCATEANYTHING_RELEASE_PACKAGE is required")
    root = Path(__file__).parents[1]
    subprocess.run(
        [
            sys.executable,
            str(root / "tools/locateanything_release_regression.py"),
            "--package",
            package,
            "--source-root",
            str(root),
        ],
        check=True,
    )
