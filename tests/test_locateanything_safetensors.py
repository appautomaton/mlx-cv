import json

from mlx_cv.hub import read_safetensors_metadata, rewrite_safetensors_metadata


def _tiny_safetensors(path, *, metadata=None):
    header = {
        "weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
        "__metadata__": metadata or {},
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + b"\0\0")


def test_metadata_rewrite_streams_tensor_bytes_and_is_atomic(tmp_path):
    source = tmp_path / "source.safetensors"
    output = tmp_path / "output.safetensors"
    _tiny_safetensors(source, metadata={"old": "value"})
    rewrite_safetensors_metadata(source, output, {"format": "new"})
    assert read_safetensors_metadata(output) == {"format": "new"}
    assert output.read_bytes()[-2:] == b"\0\0"
    assert source.exists()
