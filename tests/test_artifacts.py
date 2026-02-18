from unittest.mock import MagicMock, patch

from dialectic.artifacts import Artifact, get_artifact, map_hf_key_to_dialectic


def test_artifact_dataclass():
    a = Artifact(urls=["https://example.com/weights.bin"], filenames=["weights.bin"])
    assert a.urls == ["https://example.com/weights.bin"]
    assert a.filenames == ["weights.bin"]


def test_get_artifact_returns_cached(tmp_path):
    cached_file = tmp_path / "weights.bin"
    cached_file.write_bytes(b"fake weights")
    artifact = Artifact(
        urls=["https://example.com/weights.bin"], filenames=["weights.bin"]
    )

    with patch("dialectic.artifacts.WEIGHTS_CACHE", tmp_path):
        result = get_artifact(artifact)

    assert result == [cached_file]


def test_get_artifact_downloads_when_missing(tmp_path):
    artifact = Artifact(
        urls=["https://example.com/weights.bin"], filenames=["weights.bin"]
    )

    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]

    with (
        patch("dialectic.artifacts.WEIGHTS_CACHE", tmp_path),
        patch(
            "dialectic.artifacts.requests.get", return_value=mock_response
        ) as mock_get,
    ):
        result = get_artifact(artifact)

    mock_get.assert_called_once_with("https://example.com/weights.bin", stream=True)
    mock_response.raise_for_status.assert_called_once()
    assert result == [tmp_path / "weights.bin"]
    assert result[0].read_bytes() == b"chunk1chunk2"


def test_get_artifact_creates_parent_dirs(tmp_path):
    cache_dir = tmp_path / "nested" / "dir"
    artifact = Artifact(urls=["https://example.com/w.bin"], filenames=["w.bin"])

    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"data"]

    with (
        patch("dialectic.artifacts.WEIGHTS_CACHE", cache_dir),
        patch("dialectic.artifacts.requests.get", return_value=mock_response),
    ):
        result = get_artifact(artifact)

    assert result[0].parent.exists()
    assert result[0].read_bytes() == b"data"


def test_get_artifact_multiple_files(tmp_path):
    artifact = Artifact(
        urls=["https://example.com/a.bin", "https://example.com/b.bin"],
        filenames=["a.bin", "b.bin"],
    )

    mock_resp_a = MagicMock()
    mock_resp_a.iter_content.return_value = [b"aaa"]
    mock_resp_b = MagicMock()
    mock_resp_b.iter_content.return_value = [b"bbb"]

    with (
        patch("dialectic.artifacts.WEIGHTS_CACHE", tmp_path),
        patch(
            "dialectic.artifacts.requests.get", side_effect=[mock_resp_a, mock_resp_b]
        ),
    ):
        result = get_artifact(artifact)

    assert len(result) == 2
    assert result[0].read_bytes() == b"aaa"
    assert result[1].read_bytes() == b"bbb"


def test_get_artifact_partial_cache(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"cached")
    artifact = Artifact(
        urls=["https://example.com/a.bin", "https://example.com/b.bin"],
        filenames=["a.bin", "b.bin"],
    )

    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"downloaded"]

    with (
        patch("dialectic.artifacts.WEIGHTS_CACHE", tmp_path),
        patch(
            "dialectic.artifacts.requests.get", return_value=mock_response
        ) as mock_get,
    ):
        result = get_artifact(artifact)

    mock_get.assert_called_once_with("https://example.com/b.bin", stream=True)
    assert result[0].read_bytes() == b"cached"
    assert result[1].read_bytes() == b"downloaded"


def test_map_hf_key_strips_model_prefix():
    assert map_hf_key_to_dialectic("model.layers.0.weight") == "layers.0.weight"


def test_map_hf_key_passthrough():
    assert map_hf_key_to_dialectic("layers.0.weight") == "layers.0.weight"
