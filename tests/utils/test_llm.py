from __future__ import annotations

import pytest

from atomic_agentic.utils.llm import validate_attachment_path


class TestValidateAttachmentPath:
    def test_nonexistent_file_raises(self, tmp_path: pytest.TempPathFactory) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            validate_attachment_path(
                str(tmp_path / "missing.txt"),
                illegal_exts={".zip"},
                allowed_exts=None,
                illegal_mime_prefixes=(),
            )

    def test_illegal_extension_raises(self, tmp_path: pytest.TempPathFactory) -> None:
        path = tmp_path / "archive.zip"
        path.write_bytes(b"data")
        with pytest.raises(ValueError, match="not permitted"):
            validate_attachment_path(
                str(path),
                illegal_exts={".zip"},
                allowed_exts=None,
                illegal_mime_prefixes=(),
            )

    def test_not_in_allow_list_raises(self, tmp_path: pytest.TempPathFactory) -> None:
        path = tmp_path / "notes.md"
        path.write_text("hello")
        with pytest.raises(ValueError, match="not supported"):
            validate_attachment_path(
                str(path),
                illegal_exts={".zip"},
                allowed_exts={".txt"},
                illegal_mime_prefixes=(),
            )

    def test_allow_list_none_permits_any_ext(self, tmp_path: pytest.TempPathFactory) -> None:
        path = tmp_path / "notes.md"
        path.write_text("hello")
        validate_attachment_path(
            str(path),
            illegal_exts={".zip"},
            allowed_exts=None,
            illegal_mime_prefixes=(),
        )

    def test_illegal_mime_prefix_raises(self, tmp_path: pytest.TempPathFactory) -> None:
        path = tmp_path / "clip.mp4"
        path.write_bytes(b"data")
        with pytest.raises(ValueError, match="not permitted"):
            validate_attachment_path(
                str(path),
                illegal_exts={".zip"},
                allowed_exts=None,
                illegal_mime_prefixes=("audio/", "video/"),
            )

    def test_valid_txt_file_passes(self, tmp_path: pytest.TempPathFactory) -> None:
        path = tmp_path / "document.txt"
        path.write_text("hello")
        validate_attachment_path(
            str(path),
            illegal_exts={".zip"},
            allowed_exts={".txt"},
            illegal_mime_prefixes=("audio/", "video/"),
        )

    def test_empty_illegal_mime_prefixes_skips_mime_check(self, tmp_path: pytest.TempPathFactory) -> None:
        path = tmp_path / "video.mp4"
        path.write_bytes(b"data")
        validate_attachment_path(
            str(path),
            illegal_exts={".zip"},
            allowed_exts=None,
            illegal_mime_prefixes=(),
        )

    def test_no_extension_passes_when_not_in_illegal_set(self, tmp_path: pytest.TempPathFactory) -> None:
        path = tmp_path / "Makefile"
        path.write_text("all:")
        validate_attachment_path(
            str(path),
            illegal_exts={".zip"},
            allowed_exts=None,
            illegal_mime_prefixes=(),
        )

    def test_extension_in_allow_list_passes(self, tmp_path: pytest.TempPathFactory) -> None:
        path = tmp_path / "data.json"
        path.write_text("{}")
        validate_attachment_path(
            str(path),
            illegal_exts={".zip"},
            allowed_exts={".json", ".txt"},
            illegal_mime_prefixes=(),
        )
