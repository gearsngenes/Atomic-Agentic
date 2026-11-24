from __future__ import annotations

import pytest

from atomic_agentic.constants.llm import MISTRAL_IMAGE_EXTS, OPENAI_IMAGE_EXTS
from atomic_agentic.utils.llm import (
    classify_path,
    collect_system_text,
    inlined_file_block,
    read_text_file,
    validate_attachment_path,
)


class TestCollectSystemText:
    def test_joins_system_contents_with_blank_lines(self) -> None:
        messages = [
            {"role": "system", "content": "  One.  "},
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "Two."},
        ]

        assert collect_system_text(messages) == "One.\n\nTwo."

    def test_none_when_no_system_text(self) -> None:
        assert collect_system_text([{"role": "user", "content": "Hi"}]) is None
        assert collect_system_text([]) is None


class TestClassifyPath:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("doc.pdf", "pdf"),
            ("DOC.PDF", "pdf"),
            ("photo.png", "image"),
            ("photo.JPG", "image"),
            ("notes.txt", "text"),
            ("script.py", "text"),
            ("noext", "text"),
        ],
    )
    def test_buckets(self, name: str, expected: str) -> None:
        assert classify_path(name, OPENAI_IMAGE_EXTS) == expected

    def test_image_ext_set_is_a_parameter(self) -> None:
        assert classify_path("scan.zzimg", (".zzimg",)) == "image"
        assert classify_path("scan.zzimg", MISTRAL_IMAGE_EXTS) == "text"

    def test_image_mime_alone_makes_an_image(self) -> None:
        assert classify_path("pic.bmp", MISTRAL_IMAGE_EXTS) == "image"


class TestReadTextFile:
    def test_reads_utf8(self, tmp_path: pytest.TempPathFactory) -> None:
        path = tmp_path / "a.txt"
        path.write_text("h\u00e9llo", encoding="utf-8")

        assert read_text_file(str(path)) == "h\u00e9llo"

    def test_undecodable_bytes_are_replaced(self, tmp_path: pytest.TempPathFactory) -> None:
        path = tmp_path / "b.txt"
        path.write_bytes(b"ok \xff end")

        assert read_text_file(str(path)) == "ok \ufffd end"

    def test_io_errors_propagate(self, tmp_path: pytest.TempPathFactory) -> None:
        with pytest.raises(OSError):
            read_text_file(str(tmp_path / "missing.txt"))
        with pytest.raises(OSError):
            read_text_file(str(tmp_path))


class TestInlinedFileBlock:
    def test_header_uses_the_basename(self) -> None:
        block = inlined_file_block("/some/dir/notes.txt", "body")

        assert block == "\n[Inlined file: notes.txt]\nbody"


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
