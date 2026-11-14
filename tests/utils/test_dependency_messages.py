from __future__ import annotations

from importlib import metadata

import pytest

from atomic_agentic.utils.core import format_missing_dependency


def _patch_installed(monkeypatch: pytest.MonkeyPatch, installed: bool) -> None:
    def fake_version(package: str) -> str:
        if installed:
            return "1.0.0"
        raise metadata.PackageNotFoundError(package)

    monkeypatch.setattr(metadata, "version", fake_version)


class TestFormatMissingDependency:
    def test_absent_package_names_package_and_install_command(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_installed(monkeypatch, installed=False)

        message = format_missing_dependency("google-genai", "gemini")

        assert "`google-genai`" in message
        assert "pip install atomic-agentic[gemini]" in message
        assert "could not be imported" not in message

    def test_installed_without_cause_uses_install_form(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_installed(monkeypatch, installed=True)

        message = format_missing_dependency("openai", "openai")

        assert message.startswith("requires the `openai` package")
        assert "pip install atomic-agentic[openai]" in message

    def test_installed_with_cause_reports_import_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_installed(monkeypatch, installed=True)

        message = format_missing_dependency(
            "mistralai", "mistral", ImportError("cannot import name 'Mistral'")
        )

        assert "installed but could not be imported" in message
        assert "ImportError" in message
        assert "cannot import name 'Mistral'" in message
        assert "pip install atomic-agentic[mistral]" in message

    def test_absent_package_with_cause_uses_install_form(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_installed(monkeypatch, installed=False)

        message = format_missing_dependency(
            "litellm", "litellm", ModuleNotFoundError("No module named 'litellm'")
        )

        assert message.startswith("requires the `litellm` package")
        assert "could not be imported" not in message
