"""Asset preparation must fail safely without touching historical results."""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys
import types
import zipfile

import pytest

import prepare_assets as assets
from arousal import data


@pytest.fixture
def archives(tmp_path, monkeypatch):
    source = tmp_path / "old-project" / "data" / "raw"
    source.mkdir(parents=True)
    hashes = {}
    for name in ("DEAM_audio.zip", "DEAM_Annotations.zip", "metadata.zip"):
        with zipfile.ZipFile(source / name, "w") as archive:
            archive.writestr("content.txt", name)
        hashes[name] = assets.file_hash(source / name)
    monkeypatch.setattr(data, "FILES", hashes)
    return source, hashes


def test_missing_archives_report_all_paths(tmp_path, archives):
    root = tmp_path / "new-project" / "data"
    with pytest.raises(FileNotFoundError) as exc:
        assets.check_data(root)
    for name in archives[1]:
        assert name in str(exc.value)
    assert not (root.parent / "runs").exists()


def test_directory_named_zip_is_rejected(tmp_path, archives):
    root = tmp_path / "data"
    (root / "raw" / "DEAM_audio.zip").mkdir(parents=True)
    with pytest.raises(ValueError, match="ZIP FILE"):
        assets.check_data(root)


def test_wrong_hash_is_preserved_and_not_replaced(tmp_path, archives):
    source, _ = archives
    root = tmp_path / "new-data"
    (root / "raw").mkdir(parents=True)
    target = root / "raw" / "DEAM_audio.zip"
    target.write_bytes(b"keep this bad file for inspection")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        assets.check_data(root, reuse_raw=source)
    assert target.read_bytes() == b"keep this bad file for inspection"


def test_reuse_only_missing_originals_and_never_touches_runs(tmp_path, archives):
    source, hashes = archives
    project = tmp_path / "new-project"
    root = project / "data"
    (root / "raw").mkdir(parents=True)
    existing = root / "raw" / "DEAM_audio.zip"
    existing.write_bytes((source / existing.name).read_bytes())
    before = existing.stat().st_mtime_ns
    # An irrelevant bad old copy must not replace an already valid target.
    (source / existing.name).write_bytes(b"old bad copy")
    historical = project / "runs" / "old" / "result.json"
    historical.parent.mkdir(parents=True)
    historical.write_text("historical result", encoding="utf-8")
    result = assets.check_data(root, reuse_raw=source)
    assert result["status"] == "passed"
    assert len(result["archives"]) == 3
    assert existing.stat().st_mtime_ns == before
    assert historical.read_text(encoding="utf-8") == "historical result"
    for name, digest in hashes.items():
        assert assets.file_hash(root / "raw" / name) == digest


def test_wrong_old_source_is_not_copied(tmp_path, archives):
    source, _ = archives
    (source / "DEAM_audio.zip").write_bytes(b"wrong source")
    root = tmp_path / "new-data"
    with pytest.raises(ValueError, match="Old archive checksum mismatch"):
        assets.check_data(root, reuse_raw=source)
    assert not (root / "raw" / "DEAM_audio.zip").exists()


def test_interrupted_reuse_leaves_no_partial_final_zip(tmp_path, archives, monkeypatch):
    source, hashes = archives
    root = tmp_path / "new-data"

    def interrupted(src, dst, length):
        dst.write(src.read(8))
        raise OSError("simulated copy interruption")

    monkeypatch.setattr(assets.shutil, "copyfileobj", interrupted)
    with pytest.raises(OSError, match="simulated copy interruption"):
        assets.check_data(root, reuse_raw=source)
    assert not list((root / "raw").iterdir())
    for name, digest in hashes.items():
        assert assets.file_hash(source / name) == digest


def test_copied_bytes_are_verified_before_commit(tmp_path, archives, monkeypatch):
    source, _ = archives
    root = tmp_path / "new-data"

    def corrupt_copy(src, dst, length):
        dst.write(b"simulated faulty copy")

    monkeypatch.setattr(assets.shutil, "copyfileobj", corrupt_copy)
    with pytest.raises(ValueError, match="Copied archive failed checksum"):
        assets.check_data(root, reuse_raw=source)
    assert not list((root / "raw").iterdir())


def test_concurrent_file_is_not_overwritten_during_reuse(tmp_path, archives, monkeypatch):
    source, _ = archives
    root = tmp_path / "new-data"

    def concurrent_commit(temporary, target):
        Path(target).write_bytes(b"another process created this")
        raise FileExistsError(str(target))

    monkeypatch.setattr(assets.os, "link", concurrent_commit)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        assets.check_data(root, reuse_raw=source)
    target = root / "raw" / "DEAM_audio.zip"
    assert target.read_bytes() == b"another process created this"
    assert list((root / "raw").iterdir()) == [target]


def test_crc_failure_detected_after_matching_archive_hash(tmp_path, archives, monkeypatch):
    source, hashes = archives
    path = source / "DEAM_audio.zip"
    content = path.read_bytes().replace(b"DEAM_audio.zip", b"XEAM_audio.zip", 1)
    path.write_bytes(content)
    # Exercise CRC independently of the preceding outer SHA check.
    monkeypatch.setitem(hashes, path.name, hashlib.sha256(content).hexdigest())
    with pytest.raises(ValueError, match="Corrupt ZIP member"):
        assets.check_data(source.parent)


@pytest.fixture
def model_cache(tmp_path, monkeypatch):
    expected = {}
    paths = {}
    for name in assets.MERT_FILES:
        path = tmp_path / "hub" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((name + " fixed checkpoint").encode())
        paths[name] = path
        expected[name] = assets.file_hash(path)
    monkeypatch.setattr(assets, "MERT_FILES", expected)
    hub = types.ModuleType("huggingface_hub")
    errors = types.ModuleType("huggingface_hub.errors")

    class LocalEntryNotFoundError(Exception):
        pass

    errors.LocalEntryNotFoundError = LocalEntryNotFoundError
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    return paths, hub, LocalEntryNotFoundError


def test_cached_model_never_downloads_and_checks_offline_child(tmp_path, model_cache, monkeypatch):
    paths, hub, _ = model_cache
    calls = []
    children = []

    def download(repo, name, **kwargs):
        calls.append((repo, name, kwargs))
        assert kwargs["local_files_only"] is True
        return str(paths[name])

    hub.hf_hub_download = download
    monkeypatch.setattr(assets.subprocess, "run", lambda *a, **kw: children.append((a, kw)))
    monkeypatch.setenv("HF_HOME", "persistent-cache")
    monkeypatch.setenv("HF_HUB_CACHE", "persistent-cache/hub")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    result = assets.cache_model("cuda", tmp_path / "checks")
    assert result["offline_inference_passed"] is True
    assert len(calls) == 5
    assert {call[2]["revision"] for call in calls} == {result["revision"]}
    assert len(children) == 1
    args, kwargs = children[0]
    assert args[0][0] == sys.executable
    assert "--offline-check" in args[0]
    assert "cuda" in args[0]
    assert kwargs["check"] is True
    assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    assert kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert kwargs["env"]["HF_HOME"] == "persistent-cache"
    assert kwargs["env"]["HF_HUB_CACHE"] == "persistent-cache/hub"


def test_downloads_only_missing_fixed_file(tmp_path, model_cache, monkeypatch):
    paths, hub, missing_error = model_cache
    calls = []

    def download(repo, name, **kwargs):
        calls.append((name, kwargs))
        if name == "pytorch_model.bin" and kwargs.get("local_files_only"):
            raise missing_error("not cached")
        return str(paths[name])

    hub.hf_hub_download = download
    monkeypatch.setattr(assets.subprocess, "run", lambda *a, **kw: None)
    assets.cache_model("cpu", tmp_path / "checks")
    online = [(name, kw) for name, kw in calls if not kw.get("local_files_only")]
    assert [name for name, _ in online] == ["pytorch_model.bin"]
    assert len(calls) == 6
    assert len({kw["revision"] for _, kw in calls}) == 1


def test_wrong_model_hash_blocks_inference_without_replacement(tmp_path, model_cache, monkeypatch):
    paths, hub, _ = model_cache
    paths["config.json"].write_bytes(b"changed config")
    calls = []

    def download(repo, name, **kwargs):
        calls.append(kwargs)
        return str(paths[name])

    hub.hf_hub_download = download
    monkeypatch.setattr(assets.subprocess, "run", lambda *a, **kw: pytest.fail("must not infer with bad checkpoint"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        assets.cache_model("cuda", tmp_path / "checks")
    assert all(call["local_files_only"] for call in calls)
    assert paths["config.json"].read_bytes() == b"changed config"


def test_child_inference_failure_is_not_marked_passed(tmp_path, model_cache, monkeypatch):
    paths, hub, _ = model_cache
    hub.hf_hub_download = lambda repo, name, **kw: str(paths[name])

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(assets.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        assets.cache_model("cuda", tmp_path / "checks")


def test_offline_check_refuses_unset_flags_before_loading_model(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    with pytest.raises(RuntimeError, match="both offline flags"):
        assets.main(["--offline-check", "--checks", str(tmp_path / "checks")])
