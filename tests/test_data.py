"""Synthetic fixtures test invariants only; they are never scientific results."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from arousal import data, features
from arousal.common import file_hash, read_json, write_json
from arousal.encoders import chunk_ranges, handcrafted, load_audio


@pytest.fixture
def official_fixture(tmp_path, monkeypatch):
    """Small ZIPs exercise the official parser with explicitly patched checksums."""
    root, out = tmp_path / "data", tmp_path / "out"
    raw = root / "raw"
    raw.mkdir(parents=True)
    labels = ["song_id,arousal_mean,arousal_std,valence_mean,valence_std"]
    metadata = ["song_id,artist"]
    with zipfile.ZipFile(raw / "DEAM_audio.zip", "w") as archive:
        for sid in range(1, 21):
            # IDs 1 and 2 intentionally have the same PCM, different artists.
            source_id = 1 if sid == 2 else sid
            wave = np.random.default_rng(source_id).normal(0, .1, 24000).astype(np.float32)
            stream = io.BytesIO()
            sf.write(stream, wave, 24000, format="WAV", subtype="FLOAT")
            archive.writestr(f"MEMD_audio/{sid}.wav", stream.getvalue())
            labels.append(f"{sid},{1 + (sid % 9)},1,5,1")
            metadata.append(f"{sid},Artist {sid}")
    with zipfile.ZipFile(raw / "DEAM_Annotations.zip", "w") as archive:
        archive.writestr("annotations/static_annotations_averaged_songs_1_2000.csv", "\n".join(labels))
    with zipfile.ZipFile(raw / "metadata.zip", "w") as archive:
        archive.writestr("metadata.csv", "\n".join(metadata))
    monkeypatch.setattr(data, "FILES", {name: file_hash(raw / name) for name in data.FILES})
    cfg = {"split_seeds": [42, 43, 44], "representations": ["mir"], "device": "cpu", "_resume": True,
           "data": {"download": False, "expected_clips": 20, "duration_min": .9, "duration_max": 1.1,
                    "artist_aliases": {" ARTIST   20 ": "Artist 19"}}}
    return root, out, cfg


def test_audited_data_grouped_splits_and_split_independent_identity(official_fixture):
    root, out, cfg = official_fixture
    manifest = data.prepare_data(root, out, cfg)
    assert manifest["n_clips"] == 20
    assert all("split" not in row for row in manifest["rows"])
    rows = {row["song_id"]: row for row in manifest["rows"]}
    assert rows[1]["group_id"] == rows[2]["group_id"]
    assert rows[1]["artist_group"] != rows[2]["artist_group"]
    assert rows[20]["artist_group"] == rows[19]["artist_group"]
    for seed in cfg["split_seeds"]:
        split = read_json(out / "splits" / f"split_{seed}.json")
        assert split["assignments"]["1"] == split["assignments"]["2"]
        for field in ("artist_group", "group_id", "pcm_sha256", "audio_sha256"):
            owner = {}
            for row in rows.values():
                subset = split["assignments"][str(row["song_id"])]
                assert owner.setdefault(row[field], subset) == subset
    changed = data.prepare_data(root, out, dict(cfg, split_seeds=[123]))
    assert changed["data_content_hash"] == manifest["data_content_hash"]
    assert read_json(out / "data_audit.json")["status"] == "complete"


def test_corrupt_archive_is_rejected_and_failure_is_recorded(official_fixture):
    root, out, cfg = official_fixture
    path = root / "raw/metadata.zip"
    path.write_bytes(path.read_bytes() + b"altered")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        data.prepare_data(root, out, cfg)
    assert read_json(out / "data_audit.json")["status"] == "failed"


def test_duplicate_annotation_id_fails(official_fixture, monkeypatch):
    root, out, cfg = official_fixture
    path = root / "raw/DEAM_Annotations.zip"
    with zipfile.ZipFile(path) as source:
        name = source.namelist()[0]
        content = source.read(name).decode()
    with zipfile.ZipFile(path, "w") as target:
        target.writestr(name, content + "\n1,5,1,5,1")
    monkeypatch.setitem(data.FILES, path.name, file_hash(path))
    with pytest.raises(ValueError, match="duplicate short-song ID"):
        data.prepare_data(root, out, cfg)


def test_song_2000_is_a_short_excerpt_and_2001_is_excluded(official_fixture, monkeypatch):
    root, out, cfg = official_fixture
    raw = root / "raw"
    # Exercise the real release boundary, absent from the small 1..20 fixture.
    for filename in ("DEAM_Annotations.zip", "metadata.zip"):
        path = raw / filename
        with zipfile.ZipFile(path) as source:
            name = source.namelist()[0]
            text = source.read(name).decode().replace("\n20,", "\n2000,")
        if filename == "metadata.zip":
            text += "\n2001,unknown"
        with zipfile.ZipFile(path, "w") as target:
            target.writestr(name, text)
        monkeypatch.setitem(data.FILES, filename, file_hash(path))
    path = raw / "DEAM_audio.zip"
    with zipfile.ZipFile(path) as source:
        entries = {name: source.read(name) for name in source.namelist()}
    entries["MEMD_audio/2000.wav"] = entries.pop("MEMD_audio/20.wav")
    entries["MEMD_audio/2001.wav"] = entries["MEMD_audio/2000.wav"]
    with zipfile.ZipFile(path, "w") as target:
        for name, content in entries.items():
            target.writestr(name, content)
    monkeypatch.setitem(data.FILES, path.name, file_hash(path))
    manifest = data.prepare_data(root, out, cfg)
    rows = {row["song_id"]: row for row in manifest["rows"]}
    assert len(rows) == 20 and 2000 in rows and 2001 not in rows
    assert rows[2000]["artist_raw"] == "Artist 20"
    for seed in cfg["split_seeds"]:
        assignments = read_json(out / "splits" / f"split_{seed}.json")["assignments"]
        assert "2000" in assignments and "2001" not in assignments


@pytest.mark.parametrize("sid", [0, -1, 2001, 2058])
def test_non_short_ids_cannot_enter_static_short_labels(official_fixture, monkeypatch, sid):
    root, out, cfg = official_fixture
    path = root / "raw/DEAM_Annotations.zip"
    with zipfile.ZipFile(path) as source:
        name = source.namelist()[0]
        text = source.read(name).decode().replace("\n20,", f"\n{sid},")
    with zipfile.ZipFile(path, "w") as target:
        target.writestr(name, text)
    monkeypatch.setitem(data.FILES, path.name, file_hash(path))
    with pytest.raises(ValueError, match="Invalid or duplicate short-song ID"):
        data.prepare_data(root, out, cfg)


def test_alias_cycle_and_unknown_artist_rejected():
    with pytest.raises(ValueError, match="cycle"):
        data._artist("a", {"a": "b", "b": "a"})
    with pytest.raises(ValueError, match="Missing"):
        data._artist("unknown", {})


def test_transitive_duplicate_components():
    rows = [dict(artist_group="a", audio_sha256="file1", pcm_sha256="pcm1"),
            dict(artist_group="b", audio_sha256="file2", pcm_sha256="pcm1"),
            dict(artist_group="b", audio_sha256="file3", pcm_sha256="pcm3"),
            dict(artist_group="c", audio_sha256="file3", pcm_sha256="pcm4")]
    components = data.connected_groups(rows)
    assert len({row["group_id"] for row in rows}) == 1
    assert components[0]["artists"] == ["a", "b", "c"]


@pytest.mark.parametrize("name", ["../escape.txt", "..\\escape.txt", "C:/escape.txt"])
def test_zip_escape_rejected(tmp_path, name):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(name, "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        data.safe_extract(path, tmp_path / "output")


def test_extraction_repairs_same_size_corruption(tmp_path):
    path, target = tmp_path / "audio.zip", tmp_path / "extract"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("x.txt", "good")
    data.safe_extract(path, target)
    (target / "x.txt").write_text("evil")
    data.safe_extract(path, target)
    assert (target / "x.txt").read_text() == "good"


def test_audio_retains_gain_and_mir_is_finite(tmp_path):
    path = tmp_path / "wave.wav"
    wave = (.13 * np.sin(np.arange(24000) * 2 * np.pi * 440 / 24000)).astype(np.float32)
    sf.write(path, np.stack([wave, wave], axis=1), 24000, subtype="FLOAT")
    loaded = load_audio(path)
    np.testing.assert_array_equal(loaded, wave)
    vector = handcrafted(loaded)
    assert vector.shape == (32,)
    assert np.isfinite(vector).all()
    assert vector[0] == pytest.approx(np.sqrt(np.mean(wave ** 2)), rel=.002)


def test_chunking_preserves_every_sample_and_short_tail():
    ranges = chunk_ranges(240001, 120000, 400)
    assert ranges == [(0, 120000), (120000, 240001)]
    for length in (400, 120000, 120399, 120400, 240777):
        chunks = chunk_ranges(length, 120000, 400)
        assert sum(stop - start for start, stop in chunks) == length
        assert chunks[0][0] == 0 and chunks[-1][1] == length
        assert all(stop - start >= 400 for start, stop in chunks)
        assert all(left[1] == right[0] for left, right in zip(chunks, chunks[1:]))


def test_feature_resume_and_invalid_cache_recompute(official_fixture, monkeypatch):
    root, out, cfg = official_fixture
    data.prepare_data(root, out, cfg)
    metadata = features.extract_features(root, out, cfg)
    target = out / "features/raw_features.npz"
    original_hash = file_hash(target)
    again = features.extract_features(root, out, cfg)
    assert again["feature_hash"] == metadata["feature_hash"]
    assert file_hash(target) == original_hash
    # Damaging the assembled file permits safe rebuild from per-song caches.
    target.write_bytes(b"broken")
    rebuilt = features.extract_features(root, out, cfg)
    assert rebuilt["newly_computed"] == 0 and rebuilt["reused"] == 20
    with np.load(target, allow_pickle=False) as cache:
        assert cache["mir"].shape == (20, 32)
    # A different extractor provenance must not trust earlier per-song caches.
    monkeypatch.setattr(features, "_versions", lambda: {"numpy": "intentionally_changed_test_version"})
    refreshed = features.extract_features(root, out, cfg)
    assert refreshed["newly_computed"] == 20 and refreshed["reused"] == 0
    assert refreshed["feature_hash"] != metadata["feature_hash"]


def test_changed_audio_after_audit_is_rejected(official_fixture):
    root, out, cfg = official_fixture
    manifest = data.prepare_data(root, out, cfg)
    path = root / manifest["rows"][0]["audio"]
    path.write_bytes(path.read_bytes() + b"change")
    with pytest.raises(ValueError, match="changed since audit"):
        features.extract_features(root, out, cfg)
