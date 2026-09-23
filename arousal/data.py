"""Official DEAM acquisition, auditable labels and leakage-safe split manifests.

The release checksums are taken from the previously verified project release.
No missing labels, artist names or audio are synthesized. Audio stays local.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
import stat
import unicodedata
import urllib.request
import zipfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from .common import file_hash, read_json, stable_hash, write_json
from .encoders import load_audio

BASE = "https://cvml.unige.ch/databases/DEAM/"
FILES = {
    "DEAM_Annotations.zip": "809b0feb4ba6196c1eda9b2b7b33b892b11fa71457cfceb0d47e832ed1a5f15f",
    "metadata.zip": "3d1d5ab42e852803a770f2cfbd685c6b464e1ebe76f26b0e10954597902b90f8",
    "DEAM_audio.zip": "e2814325f573ade8caebc697f0efcfdf7207142b737d72558a416117ca591353",
}
PROTOCOL = "DEAM_static_short_full_audio_artist_duplicate_groups_v2"


def _download(url: str, path: Path, expected: str) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(path.suffix + ".part")
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        with urllib.request.urlopen(request, timeout=90) as response:
            append = bool(offset and response.status == 206)
            if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise ValueError("Download resume offset was not honored by server")
            with partial.open("ab" if append else "wb") as handle:
                while block := response.read(1024 * 1024):
                    handle.write(block)
        if file_hash(partial) != expected:
            raise ValueError(f"Checksum mismatch: {partial}. Inspect the release; do not bypass verification.")
        partial.replace(path)
    digest = file_hash(path)
    if digest != expected:
        raise ValueError(f"Checksum mismatch: {path}. Inspect the release; do not bypass verification.")
    return {"filename": path.name, "url": url, "sha256": digest, "bytes": path.stat().st_size}


def safe_extract(path: Path, destination: Path) -> None:
    """Extract only normal ZIP paths, checking reused files against ZIP CRC."""
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        names = set()
        for member in archive.infolist():
            normalized = member.filename.replace("\\", "/")
            target = (destination / normalized).resolve()
            if not target.is_relative_to(destination) or ":" in normalized or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Unsafe archive member: {member.filename}")
            if normalized in names:
                raise ValueError(f"Repeated archive path: {normalized}")
            names.add(normalized)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            valid = False
            if target.is_file() and target.stat().st_size == member.file_size:
                crc = 0
                with target.open("rb") as handle:
                    while block := handle.read(1024 * 1024):
                        crc = zlib.crc32(block, crc)
                valid = (crc & 0xFFFFFFFF) == member.CRC
            if not valid:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".extracting")
                with archive.open(member) as source, temporary.open("wb") as handle:
                    while block := source.read(1024 * 1024):
                        handle.write(block)
                temporary.replace(target)


def _csv_rows(archive: zipfile.ZipFile, name: str):
    for row in csv.DictReader(io.StringIO(archive.read(name).decode("utf-8-sig"))):
        yield {key.strip().lower(): value.strip() for key, value in row.items()
               if key is not None and isinstance(value, str)}


def artist_key(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()


def _artist(value: str, aliases: dict) -> str:
    result = artist_key(value)
    seen = set()
    while result in aliases and aliases[result] != result:
        if result in seen:
            raise ValueError(f"Artist alias cycle at {result}")
        seen.add(result)
        result = aliases[result]
    if result in {"", "unknown", "unknown artist", "n/a", "na", "none", "[unknown]"}:
        raise ValueError(f"Missing or ambiguous artist: {value!r}")
    return result


def connected_groups(rows: list[dict]) -> list[dict]:
    """Connect artists sharing identical file bytes or canonical decoded PCM.

    Artists are not relabeled as the same artist: group_id alone stores the
    connected component used for splitting, bootstrap and jackknife.
    """
    parent = {row["artist_group"]: row["artist_group"] for row in rows}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    seen = {}
    for row in rows:
        for field in ("audio_sha256", "pcm_sha256"):
            key = (field, row[field])
            if key in seen:
                a, b = find(row["artist_group"]), find(seen[key])
                parent[max(a, b)] = min(a, b)
            else:
                seen[key] = row["artist_group"]
    components = defaultdict(list)
    for artist in parent:
        components[find(artist)].append(artist)
    ids = {root: names[0] if len(names) == 1 else "duplicate_component_" + stable_hash(sorted(names))[:16]
           for root, names in components.items()}
    for row in rows:
        row["group_id"] = ids[find(row["artist_group"])]
    return [{"group_id": ids[root], "artists": sorted(names)} for root, names in components.items() if len(names) > 1]


def split_groups(rows: list[dict], seed: int) -> dict:
    groups = np.asarray([row["group_id"] for row in rows])
    if len(set(groups)) < 7:
        raise ValueError("At least seven independent artist/duplicate groups are required for a 70/15/15 split")
    indices = np.arange(len(rows))
    train, rest = next(GroupShuffleSplit(n_splits=1, train_size=.7, random_state=seed).split(indices, groups=groups))
    val_i, test_i = next(GroupShuffleSplit(n_splits=1, train_size=.5, random_state=seed + 1).split(rest, groups=groups[rest]))
    positions = {"train": train, "val": rest[val_i], "test": rest[test_i]}
    assignments = {str(rows[int(i)]["song_id"]): split for split, index in positions.items() for i in index}
    for field in ("artist_group", "group_id", "audio_sha256", "pcm_sha256"):
        owners = defaultdict(set)
        for row in rows:
            owners[row[field]].add(assignments[str(row["song_id"])])
        if any(len(value) != 1 for value in owners.values()):
            raise AssertionError(f"Cross-split leakage in {field}")
    return {"seed": seed, "assignments": assignments,
            "split_hash": stable_hash({"seed": seed, "assignments": assignments}),
            "counts": {split: len(index) for split, index in positions.items()},
            "group_counts": {split: len(set(groups[index])) for split, index in positions.items()},
            "algorithm": "GroupShuffleSplit(0.7, seed); remainder GroupShuffleSplit(0.5, seed+1)"}


def prepare_data(data_root: Path, out_dir: Path, cfg: dict) -> dict:
    """Audit official data and write data_manifest.json and split_<seed>.json.

    Re-running revalidates bytes and audio instead of trusting previous status.
    Output rows never contain a split, so all split seeds share raw features.
    """
    data_root, out_dir = Path(data_root).resolve(), Path(out_dir)
    settings = cfg.get("data", {})
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = {"status": "running", "protocol": PROTOCOL, "errors": [], "warnings": [], "archives": [],
             "excluded_full_songs": 58, "duplicate_policy": "retain_and_group; count duplicate IDs separately"}
    write_json(out_dir / "data_audit.json", audit)
    try:
        for name, digest in FILES.items():
            path = data_root / "raw" / name
            if not path.exists() and not settings.get("download", True):
                raise FileNotFoundError(f"Place official {name} in {data_root / 'raw'} or set data.download=true")
            audit["archives"].append(_download(BASE + name, path, digest))
        safe_extract(data_root / "raw/DEAM_audio.zip", data_root / "audio")
        aliases = {artist_key(key): artist_key(value) for key, value in settings.get("artist_aliases", {}).items()}
        artists, raw_artists = {}, {}
        with zipfile.ZipFile(data_root / "raw/metadata.zip") as archive:
            for name in sorted(archive.namelist()):
                if name.lower().endswith(".csv"):
                    for row in _csv_rows(archive, name):
                        sid = int(row.get("song_id", row.get("id", "")))
                        # The 2014 short-excerpt release includes song 2000.
                        # Full songs begin at 2001, not at 2000.
                        if sid > 2000:
                            continue
                        if sid < 1:
                            raise ValueError(f"Invalid metadata song ID: {sid}")
                        normalized = _artist(row.get("artist", ""), aliases)
                        if sid in artists and artists[sid] != normalized:
                            raise ValueError(f"Conflicting artist metadata for song {sid}")
                        artists[sid], raw_artists[sid] = normalized, row.get("artist", "")
        labels = {}
        with zipfile.ZipFile(data_root / "raw/DEAM_Annotations.zip") as archive:
            candidates = [name for name in archive.namelist() if name.endswith("static_annotations_averaged_songs_1_2000.csv")]
            if len(candidates) != 1:
                raise ValueError("Expected exactly one official static short-song annotation file")
            for row in _csv_rows(archive, candidates[0]):
                sid = int(row["song_id"])
                if sid in labels or not 1 <= sid <= 2000:
                    raise ValueError(f"Invalid or duplicate short-song ID: {sid}")
                values = {key: float(row[key]) for key in ("arousal_mean", "arousal_std", "valence_mean", "valence_std")}
                if not all(np.isfinite(list(values.values()))) or not all(1 <= values[key] <= 9 for key in ("arousal_mean", "valence_mean")):
                    raise ValueError(f"Invalid label scale for song {sid}")
                if not all(0 <= values[key] <= 8 for key in ("arousal_std", "valence_std")):
                    raise ValueError(f"Invalid annotation standard deviation for song {sid}")
                labels[sid] = values
        expected = int(settings.get("expected_clips", 1744))
        if len(labels) != expected:
            raise ValueError(f"Expected {expected} short-song annotations, found {len(labels)}")
        audio = {}
        for path in sorted((data_root / "audio").rglob("*")):
            if path.is_file() and path.suffix.lower() in {".mp3", ".wav", ".flac"} and path.stem.isdigit():
                sid = int(path.stem)
                if sid in labels:
                    if sid in audio:
                        raise ValueError(f"Multiple audio files have song ID {sid}")
                    audio[sid] = path
        rows = []
        duration_min, duration_max = float(settings.get("duration_min", 40)), float(settings.get("duration_max", 50))
        for number, sid in enumerate(sorted(labels)):
            if sid not in audio:
                raise FileNotFoundError(f"Missing audio for song {sid}")
            if sid not in artists:
                raise ValueError(f"Missing artist metadata for song {sid}")
            wave = load_audio(audio[sid], sample_rate=24000)
            duration = len(wave) / 24000
            if not duration_min <= duration <= duration_max:
                raise ValueError(f"Song {sid} duration {duration:.3f}s outside audited [{duration_min},{duration_max}]s; no cropping is allowed")
            pcm = hashlib.sha256(b"mono_24000_float32_le\0" + np.asarray(wave, dtype="<f4").tobytes()).hexdigest()
            label = labels[sid]
            rows.append({"song_id": sid, "artist_group": artists[sid], "artist_raw": raw_artists[sid],
                         "arousal": (label["arousal_mean"] - 1) / 8, "arousal_std": label["arousal_std"] / 8,
                         "audio": audio[sid].relative_to(data_root).as_posix(), "audio_sha256": file_hash(audio[sid]),
                         "pcm_sha256": pcm, "duration_seconds": duration, "label_source": "DEAM_human_static_mean"})
            if number == 0 or (number + 1) % 100 == 0 or number + 1 == len(labels):
                print(f"Audited audio {number + 1}/{len(labels)}", flush=True)
        audit["cross_artist_components"] = connected_groups(rows)
        for field in ("audio_sha256", "pcm_sha256"):
            by_digest = defaultdict(list)
            for row in rows:
                by_digest[row[field]].append(row["song_id"])
            audit["duplicate_" + field] = [ids for ids in by_digest.values() if len(ids) > 1]
        audit.update(status="complete", n_clips=len(rows), n_artists=len({row["artist_group"] for row in rows}),
                     n_independent_groups=len({row["group_id"] for row in rows}),
                     duration_range=[min(row["duration_seconds"] for row in rows), max(row["duration_seconds"] for row in rows)],
                     artist_aliases=aliases)
        identity = {"protocol": PROTOCOL, "normalization": "(rating-1)/8", "rows": rows,
                    "source_checksums": {item["filename"]: item["sha256"] for item in audit["archives"]},
                    "artist_aliases": aliases}
        manifest = dict(identity, data_content_hash=stable_hash(identity), status="complete", dataset="DEAM",
                        source_url=BASE, smoke_only=False, n_clips=len(rows), n_artists=audit["n_artists"],
                        n_groups=audit["n_independent_groups"])
        splits = {}
        for seed in cfg.get("split_seeds", [42, 43, 44]):
            split = split_groups(rows, int(seed))
            split["data_content_hash"] = manifest["data_content_hash"]
            write_json(out_dir / "splits" / f"split_{seed}.json", split)
            splits[str(seed)] = split["counts"]
        audit["split_counts"] = splits
        audit["data_content_hash"] = manifest["data_content_hash"]
        write_json(out_dir / "data_manifest.json", manifest)
        write_json(out_dir / "data_audit.json", audit)
        return manifest
    except Exception as exc:
        audit["status"] = "failed"
        audit["errors"].append(f"{type(exc).__name__}: {exc}")
        write_json(out_dir / "data_audit.json", audit)
        raise
