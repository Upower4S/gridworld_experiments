"""Strict read-only identity of actual local model shards and runtime metadata.

No model, tokenizer or tensors are loaded. Every invocation hashes the actual
files; a path, index file, cache entry or unchanged mtime never substitutes for
weight payload hashing. Returned identities contain no observation timestamps.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any


FORMAT = "local_model_snapshot_identity_v1"
INDEX_FILE = "model.safetensors.index.json"
REQUIRED_FILES = ("config.json", "tokenizer.json", INDEX_FILE)
READ_BUFFER_BYTES = 1024 * 1024
SHARD_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.safetensors")
TOKENIZER_FILES = frozenset({"tokenizer.model", "spiece.model", "sentencepiece.bpe.model",
                             "vocab.txt", "merges.txt", "vocab.bpe", "tokenizer.tiktoken"})


def _fingerprint(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _stat_signature(value: os.stat_result) -> tuple[int, ...]:
    # Access time may change because this function reads the file; it is not a
    # content-stability signal. ctime also detects replacements/restored mtimes.
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _file_state(path: Path) -> dict[str, Any]:
    try:
        link = path.lstat()
        resolved = path.resolve(strict=True)
        target = path.stat()
    except OSError as exc:
        raise ValueError(f"Snapshot file is missing or unreadable: {path}") from exc
    if not stat.S_ISREG(target.st_mode) or not (stat.S_ISREG(link.st_mode) or stat.S_ISLNK(link.st_mode)):
        raise ValueError(f"Snapshot entries must resolve to regular local files: {path}")
    return {"resolved_path": str(resolved), "target_signature": _stat_signature(target),
            "entry_signature": _stat_signature(link), "is_symlink": stat.S_ISLNK(link.st_mode),
            "size_bytes": target.st_size}


def _hash_file(path: Path, *, collect: bool = False) -> tuple[dict[str, Any], dict[str, Any], bytes | None]:
    before = _file_state(path)
    digest, count, pieces = hashlib.sha256(), 0, []
    try:
        with path.open("rb") as handle:
            if _stat_signature(os.fstat(handle.fileno())) != before["target_signature"]:
                raise ValueError(f"Snapshot file changed while opening: {path}")
            while chunk := handle.read(READ_BUFFER_BYTES):
                digest.update(chunk)
                count += len(chunk)
                if collect:
                    pieces.append(chunk)
            if _stat_signature(os.fstat(handle.fileno())) != before["target_signature"]:
                raise ValueError(f"Snapshot file changed during hashing: {path}")
    except OSError as exc:
        raise ValueError(f"Unable to hash local snapshot file: {path}") from exc
    if _file_state(path) != before or count != before["size_bytes"]:
        raise ValueError(f"Snapshot file changed during hashing: {path}")
    record = {"sha256": digest.hexdigest(), "resolved_path": before["resolved_path"],
              "size_bytes": count, "is_symlink": before["is_symlink"]}
    return record, before, b"".join(pieces) if collect else None


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key in weight index: {key}")
        result[key] = value
    return result


def _inventory(root: Path) -> tuple[list[str], list[str]]:
    # Index shard values are basenames. Nested safetensors files are therefore
    # unexpected inventory, rather than unrecorded optional model weights.
    weights = sorted(path.relative_to(root).as_posix() for path in root.rglob("*.safetensors"))
    metadata = set()
    for path in root.iterdir():
        if (path.suffix in {".json", ".jinja", ".jinja2", ".model", ".tiktoken"}
                or path.name in TOKENIZER_FILES):
            metadata.add(path.name)
    templates = root / "chat_templates"
    if templates.exists():
        if not templates.is_dir():
            raise ValueError("chat_templates must be a local directory when present")
        for path in templates.rglob("*"):
            if path.is_file() or path.is_symlink():
                metadata.add(path.relative_to(root).as_posix())
    return weights, sorted(metadata)


def snapshot_identity(model_path: Path) -> dict[str, Any]:
    """Hash an indexed local snapshot, checking inventory and read stability.

    Requires config.json, tokenizer.json and model.safetensors.index.json. The
    nonempty index weight_map must reference safe basename .safetensors files,
    and its unique shard set must equal the actual recursive shard inventory.
    All root JSON, template, model and tiktoken files, known vocabulary/merge
    files, and files under chat_templates are included in the metadata identity.

    Hugging Face cache symlinks are allowed: their resolved local paths and
    file sizes are bound in the returned manifest. Target and directory-entry
    inode/size/mtime/ctime are checked before/open/after reads and once more
    across the complete audit. No timestamps or elapsed times enter the result.
    Safetensors header/tensor coverage is deliberately not asserted here.
    """
    supplied_path = Path(model_path).expanduser()
    try:
        root = supplied_path.resolve(strict=True)
    except OSError as exc:
        raise ValueError("Local model snapshot directory does not exist") from exc
    if not root.is_dir():
        raise ValueError("Local model snapshot path must be a directory")
    weights, metadata = _inventory(root)
    missing_metadata = sorted(set(REQUIRED_FILES) - set(metadata))
    if missing_metadata:
        raise ValueError(f"Missing required local model metadata: {missing_metadata}")
    index_record, index_state, raw_index = _hash_file(root / INDEX_FILE, collect=True)
    try:
        index = json.loads(raw_index, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid JSON model weight index") from exc
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if (not isinstance(weight_map, dict) or not weight_map
            or any(not isinstance(tensor, str) or not tensor.strip() for tensor in weight_map)):
        raise ValueError("Weight index requires a nonempty named tensor weight_map")
    if any(not isinstance(name, str) or not SHARD_NAME.fullmatch(name) or ".." in name
           for name in weight_map.values()):
        raise ValueError("Weight index shard references must be safe basename .safetensors filenames")
    indexed = sorted(set(weight_map.values()))
    if indexed != weights:
        raise ValueError(f"Weight shard inventory mismatch: missing={sorted(set(indexed) - set(weights))}, "
                         f"unexpected={sorted(set(weights) - set(indexed))}")
    records, states = {INDEX_FILE: index_record}, {INDEX_FILE: index_state}
    for name in sorted(set(weights) | set(metadata)):
        if name == INDEX_FILE:
            continue
        records[name], states[name], _ = _hash_file(root / name)
    if supplied_path.resolve(strict=True) != root or _inventory(root) != (weights, metadata):
        raise ValueError("Snapshot directory or inventory changed during hashing")
    for name, before in states.items():
        if _file_state(root / name) != before:
            raise ValueError(f"Snapshot file changed during whole-snapshot hashing: {name}")
    # A final byte read binds the parsed index to the exact bytes used to choose
    # shards, in addition to the whole-audit inode/mtime/ctime stability checks.
    final_index, final_state, _ = _hash_file(root / INDEX_FILE)
    if final_index != index_record or final_state != index_state:
        raise ValueError("Weight index changed during whole-snapshot hashing")
    file_metadata = {name: {**{key: value for key, value in records[name].items() if key != "sha256"},
                             "role": "weight_shard" if name in indexed else "metadata_or_tokenizer"}
                     for name in sorted(records)}
    identity = {"format": FORMAT, "path": str(root),
                "files": {name: records[name]["sha256"] for name in sorted(records)},
                "file_metadata": file_metadata, "required_files": list(REQUIRED_FILES),
                "weight_index_file": INDEX_FILE, "weight_map_sha256": _fingerprint(weight_map),
                "indexed_tensor_count": len(weight_map), "indexed_shard_count": len(indexed),
                "indexed_shards": indexed, "actual_weight_files": weights,
                "total_weight_bytes": sum(records[name]["size_bytes"] for name in indexed),
                "total_metadata_bytes": sum(records[name]["size_bytes"] for name in metadata),
                "index_inventory_exact": True, "stable_during_read": True,
                "tensor_header_coverage_verified": False}
    return {**identity, "identity_sha256": _fingerprint(identity)}
