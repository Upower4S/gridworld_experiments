"""Tiny local indexed-file fixtures; no model or network access."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.activation_causality import model_identity
from src.activation_causality.model_identity import snapshot_identity


class ModelIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "snapshot"
        self.root.mkdir()
        (self.root / "config.json").write_text('{"model_type":"fixture"}')
        (self.root / "tokenizer.json").write_text('{"tokenizer":"fixture"}')
        (self.root / "tokenizer_config.json").write_text('{"chat_template":"fixture"}')
        (self.root / "chat_template.jinja").write_text("{{ messages }}")
        (self.root / "model-a.safetensors").write_bytes(b"tiny indexed shard A")
        (self.root / "model-b.safetensors").write_bytes(b"tiny indexed shard B")
        self.index = {"metadata": {"total_size": 40}, "weight_map": {
            "layer.a": "model-a.safetensors", "layer.b": "model-a.safetensors", "layer.c": "model-b.safetensors"}}
        self.write_index()

    def tearDown(self):
        self.temp.cleanup()

    def write_index(self):
        (self.root / model_identity.INDEX_FILE).write_text(json.dumps(self.index))

    def test_deterministic_complete_payload_hashes_and_no_observation_timestamps(self):
        first, second = snapshot_identity(self.root), snapshot_identity(self.root)
        self.assertEqual(first, second)
        self.assertEqual(first["format"], "local_model_snapshot_identity_v1")
        self.assertEqual(first["path"], str(self.root.resolve()))
        self.assertEqual(first["indexed_tensor_count"], 3)
        self.assertEqual(first["indexed_shard_count"], 2)
        self.assertEqual(first["indexed_shards"], first["actual_weight_files"])
        for name, digest in first["files"].items():
            self.assertEqual(digest, hashlib.sha256((self.root / name).read_bytes()).hexdigest())
            self.assertEqual(first["file_metadata"][name]["size_bytes"], (self.root / name).stat().st_size)
        self.assertEqual(first["total_weight_bytes"], 2 * len(b"tiny indexed shard A"))
        self.assertTrue(first["stable_during_read"])
        self.assertFalse(first["tensor_header_coverage_verified"])
        for forbidden in ("mtime", "ctime", "created_at", "elapsed", "timestamp"):
            self.assertNotIn(forbidden, json.dumps(first))
        self.assertEqual(first["identity_sha256"], model_identity._fingerprint({key: value for key, value in first.items() if key != "identity_sha256"}))

    def test_shard_bytes_are_rehashed_even_if_size_and_mtime_restored(self):
        before = snapshot_identity(self.root)
        path = self.root / "model-a.safetensors"
        state = path.stat()
        path.write_bytes(b"tiny indexed shard X")
        os.utime(path, ns=(state.st_atime_ns, state.st_mtime_ns))
        after = snapshot_identity(self.root)
        self.assertEqual(after["file_metadata"][path.name]["size_bytes"], before["file_metadata"][path.name]["size_bytes"])
        self.assertNotEqual(after["files"][path.name], before["files"][path.name])
        self.assertNotEqual(after["identity_sha256"], before["identity_sha256"])

    def test_required_config_tokenizer_and_index_are_all_mandatory(self):
        for name in model_identity.REQUIRED_FILES:
            path, original = self.root / name, (self.root / name).read_bytes()
            path.unlink()
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Missing required"):
                snapshot_identity(self.root)
            path.write_bytes(original)

    def test_missing_extra_and_nested_unindexed_shards_are_rejected(self):
        path = self.root / "model-a.safetensors"
        original = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "missing=.*model-a"):
            snapshot_identity(self.root)
        path.write_bytes(original)
        extra = self.root / "unexpected.safetensors"
        extra.write_bytes(b"unindexed")
        with self.assertRaisesRegex(ValueError, "unexpected=.*unexpected"):
            snapshot_identity(self.root)
        extra.unlink()
        nested = self.root / "nested" / "unexpected.safetensors"
        nested.parent.mkdir()
        nested.write_bytes(b"unindexed nested")
        with self.assertRaisesRegex(ValueError, "nested/unexpected"):
            snapshot_identity(self.root)

    def test_invalid_index_paths_empty_map_and_duplicate_tensor_keys_rejected(self):
        for value in ("../model-a.safetensors", "/tmp/model-a.safetensors", "nested/model-a.safetensors",
                      "nested\\model-a.safetensors", "https://host/file.safetensors", "model.bin", "", 123):
            self.index["weight_map"]["layer.a"] = value
            self.write_index()
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "safe basename"):
                snapshot_identity(self.root)
        self.index["weight_map"] = {}
        self.write_index()
        with self.assertRaisesRegex(ValueError, "nonempty"):
            snapshot_identity(self.root)
        (self.root / model_identity.INDEX_FILE).write_text('{"weight_map":{"a":"model-a.safetensors","a":"model-b.safetensors"}}')
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            snapshot_identity(self.root)

    def test_invalid_json_missing_map_and_nonregular_shard_rejected(self):
        path = self.root / model_identity.INDEX_FILE
        for raw in ("{broken", "[]", '{"weight_map":[]}', '{"weight_map":{"":"model-a.safetensors"}}'):
            path.write_text(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                snapshot_identity(self.root)
        self.write_index()
        shard = self.root / "model-a.safetensors"
        shard.unlink()
        shard.mkdir()
        with self.assertRaisesRegex(ValueError, "regular local files"):
            snapshot_identity(self.root)

    def test_metadata_template_and_tokenizer_inventory_changes_change_identity(self):
        before = snapshot_identity(self.root)
        (self.root / "chat_template.jinja").write_text("changed template")
        template_changed = snapshot_identity(self.root)
        self.assertNotEqual(before["identity_sha256"], template_changed["identity_sha256"])
        (self.root / "special_tokens_map.json").write_text('{"bos_token":"changed"}')
        (self.root / "vocab.txt").write_text("token\n")
        nested = self.root / "chat_templates" / "tool.jinja"
        nested.parent.mkdir()
        nested.write_text("tool template")
        added = snapshot_identity(self.root)
        self.assertIn("special_tokens_map.json", added["files"])
        self.assertIn("vocab.txt", added["files"])
        self.assertIn("chat_templates/tool.jinja", added["files"])
        self.assertNotEqual(template_changed["identity_sha256"], added["identity_sha256"])

    def test_hugging_face_style_local_symlinks_bind_resolved_payload(self):
        shard = self.root / "model-a.safetensors"
        target = Path(self.temp.name) / "blob"
        shard.rename(target)
        shard.symlink_to(target)
        identity = snapshot_identity(self.root)
        entry = identity["file_metadata"][shard.name]
        self.assertTrue(entry["is_symlink"])
        self.assertEqual(entry["resolved_path"], str(target.resolve()))
        self.assertEqual(identity["files"][shard.name], hashlib.sha256(target.read_bytes()).hexdigest())
        target.unlink()
        with self.assertRaisesRegex(ValueError, "missing or unreadable"):
            snapshot_identity(self.root)

    def test_stat_change_during_single_file_read_is_rejected(self):
        real_state = model_identity._file_state
        calls = 0
        def changed_state(path):
            nonlocal calls
            result = real_state(path)
            if path.name == "model-a.safetensors":
                calls += 1
                if calls == 2:
                    result = deepcopy(result)
                    signature = list(result["target_signature"])
                    signature[4] += 1
                    result["target_signature"] = tuple(signature)
            return result
        with patch("src.activation_causality.model_identity._file_state", side_effect=changed_state):
            with self.assertRaisesRegex(ValueError, "changed during hashing"):
                snapshot_identity(self.root)

    def test_earlier_file_change_and_inventory_change_during_audit_rejected(self):
        real_hash = model_identity._hash_file
        def change_prior_file(path, **kwargs):
            result = real_hash(path, **kwargs)
            if path.name == "model-b.safetensors":
                (self.root / "config.json").write_text('{"model_type":"changed while hashing later shard"}')
            return result
        with patch("src.activation_causality.model_identity._hash_file", side_effect=change_prior_file):
            with self.assertRaisesRegex(ValueError, "whole-snapshot"):
                snapshot_identity(self.root)
        def add_file(path, **kwargs):
            result = real_hash(path, **kwargs)
            if path.name == "model-b.safetensors":
                (self.root / "extra.safetensors").write_bytes(b"arrived during read")
            return result
        with patch("src.activation_causality.model_identity._hash_file", side_effect=add_file):
            with self.assertRaisesRegex(ValueError, "inventory changed"):
                snapshot_identity(self.root)

    def test_streaming_chunks_and_non_directory_path(self):
        with patch("src.activation_causality.model_identity.READ_BUFFER_BYTES", 3):
            identity = snapshot_identity(self.root)
        self.assertEqual(identity["files"]["model-a.safetensors"], hashlib.sha256(b"tiny indexed shard A").hexdigest())
        with self.assertRaisesRegex(ValueError, "directory"):
            snapshot_identity(self.root / "config.json")


if __name__ == "__main__":
    unittest.main()
