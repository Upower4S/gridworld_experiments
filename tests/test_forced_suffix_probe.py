from __future__ import annotations

import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.conceptual_belief_space.forced_suffix_probe import (  # noqa: E402
    ACTIONS,
    FORCED_SUFFIX,
    GPT_OSS_SUFFIX_TOKEN_IDS,
    ProbeConfig,
    ProjectedShardStore,
    ShardSpec,
    TrajectoryShardWriter,
    adjacent_layer_delta,
    aligned_cot_window,
    array_sha256,
    bf16_uint16_to_float32,
    build_probe,
    clone_cache,
    float32_to_bf16_uint16,
    forced_suffix_token_ids,
    index_rows_from_shards,
    load_shard_metadata,
    longest_common_prefix_len,
    make_orthogonal_projection,
    trim_cache,
    validate_context_suffix,
)


class FakeTokenizer:
    def __init__(self, ids):
        self.ids = list(ids)

    def encode(self, text, add_special_tokens=False):
        assert text == FORCED_SUFFIX
        assert add_special_tokens is False
        return self.ids


def test_canonical_suffix_tokenization_and_candidate_exclusion():
    ids = forced_suffix_token_ids(FakeTokenizer(GPT_OSS_SUFFIX_TOKEN_IDS))
    assert ids == list(GPT_OSS_SUFFIX_TOKEN_IDS)
    assert len(ids) == 12
    assert all(not FORCED_SUFFIX.endswith(action) for action in ACTIONS)
    with pytest.raises(ValueError, match="canonical GPT-OSS"):
        forced_suffix_token_ids(FakeTokenizer([*GPT_OSS_SUFFIX_TOKEN_IDS[:-1], 1]))


def test_context_suffix_and_common_prefix_alignment():
    suffix = list(GPT_OSS_SUFFIX_TOKEN_IDS)
    prefix = [4, 5, 6]
    assert validate_context_suffix(prefix + suffix, suffix) == prefix
    with pytest.raises(ValueError, match="does not end"):
        validate_context_suffix(prefix + suffix[:-1] + [0], suffix)
    assert longest_common_prefix_len([1, 2, 3], [1, 2, 4, 5]) == 2


def test_cot_window_uses_only_aligned_analysis_tokens_and_left_pads():
    selected, padded, mask = aligned_cot_window(
        [90, 91, 10, 11, 12], [10, 11, 12], window_size=5
    )
    assert selected == [10, 11, 12]
    assert padded == [-1, -1, 10, 11, 12]
    assert mask == [0, 0, 1, 1, 1]
    selected, padded, mask = aligned_cot_window(
        [90, 10, 11, 12, 13], [10, 11, 12, 13], window_size=2
    )
    assert selected == padded == [12, 13]
    assert mask == [1, 1]
    with pytest.raises(ValueError, match="do not exactly align"):
        aligned_cot_window([90, 10, 99], [10, 11], window_size=2)


@dataclass
class FakeCache:
    state: list[int]
    meta_state: dict[str, int]

    @classmethod
    def from_state(cls, state, meta_state):
        return cls(list(state), dict(meta_state))

    def trim(self, n):
        del self.state[-int(n) :]


def test_cache_clone_and_trim_are_independent():
    original = [FakeCache([1, 2, 3], {"offset": 3}), FakeCache([4, 5, 6], {"offset": 3})]
    copied = clone_cache(original)
    trim_cache(copied, 1)
    assert copied[0].state == [1, 2]
    assert original[0].state == [1, 2, 3]


def test_projection_is_deterministic_scaled_and_orthogonal():
    first = make_orthogonal_projection(32, 8, seed=42)
    second = make_orthogonal_projection(32, 8, seed=42)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.T @ first, np.eye(8) * 4.0, atol=1e-5)
    assert array_sha256(first) == array_sha256(second)


def test_bf16_round_trip_and_adjacent_delta():
    values = np.asarray([-3.25, -0.125, 0.0, 0.1, 2.75, 1000.0], dtype=np.float32)
    restored = bf16_uint16_to_float32(float32_to_bf16_uint16(values))
    np.testing.assert_allclose(restored, values, rtol=8e-3, atol=1e-3)
    layers = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)
    delta = adjacent_layer_delta(layers)
    np.testing.assert_array_equal(delta[0], np.zeros_like(delta[0]))
    np.testing.assert_array_equal(delta[1:], layers[1:] - layers[:-1])


def test_atomic_shard_round_trip_and_index(tmp_path):
    projection = make_orthogonal_projection(8, 4, seed=42)
    spec = ShardSpec(
        n_layers=3,
        n_input_tokens=2,
        projected_dim=4,
        hidden_dim=8,
        projection_seed=42,
        projection_sha256=array_sha256(projection),
        input_kind="cot_last_tokens",
        label_suffix_token_ids=GPT_OSS_SUFFIX_TOKEN_IDS,
        model_name_or_path="fake/model",
    )
    writer = TrajectoryShardWriter(tmp_path, "trajectory/one", spec)
    expected_states = []
    expected_rms = []
    for row in range(2):
        state = np.arange(24, dtype=np.float32).reshape(3, 2, 4) + row
        rms = np.full((3, 2), row + 0.5, dtype=np.float32)
        writer.append(state, rms, {"prefix_id": row, "trajectory_id": "trajectory/one"})
        expected_states.append(state)
        expected_rms.append(rms)
    meta_path = writer.finalize()
    metadata = load_shard_metadata(meta_path)
    assert metadata["state_shape"] == [2, 3, 2, 4]
    rows = index_rows_from_shards(tmp_path)
    assert [row["prefix_id"] for row in rows] == [0, 1]
    store = ProjectedShardStore()
    for row, expected_state, expected_scale in zip(rows, expected_states, expected_rms, strict=True):
        state, scale = store.load_row(row)
        np.testing.assert_allclose(state, expected_state, rtol=8e-3, atol=1e-3)
        np.testing.assert_allclose(scale, expected_scale, rtol=1e-3, atol=1e-3)


def test_bidirectional_probe_shapes_and_late_layer_influence():
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    config = ProbeConfig(
        projected_dim=8,
        n_layers=4,
        n_input_tokens=3,
        token_dim=4,
        d_model=12,
        encoder_layers=2,
        n_heads=3,
        dim_feedforward=24,
        dropout=0.0,
    )
    model = build_probe(config).eval()
    model.set_scalar_normalization([0.0, 0.0], [1.0, 1.0])
    projected = torch.randn(2, 4, 3, 8)
    rms = torch.ones(2, 4, 3)
    with torch.no_grad():
        token_mask = torch.ones(2, 3)
        base = model(projected, rms, token_keep_mask=token_mask)
        changed_input = projected.clone()
        changed_input[:, -1] += 5.0 * torch.randn_like(changed_input[:, -1])
        changed = model(changed_input, rms, token_keep_mask=token_mask)
    assert base["commitment"].shape == (2,)
    assert base["q_logits"].shape == (2, 4)
    assert torch.all((base["commitment"] >= 0.0) & (base["commitment"] <= 1.0))
    assert not torch.allclose(base["cls_hidden"], changed["cls_hidden"])


def test_nested_training_entrypoint_smoke(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("torch")
    projection = make_orthogonal_projection(16, 8, seed=42)
    spec = ShardSpec(
        n_layers=4,
        n_input_tokens=3,
        projected_dim=8,
        hidden_dim=16,
        projection_seed=42,
        projection_sha256=array_sha256(projection),
        input_kind="cot_last_tokens",
        label_suffix_token_ids=GPT_OSS_SUFFIX_TOKEN_IDS,
        model_name_or_path="synthetic/model",
        model_revision="test",
        mlx_lm_version="test",
    )
    beliefs = []
    prefix_id = 0
    shard_dir = tmp_path / "shards"
    for trajectory in range(6):
        trajectory_id = f"trajectory_{trajectory}"
        writer = TrajectoryShardWriter(shard_dir, trajectory_id, spec)
        for local_row in range(2):
            rng = np.random.default_rng(100 * trajectory + local_row)
            state = rng.standard_normal((4, 3, 8)).astype(np.float32)
            rms = np.sqrt(np.mean(state**2, axis=-1)).astype(np.float32)
            q = np.asarray(
                [0.1 + 0.02 * local_row, 0.2, 0.25 + 0.01 * trajectory, 0.45 - 0.01 * trajectory - 0.02 * local_row],
                dtype=np.float64,
            )
            q /= q.sum()
            writer.append(
                state,
                rms,
                {
                    "prefix_id": prefix_id,
                    "trajectory_id": trajectory_id,
                    "decision_id": f"{trajectory_id}::0",
                    "step_id": 0,
                    "prefix_token_index": local_row,
                    "input_kind": "cot_last_tokens",
                    "input_token_ids_json": "[1, 2, 3]",
                    "input_token_mask_json": "[1, 1, 1]",
                    "n_valid_input_tokens": 3,
                    "max_q_error": 0.0,
                    **{f"extracted_q_{action}": float(q[i]) for i, action in enumerate(ACTIONS)},
                },
            )
            beliefs.append(
                {
                    "prefix_id": prefix_id,
                    "trajectory_id": trajectory_id,
                    "decision_id": f"{trajectory_id}::0",
                    "progress": float((local_row + 1) / 2),
                    "prefix_token_index": local_row,
                    "num_cot_tokens": 2,
                    "strict_before_any_action_mention": local_row == 0,
                    "strict_before_final_action_mention": True,
                    "agent_action": "RIGHT",
                    "top_action": ACTIONS[int(np.argmax(q))],
                    **{f"q_{action}": float(q[i]) for i, action in enumerate(ACTIONS)},
                }
            )
            prefix_id += 1
        writer.finalize()
    index_path = tmp_path / "activation_index.parquet"
    belief_path = tmp_path / "beliefs.parquet"
    split_path = tmp_path / "splits.json"
    output_dir = tmp_path / "training"
    pd.DataFrame(index_rows_from_shards(shard_dir)).to_parquet(index_path, index=False)
    pd.DataFrame(beliefs).to_parquet(belief_path, index=False)
    split_path.write_text(
        '[{"split_id": 0, "test_groups": ["trajectory_5"]}]\n'
    )
    script = REPO_ROOT / "scripts/conceptual_belief_space/77_train_forced_suffix_commitment_probe.py"
    command = [
        sys.executable,
        str(script),
        "--activation_index",
        str(index_path),
        "--beliefs",
        str(belief_path),
        "--splits",
        str(split_path),
        "--output_dir",
        str(output_dir),
        "--encoder_layers",
        "1",
        "--dropouts",
        "0",
        "--lambda_q",
        "0",
        "--d_model",
        "12",
        "--token_dim",
        "4",
        "--n_heads",
        "3",
        "--dim_feedforward",
        "24",
        "--max_epochs",
        "1",
        "--patience",
        "1",
        "--batch_size",
        "4",
        "--baselines",
        "none",
        "--bootstrap_samples",
        "0",
        "--occlusion_max_rows",
        "0",
        "--limit_folds",
        "1",
        "--force",
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    predictions = pd.read_parquet(output_dir / "oof_predictions.parquet")
    assert len(predictions) == 2
    assert predictions["prefix_id"].is_unique
    assert set(predictions["trajectory_id"]) == {"trajectory_5"}
    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "report.md").exists()

    checkpoint = output_dir / "models/bidirectional_probe_split0.pt"
    checkpoint_mtime = checkpoint.stat().st_mtime_ns
    fold_dir = output_dir / "fold_artifacts/split0"
    (fold_dir / "fold_complete.json").unlink()
    (fold_dir / "predictions_bidirectional_transformer.parquet").unlink()
    resume_command = [argument for argument in command if argument != "--force"] + ["--resume"]
    resumed = subprocess.run(
        resume_command, cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )
    assert "restored selected Transformer checkpoint" in resumed.stdout
    assert checkpoint.stat().st_mtime_ns == checkpoint_mtime
    assert (fold_dir / "predictions_bidirectional_transformer.parquet").exists()

    resumed_again = subprocess.run(
        resume_command, cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )
    assert "restored complete fold artifacts" in resumed_again.stdout
    assert checkpoint.stat().st_mtime_ns == checkpoint_mtime
