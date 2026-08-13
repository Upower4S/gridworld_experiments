"""Shared data and model utilities for the CoT-window layer probe.

The extraction runtime is intentionally kept out of this module so that shard
validation, model tests, and probe training do not require MLX.  Activations
are stored as raw BF16 payloads with JSON sidecars and loaded lazily.

The forced suffix is used only to define the prediction label.  Probe inputs
are residual states from genuine chain-of-thought tokens that precede it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
ANALYSIS_HEADER = "<|channel|>analysis<|message|>"
FORCED_SUFFIX = '<|end|><|start|>assistant<|channel|>final<|message|>{\n  "action": "'
EXPECTED_SUFFIX_TOKENS = 12
DEFAULT_COT_WINDOW_TOKENS = 12
GPT_OSS_SUFFIX_TOKEN_IDS = (
    200007,
    200006,
    173781,
    200005,
    17196,
    200008,
    745,
    220,
    392,
    1976,
    1243,
    392,
)
SHARD_FORMAT = "cot_window_projected_bf16_v2"
LEGACY_SHARD_FORMATS = frozenset({"forced_suffix_projected_bf16_v1"})


def encode_without_special(tokenizer: Any, text: str) -> list[int]:
    """Tokenize text without automatically inserted special tokens."""
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer.encode(text)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    return [int(token_id) for token_id in encoded]


def forced_suffix_token_ids(
    tokenizer: Any,
    *,
    expected_count: int = EXPECTED_SUFFIX_TOKENS,
    expected_ids: Sequence[int] | None = GPT_OSS_SUFFIX_TOKEN_IDS,
) -> list[int]:
    """Return and validate the fixed forced-answer suffix token IDs."""
    ids = encode_without_special(tokenizer, FORCED_SUFFIX)
    if len(ids) != int(expected_count):
        raise ValueError(
            f"forced suffix encoded to {len(ids)} tokens, expected {expected_count}; "
            "the tokenizer/model revision is incompatible with this dataset"
        )
    if expected_ids is not None and ids != [int(token_id) for token_id in expected_ids]:
        raise ValueError(
            "forced suffix token IDs do not match the canonical GPT-OSS suffix; "
            "the tokenizer/model revision is incompatible with this dataset"
        )
    return ids


def validate_context_suffix(full_context_ids: Sequence[int], suffix_ids: Sequence[int]) -> list[int]:
    """Validate a complete scoring context and return its prefix-only IDs."""
    full = [int(x) for x in full_context_ids]
    suffix = [int(x) for x in suffix_ids]
    if not suffix or len(full) <= len(suffix):
        raise ValueError("scoring context must contain prefix IDs before the forced suffix")
    if full[-len(suffix) :] != suffix:
        raise ValueError("full scoring context does not end with the canonical forced suffix IDs")
    return full[: -len(suffix)]


def aligned_cot_window(
    prefix_ids: Sequence[int],
    analysis_token_ids: Sequence[int],
    *,
    window_size: int = DEFAULT_COT_WINDOW_TOKENS,
) -> tuple[list[int], list[int], list[int]]:
    """Return an exact right-aligned CoT window and its storage mask.

    ``prefix_ids`` contains prompt plus generated analysis, but the selected
    window is required to lie wholly inside ``analysis_token_ids``.  The first
    result contains only real tokens and is suitable for model inference.  The
    second and third results are fixed-width token IDs (``-1`` for padding) and
    a 0/1 validity mask suitable for shard metadata.
    """
    prefix = [int(token_id) for token_id in prefix_ids]
    analysis = [int(token_id) for token_id in analysis_token_ids]
    width = int(window_size)
    if width <= 0:
        raise ValueError("CoT window size must be positive")
    if not analysis:
        raise ValueError("cannot construct a CoT window from an empty analysis prefix")
    if len(prefix) < len(analysis) or prefix[-len(analysis) :] != analysis:
        raise ValueError("stored analysis token IDs do not exactly align with the scoring prefix")
    selected = analysis[-width:]
    padding = width - len(selected)
    padded_ids = [-1] * padding + selected
    valid_mask = [0] * padding + [1] * len(selected)
    return selected, padded_ids, valid_mask


def longest_common_prefix_len(left: Sequence[int], right: Sequence[int]) -> int:
    """Length of the shared leading token sequence."""
    n = min(len(left), len(right))
    i = 0
    while i < n and int(left[i]) == int(right[i]):
        i += 1
    return i


def clone_cache(cache: Sequence[Any]) -> list[Any]:
    """Clone an MLX-style cache without importing MLX."""
    return [entry.from_state(entry.state, entry.meta_state) for entry in cache]


def trim_cache(cache: Sequence[Any], n_tokens: int) -> None:
    """Trim tokens from every layer cache in place."""
    if int(n_tokens) <= 0:
        return
    for entry in cache:
        entry.trim(int(n_tokens))


def stable_action_probs(action_scores: Sequence[float]) -> np.ndarray:
    """Normalize four action scores over the canonical action set."""
    scores = np.asarray(action_scores, dtype=np.float64)
    if scores.shape != (len(ACTIONS),):
        raise ValueError(f"expected {len(ACTIONS)} action scores, got {scores.shape}")
    shifted = scores - np.max(scores)
    probs = np.exp(shifted)
    probs /= probs.sum()
    return probs.astype(np.float64)


def entropy_commitment(q: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """Compute 1 - normalized categorical entropy for four-action beliefs."""
    arr = np.asarray(q, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != len(ACTIONS):
        raise ValueError(f"q must have shape [N,{len(ACTIONS)}], got {arr.shape}")
    arr = np.clip(arr, eps, None)
    arr /= arr.sum(axis=1, keepdims=True)
    entropy = -(arr * np.log(arr)).sum(axis=1)
    return (1.0 - entropy / math.log(len(ACTIONS))).astype(np.float32)


def make_orthogonal_projection(hidden_dim: int, projected_dim: int, seed: int = 42) -> np.ndarray:
    """Create a scaled random orthogonal projection with deterministic seed."""
    hidden_dim = int(hidden_dim)
    projected_dim = int(projected_dim)
    if hidden_dim <= 0 or projected_dim <= 0 or projected_dim > hidden_dim:
        raise ValueError("projection dimensions must satisfy 0 < projected_dim <= hidden_dim")
    rng = np.random.default_rng(int(seed))
    gaussian = rng.standard_normal((hidden_dim, projected_dim)).astype(np.float32)
    q, r = np.linalg.qr(gaussian, mode="reduced")
    # Fix QR sign ambiguity so hashes are stable for a fixed numerical runtime.
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0).astype(np.float32)
    q = q.astype(np.float32) * signs[None, :]
    q *= math.sqrt(hidden_dim / projected_dim)
    return np.ascontiguousarray(q, dtype=np.float32)


def array_sha256(array: np.ndarray) -> str:
    """Hash an array including dtype and shape."""
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode())
    digest.update(json.dumps(list(arr.shape)).encode())
    digest.update(arr.view(np.uint8))
    return digest.hexdigest()


def float32_to_bf16_uint16(values: np.ndarray) -> np.ndarray:
    """Round float32 values to BF16 represented as little-endian uint16."""
    arr = np.ascontiguousarray(values, dtype="<f4")
    bits = arr.view("<u4")
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding_bias) >> np.uint32(16)).astype("<u2")


def bf16_uint16_to_float32(values: np.ndarray) -> np.ndarray:
    """Convert little-endian BF16 uint16 payloads to float32."""
    bits = np.asarray(values, dtype="<u2").astype("<u4") << np.uint32(16)
    return bits.view("<f4")


def adjacent_layer_delta(projected: np.ndarray) -> np.ndarray:
    """Return adjacent physical-layer deltas with a zero first layer."""
    x = np.asarray(projected)
    if x.ndim < 2:
        raise ValueError("projected activations need a layer axis")
    out = np.zeros_like(x)
    out[..., 1:, :, :] = x[..., 1:, :, :] - x[..., :-1, :, :]
    return out


def shard_slug(trajectory_id: str) -> str:
    """Filesystem-safe stable shard name for a trajectory."""
    readable = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(trajectory_id))
    digest = hashlib.sha1(str(trajectory_id).encode()).hexdigest()[:10]
    return f"{readable[:96]}_{digest}"


@dataclass(frozen=True)
class ShardSpec:
    n_layers: int
    n_input_tokens: int
    projected_dim: int
    hidden_dim: int
    projection_seed: int
    projection_sha256: str
    input_kind: str
    label_suffix_token_ids: tuple[int, ...]
    model_name_or_path: str
    model_revision: str | None = None
    mlx_lm_version: str | None = None


class TrajectoryShardWriter:
    """Stream one trajectory into atomic projected-state and RMS shards."""

    def __init__(self, output_dir: str | Path, trajectory_id: str, spec: ShardSpec):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_id = str(trajectory_id)
        self.spec = spec
        self.slug = shard_slug(self.trajectory_id)
        self.state_path = self.output_dir / f"{self.slug}.bf16"
        self.rms_path = self.output_dir / f"{self.slug}.rms.f16"
        self.meta_path = self.output_dir / f"{self.slug}.json"
        self.partial_state = self.output_dir / f"{self.slug}.bf16.partial"
        self.partial_rms = self.output_dir / f"{self.slug}.rms.f16.partial"
        self._state_handle = self.partial_state.open("wb")
        self._rms_handle = self.partial_rms.open("wb")
        self.rows: list[dict[str, Any]] = []
        self.closed = False

    def append(self, projected: np.ndarray, rms: np.ndarray, metadata: Mapping[str, Any]) -> int:
        expected_state = (
            int(self.spec.n_layers),
            int(self.spec.n_input_tokens),
            int(self.spec.projected_dim),
        )
        expected_rms = expected_state[:2]
        state = np.asarray(projected, dtype=np.float32)
        scale = np.asarray(rms, dtype=np.float32)
        if state.shape != expected_state:
            raise ValueError(f"projected row shape {state.shape} != {expected_state}")
        if scale.shape != expected_rms:
            raise ValueError(f"RMS row shape {scale.shape} != {expected_rms}")
        if not np.isfinite(state).all() or not np.isfinite(scale).all():
            raise ValueError("activation shard row contains NaN or infinity")
        row_index = len(self.rows)
        float32_to_bf16_uint16(state).tofile(self._state_handle)
        np.asarray(scale, dtype="<f2").tofile(self._rms_handle)
        row = dict(metadata)
        row["shard_row"] = int(row_index)
        self.rows.append(row)
        return row_index

    def finalize(self) -> Path:
        if self.closed:
            return self.meta_path
        self._state_handle.close()
        self._rms_handle.close()
        n_rows = len(self.rows)
        expected_state_bytes = (
            n_rows * self.spec.n_layers * self.spec.n_input_tokens * self.spec.projected_dim * 2
        )
        expected_rms_bytes = n_rows * self.spec.n_layers * self.spec.n_input_tokens * 2
        if self.partial_state.stat().st_size != expected_state_bytes:
            raise RuntimeError("projected shard byte count does not match its declared shape")
        if self.partial_rms.stat().st_size != expected_rms_bytes:
            raise RuntimeError("RMS shard byte count does not match its declared shape")
        os.replace(self.partial_state, self.state_path)
        os.replace(self.partial_rms, self.rms_path)
        metadata = {
            "format": SHARD_FORMAT,
            "trajectory_id": self.trajectory_id,
            "n_rows": n_rows,
            "state_shape": [n_rows, self.spec.n_layers, self.spec.n_input_tokens, self.spec.projected_dim],
            "rms_shape": [n_rows, self.spec.n_layers, self.spec.n_input_tokens],
            "state_dtype": "bfloat16",
            "rms_dtype": "float16",
            "state_file": self.state_path.name,
            "rms_file": self.rms_path.name,
            "spec": asdict(self.spec),
            "rows": self.rows,
        }
        temp_meta = self.meta_path.with_suffix(".json.partial")
        temp_meta.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        os.replace(temp_meta, self.meta_path)
        self.closed = True
        return self.meta_path

    def abort(self) -> None:
        if not self.closed:
            self._state_handle.close()
            self._rms_handle.close()
        self.closed = True


def load_shard_metadata(path: str | Path, *, validate_files: bool = True) -> dict[str, Any]:
    """Load and optionally validate a completed trajectory shard sidecar."""
    path = Path(path)
    metadata = json.loads(path.read_text())
    if metadata.get("format") not in {SHARD_FORMAT, *LEGACY_SHARD_FORMATS}:
        raise ValueError(f"unsupported shard format in {path}: {metadata.get('format')}")
    n_rows, n_layers, n_input, projected_dim = [int(x) for x in metadata["state_shape"]]
    if metadata["rms_shape"] != [n_rows, n_layers, n_input]:
        raise ValueError(f"RMS shape disagrees with state shape in {path}")
    if len(metadata.get("rows", [])) != n_rows:
        raise ValueError(f"row metadata count disagrees with state shape in {path}")
    if validate_files:
        state_path = path.parent / metadata["state_file"]
        rms_path = path.parent / metadata["rms_file"]
        if not state_path.exists() or state_path.stat().st_size != n_rows * n_layers * n_input * projected_dim * 2:
            raise ValueError(f"missing or malformed projected state payload for {path}")
        if not rms_path.exists() or rms_path.stat().st_size != n_rows * n_layers * n_input * 2:
            raise ValueError(f"missing or malformed RMS payload for {path}")
    return metadata


def index_rows_from_shards(shard_dir: str | Path) -> list[dict[str, Any]]:
    """Build global index records from every completed shard sidecar."""
    shard_dir = Path(shard_dir)
    rows: list[dict[str, Any]] = []
    for meta_path in sorted(shard_dir.glob("*.json")):
        metadata = load_shard_metadata(meta_path)
        shape = metadata["state_shape"]
        for row in metadata["rows"]:
            out = dict(row)
            out.update(
                {
                    "state_path": str((meta_path.parent / metadata["state_file"]).resolve()),
                    "rms_path": str((meta_path.parent / metadata["rms_file"]).resolve()),
                    "n_layers": int(shape[1]),
                    "n_input_tokens": int(shape[2]),
                    "projected_dim": int(shape[3]),
                    "shard_n_rows": int(shape[0]),
                    "shard_metadata_path": str(meta_path.resolve()),
                }
            )
            out.setdefault(
                "input_kind",
                metadata.get("spec", {}).get("input_kind", "forced_suffix_legacy"),
            )
            rows.append(out)
    return rows


class ProjectedShardStore:
    """LRU-backed random access to projected activation shards."""

    def __init__(self, *, max_open: int = 16):
        self.max_open = int(max_open)
        self._states: OrderedDict[tuple[str, tuple[int, ...]], np.memmap] = OrderedDict()
        self._rms: OrderedDict[tuple[str, tuple[int, ...]], np.memmap] = OrderedDict()

    def _open(self, cache: OrderedDict, key: tuple[str, tuple[int, ...]], dtype: str) -> np.memmap:
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        path, shape = key
        mmap = np.memmap(path, mode="r", dtype=dtype, shape=shape)
        cache[key] = mmap
        if len(cache) > self.max_open:
            cache.popitem(last=False)
        return mmap

    def load_row(self, row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        n_input_tokens = int(row.get("n_input_tokens", row.get("n_suffix_tokens", -1)))
        if n_input_tokens <= 0:
            raise ValueError("activation index row is missing a positive input-token dimension")
        state_shape = (
            int(row["shard_n_rows"]),
            int(row["n_layers"]),
            n_input_tokens,
            int(row["projected_dim"]),
        )
        rms_shape = state_shape[:3]
        state_map = self._open(self._states, (str(row["state_path"]), state_shape), "<u2")
        rms_map = self._open(self._rms, (str(row["rms_path"]), rms_shape), "<f2")
        idx = int(row["shard_row"])
        state = bf16_uint16_to_float32(np.asarray(state_map[idx])).astype(np.float32, copy=False)
        rms = np.asarray(rms_map[idx], dtype=np.float32)
        return state, rms

    def load_rows(self, rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized read for rows from one shard.

        Training batches are deliberately shard-local. Reading and converting the
        complete batch at once avoids reopening memory maps and performing one
        BF16 conversion allocation per example.
        """
        if not rows:
            raise ValueError("load_rows requires at least one row")
        first = rows[0]
        n_input_tokens = int(first.get("n_input_tokens", first.get("n_suffix_tokens", -1)))
        if n_input_tokens <= 0:
            raise ValueError("activation index row is missing a positive input-token dimension")
        state_shape = (
            int(first["shard_n_rows"]),
            int(first["n_layers"]),
            n_input_tokens,
            int(first["projected_dim"]),
        )
        state_path = str(first["state_path"])
        rms_path = str(first["rms_path"])
        for row in rows[1:]:
            candidate_input_tokens = int(row.get("n_input_tokens", row.get("n_suffix_tokens", -1)))
            candidate_shape = (
                int(row["shard_n_rows"]),
                int(row["n_layers"]),
                candidate_input_tokens,
                int(row["projected_dim"]),
            )
            if (
                str(row["state_path"]) != state_path
                or str(row["rms_path"]) != rms_path
                or candidate_shape != state_shape
            ):
                raise ValueError("load_rows batches must contain rows from exactly one shard")
        state_map = self._open(self._states, (state_path, state_shape), "<u2")
        rms_shape = state_shape[:3]
        rms_map = self._open(self._rms, (rms_path, rms_shape), "<f2")
        indices = np.fromiter((int(row["shard_row"]) for row in rows), dtype=np.int64, count=len(rows))
        state = bf16_uint16_to_float32(np.asarray(state_map[indices])).astype(np.float32, copy=False)
        rms = np.asarray(rms_map[indices], dtype=np.float32)
        return state, rms


@dataclass(frozen=True)
class ProbeConfig:
    projected_dim: int = 512
    n_layers: int = 24
    n_input_tokens: int = 12
    token_dim: int = 64
    d_model: int = 384
    encoder_layers: int = 2
    n_heads: int = 6
    dim_feedforward: int = 1536
    dropout: float = 0.1
    use_deltas: bool = True


def build_probe(config: ProbeConfig):
    """Construct the bidirectional PyTorch probe lazily."""
    import torch
    import torch.nn as nn

    class CotWindowLayerEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.state_norm = nn.LayerNorm(config.projected_dim)
            self.delta_norm = nn.LayerNorm(config.projected_dim)
            self.state_proj = nn.Linear(config.projected_dim, config.token_dim)
            self.delta_proj = nn.Linear(config.projected_dim, config.token_dim)
            self.scalar_proj = nn.Linear(2, config.token_dim)
            self.token_position_embedding = nn.Parameter(torch.zeros(config.n_input_tokens, config.token_dim))
            nn.init.normal_(self.token_position_embedding, std=0.02)
            self.layer_proj = nn.Sequential(
                nn.LayerNorm(config.n_input_tokens * config.token_dim),
                nn.Linear(config.n_input_tokens * config.token_dim, config.d_model),
                nn.GELU(),
            )
            self.register_buffer("scalar_mean", torch.zeros(2), persistent=True)
            self.register_buffer("scalar_std", torch.ones(2), persistent=True)

        def set_scalar_normalization(self, mean: Sequence[float], std: Sequence[float]) -> None:
            mean_t = torch.as_tensor(mean, dtype=self.scalar_mean.dtype, device=self.scalar_mean.device)
            std_t = torch.as_tensor(std, dtype=self.scalar_std.dtype, device=self.scalar_std.device)
            if tuple(mean_t.shape) != (2,) or tuple(std_t.shape) != (2,):
                raise ValueError("scalar normalization must contain state and delta statistics")
            self.scalar_mean.copy_(mean_t)
            self.scalar_std.copy_(std_t.clamp_min(1e-6))

        def forward(self, projected, rms, layer_keep_mask=None, token_keep_mask=None):
            if projected.ndim != 4:
                raise ValueError("projected input must have shape [B,L,A,P]")
            expected = (config.n_layers, config.n_input_tokens, config.projected_dim)
            if tuple(projected.shape[1:]) != expected:
                raise ValueError(f"projected input shape {tuple(projected.shape[1:])} != {expected}")
            if tuple(rms.shape[1:]) != expected[:2]:
                raise ValueError("RMS input must have shape [B,L,A]")
            delta = torch.zeros_like(projected)
            delta[:, 1:] = projected[:, 1:] - projected[:, :-1]
            state_features = self.state_proj(self.state_norm(projected))
            if config.use_deltas:
                delta_features = self.delta_proj(self.delta_norm(delta))
                layer_mask = torch.ones((1, config.n_layers, 1, 1), device=projected.device, dtype=projected.dtype)
                layer_mask[:, 0] = 0.0
                delta_features = delta_features * layer_mask
            else:
                delta_features = torch.zeros_like(state_features)
            state_log_rms = torch.log(rms.float().clamp_min(1e-8))
            delta_rms = torch.sqrt(torch.mean(delta.float().square(), dim=-1).clamp_min(1e-12))
            delta_log_rms = torch.log(delta_rms.clamp_min(1e-8))
            scalars = torch.stack([state_log_rms, delta_log_rms], dim=-1)
            scalars = (scalars - self.scalar_mean) / self.scalar_std
            encoded = state_features + delta_features + self.scalar_proj(scalars.to(projected.dtype))
            encoded = encoded + self.token_position_embedding.view(
                1, 1, config.n_input_tokens, config.token_dim
            )
            encoded = torch.nn.functional.gelu(encoded)
            if layer_keep_mask is not None:
                encoded = encoded * layer_keep_mask[:, :, None, None].to(encoded.dtype)
            if token_keep_mask is not None:
                encoded = encoded * token_keep_mask[:, None, :, None].to(encoded.dtype)
            return self.layer_proj(encoded.flatten(start_dim=2))

    class BidirectionalCotWindowProbe(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.input_encoder = CotWindowLayerEncoder()
            self.layer_embedding = nn.Parameter(torch.zeros(config.n_layers, config.d_model))
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
            nn.init.normal_(self.layer_embedding, std=0.02)
            nn.init.normal_(self.cls_token, std=0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.encoder_layers)
            self.output_norm = nn.LayerNorm(config.d_model)
            self.commitment_head = nn.Linear(config.d_model, 1)
            self.q_head = nn.Linear(config.d_model, len(ACTIONS))

        def set_scalar_normalization(self, mean: Sequence[float], std: Sequence[float]) -> None:
            self.input_encoder.set_scalar_normalization(mean, std)

        def forward(self, projected, rms, layer_keep_mask=None, token_keep_mask=None):
            layer_tokens = self.input_encoder(projected, rms, layer_keep_mask, token_keep_mask)
            layer_tokens = layer_tokens + self.layer_embedding.unsqueeze(0)
            cls = self.cls_token.expand(projected.shape[0], -1, -1)
            sequence = torch.cat([cls, layer_tokens], dim=1)
            # Deliberately no attention mask: every physical layer is visible to CLS.
            hidden = self.transformer(sequence)
            cls_hidden = self.output_norm(hidden[:, 0])
            return {
                "commitment": torch.sigmoid(self.commitment_head(cls_hidden).squeeze(-1)),
                "q_logits": self.q_head(cls_hidden),
                "cls_hidden": cls_hidden,
            }

    return BidirectionalCotWindowProbe()


def build_mlp_baseline(config: ProbeConfig, *, mode: str, layer_index: int | None = None):
    """Build final/single/mean/concat baselines with the shared token encoder."""
    import torch
    import torch.nn as nn

    if mode not in {"final", "single", "mean", "concat"}:
        raise ValueError(f"unknown MLP baseline mode: {mode}")
    if mode == "single" and layer_index is None:
        raise ValueError("single-layer baseline requires layer_index")

    probe = build_probe(config)
    encoder = probe.input_encoder
    input_dim = config.n_layers * config.d_model if mode == "concat" else config.d_model
    hidden_dim = config.d_model if mode != "concat" else 2 * config.d_model

    class MLPBaseline(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.input_encoder = encoder
            self.mode = mode
            self.layer_index = layer_index
            self.mlp = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden_dim, config.d_model),
                nn.GELU(),
            )
            self.output_norm = nn.LayerNorm(config.d_model)
            self.commitment_head = nn.Linear(config.d_model, 1)
            self.q_head = nn.Linear(config.d_model, len(ACTIONS))

        def set_scalar_normalization(self, mean: Sequence[float], std: Sequence[float]) -> None:
            self.input_encoder.set_scalar_normalization(mean, std)

        def forward(self, projected, rms, layer_keep_mask=None, token_keep_mask=None):
            tokens = self.input_encoder(projected, rms, layer_keep_mask, token_keep_mask)
            if self.mode == "final":
                pooled = tokens[:, -1]
            elif self.mode == "single":
                pooled = tokens[:, int(self.layer_index)]
            elif self.mode == "mean":
                pooled = tokens.mean(dim=1)
            else:
                pooled = tokens.flatten(start_dim=1)
            hidden = self.output_norm(self.mlp(pooled))
            return {
                "commitment": torch.sigmoid(self.commitment_head(hidden).squeeze(-1)),
                "q_logits": self.q_head(hidden),
                "cls_hidden": hidden,
            }

    return MLPBaseline()


def scalar_feature_stats(projected_batches: Iterable[np.ndarray], rms_batches: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Fit global log-RMS scalar normalization from train-only batches."""
    total = np.zeros(2, dtype=np.float64)
    total_sq = np.zeros(2, dtype=np.float64)
    count = 0
    for projected, rms in zip(projected_batches, rms_batches, strict=True):
        x = np.asarray(projected, dtype=np.float32)
        scale = np.asarray(rms, dtype=np.float32)
        delta = adjacent_layer_delta(x)
        delta_rms = np.sqrt(np.maximum(np.mean(delta.astype(np.float64) ** 2, axis=-1), 1e-12))
        values = np.stack(
            [np.log(np.maximum(scale, 1e-8)), np.log(np.maximum(delta_rms, 1e-8))],
            axis=-1,
        ).reshape(-1, 2)
        finite = np.isfinite(values).all(axis=1)
        values = values[finite]
        total += values.sum(axis=0)
        total_sq += (values**2).sum(axis=0)
        count += len(values)
    if count == 0:
        raise ValueError("cannot fit scalar normalization from zero finite values")
    mean = total / count
    var = np.maximum(total_sq / count - mean**2, 1e-12)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)
